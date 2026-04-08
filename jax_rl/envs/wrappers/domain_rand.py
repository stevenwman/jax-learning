"""Domain Randomization wrapper — unified episode boundary + DR.

Replaces AutoResetWrapper + EpisodeWrapper + DomainRandomizationVmapWrapper
with a single wrapper that handles:
  - Vectorization (vmap)
  - Episode length tracking + truncation
  - Auto-reset on done (two modes: per_step or syncd)
  - Per-episode domain randomization (model + runtime)
  - Clean state.info reset on episode boundary

Two reset modes:
  - "per_step": compute reset for all envs every step, select via where_done.
    Fresh IC + clean state every episode. Pays O(N) reset cost per step.
  - "syncd": all envs step for episode_length, mask post-done rewards,
    batch-reset all envs together. No per-step reset overhead. Wastes
    compute on post-done envs but consistently faster throughput.

With no DR spec: behaves as an improved AutoResetWrapper.
With DR spec: adds per-episode physics randomization.
"""

import contextlib
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import jax
from jax import numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env

from jax_rl.envs.wrappers.training import Wrapper


# ═════════════════════════════════════════════════════════════════════
# DRSpec — declarative domain randomization specification
# ═════════════════════════════════════════════════════════════════════

@dataclass
class DRSpec:
    """Single domain randomization parameter specification."""

    name: str
    """Human-readable name (e.g., "friction", "motor_strength")."""

    type: Literal["model", "runtime"]
    """Where randomization is applied.
    - "model": modifies mjx.Model fields.
    - "runtime": produces values stored in state.info[name].
    """

    # ── Model DR fields (type="model" only) ─────────────────────
    field: Optional[str] = None
    """MJX model field name (e.g., "geom_friction", "dof_damping")."""

    column: Optional[int] = None
    """If field is 2D, which column to randomize. None = entire field."""

    indices: Optional[tuple[int, int]] = None
    """Row slice to randomize as (start, stop). None = all rows."""

    operation: Literal["set", "multiply", "add"] = "multiply"
    """How to apply: multiply (scale), set (absolute), add (offset)."""

    # ── Range ───────────────────────────────────────────────────
    min: float = 0.0
    max: float = 1.0

    per_element: bool = False
    """False: one sample broadcast. True: independent sample per element."""

    # ── Runtime DR only (type="runtime") ────────────────────────
    shape: Optional[tuple] = None
    """Output shape stored in state.info[name]. None = scalar ()."""

    description: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════
# DomainRandWrapper
# ═════════════════════════════════════════════════════════════════════

class DomainRandWrapper(Wrapper):
    """Unified episode boundary manager with optional domain randomization.

    Args:
        env: Unwrapped environment (not vmapped).
        episode_length: Max steps per episode (truncation).
        mode: "per_step" or "syncd".
    """

    _KEY = '_dr'

    def __init__(
        self,
        env: Any,
        episode_length: int = 1000,
        mode: Literal["per_step", "syncd"] = "syncd",
    ):
        super().__init__(env)
        self.episode_length = episode_length
        self.mode = mode

        if hasattr(env, 'get_domain_randomization_spec'):
            self.dr_specs = env.get_domain_randomization_spec()
        else:
            self.dr_specs = []

        self._model_specs = [s for s in self.dr_specs if s.type == "model"]
        self._runtime_specs = [s for s in self.dr_specs if s.type == "runtime"]

    # ── Reset ───────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Initial reset — called once at training start."""
        rng_key = jax.vmap(jax.random.split)(rng)
        rng, key = rng_key[:, 0], rng_key[:, 1]

        if self._model_specs:
            dr_model, in_axes = self._build_dr_model(key)
            def _reset_with_model(mjx_model, rng):
                with self._swap_model(mjx_model) as v_env:
                    return v_env.reset(rng)
            state = jax.vmap(_reset_with_model, in_axes=[in_axes, 0])(dr_model, key)
        else:
            state = jax.vmap(self.env.reset)(key)

        # Apply runtime DR to state.info
        if self._runtime_specs:
            state = self._apply_runtime_dr(state, key)

        state.info[f'{self._KEY}_rng'] = rng
        state.info[f'{self._KEY}_steps'] = jp.zeros(rng.shape[0])
        state.info[f'{self._KEY}_episode_done'] = jp.zeros(rng.shape[0])
        state.info[f'{self._KEY}_done_count'] = jp.zeros(rng.shape[0])

        # EpisodeWrapper-compatible keys (training loop reads these)
        state.info['steps'] = jp.zeros(rng.shape[0])
        state.info['truncation'] = jp.zeros(rng.shape[0])

        # Episode metrics
        episode_metrics = {
            'sum_reward': jp.zeros(rng.shape[0]),
            'length': jp.zeros(rng.shape[0]),
        }
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jp.zeros(rng.shape[0])
        state.info[f'{self._KEY}_episode_metrics'] = episode_metrics

        return state

    # ── Step (dispatches to mode) ───────────────────────────────

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self.mode == "per_step":
            return self._step_per_step(state, action)
        else:
            return self._step_syncd(state, action)

    # ── Per-step reset mode ─────────────────────────────────────

    def _step_per_step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Save and strip wrapper state so tree_map doesn't see mismatched keys
        drv2_rng = state.info.pop(f'{self._KEY}_rng')
        drv2_steps = state.info.pop(f'{self._KEY}_steps')
        drv2_done_count = state.info.pop(f'{self._KEY}_done_count')
        drv2_episode_done = state.info.pop(f'{self._KEY}_episode_done')
        drv2_metrics = state.info.pop(f'{self._KEY}_episode_metrics')
        state.info.pop('steps', None)
        state.info.pop('truncation', None)

        # Prepare reset for ALL envs (JAX both-paths)
        rng_key = jax.vmap(jax.random.split)(drv2_rng)
        reset_rng, next_rng = rng_key[:, 0], rng_key[:, 1]

        if self._model_specs:
            # Build per-env randomized model, vmap reset/step over it
            dr_model, in_axes = self._build_dr_model(reset_rng)

            def _reset_with_model(mjx_model, rng):
                with self._swap_model(mjx_model) as v_env:
                    return v_env.reset(rng)

            def _step_with_model(mjx_model, s, a):
                with self._swap_model(mjx_model) as v_env:
                    return v_env.step(s, a)

            reset_state = jax.vmap(_reset_with_model, in_axes=[in_axes, 0])(
                dr_model, reset_rng
            )
        else:
            reset_state = jax.vmap(self.env.reset)(reset_rng)

        # Apply runtime DR to reset state
        if self._runtime_specs:
            reset_state = self._apply_runtime_dr(reset_state, reset_rng)

        # Step all envs
        state = state.replace(done=jp.zeros_like(state.done))
        if self._model_specs:
            state = jax.vmap(_step_with_model, in_axes=[in_axes, 0, 0])(
                dr_model, state, action
            )
        else:
            state = jax.vmap(self.env.step)(state, action)

        # Compute done
        steps = drv2_steps + 1
        truncation = (steps >= self.episode_length).astype(float)
        done = jp.maximum(state.done, truncation)

        # where_done: select reset for done envs
        data = jax.tree.map(
            lambda r, s: _where_done(done, r, s), reset_state.data, state.data
        )
        obs = jax.tree.map(
            lambda r, s: _where_done(done, r, s), reset_state.obs, state.obs
        )
        info = jax.tree.map(
            lambda r, s: _where_done(done, r, s), reset_state.info, state.info
        )

        # Update episode metrics
        drv2_metrics['sum_reward'] = jp.where(
            drv2_episode_done, 0.0, drv2_metrics['sum_reward']
        ) + state.reward
        drv2_metrics['length'] = jp.where(
            drv2_episode_done, 0.0, drv2_metrics['length']
        ) + 1

        # Truncation: done by episode length, not by env termination
        truncation = jp.where(
            truncation > 0, 1.0 - state.done, jp.zeros_like(done)
        )

        # Re-attach wrapper state
        info[f'{self._KEY}_rng'] = next_rng
        info[f'{self._KEY}_steps'] = jp.where(done, 0.0, steps)
        info[f'{self._KEY}_done_count'] = drv2_done_count + done
        info[f'{self._KEY}_episode_done'] = done
        info[f'{self._KEY}_episode_metrics'] = drv2_metrics
        info['steps'] = jp.where(done, 0.0, steps)
        info['truncation'] = truncation

        return state.replace(data=data, obs=obs, done=done, info=info)

    # ── Sync'd reset mode ───────────────────────────────────────

    def _step_syncd(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        was_done = state.info[f'{self._KEY}_episode_done']

        # Step all envs (even done ones)
        state = jax.vmap(self.env.step)(state, action)

        # Update step count and done status
        steps = state.info[f'{self._KEY}_steps'] + 1
        truncation = (steps >= self.episode_length).astype(float)
        episode_done = jp.maximum(jp.maximum(was_done, state.done), truncation)

        # Mask reward for post-done steps
        reward = jp.where(was_done, 0.0, state.reward)

        # Update episode metrics (only for non-done steps)
        metrics = state.info[f'{self._KEY}_episode_metrics']
        metrics['sum_reward'] += jp.where(was_done, 0.0, state.reward)
        metrics['length'] += jp.where(was_done, 0.0, 1.0)

        # Truncation: done by episode length, not by env termination
        trunc_flag = jp.where(
            truncation > 0, 1.0 - state.done, jp.zeros_like(episode_done)
        ) * (1.0 - was_done)  # only count first truncation

        state.info[f'{self._KEY}_steps'] = steps
        state.info[f'{self._KEY}_episode_done'] = episode_done
        state.info[f'{self._KEY}_done_count'] = (
            state.info[f'{self._KEY}_done_count'] +
            jp.maximum(state.done, truncation) * (1.0 - was_done)
        )
        state.info[f'{self._KEY}_episode_metrics'] = metrics
        state.info['steps'] = steps
        state.info['truncation'] = trunc_flag

        return state.replace(reward=reward, done=episode_done)

    def batch_reset(self, state: mjx_env.State) -> mjx_env.State:
        """Batch-reset all envs. Call between rollouts in syncd mode."""
        rng_key = jax.vmap(jax.random.split)(state.info[f'{self._KEY}_rng'])
        rng, key = rng_key[:, 0], rng_key[:, 1]
        new_state = jax.vmap(self.env.reset)(key)

        if self._runtime_specs:
            new_state = self._apply_runtime_dr(new_state, key)

        # Preserve episode metrics from completed rollout
        prev_metrics = state.info[f'{self._KEY}_episode_metrics']

        new_state.info[f'{self._KEY}_rng'] = rng
        new_state.info[f'{self._KEY}_steps'] = jp.zeros(rng.shape[0])
        new_state.info[f'{self._KEY}_episode_done'] = jp.zeros(rng.shape[0])
        new_state.info[f'{self._KEY}_done_count'] = jp.zeros(rng.shape[0])
        new_state.info[f'{self._KEY}_episode_metrics'] = {
            k: jp.zeros_like(v) for k, v in prev_metrics.items()
        }
        new_state.info['steps'] = jp.zeros(rng.shape[0])
        new_state.info['truncation'] = jp.zeros(rng.shape[0])

        return new_state

    # ── Model DR helpers ─────────────────────────────────────────

    @contextlib.contextmanager
    def _swap_model(self, mjx_model: mjx.Model):
        """Temporarily replace the env's mjx_model."""
        env = self.env.unwrapped
        old = env._mjx_model
        try:
            env._mjx_model = mjx_model
            yield env
        finally:
            env._mjx_model = old

    def _build_dr_model(self, rng: jax.Array):
        """Sample model DR and return (batched_model, in_axes).

        Args:
            rng: (num_envs, 2) per-env PRNG keys.

        Returns:
            (mjx.Model, mjx.Model) — batched model + in_axes for vmap.
        """
        if not self._model_specs:
            return None, None

        model = self.env.unwrapped.mjx_model

        @jax.vmap
        def sample_and_apply(rng):
            replacements = {}
            for spec in self._model_specs:
                rng, key = jax.random.split(rng)
                field_data = getattr(model, spec.field)

                # Determine sample shape
                if spec.indices is not None:
                    start, stop = spec.indices
                    target = field_data[start:stop] if spec.column is None else field_data[start:stop, spec.column]
                elif spec.column is not None:
                    target = field_data[:, spec.column]
                else:
                    target = field_data

                # Sample
                if spec.per_element:
                    rand = jax.random.uniform(key, target.shape, minval=spec.min, maxval=spec.max)
                else:
                    rand = jax.random.uniform(key, minval=spec.min, maxval=spec.max)

                # Apply operation
                if spec.operation == "multiply":
                    new_target = target * rand
                elif spec.operation == "add":
                    new_target = target + rand
                else:  # "set"
                    if spec.per_element:
                        new_target = rand
                    else:
                        new_target = jp.full_like(target, rand)

                # Write back to field
                if spec.indices is not None:
                    start, stop = spec.indices
                    if spec.column is not None:
                        new_field = field_data.at[start:stop, spec.column].set(new_target)
                    else:
                        new_field = field_data.at[start:stop].set(new_target)
                elif spec.column is not None:
                    new_field = field_data.at[:, spec.column].set(new_target)
                else:
                    new_field = new_target

                replacements[spec.field] = new_field

            return replacements

        batched_replacements = sample_and_apply(rng)

        # Build in_axes: randomized fields → axis 0, everything else → None
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        axis_replacements = {field: 0 for field in batched_replacements}
        in_axes = in_axes.tree_replace(axis_replacements)

        model = model.tree_replace(batched_replacements)
        return model, in_axes

    # ── Runtime DR helpers ──────────────────────────────────────

    def _apply_runtime_dr(
        self, state: mjx_env.State, rng: jax.Array
    ) -> mjx_env.State:
        """Sample and store runtime DR values in state.info."""
        for spec in self._runtime_specs:
            rng_split = jax.vmap(jax.random.split)(rng)
            rng, key = rng_split[:, 0], rng_split[:, 1]
            shape = spec.shape or ()
            if spec.per_element and shape:
                vals = jax.vmap(
                    lambda k: jax.random.uniform(k, shape, minval=spec.min, maxval=spec.max)
                )(key)
            else:
                vals = jax.vmap(
                    lambda k: jax.random.uniform(k, minval=spec.min, maxval=spec.max)
                )(key)
            state.info[spec.name] = vals
        return state


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _where_done(done, reset_val, step_val):
    """Select reset_val where done, step_val otherwise.

    Handles Warp's flattened arrays where leading dim != num_envs.
    """
    if not hasattr(step_val, 'shape') or not step_val.shape:
        return step_val
    if done.shape and done.shape[0] != step_val.shape[0]:
        return step_val  # flattened array, can't select per-env
    d = jp.reshape(done, [step_val.shape[0]] + [1] * (len(step_val.shape) - 1))
    return jp.where(d, reset_val, step_val)
