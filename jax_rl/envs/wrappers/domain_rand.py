"""Domain Randomization wrapper — unified episode boundary + DR.

A single wrapper that replaces the legacy AutoResetWrapper + EpisodeWrapper
stack for envs that need fresh-reset + per-episode DR. Handles:
  - Vectorization (vmap)
  - Episode length tracking + truncation
  - Auto-reset on done (per_step mode)
  - Per-episode domain randomization (model + runtime)
  - Clean state.info reset on episode boundary

Reset mode:
  - "per_step": compute reset for all envs every step, select via where_done.
    Fresh IC + clean state every episode. Pays O(N) reset cost per step.

DR persistence semantics (per-episode, not per-step):
  - Model DR (mjx.Model fields): sampled per env at reset, persisted in
    state.info[`_dr_dr_fields`] across steps, used for the active env.step
    path. step() also samples a fresh set for the reset-candidate path; the
    where_done merge then keeps persisted fields on active envs and swaps
    in the fresh sample on envs that just reset. Until 2026-04-27 this was
    broken: dr_model was rebuilt every step and used in the active step
    path, so non-done episodes saw physics shift every step.
  - Runtime DR (state.info entries): sampled per env at reset, kept in
    state.info between steps; reset_state gets fresh values, where_done
    merges per env on done.

With no DR spec: behaves as an improved AutoResetWrapper.
With DR spec: adds per-episode physics randomization.
"""

import contextlib
from dataclasses import dataclass
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
        mode: "per_step" (only supported mode).
    """

    _KEY = '_dr'

    def __init__(
        self,
        env: Any,
        episode_length: int = 1000,
        mode: Literal["per_step"] = "per_step",
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

        dr_fields = None
        if self._model_specs:
            dr_fields = self._sample_dr_fields(key)
            dr_model, in_axes = self._dr_model_from_fields(dr_fields)
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

        # Persist sampled per-env model DR fields across steps. step() uses
        # these for the active env.step path (so non-done episodes see
        # stable physics) and only resamples for envs that just reset.
        if dr_fields is not None:
            state.info[f'{self._KEY}_dr_fields'] = dr_fields

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

    # ── Step ─────────────────────────────────────────────────────

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Save and strip wrapper state so tree_map doesn't see mismatched keys
        drv2_rng = state.info.pop(f'{self._KEY}_rng')
        drv2_steps = state.info.pop(f'{self._KEY}_steps')
        drv2_done_count = state.info.pop(f'{self._KEY}_done_count')
        drv2_episode_done = state.info.pop(f'{self._KEY}_episode_done')
        drv2_metrics = state.info.pop(f'{self._KEY}_episode_metrics')
        # Persisted per-env model DR fields from prior step (or reset).
        drv2_dr_fields = state.info.pop(f'{self._KEY}_dr_fields', None)
        state.info.pop('steps', None)
        state.info.pop('truncation', None)

        # Prepare reset for ALL envs (JAX both-paths)
        rng_key = jax.vmap(jax.random.split)(drv2_rng)
        reset_rng, next_rng = rng_key[:, 0], rng_key[:, 1]

        reset_dr_fields = None
        if self._model_specs:
            # Sample fresh DR for the reset-candidate path. Use PERSISTED DR
            # for the active step path so non-done episodes see stable physics
            # (per-episode DR, not per-step). Where_done below merges fresh
            # into persisted for envs that just reset.
            reset_dr_fields = self._sample_dr_fields(reset_rng)
            reset_dr_model, in_axes = self._dr_model_from_fields(reset_dr_fields)
            step_dr_model, _ = self._dr_model_from_fields(drv2_dr_fields)

            def _reset_with_model(mjx_model, rng):
                with self._swap_model(mjx_model) as v_env:
                    return v_env.reset(rng)

            def _step_with_model(mjx_model, s, a):
                with self._swap_model(mjx_model) as v_env:
                    return v_env.step(s, a)

            reset_state = jax.vmap(_reset_with_model, in_axes=[in_axes, 0])(
                reset_dr_model, reset_rng
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
                step_dr_model, state, action
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

        # Per-episode model DR: keep persisted fields on active envs,
        # swap in freshly-sampled fields on envs that just reset.
        if self._model_specs:
            info[f'{self._KEY}_dr_fields'] = jax.tree.map(
                lambda r, p: _where_done(done, r, p),
                reset_dr_fields, drv2_dr_fields,
            )

        info['steps'] = jp.where(done, 0.0, steps)
        info['truncation'] = truncation

        return state.replace(data=data, obs=obs, done=done, info=info)

    # ── Model DR helpers ─────────────────────────────────────────

    @contextlib.contextmanager
    def _swap_model(self, mjx_model: mjx.Model):
        """Temporarily replace the env's mjx_model.

        Mutates the *base* env's `_mjx_model` field (since that's where the
        model lives), but yields `self.env` — the wrapped env directly below
        DomainRandWrapper — so that intermediate wrappers (FrameStack,
        ActionDelay, etc.) still apply during reset/step.
        """
        base = self.env.unwrapped
        old = base._mjx_model
        try:
            base._mjx_model = mjx_model
            yield self.env
        finally:
            base._mjx_model = old

    def _sample_dr_fields(self, rng: jax.Array):
        """Sample per-env model-DR field replacements.

        Args:
            rng: (num_envs, 2) per-env PRNG keys.

        Returns:
            dict[str, jax.Array] with one entry per randomized model field;
            each value has leading dim == num_envs. None if no model specs.
        """
        if not self._model_specs:
            return None

        model = self.env.unwrapped.mjx_model

        @jax.vmap
        def sample_and_apply(rng):
            replacements = {}
            for spec in self._model_specs:
                rng, key = jax.random.split(rng)
                # Compose: when two specs target the same model field (e.g.
                # geom_friction column 0 + column 1), each spec's op must
                # build on the previous spec's output, not the unmodified
                # model field. Without this, the second spec's `at[...].set`
                # silently discards the first's contribution.
                current_field = replacements.get(spec.field, getattr(model, spec.field))

                # Determine sample shape
                if spec.indices is not None:
                    start, stop = spec.indices
                    target = current_field[start:stop] if spec.column is None else current_field[start:stop, spec.column]
                elif spec.column is not None:
                    target = current_field[:, spec.column]
                else:
                    target = current_field

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
                        new_field = current_field.at[start:stop, spec.column].set(new_target)
                    else:
                        new_field = current_field.at[start:stop].set(new_target)
                elif spec.column is not None:
                    new_field = current_field.at[:, spec.column].set(new_target)
                else:
                    new_field = new_target

                replacements[spec.field] = new_field

            return replacements

        return sample_and_apply(rng)

    def _dr_model_from_fields(self, batched_replacements):
        """Build a vmap-ready model + in_axes from pre-sampled DR fields.

        Args:
            batched_replacements: dict from `_sample_dr_fields`. None if no
                model specs.

        Returns:
            (mjx.Model, mjx.Model) — batched model + in_axes for vmap, or
            (None, None) if `batched_replacements` is None.
        """
        if batched_replacements is None:
            return None, None

        model = self.env.unwrapped.mjx_model
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        axis_replacements = {field: 0 for field in batched_replacements}
        in_axes = in_axes.tree_replace(axis_replacements)
        model = model.tree_replace(batched_replacements)
        return model, in_axes

    def _build_dr_model(self, rng: jax.Array):
        """Legacy combined helper (sample + apply). Kept for backwards compat."""
        fields = self._sample_dr_fields(rng)
        return self._dr_model_from_fields(fields)

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
