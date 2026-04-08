"""Training wrappers — vectorization, episode management, auto-reset.

LEGACY: These wrappers are the original training stack, vendored from Brax
and MuJoCo Playground. For Go2/sim-to-real work, use DomainRandWrapper
(jax_rl/envs/wrappers/domain_rand.py) which replaces this entire stack with
per-episode DR + fresh ICs. These wrappers remain for lightweight envs
(CheetahRun, Cartpole) where the legacy stack is 3x faster.

Activate via: --reset-mode legacy (default) vs --reset-mode per_step (DomainRandWrapper).
"""

import contextlib
from typing import Any, Callable, List, Optional, Sequence, Tuple

import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np


class Wrapper(mjx_env.MjxEnv):
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Any):
        self.env = env

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return self.env.reset(rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.env.mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self.env.mjx_model

    @property
    def xml_path(self) -> str:
        return self.env.xml_path

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[
            Sequence[Callable[[mujoco.MjvScene], None]]
        ] = None,
    ) -> Sequence[np.ndarray]:
        return self.env.render(
            trajectory, height, width, camera, scene_option, modify_scene_fns
        )


class VmapWrapper(Wrapper):
    """Vectorizes env reset/step via jax.vmap."""

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return jax.vmap(self.env.step)(state, action)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode_length (truncation)."""

    def __init__(self, env: Any, episode_length: int, action_repeat: int = 1):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        state.info['steps'] = jp.zeros(rng.shape[:-1])
        state.info['truncation'] = jp.zeros(rng.shape[:-1])
        state.info['episode_done'] = jp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps

        prev_done = state.info['episode_done']
        state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
        state.info['episode_metrics']['sum_reward'] += jp.sum(rewards, axis=0)
        state.info['episode_metrics']['length'] *= (1 - prev_done)
        state.info['episode_metrics']['length'] += self.action_repeat
        for metric_name in state.metrics.keys():
            if metric_name != 'reward':
                state.info['episode_metrics'][metric_name] *= (1 - prev_done)
                state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_done'] = done
        return state.replace(done=done)


class AutoResetWrapper(Wrapper):
    """Automatically resets done envs.

    Default (fast): replays cached initial data+obs. state.info is NOT reset.
    full_reset=True: calls env.reset() per done env. Slower but resets info.
    """

    def __init__(self, env: Any, full_reset: bool = False):
        super().__init__(env)
        self._full_reset = full_reset
        self._info_key = 'AutoResetWrapper'

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(rng)
        rng, key = rng_key[..., 0], rng_key[..., 1]
        state = self.env.reset(key)
        state.info[f'{self._info_key}_first_data'] = state.data
        state.info[f'{self._info_key}_first_obs'] = state.obs
        state.info[f'{self._info_key}_rng'] = rng
        state.info[f'{self._info_key}_done_count'] = jp.zeros(
            key.shape[:-1], dtype=int
        )
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        reset_state = None
        rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
        reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
        if self._full_reset:
            reset_state = self.reset(reset_key)
            reset_data = reset_state.data
            reset_obs = reset_state.obs
        else:
            reset_data = state.info[f'{self._info_key}_first_data']
            reset_obs = state.info[f'{self._info_key}_first_obs']

        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, reset_data, state.data)
        obs = jax.tree.map(where_done, reset_obs, state.obs)

        next_info = state.info
        done_count_key = f'{self._info_key}_done_count'
        if self._full_reset and reset_state:
            next_info = jax.tree.map(where_done, reset_state.info, state.info)
            next_info[done_count_key] = state.info[done_count_key]
            if 'steps' in next_info:
                next_info['steps'] = state.info['steps']
            preserve_info_key = f'{self._info_key}_preserve_info'
            if preserve_info_key in next_info:
                next_info[preserve_info_key] = state.info[preserve_info_key]

        next_info[done_count_key] += state.done.astype(int)
        next_info[f'{self._info_key}_rng'] = reset_rng

        return state.replace(data=data, obs=obs, info=next_info)


class DomainRandomizationVmapWrapper(Wrapper):
    """Vectorized env with per-env domain randomization."""

    def __init__(
        self,
        env: Any,
        randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    ):
        super().__init__(env)
        self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

    @contextlib.contextmanager
    def _v_env_fn(self, mjx_model: mjx.Model):
        env = self.env.unwrapped
        old_mjx_model = env._mjx_model
        try:
            env.unwrapped._mjx_model = mjx_model
            yield env
        finally:
            env.unwrapped._mjx_model = old_mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        def reset(mjx_model, rng):
            with self._v_env_fn(mjx_model) as v_env:
                return v_env.reset(rng)
        return jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        def step(mjx_model, s, a):
            with self._v_env_fn(mjx_model) as v_env:
                return v_env.step(s, a)
        return jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
            self._mjx_model_v, state, action
        )


def wrap_for_training(
    env: Any,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
    """Wrap a raw env for training: vmap + episode management + auto-reset."""
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env)
    return env
