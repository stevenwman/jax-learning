"""Frame stacking wrapper — wraps any Playground env to provide temporal obs history.

Maintains a FIFO buffer of the last N observations in state.info["frame_stack"].
Newest frame at index [0:obs_dim], oldest at the end.

Works with both flat obs (array) and dict obs (stacks the "state" key only).
"""

import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper


class FrameStackWrapper(Wrapper):
    """Stack N consecutive observations for temporal context.

    Applied between env creation and wrap_for_brax_training.
    Operates per-env (pre-vmap).
    """

    def __init__(self, env: mjx_env.MjxEnv, n_frames: int = 3):
        super().__init__(env)
        self._n_frames = n_frames

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        obs = state.obs
        if isinstance(obs, dict):
            raw = obs["state"]
            stack = jp.tile(raw, self._n_frames)
            obs = {**obs, "state": stack}
        else:
            stack = jp.tile(obs, self._n_frames)
            obs = stack
        state.info["frame_stack"] = stack
        return state.replace(obs=obs)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        obs = state.obs
        if isinstance(obs, dict):
            raw = obs["state"]
        else:
            raw = obs
        raw_dim = raw.shape[-1]
        old_stack = state.info["frame_stack"]
        new_stack = jp.concatenate([raw, old_stack[:-raw_dim]])
        state.info["frame_stack"] = new_stack
        if isinstance(obs, dict):
            obs = {**obs, "state": new_stack}
        else:
            obs = new_stack
        return state.replace(obs=obs)
