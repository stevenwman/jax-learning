"""Gym backend for the env-bundle pipeline.

Wraps `gym.vector.{Sync,Async}VectorEnv` so single-process gym envs
(pusht, dm_control via gym wrapper, etc.) can plug into the same
EnvBundle interface as MJX envs.

Per-env factory functions live in `GYM_ENV_FACTORIES`. They take a
TrainConfig and return a callable `() -> gym.Env` that the vector env
spawns. PushT is the first concrete env wired up.

Vectorization:
- `num_envs` is capped at `os.cpu_count()` for AsyncVectorEnv to avoid
  oversubscription. Warning emitted if cfg.num_envs exceeds the cap.
- `num_envs == 1` auto-selects SyncVectorEnv (no IPC, easy debugging).

Action scaling assumes policy outputs in `[-1, 1]`; gym envs are
expected to declare their native action space and we use
`gym.wrappers.RescaleAction` to bridge. Each factory wraps as needed.

State adapter:
- `GymState` holds obs/reward/done/info to match the MJX-state interface
  consumed by `run_offpolicy_loop` (state.obs / state.reward / state.done /
  state.info["truncation"]).
- Auto-reset is handled by `gym.vector.VectorEnv` natively; obs returned
  after a terminal step is the *reset* obs (matches Brax convention).

Registers itself as the "gym" backend on import. Concrete env names
get added to the gym detection set via `register_gym_env_name`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import numpy as np

from jax_rl.configs.train_config import TrainConfig
from jax_rl.training.env_bundle import EnvBundle


# ── Gym state adapter ───────────────────────────────────────────────────

@dataclass
class GymState:
    """Mirrors the surface of MJX/Brax state for run_offpolicy_loop.

    Fields:
    - obs: (N, obs_dim) np.ndarray  (or dict for multi-modal obs).
    - reward: (N,)
    - done: (N,) — terminated | truncated, matches Brax convention.
    - info: dict, must populate "truncation" (N,) for SAC bootstrap.
    """
    obs: Any
    reward: np.ndarray
    done: np.ndarray
    info: dict = field(default_factory=dict)


# ── Per-env factories ───────────────────────────────────────────────────
# Each entry: (env_name) -> Callable[[TrainConfig], Callable[[], gym.Env]].
# The outer call returns a thunk that vector envs spawn.

GYM_ENV_FACTORIES: dict[str, Callable[[TrainConfig], Callable[[], Any]]] = {}


def register_gym_env(name: str, factory: Callable[[TrainConfig], Callable[[], Any]]) -> None:
    """Register a gym env factory + add the name to the detection set."""
    from jax_rl.training.env_backends import register_gym_env_name
    GYM_ENV_FACTORIES[name] = factory
    register_gym_env_name(name)


# ── PushT factory ───────────────────────────────────────────────────────

def _make_pusht_factory(cfg: TrainConfig) -> Callable[[], Any]:
    """Return a thunk that builds a single PushT env per cfg.env_kwargs.

    Defaults match the recipe used in scripts/train_pusht.py:
        obs_type="keypoints", reward_mode="contact_gated",
        coverage_shape="log_barrier", coverage_eps=0.01,
        block_shape="tee", action_repeat=2, max_episode_steps=300.

    Override any of these via cfg.env_kwargs.
    """
    import gymnasium as gym
    from jax_rl.envs.manipulation.pusht import PushTEnv

    kwargs = dict(cfg.env_kwargs)
    pusht_kwargs = dict(
        obs_type=kwargs.pop("obs_type", "keypoints"),
        reward_mode=kwargs.pop("reward_mode", "contact_gated"),
        coverage_shape=kwargs.pop("coverage_shape", "log_barrier"),
        coverage_eps=kwargs.pop("coverage_eps", 0.01),
        block_shape=kwargs.pop("block_shape", "tee"),
        render_mode=kwargs.pop("render_mode", "rgb_array"),
    )
    action_repeat = int(kwargs.pop("action_repeat", 2))
    max_episode_steps = int(kwargs.pop("max_episode_steps", 300))
    if kwargs:
        raise ValueError(f"Unknown PushT env_kwargs: {sorted(kwargs)}")

    def make_env():
        env = PushTEnv(**pusht_kwargs)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = _NormalizeObsWrapper(env)
        env = _ActionRepeatWrapper(env, k=action_repeat)
        # Policy emits [-1, 1]; gym-pusht wants [0, 512]² for actions.
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        return env

    return make_env


# ── Vendored pusht-style wrappers (port from scripts/train_pusht.py) ───

import gymnasium as _gym  # noqa: E402


class _NormalizeObsWrapper(_gym.ObservationWrapper):
    """Rescale raw pixel-coord obs [0, 512] → roughly [-1, 1]."""
    def __init__(self, env):
        super().__init__(env)
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        self._center = (low + high) / 2.0
        self._scale = (high - low) / 2.0 + 1e-6
        self.observation_space = _gym.spaces.Box(
            low=-1.0, high=1.0, shape=low.shape, dtype=np.float32,
        )

    def observation(self, obs):
        return ((np.asarray(obs, dtype=np.float32) - self._center) / self._scale).astype(np.float32)


class _ActionRepeatWrapper(_gym.Wrapper):
    """Repeat action K times, accumulate reward, return last obs."""
    def __init__(self, env, k: int):
        super().__init__(env)
        self.k = k

    def step(self, action):
        total_r = 0.0
        for _ in range(self.k):
            obs, r, term, trunc, info = self.env.step(action)
            total_r += float(r)
            if term or trunc:
                break
        return obs, total_r, term, trunc, info


register_gym_env("PushT", _make_pusht_factory)


# ── Gymnasium MuJoCo factories (HalfCheetah, Hopper, Walker2d, Humanoid…) ─

def _make_gymnasium_mujoco_factory(gym_id: str):
    """Return a factory that builds a `gymnasium[mujoco]` env from gym_id.

    These envs already expose continuous action_space in [-1, 1] and 1D
    state obs — no RescaleAction / NormalizeObs needed. Episode length
    cap is the env's own (HalfCheetah=1000 etc.), so no extra TimeLimit.

    cfg.env_kwargs (dict) is forwarded to `gym.make()`. Useful for setting
    `render_mode='rgb_array'`, `forward_reward_weight`, etc.
    """
    def factory(cfg: TrainConfig):
        import gymnasium as gym
        kwargs = dict(cfg.env_kwargs)

        def make_env():
            return gym.make(gym_id, **kwargs)

        return make_env

    return factory


# Register classics. Add more as needed.
for _name, _id in [
    ("HalfCheetah", "HalfCheetah-v5"),
    ("Hopper",       "Hopper-v5"),
    ("Walker2d",     "Walker2d-v5"),
    ("Humanoid",     "Humanoid-v5"),
    ("Ant",          "Ant-v5"),
    ("Pendulum",     "Pendulum-v1"),
    ("LunarLanderContinuous", "LunarLanderContinuous-v3"),
]:
    register_gym_env(_name, _make_gymnasium_mujoco_factory(_id))


# ── Bundle builder ──────────────────────────────────────────────────────

def make_gym_env_bundle(cfg: TrainConfig, seed: int) -> EnvBundle:
    """Build a gym EnvBundle (CPU vector env)."""
    import gymnasium as gym

    if cfg.env_name not in GYM_ENV_FACTORIES:
        raise ValueError(
            f"Gym env {cfg.env_name!r} has no registered factory. "
            f"Known: {sorted(GYM_ENV_FACTORIES)}."
        )
    make_env_thunk = GYM_ENV_FACTORIES[cfg.env_name](cfg)

    cpu = max(1, os.cpu_count() or 1)
    n = cfg.num_envs
    if n > cpu:
        print(f"  [gym backend] cfg.num_envs={n} > cpu_count={cpu}; capping to {cpu}.")
        n = cpu

    if n == 1:
        vec_env = gym.vector.SyncVectorEnv([make_env_thunk])
    else:
        vec_env = gym.vector.AsyncVectorEnv([make_env_thunk for _ in range(n)])

    obs, info = vec_env.reset(seed=seed)
    obs = _to_numpy(obs)
    init_state = GymState(
        obs=obs,
        reward=np.zeros(n, dtype=np.float32),
        done=np.zeros(n, dtype=np.float32),
        info={"truncation": np.zeros(n, dtype=np.float32)},
    )

    def env_step(state: GymState, action) -> GymState:
        # action may be JAX array (N, action_dim); gym needs numpy.
        a_np = np.asarray(action, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = vec_env.step(a_np)
        # Brax-style: done = terminated | truncated, info["truncation"] = truncated.
        done = np.asarray(terminated | truncated, dtype=np.float32)
        truncation = np.asarray(truncated, dtype=np.float32)
        gym_info = {"truncation": truncation}
        # Preserve any per-env info dicts the env emits (gym vector returns
        # batched arrays for single-key infos).
        for k, v in info.items():
            if k not in gym_info:
                gym_info[k] = v
        return GymState(
            obs=_to_numpy(next_obs),
            reward=np.asarray(reward, dtype=np.float32),
            done=done,
            info=gym_info,
        )

    eval_env = gym.vector.SyncVectorEnv([make_env_thunk])

    # Detect obs/action dims from initial state.
    dict_obs = isinstance(obs, dict)
    if dict_obs:
        obs_dim = obs["state"].shape[-1] if "state" in obs else None
        if obs_dim is None:
            raise ValueError("Gym dict-obs envs must expose 'state' key for offpolicy training.")
    else:
        obs_dim = obs.shape[-1]

    sample_action = vec_env.single_action_space.sample()
    action_dim = int(np.asarray(sample_action).shape[-1])

    has_privileged = dict_obs and "privileged_state" in obs
    critic_obs_dim = obs["privileged_state"].shape[-1] if has_privileged else None

    def render_fn(state, env_idx: int = 0):
        # AsyncVectorEnv doesn't support direct env access; use eval_env.
        try:
            return eval_env.envs[env_idx].render()
        except (AttributeError, IndexError):
            return None

    return EnvBundle(
        env=vec_env, env_step=env_step, env_state=init_state, eval_env=eval_env,
        obs_dim=obs_dim, action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        has_privileged=has_privileged,
        dict_obs=dict_obs,
        key=jax.random.PRNGKey(seed),
        backend_kind="gym",
        num_envs=n,
        render_fn=render_fn,
    )


def _to_numpy(obs):
    """Coerce vec_env obs (which may be dict-of-arrays or array) to numpy."""
    if isinstance(obs, dict):
        return {k: np.asarray(v) for k, v in obs.items()}
    return np.asarray(obs)


# Register on import.
from jax_rl.training.env_backends import register_backend
register_backend("gym", make_gym_env_bundle)
