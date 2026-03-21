"""Environment creation — replaces the 6-line block duplicated across all train scripts."""

import jax
import jax.numpy as jnp

from mujoco_playground import dm_control_suite
from mujoco_playground._src.wrapper import wrap_for_brax_training

from jax_rl.configs.train_config import TrainConfig
from jax_rl.utils.normalization import NormalizationState


def _make_nan_safe_step(raw_step):
    """Wrap env.step to guard against MJX physics NaN.

    MuJoCo's MJX backend can produce NaN obs/rewards when the simulation
    encounters extreme states (contact solver failure, singular mass matrix).
    This happens stochastically with humanoid envs at 1024 parallel worlds.

    When NaN is detected:
    - obs replaced with zeros (safe for network forward pass)
    - reward set to 0
    - done set to 1 (triggers auto-reset on next step)
    """
    @jax.jit
    def safe_step(state, action):
        # Guard NaN actions (from NaN obs → actor forward → NaN action)
        action = jnp.where(jnp.isnan(action), 0.0, action)
        state = raw_step(state, action)
        # Guard NaN obs/rewards from MJX physics failures
        has_nan = jnp.any(jnp.isnan(state.obs), axis=-1)  # (num_envs,)
        has_nan = has_nan | jnp.isnan(state.reward)  # also check reward
        safe_obs = jnp.where(has_nan[:, None], 0.0, state.obs)
        safe_reward = jnp.where(has_nan, 0.0, state.reward)
        safe_done = jnp.where(has_nan, 1.0, state.done)
        state = state.replace(obs=safe_obs, reward=safe_reward, done=safe_done)
        return state
    return safe_step


def make_envs(cfg: TrainConfig, seed: int):
    """Create training env + eval env, JIT env.step, reset training env.

    Returns:
        env: wrapped training environment
        env_step: JIT'd, NaN-safe env.step function
        env_state: initial env state (reset with num_envs)
        eval_env: separate wrapped env for evaluation
        obs_dim: observation dimensionality
        action_dim: action dimensionality
    """
    env = dm_control_suite.load(cfg.env_name)
    env = wrap_for_brax_training(env, episode_length=cfg.episode_length)
    env_step = _make_nan_safe_step(env.step)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, cfg.num_envs))

    eval_env = dm_control_suite.load(cfg.env_name)
    eval_env = wrap_for_brax_training(eval_env, episode_length=cfg.episode_length)

    obs_dim = env_state.obs.shape[-1]
    action_dim = env.action_size

    return env, env_step, env_state, eval_env, obs_dim, action_dim, key


def make_identity_norm_state(obs_dim: int) -> NormalizationState:
    """Identity norm state for off-policy algos (no obs normalization).

    Q-network LayerNorm handles input scaling instead.
    Kept as identity for checkpoint/inference compatibility with record_video.py.
    """
    return NormalizationState(
        mean=jnp.zeros(obs_dim),
        mean_of_squares=jnp.ones(obs_dim),
        count=1,
    )
