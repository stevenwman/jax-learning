"""Environment creation — replaces the 6-line block duplicated across all train scripts."""

import jax
import jax.numpy as jnp

from mujoco_playground import dm_control_suite
from mujoco_playground._src.wrapper import wrap_for_brax_training

from jax_rl.configs.train_config import TrainConfig
from jax_rl.utils.normalization import NormalizationState


def make_envs(cfg: TrainConfig, seed: int):
    """Create training env + eval env, JIT env.step, reset training env.

    Returns:
        env: wrapped training environment
        env_step: JIT'd env.step function
        env_state: initial env state (reset with num_envs)
        eval_env: separate wrapped env for evaluation
        obs_dim: observation dimensionality
        action_dim: action dimensionality
    """
    env = dm_control_suite.load(cfg.env_name)
    env = wrap_for_brax_training(env, episode_length=cfg.episode_length)
    env_step = jax.jit(env.step)

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
