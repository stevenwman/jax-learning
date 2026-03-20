"""Deterministic evaluation utility — shared across all algorithms.

Runs N episodes with deterministic policy, reports return statistics.
Uses a separate env instance so training state is not disturbed.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable


def evaluate(
    select_action_fn: Callable,
    actor_params,
    env,
    num_episodes: int = 10,
    episode_length: int = 1000,
    key: jax.Array | None = None,
    num_envs: int | None = None,
) -> dict:
    """Run deterministic evaluation episodes.

    Args:
        select_action_fn: fn(actor_params, obs, key, deterministic=True) -> action
        actor_params: current actor parameters
        env: brax-wrapped env (already wrap_for_brax_training'd)
        num_episodes: how many episodes to report stats for
        episode_length: max steps per episode
        key: PRNG key (only needed for SAC/PPO to pass to select_action)
        num_envs: if provided, pad the batch to this size to reuse training JIT
                  compilations. Avoids separate eval compilations that waste GPU memory.

    Returns:
        dict with eval_mean, eval_std, eval_min, eval_max
    """
    if key is None:
        key = jax.random.PRNGKey(999)

    # Use num_envs as batch dim if provided (matches training compilation),
    # otherwise fall back to num_episodes (creates separate compilation).
    batch_dim = num_envs if num_envs is not None else num_episodes

    # Cache the JIT'd env.step — avoid recompilation on every eval call.
    if not hasattr(evaluate, '_env_step_cache'):
        evaluate._env_step_cache = {}
    cache_key = id(env)
    if cache_key not in evaluate._env_step_cache:
        evaluate._env_step_cache[cache_key] = jax.jit(env.step)
    env_step = evaluate._env_step_cache[cache_key]

    # Reset batch_dim envs (may be larger than num_episodes to match training shape)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, batch_dim))

    episode_returns = np.zeros(batch_dim)
    episode_done = np.zeros(batch_dim, dtype=bool)

    for _ in range(episode_length):
        # Only check first num_episodes for early termination
        if episode_done[:num_episodes].all():
            break

        key, ak = jax.random.split(key)
        action = select_action_fn(actor_params, env_state.obs, ak, deterministic=True)
        env_state = env_step(env_state, action)

        rewards = np.asarray(env_state.reward)
        dones = np.asarray(env_state.done).astype(bool)

        episode_returns += rewards * (~episode_done)
        episode_done |= dones

    # Only report stats for the first num_episodes (ignore padding envs)
    results = episode_returns[:num_episodes]
    return {
        "eval_mean": float(np.mean(results)),
        "eval_std": float(np.std(results)),
        "eval_min": float(np.min(results)),
        "eval_max": float(np.max(results)),
    }
