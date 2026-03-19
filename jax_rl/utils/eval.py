"""Deterministic evaluation utility — shared across all algorithms.

Runs N episodes with deterministic policy, reports return statistics.
Uses a separate env instance so training state is not disturbed.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable


def warmup_eval(env, num_episodes: int = 5):
    """Pre-compile eval env.step JIT before training fills GPU memory.

    Call this right after creating the eval env, before training starts.
    """
    if not hasattr(evaluate, '_env_step_cache'):
        evaluate._env_step_cache = {}
    cache_key = id(env)
    if cache_key not in evaluate._env_step_cache:
        evaluate._env_step_cache[cache_key] = jax.jit(env.step)
    env_step = evaluate._env_step_cache[cache_key]

    # Trigger JIT compilation with a dummy step
    key = jax.random.PRNGKey(9999)
    env_state = env.reset(jax.random.split(key, num_episodes))
    dummy_action = jnp.zeros((num_episodes, env.action_size))
    _ = env_step(env_state, dummy_action)


def evaluate(
    select_action_fn: Callable,
    actor_params,
    env,
    num_episodes: int = 10,
    episode_length: int = 1000,
    key: jax.Array | None = None,
) -> dict:
    """Run deterministic evaluation episodes.

    Args:
        select_action_fn: fn(actor_params, obs, key, deterministic=True) → action
        actor_params: current actor parameters
        env: brax-wrapped env (already wrap_for_brax_training'd)
        num_episodes: how many episodes to run
        episode_length: max steps per episode
        key: PRNG key (only needed for SAC/PPO to pass to select_action)

    Returns:
        dict with eval_mean, eval_std, eval_min, eval_max
    """
    if key is None:
        key = jax.random.PRNGKey(999)

    # Cache the JIT'd env.step — avoid recompilation on every eval call.
    # Use env object as cache key via a module-level dict.
    if not hasattr(evaluate, '_env_step_cache'):
        evaluate._env_step_cache = {}
    cache_key = id(env)
    if cache_key not in evaluate._env_step_cache:
        evaluate._env_step_cache[cache_key] = jax.jit(env.step)
    env_step = evaluate._env_step_cache[cache_key]

    # Reset num_episodes envs
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, num_episodes))

    episode_returns = np.zeros(num_episodes)
    episode_done = np.zeros(num_episodes, dtype=bool)

    for _ in range(episode_length):
        if episode_done.all():
            break

        key, ak = jax.random.split(key)
        action = select_action_fn(actor_params, env_state.obs, ak, deterministic=True)
        env_state = env_step(env_state, action)

        rewards = np.asarray(env_state.reward)
        dones = np.asarray(env_state.done).astype(bool)

        # Accumulate rewards only for episodes not yet done
        episode_returns += rewards * (~episode_done)
        episode_done |= dones

    return {
        "eval_mean": float(np.mean(episode_returns)),
        "eval_std": float(np.std(episode_returns)),
        "eval_min": float(np.min(episode_returns)),
        "eval_max": float(np.max(episode_returns)),
    }
