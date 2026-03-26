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
    obs_normalize_fn: Callable | None = None,
    q_fn: Callable | None = None,
    gamma: float = 0.99,
    action_fn_kwargs: dict | None = None,
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
        q_fn: optional fn(obs, action) -> Q value. If provided, computes Q prediction
              accuracy vs Monte Carlo returns (bias, RMSE, correlation).
        gamma: discount factor for MC return computation (only used if q_fn is provided).

    Returns:
        dict with eval_mean, eval_std, eval_min, eval_max.
        If q_fn provided, also: q_bias, q_rmse, q_corr, q_mean, mc_mean.
    """
    if key is None:
        key = jax.random.PRNGKey(999)

    # Use num_envs as batch dim if provided (matches training compilation),
    # otherwise fall back to num_episodes (creates separate compilation).
    batch_dim = num_envs if num_envs is not None else num_episodes

    # Cache the NaN-safe JIT'd env.step — avoid recompilation on every eval call.
    # Uses same NaN guard as training to handle MJX physics failures.
    if not hasattr(evaluate, '_env_step_cache'):
        evaluate._env_step_cache = {}
    cache_key = id(env)
    if cache_key not in evaluate._env_step_cache:
        from jax_rl.training.env_setup import _make_nan_safe_step
        evaluate._env_step_cache[cache_key] = _make_nan_safe_step(env.step)
    env_step = evaluate._env_step_cache[cache_key]

    # Reset batch_dim envs (may be larger than num_episodes to match training shape)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, batch_dim))
    # NOTE: MJX env.reset() produces weak_type fields (.data.time) that cause
    # env.step() to recompile on every eval (~2 while + 2 scan). We tried casting
    # weak_types here but it didn't fix it — the issue is internal to MJX's step.
    # Mitigated by XLA_CLIENT_MEM_FRACTION=0.7. See LESSONS.md.

    _action_kwargs = action_fn_kwargs or {}
    episode_returns = np.zeros(batch_dim)
    episode_done = np.zeros(batch_dim, dtype=bool)

    # Q diagnostics storage (only if q_fn provided)
    if q_fn is not None:
        step_rewards = []   # list of (batch_dim,) arrays
        step_q_preds = []   # list of (batch_dim,) arrays
        step_active = []    # list of (batch_dim,) bool arrays (not done yet)

    for _ in range(episode_length):
        # Only check first num_episodes for early termination
        if episode_done[:num_episodes].all():
            break

        key, ak = jax.random.split(key)
        obs = env_state.obs
        if obs_normalize_fn is not None:
            obs = obs_normalize_fn(obs)
        action = select_action_fn(actor_params, obs, ak, deterministic=True, **_action_kwargs)

        # Record Q prediction before stepping
        if q_fn is not None:
            q_pred = np.asarray(q_fn(obs, action)).squeeze()
            step_q_preds.append(q_pred)
            step_active.append(~episode_done.copy())

        env_state = env_step(env_state, action)

        rewards = np.asarray(env_state.reward)
        dones = np.asarray(env_state.done).astype(bool)

        if q_fn is not None:
            step_rewards.append(rewards.copy())

        episode_returns += rewards * (~episode_done)
        episode_done |= dones

    # Only report stats for the first num_episodes (ignore padding envs)
    results = episode_returns[:num_episodes]
    metrics = {
        "eval_mean": float(np.mean(results)),
        "eval_std": float(np.std(results)),
        "eval_min": float(np.min(results)),
        "eval_max": float(np.max(results)),
    }

    # Compute Q diagnostics: compare Q predictions to MC returns
    if q_fn is not None and len(step_rewards) > 0:
        n_steps = len(step_rewards)
        rewards_arr = np.stack(step_rewards)    # (T, batch_dim)
        q_preds_arr = np.stack(step_q_preds)    # (T, batch_dim)
        active_arr = np.stack(step_active)      # (T, batch_dim)

        # Compute discounted MC return backward: G_t = r_t + gamma * G_{t+1}
        mc_returns = np.zeros_like(rewards_arr)
        mc_returns[-1] = rewards_arr[-1]
        for t in range(n_steps - 2, -1, -1):
            mc_returns[t] = rewards_arr[t] + gamma * mc_returns[t + 1]

        # Only use first num_episodes and active (not-done) steps
        mask = active_arr[:, :num_episodes]
        q_vals = q_preds_arr[:, :num_episodes][mask]
        mc_vals = mc_returns[:, :num_episodes][mask]

        if len(q_vals) > 0:
            bias = float(np.mean(q_vals - mc_vals))
            rmse = float(np.sqrt(np.mean((q_vals - mc_vals) ** 2)))
            # Correlation (guard against constant arrays)
            if np.std(q_vals) > 1e-8 and np.std(mc_vals) > 1e-8:
                corr = float(np.corrcoef(q_vals, mc_vals)[0, 1])
            else:
                corr = 0.0
            metrics.update({
                "q_bias": bias,
                "q_rmse": rmse,
                "q_corr": corr,
                "q_mean": float(np.mean(q_vals)),
                "mc_mean": float(np.mean(mc_vals)),
            })

    return metrics
