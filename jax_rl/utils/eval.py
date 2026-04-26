"""Deterministic evaluation utility — shared across all algorithms.

Runs N episodes with deterministic policy, reports return statistics.
Uses a separate env instance so training state is not disturbed.

Uses lax.scan for the eval loop to avoid Warp OOM from Python-loop
buffer accumulation (see lessons/warp.md).
"""

import functools

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
    eval_log_path: str | None = None,
    total_steps: int | None = None,
) -> dict:
    """Run deterministic evaluation episodes.

    Args:
        select_action_fn: fn(actor_params, obs, key, deterministic=True) -> action
        actor_params: current actor parameters
        env: brax-wrapped env (already wrap_for_training'd)
        num_episodes: how many episodes to report stats for
        episode_length: max steps per episode
        key: PRNG key (only needed for SAC/PPO to pass to select_action)
        num_envs: if provided, pad the batch to this size to reuse training JIT
                  compilations. Avoids separate eval compilations that waste GPU memory.
        q_fn: optional fn(obs, action) -> Q value. If provided, computes Q prediction
              accuracy vs Monte Carlo returns (bias, RMSE, correlation).
        gamma: discount factor for MC return computation (only used if q_fn is provided).
        eval_log_path: if provided, append per-episode eval data to this CSV.
            Includes episode return and init_qpos (if available in env state.info).
        total_steps: current training step count (logged in eval CSV).

    Returns:
        dict with eval_mean, eval_std, eval_min, eval_max.
        If q_fn provided, also: q_bias, q_rmse, q_corr, q_mean, mc_mean.
    """
    if key is None:
        key = jax.random.PRNGKey(999)

    batch_dim = num_envs if num_envs is not None else num_episodes

    # Cache the NaN-safe JIT'd env.step — avoid recompilation on every eval call.
    if not hasattr(evaluate, '_env_step_cache'):
        evaluate._env_step_cache = {}
    cache_key = id(env)
    if cache_key not in evaluate._env_step_cache:
        from jax_rl.training.env_setup import _make_nan_safe_step
        evaluate._env_step_cache[cache_key] = _make_nan_safe_step(env.step)
    env_step = evaluate._env_step_cache[cache_key]

    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, batch_dim))

    # Capture init_qpos before scan (if env provides it in state.info).
    init_qpos = None
    if hasattr(env_state, 'info') and isinstance(env_state.info, dict):
        if 'init_qpos' in env_state.info:
            init_qpos = np.asarray(env_state.info['init_qpos'])

    _action_kwargs = action_fn_kwargs or {}

    # Build the scan step function. Captures actor_params, env_step, etc.
    # Q diagnostics are computed inside the scan to avoid Python-loop OOM.
    has_q = q_fn is not None

    @jax.jit
    def _scan_eval(env_state, key):
        """Run episode_length steps via lax.scan, return per-step data."""

        def scan_step(carry, step_key):
            env_state, episode_returns, episode_done = carry

            obs = env_state.obs
            if obs_normalize_fn is not None:
                obs = obs_normalize_fn(obs)
            action = select_action_fn(
                actor_params, obs, step_key, deterministic=True,
                **_action_kwargs,
            )

            # Q prediction (before stepping)
            if has_q:
                q_pred = q_fn(env_state.obs, action).squeeze()
            else:
                q_pred = jnp.zeros(batch_dim)

            env_state = env_step(env_state, action)

            reward = env_state.reward
            done = env_state.done.astype(jnp.bool_)

            # Accumulate returns only for non-done episodes
            episode_returns = episode_returns + reward * (~episode_done).astype(reward.dtype)
            episode_done = episode_done | done

            carry = (env_state, episode_returns, episode_done)
            # Per-step outputs for Q diagnostics
            per_step = (reward, q_pred, ~episode_done)
            return carry, per_step

        init_returns = jnp.zeros(batch_dim)
        init_done = jnp.zeros(batch_dim, dtype=jnp.bool_)

        step_keys = jax.random.split(key, episode_length)

        (env_state, episode_returns, episode_done), per_step_data = jax.lax.scan(
            scan_step,
            (env_state, init_returns, init_done),
            step_keys,
        )

        return episode_returns, per_step_data

    episode_returns, per_step_data = _scan_eval(env_state, key)

    # Extract results (move to CPU)
    episode_returns = np.asarray(episode_returns)
    results = episode_returns[:num_episodes]

    metrics = {
        "eval_mean": float(np.mean(results)),
        "eval_std": float(np.std(results)),
        "eval_min": float(np.min(results)),
        "eval_max": float(np.max(results)),
    }

    # Q diagnostics
    if has_q:
        rewards_arr, q_preds_arr, active_arr = per_step_data
        rewards_arr = np.asarray(rewards_arr)   # (T, batch_dim)
        q_preds_arr = np.asarray(q_preds_arr)   # (T, batch_dim)
        active_arr = np.asarray(active_arr)     # (T, batch_dim)

        n_steps = rewards_arr.shape[0]

        # Discounted MC return backward: G_t = r_t + gamma * G_{t+1}
        mc_returns = np.zeros_like(rewards_arr)
        mc_returns[-1] = rewards_arr[-1]
        for t in range(n_steps - 2, -1, -1):
            mc_returns[t] = rewards_arr[t] + gamma * mc_returns[t + 1]

        # Only first num_episodes, active steps
        mask = active_arr[:, :num_episodes]
        q_vals = q_preds_arr[:, :num_episodes][mask]
        mc_vals = mc_returns[:, :num_episodes][mask]

        if len(q_vals) > 0:
            bias = float(np.mean(q_vals - mc_vals))
            rmse = float(np.sqrt(np.mean((q_vals - mc_vals) ** 2)))
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

    # Write per-episode eval log CSV (append mode).
    if eval_log_path is not None:
        import csv
        import os
        os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)
        write_header = not os.path.exists(eval_log_path)
        nq = init_qpos.shape[1] if init_qpos is not None else 0
        with open(eval_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                header = ['total_steps', 'episode_idx', 'return']
                if nq > 0:
                    header += [f'init_qpos_{i}' for i in range(nq)]
                writer.writerow(header)
            for ep_idx in range(len(results)):
                row = [
                    total_steps if total_steps is not None else -1,
                    ep_idx,
                    float(results[ep_idx]),
                ]
                if init_qpos is not None:
                    row += [float(v) for v in init_qpos[ep_idx]]
                writer.writerow(row)

    return metrics


def evaluate_gym(
    select_action_fn: Callable,
    actor_params,
    eval_env,
    num_episodes: int = 10,
    episode_length: int = 1000,
    key: jax.Array | None = None,
    obs_normalize_fn: Callable | None = None,
    action_fn_kwargs: dict | None = None,
    **_unused,
) -> dict:
    """Deterministic eval for gym vector envs (Python loop, single-env serial).

    Drop-in replacement for `evaluate()` when bundle.backend_kind == "gym".
    Skips Q-bias diagnostics (gym is mostly used for low-dim research envs;
    add later if needed).

    Args:
        eval_env: gym.vector.SyncVectorEnv (one underlying env).
        Same signature otherwise. Extra kwargs (q_fn, gamma, eval_log_path,
        total_steps, num_envs) are accepted-and-ignored for compat with
        evaluate()'s call sites.
    """
    if key is None:
        key = jax.random.PRNGKey(999)
    _kwargs = action_fn_kwargs or {}

    returns = []
    for ep in range(num_episodes):
        obs, _ = eval_env.reset(seed=int(jax.random.randint(key, (), 0, 2**31 - 1)))
        key, _ = jax.random.split(key)
        ep_return = 0.0
        for _ in range(episode_length):
            key, ak = jax.random.split(key)
            obs_jax = jnp.asarray(obs)
            if obs_normalize_fn is not None:
                obs_jax = obs_normalize_fn(obs_jax)
            action = select_action_fn(
                actor_params, obs_jax, ak, deterministic=True, **_kwargs
            )
            action_np = np.asarray(action, dtype=np.float32)
            obs, r, term, trunc, info = eval_env.step(action_np)
            # Single-env vec → reduce batch dim of size 1.
            ep_return += float(np.asarray(r).reshape(-1)[0])
            done = bool(np.asarray(term).any() or np.asarray(trunc).any())
            if done:
                break
        returns.append(ep_return)

    returns_arr = np.array(returns, dtype=np.float32)
    return {
        "eval_mean": float(returns_arr.mean()),
        "eval_std": float(returns_arr.std()),
        "eval_min": float(returns_arr.min()),
        "eval_max": float(returns_arr.max()),
    }
