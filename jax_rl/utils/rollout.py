"""Rollout step builders for video recording and evaluation.

Both builders treat norm_state as FROZEN — match training's inference-time
contract. FS>1 routes through normalize_stacked (per-frame normalization
using single-frame stats), matching onpolicy_collect's path.
"""

import jax
import jax.numpy as jnp

from jax_rl.utils.normalization import (
    normalize as norm_normalize,
    normalize_stacked as norm_normalize_stacked,
)


def _extract_policy_obs(raw_obs):
    """Extract flat policy observation, adding a batch dim.

    Handles both dict-style obs (with 'state' key) and flat arrays.
    Returns shape (1, obs_dim).
    """
    obs = raw_obs["state"] if isinstance(raw_obs, dict) else raw_obs
    return obs[None]


def _apply_norm(ns, obs, n_frame_stack):
    # Dispatch by saved-stats shape: when use_obs_norm was False in training,
    # the running stats track the full stacked obs dim (no tile needed). When
    # use_obs_norm was True, stats are at raw (single-frame) dim and require
    # tile-by-n_frame_stack.
    raw_dim = ns.mean.shape[-1]
    obs_dim = obs.shape[-1]
    if n_frame_stack > 1 and raw_dim * n_frame_stack == obs_dim:
        return norm_normalize_stacked(ns, obs, n_frame_stack)
    return norm_normalize(ns, obs)


def build_ppo_rollout_step(algo, training_state, norm_state, env_step,
                           kicks_fn=None, n_frame_stack: int = 1):
    """Build a PPO rollout step for jax.lax.scan.

    Frozen norm_state — no stats update during rollout (matches eval contract).

    Args:
        algo: PPO algo instance (used for select_action).
        training_state: Frozen training state with actor_params.
        norm_state: Running observation normalization state (frozen).
        env_step: JIT-compiled env.step function.
        kicks_fn: Optional (env_state, step_idx, key) -> (env_state, key).
        n_frame_stack: Env frame-stack depth. >1 routes through normalize_stacked.

    Returns:
        (rollout_step_fn, init_carry) where carry = (env_state, key).
    """
    frozen_state = training_state
    frozen_norm = norm_state

    def rollout_step(carry, step_idx):
        env_state, key = carry
        if kicks_fn is not None:
            env_state, key = kicks_fn(env_state, step_idx, key)
        obs = _extract_policy_obs(env_state.obs)
        normed_obs = _apply_norm(frozen_norm, obs, n_frame_stack)
        key, action_key = jax.random.split(key)
        action, _, _ = algo.select_action(frozen_state, normed_obs, action_key,
                                          deterministic=True)
        clipped_action = jnp.clip(action, -1.0, 1.0).squeeze(0)
        env_state = env_step(env_state, clipped_action)
        return (env_state, key), (env_state, clipped_action)

    return rollout_step, None  # init_carry built by caller


def build_offpolicy_rollout_step(algo, actor_params, norm_state, env_step,
                                 use_obs_norm, kicks_fn=None,
                                 n_frame_stack: int = 1):
    """Build an off-policy (SAC/TD3) rollout step for jax.lax.scan.

    Args:
        algo: SAC/TD3 algo instance (used for select_action).
        actor_params: Frozen actor parameters.
        norm_state: Frozen observation normalization state.
        env_step: JIT-compiled env.step function.
        use_obs_norm: Whether to apply obs normalization.
        kicks_fn: Optional (env_state, step_idx, key) -> (env_state, key).
        n_frame_stack: Env frame-stack depth. >1 routes through normalize_stacked.

    Returns:
        (rollout_step_fn, init_carry_fn) — caller builds init_carry as
        (env_state, key).
    """
    frozen_params = actor_params
    frozen_norm = norm_state

    def rollout_step(carry, step_idx):
        env_state, key = carry
        if kicks_fn is not None:
            env_state, key = kicks_fn(env_state, step_idx, key)
        obs = _extract_policy_obs(env_state.obs)
        if use_obs_norm:
            obs = _apply_norm(frozen_norm, obs, n_frame_stack)
        key, action_key = jax.random.split(key)
        action = algo.select_action(frozen_params, obs, action_key,
                                    deterministic=True)
        clipped_action = action.squeeze(0)
        env_state = env_step(env_state, clipped_action)
        return (env_state, key), (env_state, clipped_action)

    return rollout_step, None  # init_carry built by caller
