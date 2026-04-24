"""Factory for the PPO lax.scan collect loop with optional hooks.

Extracted from train_ppo_fast.py so PPO and PPO+contraction (and future on-policy
variants like RND / intrinsic motivation) share the scan body. The outer training
loop (eval, checkpoint, W&B) stays duplicated in the train scripts — by design,
this is a partial extraction. See .superpowers/plans/2026-04-21-onpolicy-loop-extraction.md.

Hooks:
- extra_rollout_fn(training_state, env_state, key) -> NamedTuple extras
    Called inside collect_step BEFORE env_step. Return value is stacked per step
    and accessible on the returned rollout via `rollout.extras`.
- reward_augment_fn(training_state, env_state, extras) -> (num_envs,) bonus
    Called inside collect_step AFTER env_step and AFTER reward_scaling. Added to
    the step reward. `extras` is the NamedTuple returned by extra_rollout_fn
    (same step), or None if no extras hook set.

Both hooks are compile-time constants: captured by closure in this factory.
Setting them None falls back to the bit-identical baseline scan.
"""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from jax_rl.utils.normalization import (
    normalize as norm_normalize,
    normalize_stacked as norm_normalize_stacked,
    update as norm_update,
)


class StepData(NamedTuple):
    """Per-step scan output.

    `extras` is a NamedTuple from `extra_rollout_fn`, or `()` (empty pytree)
    when no hook is set. Empty tuple is required rather than None because
    lax.scan stacks per-step outputs and can't stack None leaves.
    """
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    truncation: jax.Array
    extras: Any = ()


def make_collect(
    *,
    env_step: Callable,
    select_stochastic: Callable,
    select_deterministic: Callable,
    num_steps: int,
    num_envs: int,
    reward_scaling: float,
    handle_truncation: bool,
    get_policy_obs: Callable,
    get_critic_obs: Callable,
    n_frame_stack: int,
    policy_raw_dim: int,
    critic_raw_dim: int,
    extra_rollout_fn: Callable | None = None,
    reward_augment_fn: Callable | None = None,
):
    """Build the JIT'd `_collect` function.

    Returns:
        _collect(training_state, env_state, norm_state, critic_norm_state,
                 key, running_ep_return) -> (env_state, norm_state, critic_norm_state,
                                              key, rollout, normed_next, normed_next_critic,
                                              next_value, ep_count, ep_return_sum, running_ep_return)
    """
    has_extras = extra_rollout_fn is not None
    has_reward_aug = reward_augment_fn is not None

    @jax.jit
    def _collect(training_state, env_state, norm_state, critic_norm_state, key, running_ep_return):
        def collect_step(carry, _unused):
            env_state, ns, cns, key, ep_return, ep_count, ep_return_sum = carry

            policy_obs = get_policy_obs(env_state.obs)
            critic_obs = get_critic_obs(env_state.obs)

            normed_obs = (norm_normalize_stacked(ns, policy_obs, n_frame_stack)
                          if n_frame_stack > 1 else norm_normalize(ns, policy_obs))
            # Critic (privileged) never frame-stacked → plain normalize.
            normed_critic_obs = norm_normalize(cns, critic_obs)

            # Capture pre-step extras (matches ref ContractionPPO off-by-one semantics)
            if has_extras:
                key, ex_key = jax.random.split(key)
                extras = extra_rollout_fn(training_state, env_state, ex_key)
            else:
                extras = ()  # empty pytree; scan-safe placeholder

            key, action_key = jax.random.split(key)
            action, log_prob, value = select_stochastic(
                training_state.actor_params, training_state.critic_params,
                normed_obs, normed_critic_obs, action_key,
            )
            clipped_action = jnp.clip(action, -1.0, 1.0)

            env_state = env_step(env_state, clipped_action)

            truncation = jnp.where(
                handle_truncation,
                env_state.info["truncation"],
                jnp.zeros_like(env_state.done),
            )

            scaled_reward = env_state.reward * reward_scaling
            if has_reward_aug:
                bonus = reward_augment_fn(training_state, env_state, extras)
                scaled_reward = scaled_reward + bonus

            ep_return = ep_return + env_state.reward
            done_mask = env_state.done
            ep_return_sum = ep_return_sum + jnp.sum(ep_return * done_mask)
            ep_count = ep_count + jnp.sum(done_mask)
            ep_return = ep_return * (1.0 - done_mask)

            step_data = StepData(
                obs=normed_obs,
                critic_obs=normed_critic_obs,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=scaled_reward,
                done=env_state.done,
                truncation=truncation,
                extras=extras,
            )

            return (env_state, ns, cns, key, ep_return, ep_count, ep_return_sum), (step_data, policy_obs, critic_obs)

        init_ep_count = jnp.zeros(())
        init_ep_return_sum = jnp.zeros(())

        (env_state, norm_state, critic_norm_state, key, running_ep_return, ep_count, ep_return_sum), (rollout, raw_policy_obs, raw_critic_obs) = jax.lax.scan(
            collect_step,
            (env_state, norm_state, critic_norm_state, key, running_ep_return, init_ep_count, init_ep_return_sum),
            None,
            length=num_steps,
        )

        flat_policy_obs = raw_policy_obs.reshape(-1, raw_policy_obs.shape[-1])
        flat_critic_obs = raw_critic_obs.reshape(-1, raw_critic_obs.shape[-1])
        # Policy may be stacked (newest frame only); critic is never stacked.
        if n_frame_stack > 1:
            flat_policy_obs = flat_policy_obs[:, :policy_raw_dim]
        norm_state = norm_update(norm_state, flat_policy_obs)
        critic_norm_state = norm_update(critic_norm_state, flat_critic_obs)

        next_policy_obs = get_policy_obs(env_state.obs)
        next_critic_obs = get_critic_obs(env_state.obs)
        normed_next = (norm_normalize_stacked(norm_state, next_policy_obs, n_frame_stack)
                       if n_frame_stack > 1 else norm_normalize(norm_state, next_policy_obs))
        normed_next_critic = norm_normalize(critic_norm_state, next_critic_obs)
        _, _, next_value = select_deterministic(
            training_state.actor_params, training_state.critic_params,
            normed_next, normed_next_critic,
        )

        return (env_state, norm_state, critic_norm_state, key,
                rollout, normed_next, normed_next_critic, next_value,
                ep_count, ep_return_sum, running_ep_return)

    return _collect
