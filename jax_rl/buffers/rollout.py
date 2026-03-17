"""Rollout buffer for on-policy algorithms (PPO)."""

import jax
import jax.numpy as jnp
from typing import NamedTuple


class RolloutBatch(NamedTuple):
    """A batch of rollout data."""

    obs: jax.Array  # (num_steps, num_envs, obs_dim)
    actions: jax.Array  # (num_steps, num_envs, action_dim)
    rewards: jax.Array  # (num_steps, num_envs)
    dones: jax.Array  # (num_steps, num_envs)
    log_probs: jax.Array  # (num_steps, num_envs)
    values: jax.Array  # (num_steps, num_envs)
    advantages: jax.Array | None = None  # (num_steps, num_envs) - computed after collection
    returns: jax.Array | None = None  # (num_steps, num_envs) - computed after collection


class RolloutBuffer:
    """Buffer for storing and processing on-policy rollouts.

    Stores trajectories from parallel environments and computes GAE advantages.
    """

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int) -> None:
        """Initialize rollout buffer.

        Args:
            num_steps: Number of steps per rollout
            num_envs: Number of parallel environments
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Initialize storage
        self.obs = jnp.zeros((num_steps, num_envs, obs_dim))
        self.actions = jnp.zeros((num_steps, num_envs, action_dim))
        self.rewards = jnp.zeros((num_steps, num_envs))
        self.dones = jnp.zeros((num_steps, num_envs))
        self.truncations = jnp.zeros((num_steps, num_envs))
        self.log_probs = jnp.zeros((num_steps, num_envs))
        self.values = jnp.zeros((num_steps, num_envs))
        self.ptr = 0

    def add(
        self,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        truncation: jax.Array,
        log_prob: jax.Array,
        value: jax.Array,
    ) -> None:
        """Add a step of experience to the buffer.

        Args:
            obs: Observations, shape (num_envs, obs_dim)
            action: Actions, shape (num_envs, action_dim)
            reward: Rewards, shape (num_envs,)
            done: Done flags, shape (num_envs,)
            truncation: Truncation flags, shape (num_envs,)
            log_prob: Log probabilities, shape (num_envs,)
            value: State values, shape (num_envs,)
        """
        self.obs = self.obs.at[self.ptr].set(obs)
        self.actions = self.actions.at[self.ptr].set(action)
        self.rewards = self.rewards.at[self.ptr].set(reward)
        self.dones = self.dones.at[self.ptr].set(done)
        self.truncations = self.truncations.at[self.ptr].set(truncation)
        self.log_probs = self.log_probs.at[self.ptr].set(log_prob)
        self.values = self.values.at[self.ptr].set(value)
        self.ptr += 1

    def reset(self) -> None:
        """Reset buffer pointer."""
        self.ptr = 0

    def get(self, next_value: jax.Array, gamma: float, gae_lambda: float) -> RolloutBatch:
        """Get rollout batch with computed advantages and returns.

        Args:
            next_value: Value of next state (for bootstrapping), shape (num_envs,)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            RolloutBatch with computed advantages and returns
        """
        advantages, returns = compute_gae(
            self.rewards,
            self.values,
            self.dones,
            self.truncations,
            next_value,
            gamma,
            gae_lambda,
        )

        return RolloutBatch(
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            log_probs=self.log_probs,
            values=self.values,
            advantages=advantages,
            returns=returns,
        )


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    truncations: jax.Array,
    next_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation (GAE).

    Uses backward scan for efficient computation:
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = δ_t + (γ * λ) * (1 - done_t) * A_{t+1}

    Truncation handling:
        When an episode truncates (timeout), the env auto-resets and the stored
        next obs is the RESET obs, not the terminal obs. So values[t+1] would be
        V(reset_obs) instead of V(truncated_obs). We approximate V(truncated_obs)
        ≈ V(s_t) (the value before the action), matching Brax's approach.

    Args:
        rewards: Rewards, shape (num_steps, num_envs)
        values: State values, shape (num_steps, num_envs)
        dones: Done flags, shape (num_steps, num_envs)
        truncations: Truncation flags, shape (num_steps, num_envs)
        next_value: Value of next state after last step, shape (num_envs,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (bias-variance tradeoff)

    Returns:
        (advantages, returns) tuple:
            advantages: GAE advantages, shape (num_steps, num_envs)
            returns: TD(λ) returns, shape (num_steps, num_envs)
    """
    # Compute next values: V(s_{t+1}) for each timestep
    next_values = jnp.concatenate([values[1:], next_value[None]], axis=0)

    # For truncated transitions, replace V(reset_obs) with V(s_t) as bootstrap
    # approximation. This avoids using the value of the wrong state.
    next_values = jnp.where(truncations, values, next_values)

    # Effective done for GAE: true terminals stop bootstrap, truncations don't
    # (the bootstrap value correction above handles truncations instead)
    effective_dones = dones * (1.0 - truncations)

    def scan_fn(gae: jax.Array, t: int) -> tuple[jax.Array, jax.Array]:
        # TD error: use effective_dones (0 for truncation → enables bootstrap)
        delta = rewards[t] + gamma * next_values[t] * (1 - effective_dones[t]) - values[t]

        # GAE propagation: use actual dones (1 for truncation → stops leakage
        # from next episode's advantages into current episode)
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae

        return gae, gae

    # Run backward scan from T-1 to 0
    _, advantages = jax.lax.scan(
        scan_fn,
        init=jnp.zeros(next_value.shape[0]),
        xs=jnp.arange(rewards.shape[0])[::-1],
    )

    # Reverse advantages back to forward order
    advantages = advantages[::-1]

    # Compute returns: R_t = A_t + V(s_t)
    returns = advantages + values

    return advantages, returns
