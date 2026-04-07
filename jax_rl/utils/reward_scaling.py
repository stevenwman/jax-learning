"""Adaptive reward scaling for FlashSAC.

Tracks running discounted returns per environment and uses their variance to
scale rewards, keeping values within the distributional critic's support bounds.

Usage::

    state = init_reward_norm(num_envs=16)

    # During collection, after each env step:
    state = update_reward_stats(state, reward, terminated, truncated, gamma=0.99)

    # Scale rewards before storing in replay buffer:
    scaled_reward = scale_reward(state, reward, G_max=5.0)

Reference: FlashSAC PyTorch implementation (RewardNormalizer + RunningMeanStd).
"""

import flax
import jax.numpy as jnp


@flax.struct.dataclass
class RewardNormState:
    """Pure-functional state for adaptive reward scaling.

    Fields:
        G_r: Running discounted return per environment, shape (num_envs,).
        G_r_max: All-time max of |G_r| across all envs, scalar.
        G_mean: Welford running mean of G_r samples, scalar.
        G_var: Welford running variance of G_r samples, scalar.
        G_count: Total number of samples seen by Welford estimator, scalar.
    """
    G_r: jnp.ndarray
    G_r_max: jnp.ndarray
    G_mean: jnp.ndarray
    G_var: jnp.ndarray
    G_count: jnp.ndarray


def init_reward_norm(num_envs: int) -> RewardNormState:
    """Create a fresh RewardNormState.

    Args:
        num_envs: Number of parallel environments.

    Returns:
        RewardNormState with G_r=zeros, G_r_max=0, G_mean=0, G_var=1, G_count=0.
    """
    return RewardNormState(
        G_r=jnp.zeros(num_envs),
        G_r_max=jnp.array(0.0),
        G_mean=jnp.array(0.0),
        G_var=jnp.array(1.0),
        G_count=jnp.array(0.0),
    )


def update_reward_stats(
    state: RewardNormState,
    reward: jnp.ndarray,
    terminated: jnp.ndarray,
    truncated: jnp.ndarray,
    gamma: float,
) -> RewardNormState:
    """Update running discounted returns and Welford variance statistics.

    Both terminated and truncated signals reset the per-env return to zero
    before accumulating the new reward (done = max(terminated, truncated)).

    The Welford update uses the non-standard epsilon-stabilized formula from
    the FlashSAC reference: m_a = G_var * (G_count + 1e-4) rather than
    G_var * G_count. This avoids variance collapse early in training.

    Args:
        state: Current normalization state.
        reward: Per-env rewards, shape (num_envs,).
        terminated: Per-env terminal flags (0 or 1), shape (num_envs,).
        truncated: Per-env truncation flags (0 or 1), shape (num_envs,).
        gamma: Discount factor.

    Returns:
        Updated RewardNormState.
    """
    done = jnp.maximum(terminated, truncated)
    G_r = gamma * (1.0 - done) * state.G_r + reward
    G_r_max = jnp.maximum(state.G_r_max, jnp.max(jnp.abs(G_r)))

    # Welford batch update (non-standard epsilon variant from reference)
    sample_mean = jnp.mean(G_r)
    sample_var = jnp.var(G_r)  # biased variance (ddof=0)
    sample_count = G_r.shape[0]

    delta = sample_mean - state.G_mean
    total_count = state.G_count + sample_count
    ratio = sample_count / total_count

    new_mean = state.G_mean + delta * ratio
    m_a = state.G_var * (state.G_count + 1e-4)  # epsilon=1e-4, NOT G_count alone
    m_b = sample_var * sample_count
    M2 = m_a + m_b + delta ** 2 * state.G_count * ratio
    new_var = M2 / total_count

    return state.replace(
        G_r=G_r,
        G_r_max=G_r_max,
        G_mean=new_mean,
        G_var=new_var,
        G_count=total_count,
    )


def scale_reward(
    state: RewardNormState,
    reward: jnp.ndarray,
    G_max: float = 5.0,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Scale rewards so that G_r stays within [-G_max, G_max].

    The denominator is the larger of:
    - sqrt(G_var + eps): variance-based scaling
    - G_r_max / G_max: hard clip to keep returns in bounds

    Args:
        state: Current normalization state.
        reward: Raw rewards to scale.
        G_max: Target max magnitude for discounted returns.
        eps: Small constant for numerical stability.

    Returns:
        Scaled rewards, same shape as reward.
    """
    denominator = jnp.maximum(
        jnp.sqrt(state.G_var + eps),
        state.G_r_max / G_max,
    )
    return reward / denominator
