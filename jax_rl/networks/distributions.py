"""Distribution utilities for policies.

Provides helpers for sampling actions and computing log probabilities.
Uses distrax for clean, composable distributions.
"""

import jax
import jax.numpy as jnp
import distrax

# Epsilon to avoid numerical issues with atanh(±1) = ±∞
# When computing log_prob of tanh-squashed actions, we need atanh which has
# singularities at ±1. Clipping actions to [-1+eps, 1-eps] prevents this.
ATANH_EPSILON = 1e-6


def sample_gaussian(
    mean: jax.Array,
    log_std: jax.Array,
    key: jax.random.PRNGKey,
    squash: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Sample from Gaussian distribution (optionally squashed with tanh).

    Args:
        mean: Mean of Gaussian, shape (batch, action_dim) or (action_dim,)
        log_std: Log std of Gaussian, shape (batch, action_dim) or (action_dim,)
        key: PRNGKey for sampling
        squash: If True, apply tanh to squash to [-1, 1]

    Returns:
        (action, log_prob) tuple:
            action: Sampled action, shape (batch, action_dim) or (action_dim,)
            log_prob: Log probability of action, shape (batch,) or scalar (summed over action dims)
    """
    std = jnp.exp(log_std)
    base_dist = distrax.Normal(loc=mean, scale=std)

    if squash:
        # TanhNormal for bounded actions
        dist = distrax.Transformed(base_dist, distrax.Tanh())
    else:
        dist = base_dist

    action, log_prob = dist.sample_and_log_prob(seed=key)

    # Sum log_prob over action dimensions (assuming independent actions)
    log_prob = log_prob.sum(axis=-1)

    return action, log_prob


def gaussian_log_prob(
    mean: jax.Array,
    log_std: jax.Array,
    action: jax.Array,
    squash: bool = True,
) -> jax.Array:
    """Compute log probability of action under Gaussian distribution.

    Args:
        mean: Mean of Gaussian, shape (batch, action_dim) or (action_dim,)
        log_std: Log std of Gaussian, shape (batch, action_dim) or (action_dim,)
        action: Action to evaluate, shape (batch, action_dim) or (action_dim,)
        squash: If True, use tanh-squashed distribution

    Returns:
        Log probability, shape (batch,) or scalar
    """
    std = jnp.exp(log_std)
    base_dist = distrax.Normal(loc=mean, scale=std)

    if squash:
        # Clip actions away from ±1 to avoid atanh singularities
        action = jnp.clip(action, -1.0 + ATANH_EPSILON, 1.0 - ATANH_EPSILON)
        dist = distrax.Transformed(base_dist, distrax.Tanh())
    else:
        dist = base_dist

    log_prob = dist.log_prob(action)

    # Sum over action dimensions
    log_prob = log_prob.sum(axis=-1)

    return log_prob


def entropy_gaussian(log_std: jax.Array) -> jax.Array:
    """Compute entropy of Gaussian distribution (before tanh squashing).

    Args:
        log_std: Log std of Gaussian, shape (batch, action_dim) or (action_dim,)

    Returns:
        Entropy, shape (batch,) or scalar (summed over action dims)
    """
    # Entropy of Gaussian: 0.5 * log(2 * pi * e * std^2)
    #                    = 0.5 * (log(2*pi) + 1 + 2*log_std)
    entropy = 0.5 * (jnp.log(2 * jnp.pi) + 1 + 2 * log_std)

    # Sum over action dimensions
    return entropy.sum(axis=-1)
