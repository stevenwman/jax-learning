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
    key: jax.Array,
    squash: bool = False,
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
    squash: bool = False,
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


def entropy_gaussian(log_std: jax.Array, mean: jax.Array | None = None,
                     key: jax.Array | None = None, squash: bool = False) -> jax.Array:
    """Compute entropy of Gaussian distribution.

    When squash=False, uses the closed-form Gaussian entropy.
    When squash=True, estimates entropy of the tanh-squashed distribution
    via single-sample estimate: -E[log p(tanh(x))], matching Brax.

    Args:
        log_std: Log std of Gaussian, shape (batch, action_dim) or (action_dim,)
        mean: Mean of Gaussian (required when squash=True)
        key: PRNGKey for sampling (required when squash=True)
        squash: If True, compute tanh-corrected entropy

    Returns:
        Entropy, shape (batch,) or scalar (summed over action dims)
    """
    if squash:
        # Single-sample entropy estimate of tanh-normal (matches Brax)
        std = jnp.exp(log_std)
        dist = distrax.Normal(loc=mean, scale=std)
        raw_actions = dist.sample(seed=key)
        # Base Gaussian log_prob
        log_prob = dist.log_prob(raw_actions)
        # Tanh Jacobian correction: log |d tanh / dx| = log(1 - tanh(x)^2)
        log_prob -= jnp.log(1 - jnp.tanh(raw_actions) ** 2 + 1e-6)
        # Entropy = -E[log p(x)], summed over action dims
        return -log_prob.sum(axis=-1)
    else:
        # Closed-form Gaussian entropy: 0.5 * log(2 * pi * e * std^2)
        entropy = 0.5 * (jnp.log(2 * jnp.pi) + 1 + 2 * log_std)
        return entropy.sum(axis=-1)
