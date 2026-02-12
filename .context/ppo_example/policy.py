"""Policy networks."""

import jax
import jax.numpy as jnp
import distrax
from flax import nnx

from encoder import MLPEncoder
from heads import GaussianHead


class Policy(nnx.Module):
    """Policy that composes encoder + head.
    
    Args:
        encoder: Feature encoder (e.g., MLPEncoder)
        head: Policy head (e.g., GaussianHead)
        squash: If True, apply tanh squashing (TanhNormal for SAC)
    """
    
    def __init__(self, encoder: MLPEncoder, head: GaussianHead, squash: bool = False):
        self.encoder = encoder
        self.head = head
        self.squash = squash
    
    def _make_dist(self, mean: jax.Array, std: jax.Array) -> distrax.Distribution:
        """Create action distribution."""
        base_dist = distrax.Normal(mean, std)
        
        if self.squash:
            # TanhNormal: squash to [-1, 1]
            return distrax.Transformed(base_dist, distrax.Tanh())
        else:
            return distrax.Independent(base_dist, reinterpreted_batch_ndims=1)
    
    def sample(self, obs: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Sample action and compute log probability.
        
        Args:
            obs: Observations, shape (batch, obs_dim)
            key: PRNG key
        
        Returns:
            action: Shape (batch, action_dim)
            log_prob: Shape (batch,)
        """
        features = self.encoder(obs)
        mean, log_std = self.head(features)
        std = jnp.exp(log_std)
        
        dist = self._make_dist(mean, std)
        action = dist.sample(seed=key)
        log_prob = dist.log_prob(action)
        
        # Sum log probs for multi-dim actions if not using Independent
        if self.squash and log_prob.ndim > 1:
            log_prob = log_prob.sum(axis=-1)
        
        return action, log_prob
    
    def log_prob(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Compute log probability of action.
        
        Args:
            obs: Observations, shape (batch, obs_dim)
            action: Actions, shape (batch, action_dim)
        
        Returns:
            log_prob: Shape (batch,)
        """
        features = self.encoder(obs)
        mean, log_std = self.head(features)
        std = jnp.exp(log_std)
        
        dist = self._make_dist(mean, std)
        log_prob = dist.log_prob(action)
        
        if self.squash and log_prob.ndim > 1:
            log_prob = log_prob.sum(axis=-1)
        
        return log_prob
