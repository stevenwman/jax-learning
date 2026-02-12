"""Policy and value heads."""

import jax
import jax.numpy as jnp
from flax import nnx


class GaussianHead(nnx.Module):
    """Gaussian policy head with state-dependent log_std."""
    
    def __init__(self, feature_dim: int, action_dim: int, rngs: nnx.Rngs):
        self.mean_layer = nnx.Linear(feature_dim, action_dim, rngs=rngs)
        self.log_std_layer = nnx.Linear(feature_dim, action_dim, rngs=rngs)
    
    def __call__(self, features: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass.
        
        Args:
            features: Shape (batch, feature_dim)
        
        Returns:
            mean: Shape (batch, action_dim)
            log_std: Shape (batch, action_dim)
        """
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        # Clamp log_std for stability
        log_std = jnp.clip(log_std, -20, 2)
        return mean, log_std


class ValueHead(nnx.Module):
    """State value head V(s)."""
    
    def __init__(self, feature_dim: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(feature_dim, 1, rngs=rngs)
    
    def __call__(self, features: jax.Array) -> jax.Array:
        """Forward pass.
        
        Args:
            features: Shape (batch, feature_dim)
        
        Returns:
            value: Shape (batch, 1)
        """
        return self.linear(features)
