"""Neural network encoders."""

import jax.numpy as jnp
from flax import nnx

from configs import EncoderConfig


class MLPEncoder(nnx.Module):
    """MLP encoder with configurable hidden layers."""
    
    def __init__(self, config: EncoderConfig, rngs: nnx.Rngs):
        self.config = config
        
        # Build layers
        self.layers = nnx.List([])
        in_dim = config.obs_dim
        
        for hidden_dim in config.hidden_dim:
            self.layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            x: Observations, shape (batch, obs_dim)
        
        Returns:
            features: Shape (batch, hidden_dim[-1])
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # ReLU between layers, no activation on final
            if i < len(self.layers) - 1:
                x = nnx.relu(x)
        return x
