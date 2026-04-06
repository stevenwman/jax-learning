"""Distributional Q-value head for FastTD3/FastSAC.

Same architecture as QHead but outputs (batch, num_atoms) logits instead of scalar.
Used with C51 categorical distribution over return atoms.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax_rl.networks.activations import ACTIVATIONS


class DistributionalQHead(nn.Module):
    """MLP Q-network: concat(obs, action) → (batch, num_atoms) logits.

    Same hidden structure as QHead. Final layer outputs num_atoms logits
    instead of a single scalar.

    Attributes:
        hidden_dim: Sizes of hidden layers, e.g. (256, 256).
        num_atoms: Number of categorical atoms (default 51 for C51).
        activation: Activation function name.
        layer_norm: Whether to apply LayerNorm after each hidden layer.
    """
    hidden_dim: tuple
    num_atoms: int = 51
    activation: str = "relu"
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], axis=-1)
        act_fn = ACTIVATIONS[self.activation]
        for d in self.hidden_dim:
            x = nn.Dense(d, kernel_init=nn.initializers.lecun_uniform())(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)
        # Output: num_atoms logits (unnormalized log-probs over support)
        return nn.Dense(self.num_atoms, kernel_init=nn.initializers.lecun_uniform())(x)
