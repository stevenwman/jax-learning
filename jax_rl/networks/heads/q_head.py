"""Q-value head for SAC / off-policy methods.

Takes (obs, action) concatenated → scalar Q-value.
Supports optional LayerNorm after each hidden layer (Brax default: enabled).
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax_rl.networks.activations import ACTIVATIONS


class QHead(nn.Module):
    """MLP Q-network: concat(obs, action) → scalar Q-value.

    Attributes:
        hidden_dim: Sizes of hidden layers, e.g. (256, 256).
        activation: Activation function name (must be in ACTIVATIONS).
        layer_norm: Whether to apply LayerNorm after each hidden layer.
                    True matches Brax's q_network_layer_norm=True.
    """
    hidden_dim: tuple
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
        return jnp.squeeze(
            nn.Dense(1, kernel_init=nn.initializers.lecun_uniform())(x),
            axis=-1,
        )
