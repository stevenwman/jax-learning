"""Value head for state value estimation V(s).

Used by PPO's critic. Takes encoder features and outputs a single scalar
value estimate per state. The squeeze removes the trailing dim from
Dense(1) output: (batch, 1) → (batch,).
"""

from flax import linen as nn
import jax
import jax.numpy as jnp


class ValueHead(nn.Module):
    """Scalar value head: features → V(s).

    Input:  features of shape (batch, feature_dim) from encoder
    Output: value estimates of shape (batch,)
    """

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        # Dense(1) outputs (batch, 1), squeeze to (batch,) for loss computation
        return jnp.squeeze(
            nn.Dense(1, kernel_init=nn.initializers.lecun_uniform())(features),
            axis=-1,
        )
