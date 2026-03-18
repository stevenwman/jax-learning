"""Deterministic policy head for TD3.

Maps encoder features → bounded action via tanh.
No stochasticity — exploration is handled externally via additive noise.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn


class DeterministicHead(nn.Module):
    """Deterministic policy: features → tanh(Dense(action_dim)).

    Args:
        action_dim: Number of action dimensions.
    """
    action_dim: int

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        return jnp.tanh(
            nn.Dense(self.action_dim, kernel_init=nn.initializers.lecun_uniform())(features)
        )
