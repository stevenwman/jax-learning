"""Value head for state value estimation V(s)."""

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax_rl.configs.networks_config import ValueHeadConfig



class ValueHead(nn.Module):
    config: ValueHeadConfig

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        return jnp.squeeze(nn.Dense(1, kernel_init=nn.initializers.lecun_uniform())(features), axis=-1)