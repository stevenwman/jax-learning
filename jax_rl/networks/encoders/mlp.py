"""MLP encoder implementation."""

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax_rl.configs.networks_config import EncoderConfig


# Activation function mapping
ACTIVATIONS = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
    "gelu": jax.nn.gelu,
    "swish": jax.nn.swish,
}

class MlpEncoder(nn.Module):
    config: EncoderConfig

    @nn.compact
    def __call__(self, obs: jax.Array, context: jax.Array = None) -> jax.Array:
        if context is not None:
            obs = jnp.concatenate([obs, context], axis=-1)
        for d_out in self.config.hidden_dim:
            obs = nn.Dense(d_out, kernel_init=nn.initializers.lecun_uniform())(obs)
            if self.config.norm is not None:
                obs = nn.LayerNorm()(obs)
            obs = ACTIVATIONS[self.config.activation](obs)
        return obs

    @property
    def feature_dim(self) -> int:
        return self.config.hidden_dim[-1]