from flax import nnx
import jax
import jax.numpy as jnp
from config import EncoderConfig

class MLPEncoder(nnx.Module):

    def __init__(self, config: EncoderConfig, rngs: nnx.Rngs) -> None:
        net_dims = [config.obs_dim] + list(config.hidden_dim)
        self.layers = nnx.List([nnx.Linear(d_in, d_out, rngs=rngs) 
               for d_in, d_out in zip(net_dims[:-1], net_dims[1:])])

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = nnx.relu(x) if i < len(self.layers) - 1 else x
        return x