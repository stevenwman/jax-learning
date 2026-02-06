from config import EncoderConfig
from encoder import MLPEncoder
from flax import nnx

config = EncoderConfig(obs_dim=17)
encoder = MLPEncoder(config, rngs=nnx.Rngs(0))

# Check shapes
import jax.numpy as jnp
obs = jnp.ones((32, 17))  # batch of 32, obs_dim=17
out = encoder(obs)
print(out.shape)  # Should be (32, 256)