from encoder import MLPEncoder
from heads import GaussianHead
from config import EncoderConfig
from policy import policy
from flax import nnx
import jax.numpy as jnp
import jax

config = EncoderConfig(obs_dim=17)
rngs = nnx.Rngs(0)

encoder = MLPEncoder(config, rngs=rngs)
head = GaussianHead(feature_dim=256, action_dim=6, rngs=rngs)

obs = jnp.ones((32, 17))
features = encoder(obs)
mean, log_std = head(features)

print(features.shape)  # (32, 256)
print(mean.shape)      # (32, 6)
print(log_std.shape)   # (32, 6)

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

policy = policy(encoder, head, rngs)
sample = policy.sample(obs, subkey)

print(sample.shape)