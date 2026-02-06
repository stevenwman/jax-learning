from flax import nnx
import jax
import jax.numpy as jnp
import distrax

class policy(nnx.Module):

    def __init__(self, encoder, head, rng: nnx.Rngs):
        self.encoder = encoder
        self.head = head

    def __call__(self, obs):
        return self.head(self.encoder(obs))

    def sample(self, obs, key: jax.random.PRNGKey):
        mu, log_std = self(obs)
        distr = distrax.Normal(loc=mu, scale=jnp.exp(log_std))
        return distr.sample(seed=key)