from flax import nnx
import jax
import jax.numpy as jnp
import distrax

class Policy(nnx.Module):

    def __init__(self, encoder, head, squash: bool = False):
        self.encoder = encoder
        self.head = head
        self.squash = squash

    def __call__(self, obs):
        return self.head(self.encoder(obs))

    def sample(self, obs, key: jax.random.PRNGKey):
        mu, log_std = self(obs)
        base_distr = distrax.Normal(loc=mu, scale=jnp.exp(log_std))

        if self.squuash: 
            distr = distrax.Transformed(base_distr, distrax.Tanh())
        else:
            distr = base_distr

        action, log_prob = distr.sample_and_log_prob(seed=key)
        return action, log_prob.sum(axis=-1)  # sum over action dims