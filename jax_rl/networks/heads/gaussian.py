"""Gaussian policy head for stochastic policies (PPO, SAC)."""

import jax
from flax import linen as nn
import jax.numpy as jnp
from jax_rl.configs.networks_config import PolicyHeadConfig


class GaussianHead(nn.Module):
    config: PolicyHeadConfig

    @nn.compact
    def __call__(self, features: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = nn.Dense(self.config.action_dim)(features)
        log_std = nn.Dense(self.config.action_dim)(features)
        log_std = jnp.clip(log_std, self.config.log_std_min, self.config.log_std_max)
        return mean, log_std