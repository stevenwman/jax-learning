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

        if self.config.state_dependent_std:
            # State-dependent: Dense layer maps features → log_std (SAC)
            log_std = nn.Dense(self.config.action_dim)(features)
        else:
            # State-independent: single learned vector, same for all obs (PPO)
            # Initialized to log(init_noise_std) so std starts at init_noise_std
            log_std = self.param(
                'log_std',
                nn.initializers.constant(jnp.log(self.config.init_noise_std)),
                (self.config.action_dim,),
            )
            log_std = jnp.broadcast_to(log_std, mean.shape)

        log_std = jnp.clip(log_std, self.config.log_std_min, self.config.log_std_max)
        return mean, log_std
