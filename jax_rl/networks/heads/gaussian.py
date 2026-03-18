"""Gaussian policy head for stochastic policies (PPO, SAC)."""

import jax
from flax import linen as nn
import jax.numpy as jnp
from jax_rl.configs.networks_config import PolicyHeadConfig


class GaussianHead(nn.Module):
    config: PolicyHeadConfig

    @nn.compact
    def __call__(self, features: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = nn.Dense(self.config.action_dim, kernel_init=nn.initializers.lecun_uniform())(features)

        if self.config.state_dependent_std:
            # State-dependent: Dense → softplus + min_std (matches Brax tanh_normal)
            raw_scale = nn.Dense(self.config.action_dim, kernel_init=nn.initializers.lecun_uniform())(features)
            std = jax.nn.softplus(raw_scale) + self.config.min_std
            log_std = jnp.log(std)
        else:
            # State-independent: single learned vector, same for all obs
            log_std = self.param(
                'log_std',
                nn.initializers.constant(jnp.log(self.config.init_noise_std)),
                (self.config.action_dim,),
            )
            log_std = jnp.broadcast_to(log_std, mean.shape)
            log_std = jnp.clip(log_std, self.config.log_std_min, self.config.log_std_max)

        return mean, log_std
