"""Gaussian policy head for stochastic policies (PPO, SAC, FastSAC).

Outputs (mean, log_std) for a diagonal Gaussian action distribution.
The caller (via distributions.py) samples actions as:
    action = tanh(mean + std * noise)    [squashed Gaussian]

Two std modes:
    state_dependent_std=True:  std = softplus(Dense(features)) + min_std
    state_dependent_std=False: std = learned parameter (same for all states)

Optional DEM (Dimension-wise Entropy Modulation) for FastDSAC:
    When config.dem=True, outputs a third tensor (dem_logits) that controls
    per-action-dimension exploration weighting. See LESSONS.md "DEM".
"""

import jax
from flax import linen as nn
import jax.numpy as jnp
from jax_rl.configs.networks_config import PolicyHeadConfig


class GaussianHead(nn.Module):
    """Gaussian policy head.

    Attributes:
        config: PolicyHeadConfig specifying action_dim, std mode, noise bounds.

    Input:  features from encoder, shape (batch, feature_dim)
    Output: (mean, log_std) each shape (batch, action_dim)
            or (mean, log_std, dem_logits) if config.dem=True
    """
    config: PolicyHeadConfig

    @nn.compact
    def __call__(self, features: jax.Array) -> tuple[jax.Array, ...]:
        mean = nn.Dense(self.config.action_dim, kernel_init=nn.initializers.lecun_uniform())(features)

        if self.config.state_dependent_std:
            # State-dependent: Dense → softplus + min_std (matches Brax tanh_normal)
            raw_scale = nn.Dense(self.config.action_dim, kernel_init=nn.initializers.lecun_uniform())(features)
            std = jax.nn.softplus(raw_scale) + self.config.min_std
            # Optional max_std cap (FastSAC paper uses 1.0 to prevent excessive exploration)
            max_std = jnp.exp(self.config.log_std_max)
            std = jnp.minimum(std, max_std)
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

        if self.config.dem:
            # DEM logits — separate head for dimension-wise entropy modulation
            dem_logits = nn.Dense(
                self.config.action_dim, kernel_init=nn.initializers.zeros,
            )(features)
            return mean, log_std, dem_logits

        return mean, log_std
