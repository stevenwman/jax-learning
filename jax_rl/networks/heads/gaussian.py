"""Gaussian policy head for stochastic policies (PPO, SAC)."""

from flax import nnx
import jax
import jax.numpy as jnp
from jax_rl.configs.networks_config import PolicyHeadConfig


class GaussianHead(nnx.Module):
    """Gaussian policy head that outputs (mean, log_std).

    Implements the PolicyHead protocol.
    Used for stochastic policies in PPO and SAC.
    """

    def __init__(
        self, feature_dim: int, config: PolicyHeadConfig, rngs: nnx.Rngs
    ) -> None:
        """Initialize Gaussian policy head.

        Args:
            feature_dim: Input feature dimension from encoder
            config: Policy head configuration
            rngs: Random number generators for initialization
        """
        self.config = config
        self.mu_net = nnx.Linear(feature_dim, config.action_dim, rngs=rngs)
        self.log_std_net = nnx.Linear(feature_dim, config.action_dim, rngs=rngs)

    def __call__(self, features: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute mean and log_std from features.

        Args:
            features: Feature vector from encoder, shape (batch, feature_dim) or (feature_dim,)

        Returns:
            (mean, log_std) tuple:
                mean: shape (batch, action_dim) or (action_dim,)
                log_std: shape (batch, action_dim) or (action_dim,), clamped to [log_std_min, log_std_max]
        """
        mean = self.mu_net(features)
        log_std = self.log_std_net(features)

        # Clamp log_std for numerical stability
        log_std = jnp.clip(log_std, self.config.log_std_min, self.config.log_std_max)

        return mean, log_std
