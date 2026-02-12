"""Value head for state value estimation V(s)."""

from flax import nnx
import jax
from jax_rl.configs.networks_config import ValueHeadConfig


class ValueHead(nnx.Module):
    """Value head that outputs state value V(s).

    Implements the ValueHead protocol.
    Used for PPO critic and other value-based methods.
    """

    def __init__(
        self, feature_dim: int, config: ValueHeadConfig, rngs: nnx.Rngs
    ) -> None:
        """Initialize value head.

        Args:
            feature_dim: Input feature dimension from encoder
            config: Value head configuration
            rngs: Random number generators for initialization
        """
        self.config = config
        self.layer = nnx.Linear(feature_dim, 1, rngs=rngs)

    def __call__(self, features: jax.Array) -> jax.Array:
        """Compute state value from features.

        Args:
            features: Feature vector from encoder, shape (batch, feature_dim) or (feature_dim,)

        Returns:
            State value(s), shape (batch,) or scalar
        """
        value = self.layer(features)
        return value.squeeze(-1)  # Remove last dimension to get (batch,) or scalar
