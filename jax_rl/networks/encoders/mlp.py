"""MLP encoder — maps raw observations to a fixed-size feature vector.

The encoder is the first stage of any network (actor, critic, Q). It takes
raw observations (e.g., joint positions, velocities) and produces a feature
vector that heads (GaussianHead, ValueHead, QHead) consume.

Swappable: all algos construct encoders via builders.py, so replacing MLP
with CNN or Transformer means changing the builder, not the algo code.

Architecture::

    obs (obs_dim,) → Dense → [LayerNorm] → activation → Dense → ... → features (hidden_dim[-1],)
"""

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax_rl.configs.networks_config import EncoderConfig
from jax_rl.networks.activations import ACTIVATIONS


class MlpEncoder(nn.Module):
    """Multi-layer perceptron encoder.

    Attributes:
        config: EncoderConfig specifying hidden_dim (tuple of layer sizes),
            activation ("relu", "swish", etc.), and norm (None or "layer").

    Input:  obs of shape (batch, obs_dim)
    Output: features of shape (batch, hidden_dim[-1])

    Optional context input (for future goal-conditioned / USD):
        If context is provided, it's concatenated to obs before encoding.
        Currently unused — the context_dim field in EncoderConfig is reserved.
    """
    config: EncoderConfig

    @nn.compact
    def __call__(self, obs: jax.Array, context: jax.Array = None) -> jax.Array:
        if context is not None:
            obs = jnp.concatenate([obs, context], axis=-1)

        act_fn = ACTIVATIONS[self.config.activation]
        for d_out in self.config.hidden_dim:
            obs = nn.Dense(d_out, kernel_init=nn.initializers.lecun_uniform())(obs)
            if self.config.norm is not None:
                obs = nn.LayerNorm()(obs)
            obs = act_fn(obs)
        return obs

    @property
    def feature_dim(self) -> int:
        """Output feature dimensionality (last hidden layer size)."""
        return self.config.hidden_dim[-1]
