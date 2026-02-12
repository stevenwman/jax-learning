"""MLP encoder implementation."""

from flax import nnx
import jax
import jax.numpy as jnp
from jax_rl.configs.networks_config import EncoderConfig


# Activation function mapping
ACTIVATIONS = {
    "relu": nnx.relu,
    "tanh": jnp.tanh,
    "elu": nnx.elu,
    "gelu": nnx.gelu,
    "swish": nnx.swish,
}


class MLPEncoder(nnx.Module):
    """MLP encoder that implements the Encoder protocol.

    Encodes observations (and optionally actions/context) into feature vectors.
    Supports:
    - Configurable hidden dimensions and activation
    - Optional LayerNorm (for FastTD3/SAC)
    - Action concatenation (for Q-networks)
    - Context fusion (for goal-conditioned, USD - future)
    """

    def __init__(self, config: EncoderConfig, rngs: nnx.Rngs) -> None:
        """Initialize MLP encoder.

        Args:
            config: Encoder configuration
            rngs: Random number generators for initialization
        """
        self.config = config
        self.activation_fn = ACTIVATIONS.get(config.activation, nnx.relu)

        # Build layer dimensions: obs_dim -> hidden -> hidden -> ... -> hidden[-1]
        # Note: action/context are concatenated to obs before first layer if provided
        input_dim = config.obs_dim
        net_dims = [input_dim] + list(config.hidden_dim)

        # Create linear layers
        self.layers = nnx.List(
            [
                nnx.Linear(d_in, d_out, rngs=rngs)
                for d_in, d_out in zip(net_dims[:-1], net_dims[1:])
            ]
        )

        # Optional layer normalization (for FastTD3/SAC)
        if config.norm == "layer":
            self.norms = nnx.List(
                [nnx.LayerNorm(d_out, rngs=rngs) for d_out in net_dims[1:]]
            )
        else:
            self.norms = None

    def __call__(
        self,
        obs: jax.Array,
        action: jax.Array | None = None,
        context: jax.Array | None = None,
    ) -> jax.Array:
        """Encode observation (and optionally action/context) to features.

        Args:
            obs: Observation, shape (batch, obs_dim) or (obs_dim,)
            action: Optional action for Q-networks, shape (batch, action_dim) or (action_dim,)
            context: Optional context (goal, skill), shape (batch, context_dim) or (context_dim,)

        Returns:
            Feature vector, shape (batch, feature_dim) or (feature_dim,)
        """
        x = obs

        # Concatenate action if provided (for Q-networks)
        if action is not None:
            x = jnp.concatenate([x, action], axis=-1)

        # Concatenate context if provided (for goal-conditioned, USD)
        if context is not None:
            x = jnp.concatenate([x, context], axis=-1)

        # Forward through MLP layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply normalization if enabled
            if self.norms is not None:
                if self.config.norm_placement == "pre":
                    x = self.norms[i](x)

            # Apply activation (except on last layer - that's the feature output)
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)

            # Post-activation normalization
            if self.norms is not None and i < len(self.layers) - 1:
                if self.config.norm_placement == "post":
                    x = self.norms[i](x)

        return x

    @property
    def feature_dim(self) -> int:
        """Output feature dimension."""
        return self.config.hidden_dim[-1]
