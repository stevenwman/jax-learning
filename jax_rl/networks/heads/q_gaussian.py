"""Gaussian distributional Q-head for FastDSAC.

Outputs (mean, variance) instead of C51's discrete logits.
Variance is parameterized via softplus for positivity.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

ACTIVATIONS = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
    "gelu": jax.nn.gelu,
    "swish": jax.nn.swish,
}


class GaussianQHead(nn.Module):
    """MLP Q-network: concat(obs, action) → (mean, variance).

    Args:
        hidden_dim: Sizes of hidden layers, e.g. (512, 512).
        activation: Activation function name.
        layer_norm: Whether to apply LayerNorm after each hidden layer.
    """
    hidden_dim: tuple
    activation: str = "relu"
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([obs, action], axis=-1)
        act_fn = ACTIVATIONS[self.activation]
        for d in self.hidden_dim:
            x = nn.Dense(d, kernel_init=nn.initializers.lecun_uniform())(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)
        # Mean head
        mean = jnp.squeeze(
            nn.Dense(1, kernel_init=nn.initializers.lecun_uniform())(x),
            axis=-1,
        )
        # Variance head: log-variance with clamping for numerical stability.
        # Softplus + tiny eps was causing NaN on HumanoidRun (variance → 0 → 1/var explosion).
        # Log-variance clamped to [-10, 2] gives variance in [4.5e-5, 7.4].
        log_var = jnp.squeeze(
            nn.Dense(1, kernel_init=nn.initializers.zeros_init())(x),  # init at 0 → var=1
            axis=-1,
        )
        log_var = jnp.clip(log_var, -10.0, 2.0)
        variance = jnp.exp(log_var)
        return mean, variance
