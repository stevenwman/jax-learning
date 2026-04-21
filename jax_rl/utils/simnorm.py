"""SimNorm — TD-MPC2's latent normalization activation.

Reshapes the last dim into chunks of size V, applies softmax within each chunk,
then flattens back. Produces an L1-normalized, structurally non-collapsing latent.

Source: /tmp/tdmpc2/tdmpc2/common/layers.py:74-91
"""
import jax
import jax.numpy as jnp


def simnorm(x: jax.Array, V: int = 8) -> jax.Array:
    """Apply SimNorm to the last dimension of x.

    Args:
        x: (..., d) where d % V == 0
        V: simplex dimension (chunk size)

    Returns:
        (..., d) with each d/V chunk along last dim summing to 1.
    """
    d = x.shape[-1]
    assert d % V == 0, f"Last dim {d} must be divisible by V={V}"
    shape = x.shape[:-1] + (d // V, V)
    x_chunked = x.reshape(shape)
    y = jax.nn.softmax(x_chunked, axis=-1)
    return y.reshape(x.shape)
