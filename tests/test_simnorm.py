"""Tests for SimNorm activation (TD-MPC2 latent normalization)."""
import jax
import jax.numpy as jnp
import pytest

from jax_rl.utils.simnorm import simnorm


def test_simnorm_output_sums_to_d_over_v():
    # For latent_dim d with simplex-dim V, output has d/V chunks each summing to 1
    d, V = 512, 8
    x = jax.random.normal(jax.random.PRNGKey(0), (4, d))
    y = simnorm(x, V=V)
    assert y.shape == (4, d)
    # Each chunk of size V along last dim sums to 1 (per batch element)
    chunks = y.reshape(4, d // V, V)
    sums = chunks.sum(axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5)

def test_simnorm_handles_zero_input():
    # Zero input → each chunk is uniform 1/V (no NaN)
    d, V = 16, 4
    x = jnp.zeros((2, d))
    y = simnorm(x, V=V)
    assert jnp.all(jnp.isfinite(y))
    chunks = y.reshape(2, d // V, V)
    assert jnp.allclose(chunks, 1.0 / V, atol=1e-6)

def test_simnorm_gradient_flows():
    d, V = 16, 4
    def loss_fn(x):
        return simnorm(x, V=V).sum()
    x = jax.random.normal(jax.random.PRNGKey(1), (2, d))
    g = jax.grad(loss_fn)(x)
    assert jnp.all(jnp.isfinite(g))

def test_simnorm_requires_divisible():
    d, V = 16, 5  # 16 % 5 != 0
    x = jnp.zeros((2, d))
    with pytest.raises(AssertionError):
        simnorm(x, V=V)
