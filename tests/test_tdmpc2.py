"""Tests for TD-MPC2 networks and loss components."""
import jax
import jax.numpy as jnp
from flax import linen as nn

from jax_rl.algos.tdmpc2 import NormedLinear


def test_normed_linear_shape_and_activation():
    layer = NormedLinear(features=32)
    params = layer.init(jax.random.PRNGKey(0), jnp.zeros((4, 16)))
    y = layer.apply(params, jnp.ones((4, 16)))
    assert y.shape == (4, 32)
    assert jnp.all(jnp.isfinite(y))


def test_normed_linear_truncnormal_init():
    """Kernel init should be trunc_normal(std=0.02); bias zero."""
    layer = NormedLinear(features=64)
    params = layer.init(jax.random.PRNGKey(0), jnp.zeros((1, 32)))
    # Kernel params exist and are small (std=0.02)
    kernel = params["params"]["Dense_0"]["kernel"]
    assert kernel.shape == (32, 64)
    assert abs(float(kernel.std())) < 0.1  # far under 1.0 stdlib default
    # Bias should be zero
    bias = params["params"]["Dense_0"]["bias"]
    assert jnp.allclose(bias, 0.0)


def test_normed_linear_dropout_off_by_default():
    """Default dropout=0.0, so output deterministic regardless of deterministic flag."""
    layer = NormedLinear(features=8)
    params = layer.init(jax.random.PRNGKey(0), jnp.zeros((2, 4)))
    y1 = layer.apply(params, jnp.ones((2, 4)))
    y2 = layer.apply(params, jnp.ones((2, 4)))
    assert jnp.allclose(y1, y2)
