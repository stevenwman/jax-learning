"""Tests for two-hot encoding, symlog, symexp (TD-MPC2 reward/Q categorical targets)."""
import jax
import jax.numpy as jnp

from jax_rl.utils.twohot import symlog, symexp, two_hot, two_hot_inv


def test_symlog_symexp_roundtrip():
    x = jnp.array([-1e4, -10.0, -1.0, 0.0, 1.0, 10.0, 1e4])
    assert jnp.allclose(symexp(symlog(x)), x, atol=1e-3)

def test_symlog_zero_is_zero():
    assert float(symlog(jnp.array(0.0))) == 0.0

def test_two_hot_sums_to_one():
    x = jnp.array([[0.0], [5.0], [-3.0]])
    enc = two_hot(x, vmin=-10.0, vmax=10.0, num_bins=101)
    assert enc.shape == (3, 101)
    assert jnp.allclose(enc.sum(axis=-1), 1.0, atol=1e-5)

def test_two_hot_decode_roundtrip():
    for x_val in [-5.0, 0.0, 2.5, 8.9]:
        x = jnp.array([[x_val]])
        enc = two_hot(x, vmin=-10.0, vmax=10.0, num_bins=101)
        dec = two_hot_inv(enc, vmin=-10.0, vmax=10.0, num_bins=101, apply_symexp=False)
        assert jnp.allclose(dec, x, atol=1e-4), f"Failed at {x_val}: got {float(dec[0, 0])}"

def test_two_hot_inv_with_symexp():
    r = jnp.array([[100.0], [-50.0], [0.0]])
    enc = two_hot(symlog(r), vmin=-10.0, vmax=10.0, num_bins=101)
    r_recovered = two_hot_inv(enc, vmin=-10.0, vmax=10.0, num_bins=101, apply_symexp=True)
    assert jnp.allclose(r_recovered, r, atol=1.0)

def test_two_hot_clamps_out_of_range():
    x = jnp.array([[-100.0], [100.0]])
    enc = two_hot(x, vmin=-10.0, vmax=10.0, num_bins=101)
    assert jnp.all(jnp.isfinite(enc))
    assert enc[0, 0] > 0.9
    assert enc[1, -1] > 0.9
