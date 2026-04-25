"""Test the polyak.soft_update extracted helper.

Verification strategy:
  - Math identities at boundary tau values (0, 1, 0.5).
  - Pytree handling (nested dicts of arrays).
  - Within-run equivalence to the inline lambda the 5 algos used to define
    privately. Proves the extraction is mathematically equivalent on a
    single run (GPU bit-identity across runs is not relied on; see memory
    feedback_gpu_nondeterminism.md).
"""

import jax
import jax.numpy as jnp
import pytest

from jax_rl.utils.polyak import soft_update


def _inline_lambda(online, target, tau):
    """The exact body the 5 algos used to inline as a private _soft_update."""
    return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)


def test_tau_zero_returns_target():
    online = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([10.0, 20.0, 30.0])
    out = soft_update(online, target, tau=0.0)
    assert jnp.array_equal(out, target)


def test_tau_one_returns_online():
    online = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([10.0, 20.0, 30.0])
    out = soft_update(online, target, tau=1.0)
    assert jnp.array_equal(out, online)


def test_tau_half_is_midpoint():
    online = jnp.array([0.0, 2.0, 4.0])
    target = jnp.array([10.0, 12.0, 14.0])
    out = soft_update(online, target, tau=0.5)
    expected = jnp.array([5.0, 7.0, 9.0])
    assert jnp.allclose(out, expected, atol=1e-6)


def test_arbitrary_tau_matches_inline_lambda():
    """Within-run equivalence: same inputs → exact same outputs."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    online = jax.random.normal(k1, (4, 8))
    target = jax.random.normal(k2, (4, 8))
    for tau in (0.0, 0.005, 0.125, 0.5, 0.9, 1.0):
        a = soft_update(online, target, tau=tau)
        b = _inline_lambda(online, target, tau)
        assert jnp.array_equal(a, b), f"divergence at tau={tau}"


def test_pytree_dict_of_arrays():
    """Nested pytree (Flax-style param dict) handled correctly."""
    online = {
        "Dense_0": {"kernel": jnp.ones((4, 8)), "bias": jnp.ones((8,))},
        "Dense_1": {"kernel": jnp.ones((8, 2)) * 2.0, "bias": jnp.ones((2,)) * 3.0},
    }
    target = jax.tree.map(jnp.zeros_like, online)
    out = soft_update(online, target, tau=0.5)
    assert jnp.allclose(out["Dense_0"]["kernel"], 0.5)
    assert jnp.allclose(out["Dense_0"]["bias"], 0.5)
    assert jnp.allclose(out["Dense_1"]["kernel"], 1.0)
    assert jnp.allclose(out["Dense_1"]["bias"], 1.5)


def test_jit_compiles():
    """Wrapped in jit, soft_update is closure-trace-stable."""
    online = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([4.0, 5.0, 6.0])

    @jax.jit
    def step(o, t, tau):
        return soft_update(o, t, tau)

    out = step(online, target, 0.5)
    expected = jnp.array([2.5, 3.5, 4.5])
    assert jnp.allclose(out, expected, atol=1e-6)


def test_static_tau_in_closure():
    """tau captured by closure (algo pattern) — must produce same result."""
    online = jnp.array([1.0, 2.0])
    target = jnp.array([5.0, 5.0])
    tau = 0.1

    def _closure_form(o, t):
        return jax.tree.map(lambda a, b: tau * a + (1.0 - tau) * b, o, t)

    a = soft_update(online, target, tau)
    b = _closure_form(online, target)
    assert jnp.array_equal(a, b)


def test_shape_preserved():
    online = jnp.zeros((3, 5, 7))
    target = jnp.ones((3, 5, 7))
    out = soft_update(online, target, tau=0.3)
    assert out.shape == (3, 5, 7)
    assert jnp.allclose(out, 0.7)
