"""Tests for jax_rl.utils.distributional helpers.

Focus on the two helpers extracted in B5.7 (safe_log_softmax,
cross_entropy_categorical), plus a within-run equivalence proof against
the original inline pattern that 8 sites across fast_sac/fast_td3/flash_sac
used to repeat.

GPU bit-identity across runs is not relied on; see memory
feedback_gpu_nondeterminism.md.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_rl.utils.distributional import (
    cross_entropy_categorical,
    safe_log_softmax,
)


def _inline_log_probs(logits, axis=-1, min_log=-30.0):
    """The exact inline pattern used in fast_td3:159, fast_sac:164, etc."""
    return jnp.maximum(jax.nn.log_softmax(logits, axis=axis), min_log)


def _inline_cross_entropy(target_probs, logits, axis=-1, min_log=-30.0):
    """The exact pattern used in q1_per_sample / q2_per_sample lines."""
    log_probs = _inline_log_probs(logits, axis=axis, min_log=min_log)
    return -jnp.sum(target_probs * log_probs, axis=axis)


# ---------------------------------------------------------------------------
# safe_log_softmax
# ---------------------------------------------------------------------------


def test_safe_log_softmax_matches_inline():
    """Within-run equivalence to the inline jnp.maximum(log_softmax, -30) form."""
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (16, 51))
    a = safe_log_softmax(logits)
    b = _inline_log_probs(logits)
    assert jnp.array_equal(a, b)


def test_safe_log_softmax_clamps_at_min():
    """Extremely peaked logits should produce log_probs floored at min_log."""
    # Single atom strongly preferred → other atoms have prob ~0 → log_prob → -inf.
    logits = jnp.array([[100.0, -100.0, -100.0]])
    out = safe_log_softmax(logits, min_log=-30.0)
    # Non-preferred atoms should clamp at -30, not -inf.
    assert jnp.all(out >= -30.0)
    assert not jnp.any(jnp.isinf(out))
    assert not jnp.any(jnp.isnan(out))


def test_safe_log_softmax_normal_case_unaffected():
    """Mild logits → log_softmax never hits the clamp → output identical to log_softmax."""
    logits = jnp.array([[1.0, 2.0, 3.0, 1.5]])
    safe = safe_log_softmax(logits)
    raw = jax.nn.log_softmax(logits, axis=-1)
    assert jnp.allclose(safe, raw, atol=1e-6)


def test_safe_log_softmax_custom_min_log():
    """Custom min_log floor applies."""
    logits = jnp.array([[100.0, -100.0]])
    out = safe_log_softmax(logits, min_log=-15.0)
    assert jnp.all(out >= -15.0)


# ---------------------------------------------------------------------------
# cross_entropy_categorical
# ---------------------------------------------------------------------------


def test_cross_entropy_matches_inline_pattern():
    """Within-run equivalence to the original 2-line inline pattern."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    logits = jax.random.normal(k1, (32, 51))
    target_probs = jax.nn.softmax(jax.random.normal(k2, (32, 51)), axis=-1)

    a = cross_entropy_categorical(target_probs, logits)
    b = _inline_cross_entropy(target_probs, logits)
    assert jnp.array_equal(a, b)


def test_cross_entropy_no_nan_at_extreme_logits():
    """The whole point of safe_log_softmax: no NaN even when atoms are zeros."""
    target_probs = jnp.array([[0.0, 1.0, 0.0]])  # delta on atom 1
    logits = jnp.array([[100.0, -100.0, -100.0]])  # peaked elsewhere
    out = cross_entropy_categorical(target_probs, logits)
    # Without clamp: target[0]=0 * log_prob[0]=-200 → 0 * -inf would be NaN.
    # With clamp: 0 * -30 = 0, contribution = -1 * -200 = 200 (clamp wins for the supported atom).
    assert not jnp.any(jnp.isnan(out))
    assert not jnp.any(jnp.isinf(out))


def test_cross_entropy_self_distribution_is_entropy():
    """CE(p, log p) = -sum(p log p) = H(p)."""
    target_probs = jnp.array([[0.25, 0.25, 0.5]])
    # Use logits that produce the same softmax distribution.
    logits = jnp.log(target_probs)
    ce = cross_entropy_categorical(target_probs, logits)
    expected_entropy = -jnp.sum(target_probs * jnp.log(target_probs), axis=-1)
    assert jnp.allclose(ce, expected_entropy, atol=1e-5)


def test_cross_entropy_batch_shape_preserved():
    """Reduces only the atoms axis, batch dims preserved."""
    target_probs = jnp.full((4, 8, 51), 1.0 / 51)
    logits = jnp.zeros((4, 8, 51))
    out = cross_entropy_categorical(target_probs, logits)
    assert out.shape == (4, 8)


def test_jit_compiles():
    """Wrapped in jit, helper is closure-trace-stable."""
    target_probs = jnp.full((2, 51), 1.0 / 51)
    logits = jnp.zeros((2, 51))

    @jax.jit
    def step(t, l):
        return cross_entropy_categorical(t, l)

    out = step(target_probs, logits)
    # Uniform target × uniform pred → CE = log(num_atoms) = log(51).
    expected = jnp.log(51.0)
    assert jnp.allclose(out, expected, atol=1e-5)
