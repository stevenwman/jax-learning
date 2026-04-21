"""Tests for contraction fields on RolloutBatch (Task 4)."""

import jax.numpy as jnp

from jax_rl.buffers.rollout import RolloutBatch


def _mkbatch(**overrides):
    T, E, O, A = 4, 2, 8, 3
    base = dict(
        obs=jnp.zeros((T, E, O)),
        actions=jnp.zeros((T, E, A)),
        rewards=jnp.zeros((T, E)),
        dones=jnp.zeros((T, E)),
        truncations=jnp.zeros((T, E)),
        log_probs=jnp.zeros((T, E)),
        values=jnp.zeros((T, E)),
    )
    base.update(overrides)
    return RolloutBatch(**base)


def test_rollout_batch_default_contraction_is_none():
    """Backward compat: constructing without contraction fields yields None."""
    b = _mkbatch()
    assert b.contraction_c is None
    assert b.contraction_c_dot is None


def test_rollout_batch_accepts_contraction_arrays():
    T, E, C = 4, 2, 3
    c = jnp.ones((T, E, C))
    c_dot = jnp.ones((T, E, C)) * 2
    b = _mkbatch(contraction_c=c, contraction_c_dot=c_dot)
    assert b.contraction_c.shape == (T, E, C)
    assert b.contraction_c_dot.shape == (T, E, C)
