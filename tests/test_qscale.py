"""Tests for Q-scale running percentile EMA (TD-MPC2 policy loss scaling)."""
import jax
import jax.numpy as jnp

from jax_rl.utils.qscale import QScaleState, qscale_init, qscale_update, qscale_apply


def test_qscale_init_is_one():
    state = qscale_init()
    assert float(state.range_ema) == 1.0

def test_qscale_clamps_min_1():
    state = qscale_init()
    q = jnp.linspace(0.499, 0.501, 100)
    new_state = qscale_update(state, q, tau=0.01)
    assert float(new_state.range_ema) >= 1.0 - 1e-5

def test_qscale_moves_toward_actual_range():
    state = qscale_init()
    q = jnp.linspace(-100.0, 100.0, 1000)
    for _ in range(200):
        state = qscale_update(state, q, tau=0.01)
    assert float(state.range_ema) > 50.0

def test_qscale_apply_divides():
    state = QScaleState(range_ema=jnp.array(10.0))
    q = jnp.array([5.0, 10.0, 20.0])
    scaled = qscale_apply(state, q)
    assert jnp.allclose(scaled, jnp.array([0.5, 1.0, 2.0]), atol=1e-5)
