"""Tests for contraction_state obs group on go2_bongo_handstand."""

import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]

from jax_rl.envs.locomotion.go2_bongo_handstand import (
    BongoHandstand,
    default_config,
)


@pytest.fixture
def env():
    cfg = default_config()
    cfg.observe_contraction = True
    return BongoHandstand(task="bongo_handstand", config=cfg)


def test_contraction_state_absent_by_default():
    """Default env (no opt-in) must not emit contraction_state."""
    env_default = BongoHandstand(task="bongo_handstand")
    state = env_default.reset(jax.random.PRNGKey(0))
    assert "contraction_state" not in state.obs


def test_contraction_state_shape(env):
    """Group hstacks [c (3,), c_dot (3,)] → (6,)."""
    state = env.reset(jax.random.PRNGKey(0))
    assert "contraction_state" in state.obs
    assert state.obs["contraction_state"].shape == (6,)


def test_contraction_state_finite_after_step(env):
    state = env.reset(jax.random.PRNGKey(0))
    zero_action = jnp.zeros(env.action_size)
    next_state = env.step(state, zero_action)
    cs = next_state.obs["contraction_state"]
    assert cs.shape == (6,)
    assert not jnp.any(jnp.isnan(cs))
    assert jnp.all(jnp.isfinite(cs))


def test_c_is_gravity_residual(env):
    """c = get_gravity(data) - [1,0,0]. At reset, gravity ≈ [0,0,-1] in body
    frame (robot upright) → c ≈ [-1, 0, -1]."""
    state = env.reset(jax.random.PRNGKey(0))
    c = state.obs["contraction_state"][:3]
    # body-frame gravity at upright start is about -z_body ≈ [0,0,-1]
    # target is [1,0,0] → residual ≈ [-1,0,-1]
    expected = env.get_gravity(state.data) - jnp.array([1.0, 0.0, 0.0])
    assert jnp.allclose(c, expected, atol=1e-5)


def test_c_dot_at_reset_is_zero(env):
    """At t=0, angular velocity is zero → ċ = -ω × g = 0."""
    state = env.reset(jax.random.PRNGKey(0))
    c_dot = state.obs["contraction_state"][3:]
    assert jnp.allclose(c_dot, 0.0, atol=1e-6), f"c_dot = {c_dot}"
