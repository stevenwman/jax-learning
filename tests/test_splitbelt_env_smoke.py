"""GPU/Warp smoke tests for SplitbeltTreadmill env (S§10.2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]


@pytest.fixture
def env():
    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    return Go2WarpSplitbeltEnv()


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


def test_reset_returns_finite_state(env, rng):
    state = env.reset(rng)
    assert jnp.all(jnp.isfinite(state.obs["state"]))
    assert jnp.all(jnp.isfinite(state.obs["privileged_state"]))
    # term_cause starts zeroed (S§6.2).
    assert int(state.info["splitbelt"]["term_cause"]) == 0


def test_single_step_no_nan(env, rng):
    state = env.reset(rng)
    action = jnp.zeros((env._action_dim,))
    state2 = env.step(state, action)
    assert jnp.all(jnp.isfinite(state2.obs["state"]))
    assert jnp.all(jnp.isfinite(state2.reward))


def test_belt_qvel_matches_schedule(env, rng):
    """Schedule plumbing: belt joint qvel after one step matches -schedule_table[0].

    Joint qvel is NEGATED relative to schedule because schedule semantics is
    "drag speed" (biomech) — positive value = belt drags foot backward — so
    the slide-joint along +x has qvel = -schedule_speed (slab moves -x).
    """
    state = env.reset(rng)
    action = jnp.zeros((env._action_dim,))
    state2 = env.step(state, action)
    schedule_step0 = state.info["belt_schedule"][0]
    actual_left = state2.data.qvel[env._left_belt_dofadr]
    actual_right = state2.data.qvel[env._right_belt_dofadr]
    # 10% tolerance for one-step transient (kv=200 should achieve this).
    assert jnp.abs(actual_left - (-schedule_step0[0])) < 0.1
    assert jnp.abs(actual_right - (-schedule_step0[1])) < 0.1


def test_off_belt_termination(env, rng):
    """Force a foot off-belt by shifting robot in y; assert term_cause = 2."""
    state = env.reset(rng)
    # Replace base y in qpos to put robot fully off the right side of belts.
    qpos = state.data.qpos.at[1].set(2.0)  # y = 2 m, well outside belts
    new_data = state.data.replace(qpos=qpos)
    state = state.replace(data=new_data)
    action = jnp.zeros((env._action_dim,))
    s2 = env.step(state, action)
    # foot × fallback_floor should fire.
    assert bool(s2.done)
    assert int(s2.info["splitbelt"]["term_cause"]) == 2  # off-belt
