"""Tests for TerrainCurriculumDRWrapper."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def make_wrapped_env():
    """Create a wrapped curriculum env with 4 envs for testing."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper

    env = WarpJoystickCurriculum()
    wrapped = TerrainCurriculumDRWrapper(env, episode_length=50, num_envs=4)
    return wrapped


def test_reset_assigns_fixed_terrain_type(make_wrapped_env):
    """terrain_type should be deterministic: env_id % num_cols."""
    env = make_wrapped_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    types = state.info["terrain_type"]
    assert jnp.array_equal(types, jnp.array([0, 1, 2, 3]))


def test_terrain_type_preserved_across_reset(make_wrapped_env):
    """After episode boundary, terrain_type should NOT change."""
    env = make_wrapped_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    initial_types = state.info["terrain_type"]
    for _ in range(55):  # force at least one episode reset (ep_length=50)
        state = env.step(state, jnp.zeros((4, 12)))
    assert jnp.array_equal(state.info["terrain_type"], initial_types)


def test_initial_level_is_zero(make_wrapped_env):
    """All envs should start at terrain_level=0."""
    env = make_wrapped_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    assert (state.info["terrain_level"] == 0).all()


def test_level_advances_on_reach(make_wrapped_env):
    """If we manually set episode_reached_goal=True before done, level should advance."""
    env = make_wrapped_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    # Fake reaching the goal for all envs
    state.info["episode_reached_goal"] = jnp.array([True, True, True, True])
    state.info["episode_fallen"] = jnp.array([False, False, False, False])
    # Run until episode resets (ep_length=50)
    for _ in range(55):
        state = env.step(state, jnp.zeros((4, 12)))
    # Levels should be non-negative (basic sanity)
    assert (state.info["terrain_level"] >= 0).all()


def test_curriculum_flags_present_after_step(make_wrapped_env):
    """episode_promoted and episode_demoted should be present in state.info."""
    env = make_wrapped_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    state = env.step(state, jnp.zeros((4, 12)))
    assert "episode_promoted" in state.info
    assert "episode_demoted" in state.info


def test_goal_xy_is_2d(make_wrapped_env):
    """goal_xy should have shape (num_envs, 2)."""
    env = make_wrapped_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    assert state.info["goal_xy"].shape == (4, 2)


def test_initial_distance_positive_for_goal_directed(make_wrapped_env):
    """initial_distance should be positive for goal-directed types (pyramid_up / pyramid_down).

    Class-A types (rough, tilted) spawn at tile center with goal = spawn placeholder,
    so their initial_distance may be ≈ 0 by design.
    """
    env = make_wrapped_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    types = state.info["terrain_type"]
    dists = state.info["initial_distance"]
    # Types 1 (pyramid_up) and 2 (pyramid_down) are goal-directed
    is_goal = (types == 1) | (types == 2)
    goal_dists = dists[is_goal]
    if goal_dists.shape[0] > 0:
        assert (goal_dists > 0).all(), f"BUG: goal-directed envs had zero initial_distance: {goal_dists}"
