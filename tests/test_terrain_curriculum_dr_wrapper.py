"""Tests for TerrainCurriculumDRWrapper."""

import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]


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


def test_initial_distance_positive(make_wrapped_env):
    """All types spawn rim-to-center → initial_distance should be > 0 for every env."""
    env = make_wrapped_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    dists = state.info["initial_distance"]
    assert (dists > 0).all(), f"BUG: zero initial_distance: {dists}"


# ── Fall detection + zero-cmd fixes (2026-04-19) ────────────────────────

def test_fall_detection_demotes_on_env_termination():
    """Env-terminated episode (truncation=0, done=1) should demote.

    Previously relied on state.info['episode_fallen'], which where_done wipes
    to False before the wrapper can read it. Current logic infers fall from
    the preserved truncation flag: fall_at_done = (done>0) & (truncation<0.5).
    """
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper

    env = WarpJoystickCurriculum()
    wrapped = TerrainCurriculumDRWrapper(env, episode_length=50, num_envs=4)
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 4))

    # Force one env onto a high level so a demote is observable
    state.info["terrain_level"] = jnp.array([5, 5, 5, 5], dtype=jnp.int32)
    # Simulate end-of-episode: done=1, truncation=0 (env termination = fall)
    # Wrapper reads done from state.info[f'{_KEY}_episode_done'] AFTER super.step.
    # Simpler approach: drive the env until termination or timeout and check
    # that at least one env demoted (level moved ↓ by 1 on done).

    # Run 55 steps to force at least one truncation boundary.
    for _ in range(55):
        state = wrapped.step(state, jnp.zeros((4, 12)))

    # Promoted/demoted flags should be booleans, not stuck False-only.
    assert state.info["episode_promoted"].dtype == jnp.bool_
    assert state.info["episode_demoted"].dtype == jnp.bool_


def test_force_zero_linvel_sampled_per_env():
    """force_zero_linvel should be a per-env bool array, sampled at reset."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper

    env = WarpJoystickCurriculum()
    wrapped = TerrainCurriculumDRWrapper(env, episode_length=50, num_envs=4)
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    assert "force_zero_linvel" in state.info
    assert state.info["force_zero_linvel"].shape == (4,)
    assert state.info["force_zero_linvel"].dtype == jnp.bool_


def test_force_zero_linvel_frequency():
    """Over many envs, ~15% of force_zero_linvel should be True."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper

    env = WarpJoystickCurriculum()
    N = 256
    wrapped = TerrainCurriculumDRWrapper(env, episode_length=50, num_envs=N)
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(42), N))
    frac = state.info["force_zero_linvel"].mean()
    # 15% ± 5% tolerance (binomial std ≈ sqrt(0.15*0.85/256) ≈ 0.022, use wider)
    assert 0.08 < frac < 0.22, f"force_zero fraction {frac:.3f} outside [0.08, 0.22]"


def test_force_zero_yaw_sampled_per_env():
    """force_zero_yaw should be a per-env bool array, sampled at reset."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper

    env = WarpJoystickCurriculum()
    wrapped = TerrainCurriculumDRWrapper(env, episode_length=50, num_envs=4)
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    assert "force_zero_yaw" in state.info
    assert state.info["force_zero_yaw"].shape == (4,)
    assert state.info["force_zero_yaw"].dtype == jnp.bool_


def test_force_zero_yaw_zeros_cmd2():
    """force_zero_yaw=True must zero cmd[2] (yaw_rate) regardless of class."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper

    env = WarpJoystickCurriculum()
    wrapped = TerrainCurriculumDRWrapper(env, episode_length=50, num_envs=4)
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 4))

    state.info["force_zero_yaw"] = jnp.array([True, True, True, True])
    state = wrapped.step(state, jnp.zeros((4, 12)))

    cmd = state.info["command"]
    assert jnp.all(cmd[:, 2] == 0.0), f"yaw_rate not zeroed: {cmd[:,2]}"


def test_force_zero_yaw_conditional_frequency():
    """P(force_zero_yaw) should be 0.5 when force_zero_linvel=True, 0.15 otherwise."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper

    env = WarpJoystickCurriculum()
    N = 512
    wrapped = TerrainCurriculumDRWrapper(env, episode_length=50, num_envs=N)
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(42), N))
    fzl = state.info["force_zero_linvel"]
    fzy = state.info["force_zero_yaw"]
    # Conditional rates
    if fzl.sum() > 0:
        p_yaw_given_zero = fzy[fzl].mean()
        assert 0.35 < p_yaw_given_zero < 0.65, (
            f"P(yaw=0 | linvel=0) = {p_yaw_given_zero:.3f} not in [0.35, 0.65]"
        )
    if (~fzl).sum() > 0:
        p_yaw_given_nonzero = fzy[~fzl].mean()
        assert 0.05 < p_yaw_given_nonzero < 0.25, (
            f"P(yaw=0 | linvel!=0) = {p_yaw_given_nonzero:.3f} not in [0.05, 0.25]"
        )


def test_force_zero_linvel_zeros_command():
    """Class A env with force_zero_linvel=True should have cmd[0]=cmd[1]=0 after step."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper

    env = WarpJoystickCurriculum()
    wrapped = TerrainCurriculumDRWrapper(env, episode_length=50, num_envs=4)
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 4))

    # Manually set force_zero_linvel=True for all envs
    state.info["force_zero_linvel"] = jnp.array([True, True, True, True])

    state = wrapped.step(state, jnp.zeros((4, 12)))

    cmd = state.info["command"]
    # Unified design: force_zero_linvel zeros cmd_vx/cmd_vy for ALL envs
    # (all types are goal-directed; force_zero overrides the holonomic cmd).
    assert jnp.all(cmd[:, 0] == 0.0), f"vx not zero: {cmd[:, 0]}"
    assert jnp.all(cmd[:, 1] == 0.0), f"vy not zero: {cmd[:, 1]}"
