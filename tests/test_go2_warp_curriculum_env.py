import jax
import jax.numpy as jnp
import pytest


def test_env_loads():
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    assert env.action_size == 12
    assert env._terrain_origins.shape == (10, 4, 3)
    assert env._num_rows == 10
    assert env._num_cols == 4


def test_env_model_has_terrain_geoms():
    """Composed MJCF should have many more geoms than flat env."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    # Base robot has ~100 geoms; terrain adds ~1400 (40 tiles × avg ~35 geoms)
    assert env.mj_model.ngeom > 500


def test_generated_scene_file_written():
    """Confirm the generated MJCF file exists after env init."""
    from pathlib import Path
    import os
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    expected = Path(__file__).parent.parent / "jax_rl" / "envs" / "locomotion" / "xmls" / f"_generated_curriculum_scene_{os.getpid()}.xml"
    assert expected.exists()


# ── Task 2.3 tests ────────────────────────────────────────────────────────────

def test_reset_sets_curriculum_info():
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    for key in [
        "terrain_level", "terrain_type", "goal_xy", "initial_distance",
        "episode_reached_goal", "episode_min_distance", "episode_fallen",
        "target_speed",
    ]:
        assert key in state.info, f"{key} missing from state.info"
    assert int(state.info["terrain_level"]) == 0
    assert 0 <= int(state.info["terrain_type"]) < env._num_cols


def test_reset_target_speed_scales_with_level():
    """Level 0 should give target_speed=0.5."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    assert float(state.info["target_speed"]) == pytest.approx(0.5, abs=0.01)


def test_reset_initial_distance_matches_spawn_goal():
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    dist_computed = float(jnp.linalg.norm(state.data.qpos[:2] - state.info["goal_xy"]))
    assert float(state.info["initial_distance"]) == pytest.approx(dist_computed, abs=0.01)


# ── Task 2.4 tests ────────────────────────────────────────────────────────────

def test_step_computes_command_from_goal():
    """Holonomic scheme: magnitude of (vx, vy) ≈ target_speed (direction depends on yaw)."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    next_state = env.step(state, jnp.zeros(12))
    cmd = next_state.info["command"]
    assert cmd.shape == (3,)
    speed_mag = float(jnp.sqrt(cmd[0] ** 2 + cmd[1] ** 2))
    assert speed_mag == pytest.approx(float(state.info["target_speed"]), abs=0.02)


def test_step_tracks_min_distance():
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    initial = float(state.info["episode_min_distance"])
    for _ in range(5):
        state = env.step(state, jnp.zeros(12))
    assert float(state.info["episode_min_distance"]) <= initial


def test_step_reach_flag_set_when_near_goal():
    """Teleport robot to goal and verify reach flag flips."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    goal = state.info["goal_xy"]
    new_qpos = state.data.qpos.at[0].set(goal[0]).at[1].set(goal[1])
    state = state.replace(data=state.data.replace(qpos=new_qpos))
    state = env.step(state, jnp.zeros(12))
    assert bool(state.info["episode_reached_goal"])
