import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]


def test_env_loads():
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    assert env.action_size == 12
    assert env._terrain_origins.shape == (6, 5, 3)
    assert env._num_rows == 6
    assert env._num_cols == 5


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


def test_flipped_robot_triggers_termination():
    """Rotate base 180° about x (torso on back, touching ground) → done=1."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    # Rotate 180° about x axis: quat (0, 1, 0, 0) wxyz, lower z so base touches ground.
    qpos = state.data.qpos
    qpos = qpos.at[2].set(state.data.qpos[2] - 0.2)  # drop 20cm
    qpos = qpos.at[3].set(0.0)  # qw
    qpos = qpos.at[4].set(1.0)  # qx (180° about x)
    qpos = qpos.at[5].set(0.0)
    qpos = qpos.at[6].set(0.0)
    state = state.replace(data=state.data.replace(qpos=qpos))
    # Step a few times to let contact settle
    for _ in range(3):
        state = env.step(state, jnp.zeros(12))
    assert bool(state.done), "flipped + torso-on-ground did not terminate"


def test_standing_in_bowl_does_not_terminate():
    """Pyramid_down L5: robot standing at pit bottom (negative world-z) should NOT auto-terminate.

    Old termination: base_z < 0.18m → false-positive at pit bottom.
    New: base_contact > 0 → only fires when torso actually touches ground.
    """
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    # Teleport to pyramid_down (type=2) L5 tile center, upright
    tile_origin = env._terrain_origins[5, 2]
    qpos = state.data.qpos
    qpos = qpos.at[0].set(tile_origin[0])
    qpos = qpos.at[1].set(tile_origin[1])
    # Place torso at reasonable stance height ABOVE tile origin z (which may be negative)
    qpos = qpos.at[2].set(tile_origin[2] + 0.3)
    qpos = qpos.at[3].set(1.0)
    qpos = qpos.at[4].set(0.0)
    qpos = qpos.at[5].set(0.0)
    qpos = qpos.at[6].set(0.0)
    state = state.replace(
        data=state.data.replace(qpos=qpos),
        info={**state.info, "terrain_type": jnp.int32(2), "terrain_level": jnp.int32(5)},
    )
    state = env.step(state, jnp.zeros(12))
    # If base_z alone were used, low world-z would falsely trigger done. Contact-based
    # termination: torso not touching ground → no termination.
    assert not bool(state.done), f"Standing in bowl triggered false-positive termination (done={state.done})"
