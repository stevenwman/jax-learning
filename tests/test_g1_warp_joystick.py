"""GPU-only smoke + obs-schema tests for G1WarpJoystick.

Hermetic-CPU is impractical for Warp envs (need GPU device). Marks: gpu, warp.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp]


def test_g1_env_constructs_29dof():
    """Env builds, action_size=29 (full body)."""
    import jax_rl.training.env_setup  # noqa
    from mujoco_playground import registry as pg_registry
    env = pg_registry.load("G1WarpJoystickFlat")
    assert env.action_size == 29
    assert env.mjx_model.nu == 29


def test_g1_env_reset_step_smoke():
    """reset → step produces finite obs, finite reward, no immediate termination."""
    import jax
    import jax.numpy as jp
    import jax_rl.training.env_setup  # noqa
    from mujoco_playground import registry as pg_registry
    env = pg_registry.load("G1WarpJoystickFlat")
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs["state"].shape[-1] > 0
    assert state.obs["privileged_state"].shape[-1] > 0
    assert np.all(np.isfinite(state.obs["state"]))
    state2 = env.step(state, jp.zeros(env.action_size))
    assert np.isfinite(float(state2.reward))
    assert state2.data.qpos.shape == state.data.qpos.shape


def test_g1_obs_schema_state_dim():
    """Actor state obs = 96d (gyro 3 + gravity 3 + joint_pos 29 + joint_vel 29 + last_act 29 + cmd 3)."""
    import jax
    import jax_rl.training.env_setup  # noqa
    from mujoco_playground import registry as pg_registry
    env = pg_registry.load("G1WarpJoystickFlat")
    state = env.reset(jax.random.PRNGKey(0))
    expected = 3 + 3 + 29 + 29 + 29 + 3
    assert state.obs["state"].shape[-1] == expected, (
        f"state obs dim {state.obs['state'].shape[-1]} != {expected}"
    )


def test_g1_pelvis_at_keyframe_height():
    """Spawn keyframe puts pelvis at ~0.78m."""
    import jax
    import jax_rl.training.env_setup  # noqa
    from mujoco_playground import registry as pg_registry
    env = pg_registry.load("G1WarpJoystickFlat")
    state = env.reset(jax.random.PRNGKey(0))
    pelvis_z = float(state.data.qpos[2])
    assert 0.7 < pelvis_z < 0.85, f"pelvis_z={pelvis_z} not near 0.78"


def test_g1_control_metadata_shape():
    """get_control_metadata returns 29-element default_pose."""
    import jax_rl.training.env_setup  # noqa
    from mujoco_playground import registry as pg_registry
    env = pg_registry.load("G1WarpJoystickFlat")
    meta = env.get_control_metadata()
    assert len(meta["default_pose_policy"]) == 29
    assert len(meta["policy_joint_names"]) == 29
    assert "left_hip_pitch_joint" in meta["policy_joint_names"]
    assert "right_wrist_yaw_joint" in meta["policy_joint_names"]


def test_g1_no_immediate_termination():
    """Spawn at default pose: should not terminate within 10 steps with zero action."""
    import jax
    import jax.numpy as jp
    import jax_rl.training.env_setup  # noqa
    from mujoco_playground import registry as pg_registry
    env = pg_registry.load("G1WarpJoystickFlat")
    state = env.reset(jax.random.PRNGKey(0))
    for _ in range(10):
        state = env.step(state, jp.zeros(env.action_size))
    assert not bool(state.done), "G1 fell over within 10 zero-action steps; spawn unstable"
