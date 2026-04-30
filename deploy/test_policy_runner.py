"""Test PolicyRunner loads checkpoints and produces valid actions."""
import json
import numpy as np
import os
import tempfile
import sys

import pytest

pytestmark = pytest.mark.deploy


def test_policy_runner_loads_and_infers():
    """PolicyRunner should accept obs and produce action in [-1, 1]."""
    from deploy.policy_runner import PolicyRunner

    ckpt_dir = None
    if os.path.isdir("checkpoints"):
        for d in sorted(os.listdir("checkpoints")):
            best = os.path.join("checkpoints", d, "best")
            if os.path.isdir(best) and os.path.exists(os.path.join(best, "actor_params.npy")):
                ckpt_dir = best
                break

    if ckpt_dir is None:
        pytest.skip("No checkpoint found in checkpoints/")

    runner = PolicyRunner(ckpt_dir)
    print(f"Loaded: algo={runner.algo}, obs_dim={runner.obs_dim}, action_dim={runner.action_dim}")

    obs = np.zeros(runner.obs_dim, dtype=np.float32)
    action = runner.get_action(obs)

    assert action.shape == (runner.action_dim,)
    assert np.all(np.abs(action) <= 1.0 + 1e-6)
    assert not np.any(np.isnan(action))

    obs_rand = np.random.randn(runner.obs_dim).astype(np.float32)
    action_rand = runner.get_action(obs_rand)
    assert action_rand.shape == (runner.action_dim,)
    assert not np.any(np.isnan(action_rand))
    assert not np.allclose(action, action_rand)

    print("PASS")


def _build_kwargs():
    """Common build() kwargs for tests."""
    from deploy.go2_constants import NUM_JOINTS
    return dict(
        joint_pos_sdk=np.zeros(NUM_JOINTS, dtype=np.float32),
        joint_vel_sdk=np.zeros(NUM_JOINTS, dtype=np.float32),
        gyroscope=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        accelerometer=np.array([0.4, -0.5, 9.7], dtype=np.float32),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        command=np.array([0.5, 0.0, 0.1], dtype=np.float32),
    )


def test_default_schema_produces_48d_with_correct_layout():
    """Default schema (no schema arg) matches sim WarpJoystick state group exactly."""
    from deploy.obs_builder import ObsBuilder

    builder = ObsBuilder()
    obs = builder.build(**_build_kwargs())

    assert obs.shape == (48,)
    np.testing.assert_array_almost_equal(obs[0:3], [0.1, 0.2, 0.3])     # gyro
    np.testing.assert_array_almost_equal(obs[3:6], [0.4, -0.5, 9.7])    # accel
    np.testing.assert_array_almost_equal(obs[6:9], [0.0, 0.0, -1.0], decimal=5)  # gravity
    expected_offsets = np.array([0, -0.9, 1.8] * 4, dtype=np.float32)
    np.testing.assert_array_almost_equal(obs[9:21], expected_offsets)
    np.testing.assert_array_almost_equal(obs[45:48], [0.5, 0.0, 0.1])
    print("test_default_schema_produces_48d_with_correct_layout PASS")


def test_gravity_tilted():
    """Projected gravity should change when robot is tilted."""
    from deploy.obs_builder import ObsBuilder

    builder = ObsBuilder()
    angle = np.pi / 2
    quat_pitched = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0], dtype=np.float32)
    kwargs = _build_kwargs()
    kwargs["quaternion"] = quat_pitched
    obs = builder.build(**kwargs)
    proj_grav = obs[6:9]
    assert abs(proj_grav[2]) < 0.1
    assert np.linalg.norm(proj_grav) > 0.9
    print(f"test_gravity_tilted PASS (proj_gravity={proj_grav})")


def test_schema_driven_drop_accel():
    """state_schema without 'accelerometer' shrinks obs by 3d."""
    from deploy.obs_builder import ObsBuilder

    schema = ["gyro", "gravity", "joint_pos_offset", "joint_vel", "last_act", "command"]
    builder = ObsBuilder(state_schema=schema)
    assert builder.raw_dim == 45

    obs = builder.build(**_build_kwargs())
    assert obs.shape == (45,)
    np.testing.assert_array_almost_equal(obs[0:3], [0.1, 0.2, 0.3])  # gyro at [0:3]
    np.testing.assert_array_almost_equal(obs[3:6], [0.0, 0.0, -1.0], decimal=5)  # gravity at [3:6]
    # accelerometer (0.4, -0.5, 9.7) should NOT appear anywhere
    assert not np.any(np.isclose(obs, 9.7))
    print("test_schema_driven_drop_accel PASS")


def test_schema_driven_reorder():
    """Different schema order produces obs in that order."""
    from deploy.obs_builder import ObsBuilder

    schema = ["command", "gyro", "gravity", "joint_pos_offset", "joint_vel", "last_act"]
    builder = ObsBuilder(state_schema=schema)
    obs = builder.build(**_build_kwargs())

    # First 3d is command (was last in default)
    np.testing.assert_array_almost_equal(obs[0:3], [0.5, 0.0, 0.1])
    # Then gyro
    np.testing.assert_array_almost_equal(obs[3:6], [0.1, 0.2, 0.3])
    print("test_schema_driven_reorder PASS")


def test_unknown_term_raises():
    """Unknown schema term should raise ValueError with helpful message."""
    from deploy.obs_builder import ObsBuilder
    import pytest

    with pytest.raises(ValueError, match="unknown term"):
        ObsBuilder(state_schema=["gyro", "magnetometer"])
    print("test_unknown_term_raises PASS")


def test_from_checkpoint_with_schema():
    """from_checkpoint reads obs_schema from meta.json."""
    from deploy.obs_builder import ObsBuilder

    with tempfile.TemporaryDirectory() as td:
        meta = {
            "obs_dim": 45, "action_dim": 12, "algo": "fast_sac",
            "train_config": {"n_frame_stack": 1},
            "obs_schema": {
                "state": ["gyro", "gravity", "joint_pos_offset",
                          "joint_vel", "last_act", "command"],
            },
        }
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump(meta, f)
        builder = ObsBuilder.from_checkpoint(td)
        assert builder.state_schema == meta["obs_schema"]["state"]
        assert builder.raw_dim == 45
    print("test_from_checkpoint_with_schema PASS")


def test_from_checkpoint_fallback_no_schema():
    """from_checkpoint falls back to DEFAULT_STATE_SCHEMA if obs_schema absent."""
    from deploy.obs_builder import ObsBuilder, DEFAULT_STATE_SCHEMA

    with tempfile.TemporaryDirectory() as td:
        meta = {"obs_dim": 48, "action_dim": 12, "algo": "fast_sac",
                "train_config": {"n_frame_stack": 1}}
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump(meta, f)
        builder = ObsBuilder.from_checkpoint(td)
        assert builder.state_schema == DEFAULT_STATE_SCHEMA
        assert builder.raw_dim == 48
    print("test_from_checkpoint_fallback_no_schema PASS")


def test_schema_extractor_resolves_include_group():
    """schema_from_obs_groups recursively expands IncludeGroup references."""
    from jax_rl.envs.obs_spec import schema_from_obs_groups, ObsTerm, IncludeGroup

    groups = {
        "state": [
            ObsTerm("gyro", lambda **kw: None),
            ObsTerm("gravity", lambda **kw: None),
        ],
        "privileged_state": [
            IncludeGroup("state"),
            ObsTerm("xfrc", lambda **kw: None),
        ],
    }
    schema = schema_from_obs_groups(groups)
    assert schema == {
        "state": ["gyro", "gravity"],
        "privileged_state": ["gyro", "gravity", "xfrc"],
    }
    print("test_schema_extractor_resolves_include_group PASS")


if __name__ == "__main__":
    test_policy_runner_loads_and_infers()
    test_default_schema_produces_48d_with_correct_layout()
    test_gravity_tilted()
    test_schema_driven_drop_accel()
    test_schema_driven_reorder()
    test_unknown_term_raises()
    test_from_checkpoint_with_schema()
    test_from_checkpoint_fallback_no_schema()
    test_schema_extractor_resolves_include_group()
