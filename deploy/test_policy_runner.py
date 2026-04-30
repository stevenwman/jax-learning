"""Test PolicyRunner loads checkpoints and produces valid actions."""
import json
import os
import tempfile

import numpy as np

import pytest

pytestmark = pytest.mark.deploy


def _make_synthetic_shared_actor_ckpt(
    td: str,
    *,
    obs_dim: int = 48,
    action_dim: int = 12,
    algo: str = "fast_sac",
    hidden: int = 16,
):
    """Build a minimal valid shared-actor ckpt at `td`."""
    from jax_rl.training.artifact_contract import KIND_SHARED_ACTOR, stamp_meta

    meta = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "algo": algo,
        "train_config": {"n_frame_stack": 1},
        "fast_sac_config": {"hidden_dim": [hidden, hidden], "activation": "relu"},
        "obs_schema": {
            "state": [
                "gyro",
                "accelerometer",
                "gravity",
                "joint_pos_offset",
                "joint_vel",
                "last_act",
                "command",
            ],
        },
    }
    meta = stamp_meta(meta, KIND_SHARED_ACTOR)
    with open(os.path.join(td, "meta.json"), "w") as f:
        json.dump(meta, f)

    rng = np.random.default_rng(0)
    actor_params = {
        "params": {
            "MlpEncoder_0": {
                "Dense_0": {
                    "kernel": rng.normal(0.0, 0.02, size=(obs_dim, hidden)).astype(np.float32),
                    "bias": np.zeros(hidden, dtype=np.float32),
                },
                "Dense_1": {
                    "kernel": rng.normal(0.0, 0.02, size=(hidden, hidden)).astype(np.float32),
                    "bias": np.zeros(hidden, dtype=np.float32),
                },
            },
            "GaussianHead_0": {
                "Dense_0": {
                    "kernel": rng.normal(0.0, 0.02, size=(hidden, action_dim)).astype(np.float32),
                    "bias": np.zeros(action_dim, dtype=np.float32),
                },
            },
        },
    }
    np.save(
        os.path.join(td, "actor_params.npy"),
        {"actor_params": actor_params},
        allow_pickle=True,
    )
    os.makedirs(os.path.join(td, "orbax"), exist_ok=True)


def _make_synthetic_tdmpc2_ckpt(td: str):
    """Build a minimal TDMPC2 ckpt — should be rejected by PolicyRunner."""
    from jax_rl.training.artifact_contract import KIND_TDMPC2, stamp_meta

    meta = {
        "obs_dim": 24,
        "action_dim": 6,
        "algo": "tdmpc2",
        "train_config": {"n_frame_stack": 1},
    }
    meta = stamp_meta(meta, KIND_TDMPC2)
    with open(os.path.join(td, "meta.json"), "w") as f:
        json.dump(meta, f)
    np.savez(os.path.join(td, "actor_params.npz"))
    np.savez(os.path.join(td, "world_model_params.npz"))


def test_policy_runner_loads_synthetic_shared_actor_ckpt(tmp_path):
    """PolicyRunner loads a hermetic synthetic shared-actor ckpt and infers."""
    from deploy.policy_runner import PolicyRunner

    _make_synthetic_shared_actor_ckpt(str(tmp_path), obs_dim=48, action_dim=12)
    runner = PolicyRunner(str(tmp_path))

    assert runner.obs_dim == 48
    assert runner.action_dim == 12

    obs = np.zeros(48, dtype=np.float32)
    action = runner.get_action(obs)
    assert action.shape == (12,)
    assert np.all(np.abs(action) <= 1.0 + 1e-6)
    assert not np.any(np.isnan(action))


def test_policy_runner_rejects_tdmpc2_ckpt(tmp_path):
    """PolicyRunner refuses a TDMPC2 ckpt with a redirect message."""
    from deploy.policy_runner import PolicyRunner

    _make_synthetic_tdmpc2_ckpt(str(tmp_path))

    with pytest.raises(ValueError) as exc:
        PolicyRunner(str(tmp_path))
    msg = str(exc.value).lower()
    assert "tdmpc2_v1" in msg
    assert "shared-actor checkpoints" in msg


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
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        test_policy_runner_loads_synthetic_shared_actor_ckpt(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_policy_runner_rejects_tdmpc2_ckpt(Path(td))
    test_default_schema_produces_48d_with_correct_layout()
    test_gravity_tilted()
    test_schema_driven_drop_accel()
    test_schema_driven_reorder()
    test_unknown_term_raises()
    test_from_checkpoint_with_schema()
    test_from_checkpoint_fallback_no_schema()
    test_schema_extractor_resolves_include_group()
