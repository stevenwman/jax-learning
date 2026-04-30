"""End-to-end deploy contract parity tests.

Validates that the same robot state produces consistent obs, action, and
SDK joint targets across the deploy contract surface:

- Zero action → SDK targets equal default_pose_sdk.
- One-hot action → exactly one named SDK joint moves by action_scale.
- Schema/control metadata round-trips through meta.json without drift.
- ObsBuilder strict mode refuses to load legacy ckpts.
- Env get_control_metadata stays consistent with deploy/go2_constants.py
  (current state) so deploy paths don't silently disagree.

Closes codex-audit P0 findings on default-pose mismatch and
robot_interface bypassing meta. See deploy/test_time_validate.md for
the runtime version of these checks.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest


# ---------- helpers ---------------------------------------------------------

def _make_meta(
    obs_schema=None,
    control=None,
    obs_dim=48,
    action_dim=12,
    n_frame_stack=1,
):
    """Build a fully-populated meta.json dict for round-trip tests."""
    meta = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "algo": "fast_sac",
        "train_config": {"n_frame_stack": n_frame_stack},
    }
    if obs_schema is not None:
        meta["obs_schema"] = obs_schema
    if control is not None:
        meta["control"] = control
    return meta


def _go2_default_control():
    """The control block a current Go2WarpJoystick checkpoint would write."""
    default_pose_policy = [0.0, 0.9, -1.8] * 4
    policy_to_sdk = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    default_pose_sdk = np.asarray(default_pose_policy)[policy_to_sdk].tolist()
    return {
        "Kp": 20.0,
        "Kd": 0.5,
        "action_scale": 0.5,
        "policy_dt": 0.02,
        "physics_dt": 0.004,
        "action_repeat": 1,
        "contact_mode": "training",
        "torque_speed_model": False,
        "impl": "warp",
        "joint_order": "policy_FL_FR_RL_RR",
        "action_order": "policy_FL_FR_RL_RR",
        "default_pose_policy": default_pose_policy,
        "default_pose_sdk": default_pose_sdk,
        "policy_joint_names": [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ],
        "sdk_joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "policy_to_sdk": policy_to_sdk,
        "sdk_to_policy": policy_to_sdk,
    }


def _action_to_sdk_targets(action_policy, control):
    """Replicate Go2Interface.send_action arithmetic without DDS."""
    policy_to_sdk = np.asarray(control["policy_to_sdk"])
    default_pose_sdk = np.asarray(control["default_pose_sdk"])
    action_scale = float(control["action_scale"])
    action_sdk = action_policy[policy_to_sdk]
    return default_pose_sdk + action_sdk * action_scale


# ---------- env-side: get_control_metadata stays consistent -----------------

@pytest.mark.gpu
@pytest.mark.warp
@pytest.mark.go2
def test_bongo_metadata_uses_handstand_pose_not_home():
    """Codex review fix 2: get_control_metadata reads self._default_pose
    (set by subclass _post_init), not a hardcoded keyframe('home'). Bongo
    uses keyframe('handstand'), so its stamped default_pose must reflect
    that — not the home keyframe — or deploy will track the wrong target.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("warp")

    from jax_rl.envs.locomotion.go2_bongo_handstand import BongoHandstand

    env = BongoHandstand()
    m = env.get_control_metadata()

    handstand_pose = np.asarray(env._mj_model.keyframe("handstand").qpos[7:19])
    home_pose = np.asarray(env._mj_model.keyframe("home").qpos[7:19])

    np.testing.assert_array_almost_equal(
        m["default_pose_policy"], handstand_pose,
        err_msg="Bongo stamped wrong default_pose — should be handstand keyframe",
    )
    # Sanity: the two keyframes are actually different (test would be
    # vacuous otherwise).
    assert not np.allclose(handstand_pose, home_pose), (
        "Test setup invalid: home == handstand in this XML."
    )


@pytest.mark.gpu
@pytest.mark.warp
@pytest.mark.go2
def test_env_metadata_matches_deploy_constants():
    """Go2WarpEnv.get_control_metadata() must equal deploy/go2_constants.py.

    They will diverge eventually (XML keyframe edit, SDK convention change),
    but today both agree. This test catches silent drift between training
    env and deploy stack — exactly what bricked 2026-04-10 → 2026-04-24.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("warp")

    from jax_rl.envs.locomotion.go2_warp_joystick import (
        WarpJoystick, default_config,
    )
    from deploy.go2_constants import (
        DEFAULT_POSE_POLICY, DEFAULT_POSE_SDK, POLICY_TO_SDK, SDK_TO_POLICY,
        ACTION_SCALE,
    )

    env = WarpJoystick(default_config())
    m = env.get_control_metadata()

    np.testing.assert_array_almost_equal(
        m["default_pose_policy"], DEFAULT_POSE_POLICY,
        err_msg="env XML keyframe ≠ deploy/go2_constants.py — update one.",
    )
    np.testing.assert_array_almost_equal(
        m["default_pose_sdk"], DEFAULT_POSE_SDK,
        err_msg="env-derived default_pose_sdk drifted from constants.",
    )
    np.testing.assert_array_equal(m["policy_to_sdk"], POLICY_TO_SDK)
    np.testing.assert_array_equal(m["sdk_to_policy"], SDK_TO_POLICY)
    assert np.isclose(m["action_scale"], ACTION_SCALE), (
        f"action_scale drift: env={m['action_scale']} vs constants={ACTION_SCALE}"
    )


# ---------- action mapping: zero action → default pose ---------------------

def test_zero_action_maps_to_default_pose_sdk():
    """action_policy = zeros(12) → SDK targets == default_pose_sdk."""
    control = _go2_default_control()
    action = np.zeros(12, dtype=np.float32)
    targets_sdk = _action_to_sdk_targets(action, control)
    np.testing.assert_array_almost_equal(
        targets_sdk, control["default_pose_sdk"],
        err_msg="Zero action must produce exactly default_pose_sdk.",
    )


def test_one_hot_action_targets_correct_named_joint():
    """Setting action[k]=1 in policy order → only sdk_joint_names[?] changes.

    For each policy index k, find which SDK index it maps to and assert that
    the corresponding SDK joint name matches.
    """
    control = _go2_default_control()
    policy_names = control["policy_joint_names"]
    sdk_names = control["sdk_joint_names"]
    policy_to_sdk = np.asarray(control["policy_to_sdk"])
    action_scale = float(control["action_scale"])

    # policy_to_sdk[i] = which policy index goes into SDK slot i.
    # So policy index k lands at SDK slot j where policy_to_sdk[j] == k.
    for k, p_name in enumerate(policy_names):
        action = np.zeros(12, dtype=np.float32)
        action[k] = 1.0
        targets_sdk = _action_to_sdk_targets(action, control)
        delta = targets_sdk - np.asarray(control["default_pose_sdk"])

        moved = np.where(np.abs(delta) > 1e-6)[0]
        assert len(moved) == 1, (
            f"action[{p_name}]=1 moved {len(moved)} SDK joints, expected 1"
        )
        sdk_slot = int(moved[0])
        assert np.isclose(delta[sdk_slot], action_scale), (
            f"delta[{sdk_slot}]={delta[sdk_slot]:.4f} ≠ action_scale={action_scale}"
        )
        # The SDK slot's joint name must equal the policy joint we activated.
        assert sdk_names[sdk_slot] == p_name, (
            f"action[{p_name}]=1 moved SDK joint '{sdk_names[sdk_slot]}' "
            f"(expected '{p_name}'). Joint remap is wrong."
        )


# ---------- ObsBuilder strict mode refuses legacy ckpts --------------------

def test_obs_builder_strict_refuses_missing_schema():
    from deploy.obs_builder import ObsBuilder
    with tempfile.TemporaryDirectory() as td:
        meta = _make_meta(obs_schema=None, control=_go2_default_control())
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump(meta, f)
        with pytest.raises(RuntimeError, match="missing 'obs_schema'"):
            ObsBuilder.from_checkpoint(td, strict=True)


def test_obs_builder_strict_refuses_missing_control():
    from deploy.obs_builder import ObsBuilder
    schema = {"state": ["gyro", "gravity", "joint_pos_offset",
                        "joint_vel", "last_act", "command"]}
    with tempfile.TemporaryDirectory() as td:
        meta = _make_meta(obs_schema=schema, control=None, obs_dim=45)
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump(meta, f)
        with pytest.raises(RuntimeError, match="missing 'control'"):
            ObsBuilder.from_checkpoint(td, strict=True)


def test_obs_builder_strict_refuses_partial_control_block():
    """Codex review fix 1: a `control` block missing default_pose_policy /
    sdk_to_policy / etc. would otherwise sneak past obs_builder and KeyError
    later in robot_interface.send_action. Strict mode must catch it early."""
    from deploy.obs_builder import ObsBuilder
    schema = {"state": ["gyro", "gravity", "joint_pos_offset",
                        "joint_vel", "last_act", "command"]}
    # Legacy partial: only Kp/Kd/action_scale/policy_dt (pre-2026-04-27 stamp).
    legacy_partial = {
        "Kp": 20.0, "Kd": 0.5, "action_scale": 0.5,
        "policy_dt": 0.02, "physics_dt": 0.004,
    }
    with tempfile.TemporaryDirectory() as td:
        meta = _make_meta(obs_schema=schema, control=legacy_partial, obs_dim=45)
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump(meta, f)
        with pytest.raises(RuntimeError, match="partial legacy 'control'"):
            ObsBuilder.from_checkpoint(td, strict=True)


def test_obs_builder_strict_uses_meta_default_pose():
    """When control['default_pose_policy'] differs from constants, strict
    mode uses meta — proves the obs builder doesn't silently fall back."""
    from deploy.obs_builder import ObsBuilder
    schema = {"state": ["gyro", "gravity", "joint_pos_offset",
                        "joint_vel", "last_act", "command"]}
    custom_pose = [0.1, 0.7, -1.5] * 4  # deliberately != go2_constants
    control = _go2_default_control()
    control["default_pose_policy"] = custom_pose

    with tempfile.TemporaryDirectory() as td:
        meta = _make_meta(obs_schema=schema, control=control, obs_dim=45)
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump(meta, f)
        builder = ObsBuilder.from_checkpoint(td, strict=True)
        np.testing.assert_array_almost_equal(builder.default_pose_policy, custom_pose)


# ---------- Go2Interface meta consumption ----------------------------------

def test_go2_interface_uses_control_meta_values():
    """Go2Interface(control_meta=...) reads Kp/Kd/action_scale/default_pose
    from meta, not constants."""
    # Mock unitree_sdk2py imports — Go2Interface __init__ doesn't connect.
    sys.modules.setdefault("unitree_sdk2py", _UnitreeMock())
    sys.modules.setdefault("unitree_sdk2py.core", _UnitreeMock())
    sys.modules.setdefault("unitree_sdk2py.core.channel", _UnitreeMock())
    sys.modules.setdefault("unitree_sdk2py.idl", _UnitreeMock())
    sys.modules.setdefault("unitree_sdk2py.idl.default", _UnitreeMock())
    sys.modules.setdefault("unitree_sdk2py.idl.unitree_go", _UnitreeMock())
    sys.modules.setdefault("unitree_sdk2py.idl.unitree_go.msg", _UnitreeMock())
    sys.modules.setdefault("unitree_sdk2py.idl.unitree_go.msg.dds_", _UnitreeMock())
    sys.modules.setdefault("unitree_sdk2py.utils", _UnitreeMock())
    sys.modules.setdefault("unitree_sdk2py.utils.crc", _UnitreeMock())

    try:
        from deploy.robot_interface import Go2Interface
    except Exception as e:
        pytest.skip(f"unitree_sdk2py not available, skipping: {e}")

    control = _go2_default_control()
    control["Kp"] = 12.5  # non-default to prove meta wins
    control["action_scale"] = 0.25

    iface = Go2Interface(sim=True, control_meta=control)
    assert iface.kp == 12.5
    assert iface.action_scale == 0.25
    np.testing.assert_array_almost_equal(
        iface.default_pose_sdk, control["default_pose_sdk"]
    )


class _UnitreeMock:
    """Minimal stand-in for unitree_sdk2py modules so robot_interface imports."""
    def __getattr__(self, name):
        # ChannelPublisher/CRC/etc. are looked up but not called in __init__.
        return _UnitreeMock()

    def __call__(self, *args, **kwargs):
        return _UnitreeMock()


# ---------- ObsBuilder integrates with control meta -------------------------

def test_obs_builder_signals_use_meta_default_pose():
    """When ObsBuilder is built with control['default_pose_policy'], the
    joint_pos_offset term is computed against THAT pose, not constants."""
    from deploy.obs_builder import ObsBuilder

    custom_pose = np.array([0.1, 0.7, -1.5] * 4, dtype=np.float32)
    sdk_to_policy = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
    builder = ObsBuilder(
        state_schema=["joint_pos_offset"],
        default_pose_policy=custom_pose,
        sdk_to_policy=sdk_to_policy,
    )

    # Joint pos in SDK order; after sdk_to_policy → all zeros in policy order.
    joint_pos_sdk = np.zeros(12, dtype=np.float32)
    obs = builder.build(
        joint_pos_sdk=joint_pos_sdk,
        joint_vel_sdk=np.zeros(12, dtype=np.float32),
        gyroscope=np.zeros(3, dtype=np.float32),
        accelerometer=np.array([0.0, 0.0, 9.8], dtype=np.float32),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        command=np.zeros(3, dtype=np.float32),
    )
    # joint_pos_offset = joint_pos_policy - default_pose_policy
    # joint_pos_policy = zeros, so offset = -custom_pose
    np.testing.assert_array_almost_equal(obs, -custom_pose)


# ---------- Round-trip: env → meta → ObsBuilder → reproduces obs -----------

def test_meta_round_trip_through_save_checkpoint_format():
    """A meta.json written with Go2 control block round-trips through
    ObsBuilder.from_checkpoint(strict=True) without drift."""
    from deploy.obs_builder import ObsBuilder

    schema = {"state": [
        "gyro", "accelerometer", "gravity",
        "joint_pos_offset", "joint_vel", "last_act", "command",
    ]}
    control = _go2_default_control()

    with tempfile.TemporaryDirectory() as td:
        meta = _make_meta(obs_schema=schema, control=control, obs_dim=48)
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump(meta, f)

        builder = ObsBuilder.from_checkpoint(td, strict=True)
        assert builder.state_schema == schema["state"]
        assert builder.raw_dim == 48
        np.testing.assert_array_almost_equal(
            builder.default_pose_policy, control["default_pose_policy"]
        )
        np.testing.assert_array_equal(
            builder.sdk_to_policy, control["sdk_to_policy"]
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
