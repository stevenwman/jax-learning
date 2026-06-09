"""Unit tests for the DC-motor torque-speed clip (mjlab `_clip_effort` port).

Pure-function tests on CPU — no Go2, no Warp, no GPU. The function caps a
commanded joint torque by a linear DC-motor torque-speed curve:

  - At |dq| = 0:            torque limited to ±effort_limit (continuous cap).
  - Driving at |dq| = vlim: drive torque → 0 (can't push past no-load speed);
                            braking torque still allowed.
  - |dq| ≥ vel_at_eff:      torque FORCED to full braking (active deceleration).
  - effort_limit < stall:   flat top sits at effort_limit (thermal derating).

These are the defining 4-quadrant properties of `DcMotorActuator._clip_effort`
in mjlab (`src/mjlab/actuator/dc_actuator.py`). Reference math:
    vel_at_eff = vlim * (1 + effort/stall)
    top = stall*(1 - v/vlim);  bot = stall*(-1 - v/vlim)   (v clipped to ±vel_at_eff)
    allow ∈ [max(bot,-effort), min(top, effort)]
"""
import jax.numpy as jp
import numpy as np
import pytest

from jax_rl.envs.locomotion.go2_warp_base import torque_speed_clip

# Go2 calf (knee) numbers: stall = MJCF ctrlrange, vlim = URDF velocity limit.
STALL = 45.43
VLIM = 20.07


def _clip(tau, dq, stall=STALL, vlim=VLIM, effort=None):
    effort = stall if effort is None else effort
    return float(
        torque_speed_clip(
            jp.array([tau]), jp.array([dq]),
            jp.array([stall]), jp.array([vlim]), jp.array([effort]),
        )[0]
    )


def test_zero_velocity_passes_within_effort_limit():
    # At dq=0 the cap is the continuous effort_limit (== stall here).
    assert _clip(10.0, 0.0) == pytest.approx(10.0)
    assert _clip(100.0, 0.0) == pytest.approx(STALL)      # clamps to stall
    assert _clip(-100.0, 0.0) == pytest.approx(-STALL)


def test_driving_at_no_load_speed_gives_zero_drive_but_allows_braking():
    # dq = +vlim: forward drive torque must be killed; braking (-) still allowed.
    assert _clip(30.0, VLIM) == pytest.approx(0.0, abs=1e-4)   # drive → 0
    assert _clip(-30.0, VLIM) == pytest.approx(-30.0)          # brake allowed


def test_overspeed_forces_full_braking():
    # dq = 2*vlim (== vel_at_eff when effort==stall): allowed window collapses
    # to exactly -stall, so the motor FORCES a full braking torque regardless
    # of command. This is what actively decelerates an over-spun calf.
    assert _clip(30.0, 2 * VLIM) == pytest.approx(-STALL)
    assert _clip(0.0, 2 * VLIM) == pytest.approx(-STALL)


def test_negative_velocity_is_symmetric():
    # dq = -vlim: reverse drive (-) killed, forward braking (+) allowed.
    assert _clip(-30.0, -VLIM) == pytest.approx(0.0, abs=1e-4)
    assert _clip(30.0, -VLIM) == pytest.approx(30.0)


def test_continuous_effort_limit_below_stall_caps_flat_top():
    # effort_limit < stall (thermal derating): at dq=0 the cap is effort_limit,
    # NOT the higher stall torque.
    eff = 35.55
    assert _clip(40.0, 0.0, effort=eff) == pytest.approx(eff)
    assert _clip(20.0, 0.0, effort=eff) == pytest.approx(20.0)


def test_vectorized_over_joints():
    # Per-joint stall/vlim/effort, batched (the real call shape: 12 joints).
    tau = jp.array([100.0, 100.0, 100.0])
    dq = jp.array([0.0, VLIM, 2 * VLIM])
    stall = jp.array([STALL, STALL, STALL])
    vlim = jp.array([VLIM, VLIM, VLIM])
    eff = jp.array([STALL, STALL, STALL])
    out = np.asarray(torque_speed_clip(tau, dq, stall, vlim, eff))
    np.testing.assert_allclose(out, [STALL, 0.0, -STALL], atol=1e-4)


# ── Physical (mjlab-matched) per-joint armature ──────────────────────────────
# Reflected rotor inertia: rotor 0.000111842; hip/thigh gear 6 → ·36; knee gear
# 9 (1.5× cam) → ·81. mjlab go1 values (same motor family as go2).
ARM_HIP = 0.000111842 * 36   # 0.00402631
ARM_KNEE = 0.000111842 * 81  # 0.00905920


def _load_go2_mj_model():
    import mujoco
    from jax_rl.envs.locomotion.go2_warp_base import get_warp_assets
    from jax_rl.envs.locomotion import go2_constants as consts
    return mujoco.MjModel.from_xml_string(
        consts.WARP_SCENE_FLAT_XML.read_text(), assets=get_warp_assets()
    )


def test_physical_armature_sets_mjlab_per_joint_values():
    from jax_rl.envs.locomotion.go2_warp_base import physical_armature
    m = _load_go2_mj_model()
    arm = physical_armature(m)

    # The 6 freejoint dofs (base) must be untouched.
    np.testing.assert_array_equal(arm[:6], m.dof_armature[:6])

    # Each actuated joint: calf → knee armature, hip/thigh → hip armature.
    n_knee = n_hip = 0
    for i in range(m.nu):
        adr = m.jnt_dofadr[m.actuator_trnid[i, 0]]
        name = m.jnt(m.actuator_trnid[i, 0]).name
        if name.endswith("calf_joint"):
            assert arm[adr] == pytest.approx(ARM_KNEE); n_knee += 1
        else:
            assert arm[adr] == pytest.approx(ARM_HIP); n_hip += 1
    assert (n_hip, n_knee) == (8, 4)   # 4 legs × (hip,thigh) ; 4 calves
