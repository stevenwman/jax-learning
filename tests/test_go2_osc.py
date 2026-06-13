"""Unit tests for the per-leg Cartesian impedance / OSC controller.

Runs on CPU against a *synthetic* 3-DoF leg (no Go2, no Warp, no GPU) so the
controller math is tested in isolation. Gravity is disabled in the test model
so that q̈ = M⁻¹·τ exactly, letting us predict foot acceleration from physics
first principles rather than from the controller's own internals.

Key physics predictions used as assertions (impl-independent):
  - Equilibrium (target == current foot, q̇ == 0) → τ == 0.
  - Full OSC (Λ-weighted): closed-loop foot accel q̈_foot = J·M⁻¹·τ equals
    kp ⊙ err EXACTLY (Λ cancels the configuration-dependent inertia → unit
    mass). This is the defining property of operational-space control.
  - Jacobian-transpose mode: foot accel shares the SIGN of err per-axis
    (J·M⁻¹·Jᵀ is positive-definite) but is not inertia-normalized.
  - Velocity damping opposes leg-induced foot motion.
  - Torques clip to the per-joint limit.
"""
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from jax_rl.envs.locomotion.go2_osc import compute_leg_impedance_torque

# Spatial 3R leg: hip abduction (axis x), thigh + calf flexion (axis y).
# Foot position spans 3D (generic rank-3 Jacobian), mimicking a Go2 leg.
# Trunk is a FIXED body (no freejoint) so nv == 3 (the three hinges).
_LEG_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="trunk" pos="0 0 0.6">
      <geom type="box" size="0.1 0.06 0.04" mass="2"/>
      <site name="trunk_site" pos="0 0 0"/>
      <body name="hip" pos="0 0 0">
        <joint name="hip" type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.08" size="0.02" mass="0.2"/>
        <body name="thigh" pos="0 0 -0.08">
          <joint name="thigh" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.3"/>
          <body name="calf" pos="0 0 -0.2">
            <joint name="calf" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.018" mass="0.15"/>
            <site name="foot" pos="0 0 -0.2"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hip"/>
    <motor joint="thigh"/>
    <motor joint="calf"/>
  </actuator>
</mujoco>
"""


def _setup(qpos, qvel=None):
    """Build the synthetic leg at a given config; return (model, data, ids)."""
    m = mujoco.MjModel.from_xml_string(_LEG_XML)
    mx = mjx.put_model(m)
    d = mjx.make_data(mx)
    d = d.replace(qpos=jp.asarray(qpos, dtype=d.qpos.dtype))
    if qvel is not None:
        d = d.replace(qvel=jp.asarray(qvel, dtype=d.qvel.dtype))
    d = mjx.forward(mx, d)
    ids = dict(
        foot_site=np.array([m.site("foot").id]),
        leg_dofs=np.array([[0, 1, 2]]),
        body=m.body("trunk").id,
    )
    return mx, d, ids


def _foot_body_pos(mx, d, ids):
    """Current foot position expressed in the trunk frame."""
    body_pos = np.asarray(d.xpos[ids["body"]])
    R = np.asarray(d.xmat[ids["body"]]).reshape(3, 3)
    foot_w = np.asarray(d.site_xpos[ids["foot_site"][0]])
    return R.T @ (foot_w - body_pos)


def _foot_accel(mx, d, ids, tau):
    """Closed-loop foot acceleration q̈_foot = J·M⁻¹·τ (gravity-free model)."""
    site = ids["foot_site"][0]
    dofs = ids["leg_dofs"][0]
    jacp, _ = mjx.jac(mx, d, d.site_xpos[site], mx.site_bodyid[site])
    J = np.asarray(jacp[dofs].T)               # (3,3)
    M = np.asarray(mjx.full_m(mx, d))[np.ix_(dofs, dofs)]
    qacc = np.linalg.solve(M, np.asarray(tau))
    return J @ qacc


_QPOS = [0.1, 0.7, -1.4]   # a non-singular, leg-like config
_KP = jp.array([120.0, 120.0, 150.0])
_KD = jp.array([6.0, 6.0, 7.0])
_BIG_LIMIT = jp.full(3, 1e6)


def test_zero_torque_at_equilibrium():
    """Target == current foot, q̇ == 0 → controller commands no torque."""
    mx, d, ids = _setup(_QPOS)
    tgt = jp.asarray(_foot_body_pos(mx, d, ids))[None]   # (1,3)
    for use_lambda in (True, False):
        tau = compute_leg_impedance_torque(
            mx, d, ids["foot_site"], ids["leg_dofs"], ids["body"],
            tgt, _KP, _KD, _BIG_LIMIT, use_op_space_inertia=use_lambda,
        )
        assert np.allclose(np.asarray(tau), 0.0, atol=1e-5), (use_lambda, tau)


def test_jax_numpy_osc_parity():
    """The training jax `compute_leg_impedance_torque` and the Newton-eval numpy
    `mud_osc.osc_torque` must give the SAME torque for identical inputs — pins the
    two OSC ports equal (the math that drifted: Λ-weighting, mass A·ẍ, gain decode).
    Covers bare + Λ and a nonzero accel_force."""
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "projects" / "mud_eval"))
    import mud_osc

    m = mujoco.MjModel.from_xml_string(_LEG_XML)
    qpos = np.array(_QPOS); qvel = np.array([0.2, -0.1, 0.3])
    # CPU mujoco (numpy path)
    dc = mujoco.MjData(m); dc.qpos[:] = qpos; dc.qvel[:] = qvel; mujoco.mj_forward(m, dc)
    # mjx (jax path)
    mx = mjx.put_model(m)
    dx = mjx.make_data(mx).replace(qpos=jp.asarray(qpos), qvel=jp.asarray(qvel))
    dx = mjx.forward(mx, dx)

    site = np.array([m.site("foot").id]); dofs = np.array([[0, 1, 2]])
    body = m.body("trunk").id
    tgt = jp.asarray(_foot_body_pos(mx, dx, dict(foot_site=site, leg_dofs=dofs, body=body))
                     + np.array([0.02, -0.01, 0.015]))[None]   # (1,3)
    kp = jp.array([[120.0, 120.0, 150.0]]); kd = jp.array([[6.0, 6.0, 7.0]])
    af = jp.array([[0.4, -0.2, 0.3]])
    for use_lambda in (False, True):
        tj = compute_leg_impedance_torque(
            mx, dx, site, dofs, body, tgt, kp, kd, _BIG_LIMIT,
            use_op_space_inertia=use_lambda, accel_force=af)
        tn = mud_osc.osc_torque(
            m, dc, site, dofs, body, np.asarray(tgt), np.asarray(kp), np.asarray(kd),
            np.asarray(_BIG_LIMIT), use_op_space_inertia=use_lambda, accel_force=np.asarray(af))
        assert np.allclose(np.asarray(tj), tn, atol=1e-4), (use_lambda, np.asarray(tj), tn)


def test_jax_numpy_gain_decode_parity():
    """impedance_gains (jax) == impedance_gains_np (mud_osc) for the same action."""
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "projects" / "mud_eval"))
    import mud_osc
    from jax_rl.envs.locomotion.go2_warp_components import impedance_gains
    kpb = jp.array([3000.0, 3000.0, 4000.0]); kdb = jp.array([110.0, 110.0, 130.0])
    rng = np.random.RandomState(0)
    act = rng.uniform(-1, 1, 36)   # per_axis + damping
    kp_j, kd_j = impedance_gains(jp.asarray(act), kpb, kdb, granularity="per_axis",
                                 s_min=0.25, s_max=2.0, damping_action=True, z_min=0.5, z_max=2.0)
    kp_n, kd_n = mud_osc.impedance_gains_np(act, np.asarray(kpb), np.asarray(kdb),
                                            "per_axis", 0.25, 2.0, True, 0.5, 2.0)
    assert np.allclose(np.asarray(kp_j), kp_n, atol=1e-5)
    assert np.allclose(np.asarray(kd_j), kd_n, atol=1e-5)


def test_accel_force_enters_as_jt_a_xdd():
    """Virtual-mass law: passing accel_force=A·ẍ adds EXACTLY Jᵀ·(A·ẍ) to the
    joint torque (the F = wrench + A·ẍ term, NOT Λ-weighted). Isolated by the
    with/without-accel_force delta so the proof is independent of the wrench."""
    mx, d, ids = _setup(_QPOS, qvel=[0.2, -0.1, 0.3])
    err = np.array([0.02, -0.01, 0.015])
    tgt = jp.asarray(_foot_body_pos(mx, d, ids) + err)[None]
    af = jp.array([[0.5, -0.3, 0.4]])              # (1,3) A·ẍ, small (no clip)
    common = dict(use_op_space_inertia=False)
    tau0 = compute_leg_impedance_torque(
        mx, d, ids["foot_site"], ids["leg_dofs"], ids["body"],
        tgt, _KP, _KD, _BIG_LIMIT, accel_force=None, **common)
    tau1 = compute_leg_impedance_torque(
        mx, d, ids["foot_site"], ids["leg_dofs"], ids["body"],
        tgt, _KP, _KD, _BIG_LIMIT, accel_force=af, **common)
    site, dofs = ids["foot_site"][0], ids["leg_dofs"][0]
    jacp, _ = mjx.jac(mx, d, d.site_xpos[site], mx.site_bodyid[site])
    J = np.asarray(jacp[dofs].T)
    expected = J.T @ np.asarray(af[0])             # Jᵀ·A·ẍ
    assert np.allclose(np.asarray(tau1 - tau0), expected, atol=1e-5), (
        np.asarray(tau1 - tau0), expected)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_osc_mode_foot_accel_equals_kp_err(axis):
    """Defining OSC property: q̈_foot == kp ⊙ err (Λ cancels inertia)."""
    mx, d, ids = _setup(_QPOS)
    err = np.zeros(3)
    err[axis] = 0.03
    tgt = jp.asarray(_foot_body_pos(mx, d, ids) + err)[None]
    tau = compute_leg_impedance_torque(
        mx, d, ids["foot_site"], ids["leg_dofs"], ids["body"],
        tgt, _KP, _KD, _BIG_LIMIT, use_op_space_inertia=True,
    )
    acc = _foot_accel(mx, d, ids, tau)
    expected = np.asarray(_KP) * err
    assert np.allclose(acc, expected, atol=1e-2), (acc, expected)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_jt_mode_foot_accel_sign_matches_error(axis):
    """Jacobian-transpose mode: accel shares the error's sign on that axis."""
    mx, d, ids = _setup(_QPOS)
    err = np.zeros(3)
    err[axis] = 0.03
    tgt = jp.asarray(_foot_body_pos(mx, d, ids) + err)[None]
    tau = compute_leg_impedance_torque(
        mx, d, ids["foot_site"], ids["leg_dofs"], ids["body"],
        tgt, _KP, _KD, _BIG_LIMIT, use_op_space_inertia=False,
    )
    acc = _foot_accel(mx, d, ids, tau)
    assert acc[axis] > 0.0, (axis, acc)


def test_velocity_damping_opposes_motion():
    """Zero position error but foot moving → torque damps it (accel opposes)."""
    mx, d, ids = _setup(_QPOS)
    tgt = jp.asarray(_foot_body_pos(mx, d, ids))[None]  # zero pos error
    # Give the calf a velocity so the foot moves; measure leg-induced foot vel.
    qvel = [0.0, 0.0, 1.0]
    mx, d, ids = _setup(_QPOS, qvel)
    site, dofs = ids["foot_site"][0], ids["leg_dofs"][0]
    jacp, _ = mjx.jac(mx, d, d.site_xpos[site], mx.site_bodyid[site])
    v_leg = np.asarray(jacp[dofs].T) @ np.asarray(qvel)
    tau = compute_leg_impedance_torque(
        mx, d, ids["foot_site"], ids["leg_dofs"], ids["body"],
        tgt, _KP, _KD, _BIG_LIMIT, use_op_space_inertia=True,
    )
    acc = _foot_accel(mx, d, ids, tau)
    # Damping force opposes velocity → accel·v_leg < 0.
    assert float(acc @ v_leg) < 0.0, (acc, v_leg)


def test_per_leg_gains_match_shared():
    """Passing per-leg (n_legs,3) gains == shared (3,) gains for one leg."""
    mx, d, ids = _setup(_QPOS)
    err = np.array([0.0, 0.0, 0.03])
    tgt = jp.asarray(_foot_body_pos(mx, d, ids) + err)[None]
    args = (mx, d, ids["foot_site"], ids["leg_dofs"], ids["body"], tgt)
    tau_shared = compute_leg_impedance_torque(*args, _KP, _KD, _BIG_LIMIT)
    tau_perleg = compute_leg_impedance_torque(*args, _KP[None], _KD[None], _BIG_LIMIT)
    assert np.allclose(np.asarray(tau_shared), np.asarray(tau_perleg), atol=1e-6)


def test_torque_clips_to_limit():
    """A huge target error saturates every joint torque at the limit."""
    mx, d, ids = _setup(_QPOS)
    tgt = jp.asarray(_foot_body_pos(mx, d, ids) + np.array([10.0, 10.0, 10.0]))[None]
    limit = jp.array([5.0, 8.0, 8.0])
    tau = compute_leg_impedance_torque(
        mx, d, ids["foot_site"], ids["leg_dofs"], ids["body"],
        tgt, _KP, _KD, limit, use_op_space_inertia=True,
    )
    assert np.allclose(np.abs(np.asarray(tau)), np.asarray(limit), atol=1e-5), tau
