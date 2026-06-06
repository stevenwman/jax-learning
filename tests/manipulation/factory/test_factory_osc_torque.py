"""Task 2.3 — hermetic test: compute_osc_torque shape + equilibrium + linearity.

OSC controller composes Jacobian + pose error + task wrench + nullspace.
At equilibrium (q=q_default, qdot=0, target pose == current pose), all
contributions are zero so tau = 0. Off equilibrium, torque magnitude scales
linearly with pose error (the wrench is linear in err, Λ is q-only).
"""
from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np
import pytest


@pytest.fixture(scope="module")
def panda_at_default():
    """Panda standalone CPU model at IsaacLab reset_joints pose."""
    from mujoco import mjx
    xml_path = (
        Path(__file__).parents[3]
        / "jax_rl" / "envs" / "manipulation" / "factory"
        / "assets" / "franka_panda" / "panda.xml"
    )
    m = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    d = mujoco.MjData(m)
    q_default = np.array(
        [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761,
         -2.7717e-04, 1.7796, 7.8556e-01]
    )
    d.qpos[:7] = q_default
    d.qvel[:7] = 0.0
    mujoco.mj_forward(m, d)
    mjx_model = mjx.put_model(m)
    mjx_data = mjx.put_data(m, d)
    return m, d, mjx_model, mjx_data, q_default


def _site_pose(d, m, site_id):
    """Read site pose from CPU MjData (post-mj_forward)."""
    pos = np.asarray(d.site_xpos[site_id])
    mat = np.asarray(d.site_xmat[site_id]).reshape(3, 3)
    # Convert via mujoco's quat utility for ground-truth orientation.
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return pos, quat


def test_osc_torque_shape(panda_at_default):
    from jax_rl.envs.manipulation.factory.controller.osc import compute_osc_torque
    m, d, mjx_model, mjx_data, q_default = panda_at_default
    site_id = m.site("fingertip_centered").id
    pos, quat = _site_pose(d, m, site_id)
    arm_ids = jp.arange(7)
    tau = compute_osc_torque(
        mjx_model, mjx_data,
        target_pos=jp.asarray(pos),
        target_quat=jp.asarray(quat),
        site_id=site_id,
        arm_dof_ids=arm_ids,
        arm_qpos_ids=arm_ids,
        kp_task=jp.array([100., 100., 100., 30., 30., 30.]),
        kd_task=jp.array([20., 20., 20., 5., 5., 5.]),
        q_default=jp.asarray(q_default),
        kp_null=10.0, kd_null=6.32,
        torque_limit=100.0,
    )
    assert tau.shape == (7,)


def test_osc_torque_zero_at_equilibrium(panda_at_default):
    """q=q_default, qdot=0, tgt=current pose → tau ≈ 0 (no error to drive)."""
    from jax_rl.envs.manipulation.factory.controller.osc import compute_osc_torque
    m, d, mjx_model, mjx_data, q_default = panda_at_default
    site_id = m.site("fingertip_centered").id
    pos, quat = _site_pose(d, m, site_id)
    arm_ids = jp.arange(7)
    tau = compute_osc_torque(
        mjx_model, mjx_data,
        target_pos=jp.asarray(pos),
        target_quat=jp.asarray(quat),
        site_id=site_id,
        arm_dof_ids=arm_ids,
        arm_qpos_ids=arm_ids,
        kp_task=jp.array([100., 100., 100., 30., 30., 30.]),
        kd_task=jp.array([20., 20., 20., 5., 5., 5.]),
        q_default=jp.asarray(q_default),
        kp_null=10.0, kd_null=6.32,
        torque_limit=100.0,
    )
    np.testing.assert_allclose(np.asarray(tau), np.zeros(7), atol=1e-5)


def test_osc_torque_nonzero_with_position_error(panda_at_default):
    """Target offset by +1cm in x → arm torque nonzero (wrench drives EE +x)."""
    from jax_rl.envs.manipulation.factory.controller.osc import compute_osc_torque
    m, d, mjx_model, mjx_data, q_default = panda_at_default
    site_id = m.site("fingertip_centered").id
    pos, quat = _site_pose(d, m, site_id)
    arm_ids = jp.arange(7)
    tau = compute_osc_torque(
        mjx_model, mjx_data,
        target_pos=jp.asarray(pos) + jp.array([0.01, 0.0, 0.0]),
        target_quat=jp.asarray(quat),
        site_id=site_id,
        arm_dof_ids=arm_ids,
        arm_qpos_ids=arm_ids,
        kp_task=jp.array([100., 100., 100., 30., 30., 30.]),
        kd_task=jp.array([20., 20., 20., 5., 5., 5.]),
        q_default=jp.asarray(q_default),
        kp_null=10.0, kd_null=6.32,
        torque_limit=100.0,
    )
    tau_np = np.asarray(tau)
    assert np.linalg.norm(tau_np) > 0.01, f"tau norm should be nonzero: {tau_np}"


def test_osc_torque_linear_in_position_error(panda_at_default):
    """Doubling pos error doubles torque (linear in wrench, Λ and J fixed at q)."""
    from jax_rl.envs.manipulation.factory.controller.osc import compute_osc_torque
    m, d, mjx_model, mjx_data, q_default = panda_at_default
    site_id = m.site("fingertip_centered").id
    pos, quat = _site_pose(d, m, site_id)
    arm_ids = jp.arange(7)

    def torque_with_err(delta):
        return compute_osc_torque(
            mjx_model, mjx_data,
            target_pos=jp.asarray(pos) + jp.array([delta, 0.0, 0.0]),
            target_quat=jp.asarray(quat),
            site_id=site_id,
            arm_dof_ids=arm_ids,
            arm_qpos_ids=arm_ids,
            kp_task=jp.array([100., 100., 100., 30., 30., 30.]),
            kd_task=jp.array([20., 20., 20., 5., 5., 5.]),
            q_default=jp.asarray(q_default),
            kp_null=10.0, kd_null=6.32,
            torque_limit=1e6,                          # disable clamp
        )

    tau_1 = np.asarray(torque_with_err(0.005))
    tau_2 = np.asarray(torque_with_err(0.010))
    np.testing.assert_allclose(tau_2, 2.0 * tau_1, atol=1e-4)


def test_osc_torque_clamped_at_limit(panda_at_default):
    """Large error must be clamped to ±torque_limit element-wise."""
    from jax_rl.envs.manipulation.factory.controller.osc import compute_osc_torque
    m, d, mjx_model, mjx_data, q_default = panda_at_default
    site_id = m.site("fingertip_centered").id
    pos, quat = _site_pose(d, m, site_id)
    arm_ids = jp.arange(7)
    tau = compute_osc_torque(
        mjx_model, mjx_data,
        target_pos=jp.asarray(pos) + jp.array([10.0, 10.0, 10.0]),  # absurd
        target_quat=jp.asarray(quat),
        site_id=site_id,
        arm_dof_ids=arm_ids,
        arm_qpos_ids=arm_ids,
        kp_task=jp.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6]),
        kd_task=jp.array([20., 20., 20., 5., 5., 5.]),
        q_default=jp.asarray(q_default),
        kp_null=10.0, kd_null=6.32,
        torque_limit=50.0,
    )
    tau_np = np.asarray(tau)
    assert np.all(np.abs(tau_np) <= 50.0 + 1e-5), f"tau exceeds clamp: {tau_np}"
    # At least one joint should be saturated.
    assert np.max(np.abs(tau_np)) > 49.0, f"expected saturation: {tau_np}"
