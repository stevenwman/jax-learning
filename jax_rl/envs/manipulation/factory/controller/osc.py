"""Khatib operational-space control (OSC) for the Factory PegInsert arm.

Pipeline (per .superpowers/specs/2026-05-27-factory-mjx-warp-port.md):
  - Forward kinematics → fingertip pose (from mjx data.site_xpos/xmat)
  - Pose error: pos delta + axis-angle rotation
  - Task wrench:  F = Kp·err - Kd·vel
  - Task torque:  τ_task = J^T · Λ · wrench           where Λ = (J·M^-1·J^T)^-1
  - Nullspace:    τ_null = (I - J^T·Λ·J·M^-1) · M · (kp_null·(q_def - q) - kd_null·qdot)
  - Final:        τ = clamp(τ_task + τ_null, ±torque_limit)

fp64 escape hatch: wrap the controller in `jax.experimental.enable_x64` if Λ
inverse goes ill-conditioned near singular configs.
"""
from typing import Tuple

import jax.numpy as jp
from mujoco import mjx


def compute_site_jacobian(model, data, site_id: int) -> jp.ndarray:
    """Geometric Jacobian of a site, world frame, shape (6, nv).

    Row block 0:3 = linear velocity (matches mj_jacSite's jacp).
    Row block 3:6 = angular velocity (matches mj_jacSite's jacr).

    Implementation notes (verified against mujoco.mjx._src.support.jac):
      - `mjx.jac(model, data, point, body_id)` returns `(jacp, jacr)`
        each of shape `(NV, 3)` — note the order is (NV, 3), NOT (3, NV).
      - `site_bodyid` lives on Model, not Data.
    """
    jacp, jacr = mjx.jac(
        model,
        data,
        data.site_xpos[site_id],     # point in world frame
        model.site_bodyid[site_id],  # parent body id (Model field, NOT Data)
    )
    return jp.concatenate([jacp.T, jacr.T], axis=0)


def compute_pose_error(
    cur_pos: jp.ndarray, cur_quat: jp.ndarray,
    tgt_pos: jp.ndarray, tgt_quat: jp.ndarray,
) -> Tuple[jp.ndarray, jp.ndarray]:
    """OSC pose error: position delta (linear) + axis-angle rotation.

    pos_err = tgt_pos - cur_pos                    (points toward target)
    rot_err = log_quat(tgt_quat ⊗ conj(cur_quat))  (rotates cur → tgt)
    """
    pos_err = tgt_pos - cur_pos

    err_quat = _quat_mul(tgt_quat, _quat_conj(cur_quat))
    # Axis-angle, small-angle-stable via atan2(sin_half, cos_half).
    sin_half = jp.linalg.norm(err_quat[1:])
    cos_half = err_quat[0]
    angle = 2.0 * jp.arctan2(sin_half, cos_half)
    safe_denom = jp.where(sin_half > 1e-8, sin_half, 1.0)
    axis = jp.where(sin_half > 1e-8, err_quat[1:] / safe_denom, jp.zeros(3))
    rot_err = axis * angle
    return pos_err, rot_err


def _quat_mul(a: jp.ndarray, b: jp.ndarray) -> jp.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conj(q: jp.ndarray) -> jp.ndarray:
    return jp.array([q[0], -q[1], -q[2], -q[3]])


def _site_quat(data, site_id: int) -> jp.ndarray:
    """Read fingertip orientation as a (w,x,y,z) quat from site_xmat."""
    m = data.site_xmat[site_id].reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    import jax

    def trace_pos(_):
        s = jp.sqrt(trace + 1.0) * 2
        return jp.array([0.25 * s,
                         (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s])

    def case_xx(_):
        s = jp.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return jp.array([(m[2, 1] - m[1, 2]) / s,
                         0.25 * s,
                         (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s])

    def case_yy(_):
        s = jp.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return jp.array([(m[0, 2] - m[2, 0]) / s,
                         (m[0, 1] + m[1, 0]) / s,
                         0.25 * s,
                         (m[1, 2] + m[2, 1]) / s])

    def case_zz(_):
        s = jp.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        return jp.array([(m[1, 0] - m[0, 1]) / s,
                         (m[0, 2] + m[2, 0]) / s,
                         (m[1, 2] + m[2, 1]) / s,
                         0.25 * s])

    xx, yy, zz = m[0, 0], m[1, 1], m[2, 2]

    def trace_neg(_):
        return jax.lax.cond(
            (xx > yy) & (xx > zz),
            case_xx,
            lambda _: jax.lax.cond(yy > zz, case_yy, case_zz, operand=None),
            operand=None,
        )

    return jax.lax.cond(trace > 0, trace_pos, trace_neg, operand=None)


def compute_osc_torque(
    model, data,
    target_pos: jp.ndarray, target_quat: jp.ndarray,
    site_id: int,
    arm_dof_ids: jp.ndarray,    # (7,) indices into qvel
    arm_qpos_ids: jp.ndarray,   # (7,) indices into qpos
    kp_task: jp.ndarray,        # (6,) pos(3) + rot(3)
    kd_task: jp.ndarray,        # (6,)
    q_default: jp.ndarray,      # (7,)
    kp_null: float, kd_null: float,
    torque_limit: float = 100.0,
    feedforward=None,           # (7,) optional FF torque (e.g. gravity comp)
) -> jp.ndarray:
    """Khatib OSC arm torque (7,). Caller writes into ctrl[arm_act_ids].

    NOTE: returns shape (7,) — caller maps to actuator indices. We avoid
    constructing a full nu-length vector here so the same controller works
    for the panda-alone test (no freejoint peg in qvel) and the full
    Factory scene (peg freejoint at qvel[0:6], arm at qvel[6:13]).
    """
    # 1. FK + Jacobian of fingertip site
    fingertip_pos = data.site_xpos[site_id]
    fingertip_quat = _site_quat(data, site_id)

    J_full = compute_site_jacobian(model, data, site_id)      # (6, nv)
    J_arm = J_full[:, arm_dof_ids]                            # (6, 7)
    qvel_arm = data.qvel[arm_dof_ids]                         # (7,)
    fingertip_vel = J_arm @ qvel_arm                          # (6,)

    # 2. Pose error
    pos_err, rot_err = compute_pose_error(
        fingertip_pos, fingertip_quat, target_pos, target_quat
    )

    # 3. Task wrench (PD on pose error, damping on velocity)
    wrench = jp.concatenate([
        kp_task[:3] * pos_err - kd_task[:3] * fingertip_vel[:3],
        kp_task[3:] * rot_err - kd_task[3:] * fingertip_vel[3:],
    ])

    # 4. Mass matrix slice (arm × arm submatrix)
    M_full = mjx.full_m(model, data)
    M_arm = M_full[jp.ix_(arm_dof_ids, arm_dof_ids)]          # (7, 7)
    M_inv_arm = jp.linalg.inv(M_arm)

    # 5. Operational-space mass matrix Λ and DC nullspace projection.
    # Damp the inversion (1e-4 ridge) to keep Λ well-conditioned near
    # kinematic singularities — cuSolver crashed on undamped inv at some
    # arm configs reached after the IK reset.
    Lambda_inv = J_arm @ M_inv_arm @ J_arm.T + 1e-4 * jp.eye(6)
    Lambda = jp.linalg.inv(Lambda_inv)                         # (6, 6)
    J_bar = M_inv_arm @ J_arm.T @ Lambda                       # (7, 6) DC pseudo-inv
    null_proj = jp.eye(7) - J_bar @ J_arm                      # (7, 7)

    # 6. Nullspace posture term — drag toward q_default in joint space
    qpos_arm = data.qpos[arm_qpos_ids]
    tau_null_raw = M_arm @ (kp_null * (q_default - qpos_arm) - kd_null * qvel_arm)
    tau_null = null_proj @ tau_null_raw

    # 7. Task torque
    tau_task = J_arm.T @ Lambda @ wrench                       # (7,)

    tau = tau_task + tau_null
    if feedforward is not None:
        tau = tau + feedforward
    return jp.clip(tau, -torque_limit, torque_limit)
