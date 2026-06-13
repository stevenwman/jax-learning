"""Per-leg Cartesian impedance / OSC controller for the Go2 quadruped.

This is the quadruped analogue of the Factory arm OSC
(`jax_rl/envs/manipulation/factory/controller/osc.py`), simplified for the
legged case:

  - **Per leg, 3-DoF.** Each leg (hip, thigh, calf) drives one foot. The foot
    is the end-effector. A 3-DoF leg driving a 3-DoF foot *position* task has
    NO redundancy, so there is NO nullspace term (unlike the 7-DoF arm).
  - **Position only.** No orientation/quaternion machinery — feet are points.
  - **Body-frame target, world-frame error.** Targets are specified in the
    trunk (base_link) frame so the robot can walk (a world-fixed foot target
    would pin the robot in place). The target is rotated into world via the
    trunk rotation `R_body`; the impedance error is then a plain world-frame
    vector. With isotropic gains (kp_x = kp_y) the frame the wrench lives in
    does not matter; for anisotropic gains this treats them as *world* axes,
    which ≈ body axes while the trunk is near-upright. Documented, not hidden.
  - **Velocity damping is leg-relative.** We damp `J_leg · q̇_leg` (the foot
    velocity the leg itself produces), NOT the full world foot velocity — the
    base-induced component is not controllable by leg torques and must not be
    fought.
  - **No gravity feedforward in the MVP.** On a floating base, `qfrc_bias`
    sliced to the leg DoFs only cancels the leg's own link gravity (not
    body-weight support, which rides through contact), and couples to base
    attitude. We run pure impedance: the spring bears the load. A `feedforward`
    hook is left in the signature so the ablation is a one-line change later.

The Khatib operational-space mass matrix Λ = (J·M⁻¹·Jᵀ)⁻¹ inertia-weights the
wrench so the foot behaves (approximately) like a unit mass in every direction.
Set `use_op_space_inertia=False` to drop Λ and get a classic Jacobian-transpose
Cartesian impedance (τ = Jᵀ·(Kp·err − Kd·ẋ)), the standard legged-robot form
with honest N/m gains. Both are exposed for ablation.

CAVEAT (floating base): we build Λ from `M_leg`, the leg-block slice of the
mass matrix — it ignores base↔leg inertial coupling. On a fixed-base arm this
is exact; on the Go2's floating base it is an approximation, so the unit-mass
decoupling is only approximate (audited: realized foot accel deviates ~10–20%
from the ideal kp·err, sign always correct, loop stable). Exact decoupling
would need the full constrained/whole-body inertia — out of scope for the MVP.
"""
from typing import Optional

import jax.numpy as jp
from mujoco import mjx


def compute_leg_impedance_torque(
    model,
    data,
    foot_site_ids,          # (n_legs,) int — site id per foot, npy/concrete ints
    leg_dof_ids,            # (n_legs, 3) int — qvel indices of each leg's 3 joints
    body_id: int,           # trunk body id (targets are in this body's frame)
    target_foot_body,       # (n_legs, 3) desired foot pos in trunk frame
    kp,                     # (3,) shared OR (n_legs, 3) per-leg Cartesian stiffness
    kd,                     # (3,) shared OR (n_legs, 3) per-leg Cartesian damping
    torque_limit,           # (3*n_legs,) per-joint symmetric limit, leg/joint order
    use_op_space_inertia: bool = True,
    ridge: float = 1e-4,
    feedforward: Optional[jp.ndarray] = None,  # (3*n_legs,) optional FF torque
    accel_force: Optional[jp.ndarray] = None,  # (n_legs, 3) virtual-mass task force A·ẍ
) -> jp.ndarray:
    """Cartesian impedance torque for all legs, shape (3*n_legs,).

    Returned torques are in leg/joint order (leg 0's hip,thigh,calf, then leg
    1, ...). The caller remaps to actuator order before writing `ctrl`.
    """
    body_pos = data.xpos[body_id]                 # (3,) trunk origin, world
    R = data.xmat[body_id].reshape(3, 3)          # (3,3) body→world

    M_full = mjx.full_m(model, data)              # (nv, nv)

    n_legs = len(foot_site_ids)
    taus = []
    for i in range(n_legs):
        site = foot_site_ids[i]
        dofs = leg_dof_ids[i]                      # (3,) qvel indices

        # Linear Jacobian of this foot site. mjx.jac returns jacp (nv, 3);
        # jacp[k, c] = d(foot_c)/d(qvel_k), so jacp[dofs].T is (3, 3) with
        # rows = xyz, cols = this leg's 3 DoFs.
        jacp, _ = mjx.jac(model, data, data.site_xpos[site], model.site_bodyid[site])
        J = jacp[dofs].T                           # (3, 3)

        foot_w = data.site_xpos[site]              # (3,)
        qvel_leg = data.qvel[dofs]                 # (3,)
        v_leg_w = J @ qvel_leg                     # (3,) leg-induced foot vel

        desired_w = body_pos + R @ target_foot_body[i]   # body-frame tgt → world
        err_w = desired_w - foot_w                 # (3,) world-frame error

        # Per-leg gains (variable impedance) when kp/kd are (n_legs, 3); a plain
        # (3,) is shared across legs (fixed impedance). ndim is static at trace.
        kp_i = kp[i] if kp.ndim == 2 else kp       # (3,)
        kd_i = kd[i] if kd.ndim == 2 else kd       # (3,)
        wrench = kp_i * err_w - kd_i * v_leg_w     # (3,)

        if use_op_space_inertia:
            M_leg = M_full[jp.ix_(dofs, dofs)]     # (3, 3)
            M_inv = jp.linalg.inv(M_leg)
            # Ridge keeps Λ well-conditioned at stretched/folded leg configs.
            Lambda = jp.linalg.inv(J @ M_inv @ J.T + ridge * jp.eye(3))
            F = Lambda @ wrench                     # operational-space force
        else:
            F = wrench

        # Virtual-mass acceleration-feedback term: add A·ẍ as a task-space force
        # (already in force units), so it is NOT Λ-weighted — the bare F = wrench
        # + A·ẍ law (use_op_space_inertia=False). See the virtual-mass spec.
        if accel_force is not None:
            F = F + accel_force[i]                  # (3,)

        taus.append(J.T @ F)                        # (3,) joint torque

    tau = jp.concatenate(taus)                      # (3*n_legs,)
    if feedforward is not None:
        tau = tau + feedforward
    return jp.clip(tau, -torque_limit, torque_limit)
