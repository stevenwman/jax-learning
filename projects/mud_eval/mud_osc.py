"""Operational-space (Cartesian impedance) control for the Newton mud eval —
a numpy port of jax_rl/envs/locomotion/go2_osc.py::compute_leg_impedance_torque,
driven by Newton's OWN cpu mujoco (solver.mj_model / solver.mj_data) via
mj_jacSite (foot Jacobian) + mj_fullM (mass matrix). Same math as training so the
eval matches; the result is injected as control.joint_f each substep.

Site names don't survive the Newton->mjModel conversion, so foot sites are mapped
by BODY (the 4 non-trunk sites, ordered by body index = leg order FL,FR,RL,RR).
"""
from __future__ import annotations

import numpy as np
import mujoco


def find_legs(mj_model):
    """Return (foot_site_ids[4], trunk_body_id, leg_dof_ids[4,3]) for the go2
    mjModel built by Newton. Sites are unnamed post-conversion → the trunk is the
    floating base (the body the free joint drives), foot sites are the other 4
    sites, ordered by their body index (= leg order FL,FR,RL,RR)."""
    # trunk = the body with the free joint (jnt_type==mjJNT_FREE)
    free = np.where(mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    trunk_body_id = int(mj_model.jnt_bodyid[free[0]]) if len(free) else 1

    site_body = mj_model.site_bodyid
    foot_sites = [s for s in range(mj_model.nsite) if int(site_body[s]) != trunk_body_id]
    foot_sites.sort(key=lambda s: int(site_body[s]))   # body index ↑ = FL,FR,RL,RR
    foot_site_ids = np.asarray(foot_sites[:4], dtype=int)

    # qvel dofs per leg: free joint = 6, then joints in order. Each foot's leg is
    # the 3 dofs of the chain hip,thigh,calf — read from each foot body's ancestor
    # joints. Simpler + matches training: dofs [6+3i, +1, +2] for leg i.
    leg_dof_ids = np.array([[6 + 3 * i + j for j in range(3)] for i in range(4)], dtype=int)
    return foot_site_ids, trunk_body_id, leg_dof_ids


def nominal_foot_body(mj_model, foot_site_ids, trunk_body_id, home_qpos):
    """Foot positions in the trunk frame at the home pose (FK), (4,3). Mirrors
    go2_warp_joystick._compute_nominal_foot_body."""
    d = mujoco.MjData(mj_model)
    d.qpos[:] = home_qpos
    mujoco.mj_forward(mj_model, d)
    body_pos = d.xpos[trunk_body_id]
    R = d.xmat[trunk_body_id].reshape(3, 3)
    feet_w = d.site_xpos[foot_site_ids]            # (4,3) world
    return (feet_w - body_pos) @ R                 # rows: Rᵀ·(foot−trunk)


def sync_mjdata(mj_data, joint_q, joint_qd):
    """Newton state -> mujoco qpos/qvel. Newton free quat is (x,y,z,w); mujoco
    qpos quat is (w,x,y,z)."""
    jq = np.asarray(joint_q); jqd = np.asarray(joint_qd)
    mj_data.qpos[0:3] = jq[0:3]
    mj_data.qpos[3:7] = [jq[6], jq[3], jq[4], jq[5]]   # xyzw -> wxyz
    mj_data.qpos[7:7 + 12] = jq[7:19]
    mj_data.qvel[0:6] = jqd[0:6]
    mj_data.qvel[6:6 + 12] = jqd[6:18]


class MudOscController:
    """Per-substep operational-space controller for the Newton co-step loop. Holds
    the OSC setup (foot sites, leg dofs, nominal foot, gains) and, given the live
    Newton state + the policy's held foot deltas, computes the generalized joint
    force (control.joint_f) by reading J/M from the solver's OWN cpu mujoco.
    Requires use_mujoco_cpu (solver.mj_data is the actively-stepped data)."""

    def __init__(self, solver, kp, kd, torque_limit, use_op_space_inertia=True,
                 ridge=1e-4, home_joints=None):
        self.m = solver.mj_model
        self.d = solver.mj_data
        self.foot_sites, self.trunk, self.leg_dofs = find_legs(self.m)
        self.nv = int(self.m.nv)
        hj = list(home_joints) if home_joints is not None else [0.0, 0.9, -1.8] * 4
        home_qpos = np.zeros(self.m.nq)
        home_qpos[2] = 0.3; home_qpos[3] = 1.0          # base z + quat w (trunk-frame, pose-invariant)
        home_qpos[7:7 + 12] = hj
        self.nominal = nominal_foot_body(self.m, self.foot_sites, self.trunk, home_qpos)
        self.kp = np.asarray(kp, float); self.kd = np.asarray(kd, float)
        self.torque_limit = np.asarray(torque_limit, float)
        self.use_lambda = bool(use_op_space_inertia); self.ridge = float(ridge)

    def compute_joint_f(self, state, deltas) -> np.ndarray:
        """deltas: (4,3) foot-position deltas in the trunk frame (metres). Returns
        the (nv,) generalized joint force, OSC torque in the 12 actuated dofs."""
        jq = np.asarray(state.joint_q.numpy()); jqd = np.asarray(state.joint_qd.numpy())
        sync_mjdata(self.d, jq, jqd)
        mujoco.mj_forward(self.m, self.d)
        targets = self.nominal + np.asarray(deltas).reshape(4, 3)
        tau = osc_torque(self.m, self.d, self.foot_sites, self.leg_dofs, self.trunk,
                         targets, self.kp, self.kd, self.torque_limit,
                         use_op_space_inertia=self.use_lambda, ridge=self.ridge)
        jf = np.zeros(self.nv, np.float32)
        jf[6:6 + 12] = tau                              # 12 actuated dofs (FL,FR,RL,RR)
        return jf


def osc_torque(mj_model, mj_data, foot_site_ids, leg_dof_ids, trunk_body_id,
               target_foot_body, kp, kd, torque_limit,
               use_op_space_inertia=True, ridge=1e-4):
    """Cartesian-impedance joint torque, (12,) leg/joint order. Numpy port of
    compute_leg_impedance_torque. Requires mj_data already forwarded (mj_forward)
    so xpos/xmat/site_xpos/qvel/qM are current."""
    body_pos = mj_data.xpos[trunk_body_id]
    R = mj_data.xmat[trunk_body_id].reshape(3, 3)

    M_full = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M_full, mj_data.qM)

    kp = np.asarray(kp, float); kd = np.asarray(kd, float)
    jacp = np.zeros((3, mj_model.nv))
    taus = []
    for i in range(len(foot_site_ids)):
        site = int(foot_site_ids[i])
        dofs = leg_dof_ids[i]
        mujoco.mj_jacSite(mj_model, mj_data, jacp, None, site)
        J = jacp[:, dofs]                              # (3 xyz, 3 dof)

        foot_w = mj_data.site_xpos[site]
        v_leg_w = J @ mj_data.qvel[dofs]               # leg-induced foot vel
        desired_w = body_pos + R @ target_foot_body[i]
        err_w = desired_w - foot_w

        kp_i = kp[i] if kp.ndim == 2 else kp
        kd_i = kd[i] if kd.ndim == 2 else kd
        wrench = kp_i * err_w - kd_i * v_leg_w

        if use_op_space_inertia:
            M_leg = M_full[np.ix_(dofs, dofs)]
            Lam = np.linalg.inv(J @ np.linalg.inv(M_leg) @ J.T + ridge * np.eye(3))
            F = Lam @ wrench
        else:
            F = wrench
        taus.append(J.T @ F)                           # (3,) joint torque

    tau = np.concatenate(taus)
    return np.clip(tau, -torque_limit, torque_limit)
