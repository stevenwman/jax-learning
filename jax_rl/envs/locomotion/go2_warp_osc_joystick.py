"""Cartesian impedance / OSC joystick env for Go2 (Warp backend).

Same task, observations, rewards, command sampling, termination and domain
randomization as :class:`WarpJoystick` — the ONLY change is the low-level
controller. Instead of joint-space PD, the policy's 12-d action is interpreted
as four foot position targets in the trunk frame, and a per-leg Cartesian
impedance / OSC controller (:mod:`go2_osc`) drives the feet there.

This is the MVP for the OSC-impedance research line: fixed impedance, foot
position targets, velocity-tracking task. Downstream work adds the per-foot
stiffness to the action space (variable impedance) + a curriculum.

Design notes live in
``.superpowers/specs/2026-06-08-go2-osc-impedance-design.md``.
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.locomotion import go2_osc
from jax_rl.envs.locomotion.go2_warp_joystick import (
    WarpJoystick,
    default_config as _joystick_default_config,
)


def default_config() -> config_dict.ConfigDict:
    cfg = _joystick_default_config()
    # Foot-target action scale: action ∈ ~[-1,1] maps to a ±action_scale metre
    # box around the nominal foot (trunk frame). 0.5 rad suited joint targets;
    # for Cartesian foot targets that would be absurd — feet only reach
    # ~0.10–0.15 m from nominal, so 0.12 m gives a generous-but-sane box.
    cfg.action_scale = 0.12
    cfg.osc = config_dict.create(
        # target_mode:
        #   abs_body      — target = nominal_foot + action·scale (fixed trunk
        #                   anchor). Default. Holds a stand under zero action.
        #   delta_current — target = current_foot + action·scale (integrative,
        #                   moving anchor). WARNING: has NO restoring force vs
        #                   gravity under zero action (target tracks the sagging
        #                   foot), so a passive/zero-action policy COLLAPSES and
        #                   terminates in ~0.25 s (audited). A policy must
        #                   actively command an upward bias to stand. Provided
        #                   as an exploration-dynamics ablation, not a drop-in.
        target_mode="abs_body",        # {abs_body, delta_current}
        use_op_space_inertia=True,     # True = Khatib OSC (Λ); False = Jᵀ impedance
        gravity_ff="none",             # MVP: pure impedance, spring bears load
        ridge=1e-4,                    # Λ inversion ridge
        # Cartesian gains. Tuned via a zero-action hold probe (no gravity FF, so
        # the spring alone bears body weight): kp=[800,800,1000] sagged the base
        # to the 0.18 m termination floor; [3000,3000,4000] holds ~0.27 m with
        # mild compliance. kd ≈ 2·sqrt(kp) ≈ critical damping of the unit-mass
        # OSC closed loop (q̈_foot = kp·err − kd·ẋ). Re-tune if walking is harsh.
        kp=[3000.0, 3000.0, 4000.0],
        kd=[110.0, 110.0, 130.0],
    )
    return cfg


class WarpOscJoystick(WarpJoystick):
    """Joystick velocity tracking with a per-leg Cartesian impedance controller."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            task=task, config=config, config_overrides=config_overrides
        )

    def _post_init(self) -> None:
        super()._post_init()
        osc = self._config.osc
        if str(osc.gravity_ff) != "none":
            raise NotImplementedError(
                f"gravity_ff={osc.gravity_ff!r} not implemented in the MVP; "
                "only 'none' (pure impedance) is supported."
            )
        if str(osc.target_mode) not in ("abs_body", "delta_current"):
            raise ValueError(f"unknown target_mode {osc.target_mode!r}")

        self._osc_kp = jp.array(osc.kp)
        self._osc_kd = jp.array(osc.kd)
        self._osc_use_lambda = bool(osc.use_op_space_inertia)
        self._osc_ridge = float(osc.ridge)
        self._osc_target_mode = str(osc.target_mode)

        # Per-leg qvel DoF indices. qpos[7:]/qvel[6:] are joint order
        # FL,FR,RL,RR (hip,thigh,calf), and FEET_SITES is the same leg order,
        # so foot i is driven by qvel dofs [6+3i, 6+3i+1, 6+3i+2]. Verified by
        # test_osc_jacobian_block_isolation (each foot moves only with its own
        # 3 dofs).
        self._leg_dof_ids = np.array(
            [[6 + 3 * i + j for j in range(3)] for i in range(4)]
        )
        self._osc_foot_site_ids = np.asarray(self._feet_site_id)  # (4,) FL,FR,RL,RR

        # Per-joint symmetric torque limit, joint order — base class already
        # stores stall torque (= MJCF ctrlrange max) in joint order.
        self._osc_torque_limit = self._stall_torque

        # Nominal foot positions in the trunk frame at the home keyframe.
        self._nominal_foot_body = jp.array(self._compute_nominal_foot_body())

    def _compute_nominal_foot_body(self) -> np.ndarray:
        """Foot positions in the trunk frame at the 'home' keyframe (numpy FK)."""
        m = self._mj_model
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe("home").qpos
        mujoco.mj_forward(m, d)
        body_pos = d.xpos[self._torso_body_id]
        R = d.xmat[self._torso_body_id].reshape(3, 3)
        feet_w = d.site_xpos[self._osc_foot_site_ids]   # (4,3) world
        return (feet_w - body_pos) @ R                  # rows: Rᵀ·(foot−trunk)

    def _feet_in_body(self, data: mjx.Data) -> jax.Array:
        """Current foot positions expressed in the trunk frame, (4,3)."""
        body_pos = data.xpos[self._torso_body_id]
        R = data.xmat[self._torso_body_id].reshape(3, 3)
        feet_w = data.site_xpos[self._osc_foot_site_ids]
        return (feet_w - body_pos) @ R

    def _apply_control(self, data: mjx.Data, action: jax.Array) -> mjx.Data:
        """Per-leg Cartesian impedance / OSC controller, run at physics rate.

        action (12,) → four foot position deltas (trunk frame) → impedance
        torque → mjx.step, repeated n_substeps times.
        """
        deltas = action.reshape(4, 3) * self._config.action_scale   # (4,3) metres
        dynamic = self._osc_target_mode == "delta_current"
        base_targets = self._nominal_foot_body + deltas             # used if static

        model = self.mjx_model
        a2j = self._act_to_joint

        # STABILITY-CRITICAL: the impedance torque is recomputed every physics
        # substep (250 Hz), NOT once per 50 Hz control step. The unit-mass loop
        # q̈ = kp·err − kd·ẋ with kp up to 4000 has ω_n ≈ 63 rad/s; held over a
        # 50 Hz step (h=0.02) the discrete loop diverges (ρ≈2.8), but at the
        # 250 Hz substep (h=0.004) ρ≈0.8 (stable, audited). Do NOT hoist the
        # torque computation out of this scan.
        def substep(data, _):
            if dynamic:
                # Chase a moving anchor: target = current foot + delta, in
                # trunk frame, recomputed each substep.
                targets = self._feet_in_body(data) + deltas
            else:
                targets = base_targets
            tau_joint = go2_osc.compute_leg_impedance_torque(
                model, data,
                self._osc_foot_site_ids, self._leg_dof_ids, self._torso_body_id,
                targets, self._osc_kp, self._osc_kd, self._osc_torque_limit,
                use_op_space_inertia=self._osc_use_lambda, ridge=self._osc_ridge,
            )
            tau_joint = self._apply_torque_speed_limit(tau_joint, data.qvel[6:])
            tau_act = tau_joint[a2j]     # joint order → actuator order
            data = data.replace(ctrl=tau_act)
            return mjx.step(model, data), None

        return jax.lax.scan(substep, data, (), self.n_substeps)[0]
