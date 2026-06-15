"""Controller component — turns the policy action into joint torques each physics
substep (JointPD | OSC | VarImpedance | VarImpedanceMass), incl. the OSC
mechanics and variable-impedance action decoding. One of the four orthogonal
axes the Go2 Warp env delegates to (see the ``components`` package docstring).
Logic moved VERBATIM from the former ``go2_warp_components`` — no behaviour
change.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from jax_rl.envs.locomotion import go2_osc


# Variable-impedance action decoding (moved verbatim from the former
# go2_warp_osc_var_impedance).
_N_STIFFNESS = {"per_foot": 4, "per_axis": 12}


def log_action_scale(a: jax.Array, lo: float, hi: float) -> jax.Array:
    """Map action a∈[-1,1] → [lo,hi] log-spaced (a=0 → geometric mean √(lo·hi))."""
    u = 0.5 * (jp.clip(a, -1.0, 1.0) + 1.0)
    return lo * (hi / lo) ** u


def lin_action_scale(a: jax.Array, lo: float, hi: float) -> jax.Array:
    """Map action a∈[-1,1] → [lo,hi] LINEARLY (a=0 → midpoint). Used for the
    virtual mass A, whose range may include 0 (log-spacing can't)."""
    u = 0.5 * (jp.clip(a, -1.0, 1.0) + 1.0)
    return lo + (hi - lo) * u


def var_action_size(granularity: str, damping_action: bool,
                    mass_action: bool = False) -> int:
    """12 foot targets + stiffness (+ damping) (+ mass), per granularity.

    Each enabled block adds ``n = _N_STIFFNESS[granularity]`` dims: stiffness
    always, damping when ``damping_action``, virtual mass when ``mass_action``.
    """
    n = _N_STIFFNESS[granularity]
    return 12 + n * (1 + int(damping_action) + int(mass_action))


def impedance_gains(action, kp_base, kd_base, *, granularity, s_min, s_max,
                    damping_action, z_min, z_max):
    """Decode per-leg Cartesian gains (kp, kd), each (4,3), from the action tail.

        kp = s · kp_base ,   kd = ζ · √s · kd_base

    s from action[12:12+n] (log → [s_min,s_max]). ζ from the next n entries when
    ``damping_action`` (log → [z_min,z_max]), else ζ=1 (locked critical). Because
    kd_base = 2√kp_base is already critical, ζ is literally the damping ratio:
    ζ<1 underdamped/springy, ζ>1 overdamped. per_foot → one value per foot
    (broadcast over xyz); per_axis → per foot AND axis. ζ touches kd only — kp is
    independent of the damping action (the decoupling).
    """
    n = _N_STIFFNESS[granularity]
    s = log_action_scale(action[12:12 + n], s_min, s_max)
    if damping_action:
        zeta = log_action_scale(action[12 + n:12 + 2 * n], z_min, z_max)
    else:
        zeta = jp.ones_like(s)
    if granularity == "per_foot":
        s = s[:, None]
        zeta = zeta[:, None]              # (4,1) — one value per foot
    else:
        s = s.reshape(4, 3)
        zeta = zeta.reshape(4, 3)         # (4,3) — per foot AND axis
    kp = s * kp_base
    kd = zeta * jp.sqrt(s) * kd_base
    return kp, kd


class Controller:
    """Low-level control strategy: turns the policy action into joint torques
    each physics substep. ``action_size`` is the policy action dim; ``setup``
    caches controller geometry/gains on the controller itself (run once, after
    the host's task setup); ``apply`` runs the decimation loop.

    Methods take ``(self, env, ...)`` and treat env as READ-ONLY — they may read
    the model handles (``mjx_model``, ``_mj_model``, ``n_substeps``, ``_config``)
    and host-cached geometry/gains (``_torso_body_id``, ``_feet_site_id``,
    ``_act_to_joint``, ``_default_pose``, ``_stall_torque``, ``_kp``/``_kd``)
    and call ``env._apply_torque_speed_limit``; all controller state lives on
    ``self`` (set in ``setup``)."""

    def action_size(self, env) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def setup(self, env) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def apply(self, env, data, action):  # pragma: no cover - interface
        raise NotImplementedError


class JointPD(Controller):
    """Joint-space PD at physics rate (the original WarpJoystick controller).

    action → joint position targets (offset from default pose) → PD torque →
    mjx.step, repeated n_substeps. qpos[7:] is joint order (FL,FR,RL,RR), ctrl is
    actuator order (FR,FL,RR,RL); torques are remapped before writing to ctrl."""

    def action_size(self, env) -> int:
        return env.mjx_model.nu

    def setup(self, env) -> None:
        pass

    def apply(self, env, data, action, info=None):
        motor_targets = env._default_pose + action * env._config.action_scale
        kp = env._kp
        kd = env._kd
        model = env.mjx_model
        a2j = env._act_to_joint

        def substep(data, _):
            data = env._force_field.apply_substep(env, data, info)  # mud &c; NoField = identity
            current_q = data.qpos[7:]   # joint order (FL,FR,RL,RR)
            current_dq = data.qvel[6:]  # joint order
            tau_joint = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
            tau_joint = env._apply_torque_speed_limit(tau_joint, current_dq)
            tau_act = tau_joint[a2j]     # ctrl[a] = tau_joint[act_to_joint[a]]
            data = data.replace(ctrl=tau_act)
            return mjx.step(model, data), None

        return jax.lax.scan(substep, data, (), env.n_substeps)[0]


class OSC(Controller):
    """Per-leg Cartesian impedance / OSC controller (fixed gains).

    action (12,) → four foot position deltas (trunk frame) → impedance torque via
    the ``_run_osc`` decimation loop. ``setup`` caches the OSC gains, leg DoF
    map, foot sites and nominal foot positions on the controller."""

    def action_size(self, env) -> int:
        return env.mjx_model.nu

    def setup(self, env) -> None:
        osc = env._config.osc
        if str(osc.gravity_ff) != "none":
            raise NotImplementedError(
                f"gravity_ff={osc.gravity_ff!r} not implemented in the MVP; "
                "only 'none' (pure impedance) is supported."
            )
        if str(osc.target_mode) not in ("abs_body", "delta_current"):
            raise ValueError(f"unknown target_mode {osc.target_mode!r}")

        # Cartesian impedance gains (3,) — distinct from the host's joint-PD
        # env._kp/env._kd used by JointPD.
        self._kp = jp.array(osc.kp)
        self._kd = jp.array(osc.kd)
        self._use_lambda = bool(osc.use_op_space_inertia)
        self._ridge = float(osc.ridge)
        self._target_mode = str(osc.target_mode)

        # Per-leg qvel DoF indices. qpos[7:]/qvel[6:] are joint order
        # FL,FR,RL,RR (hip,thigh,calf), and FEET_SITES is the same leg order,
        # so foot i is driven by qvel dofs [6+3i, 6+3i+1, 6+3i+2].
        self._leg_dof_ids = np.array(
            [[6 + 3 * i + j for j in range(3)] for i in range(4)]
        )
        self._foot_site_ids = np.asarray(env._feet_site_id)  # (4,) FL,FR,RL,RR
        # Per-joint symmetric torque limit, joint order (base stores stall torque).
        self._torque_limit = env._stall_torque
        # Nominal foot positions in the trunk frame at the home keyframe.
        self._nominal_foot_body = jp.array(self._compute_nominal_foot_body(env))

    # ── OSC mechanics (moved verbatim from the former host-owned section) ───

    def _compute_nominal_foot_body(self, env) -> np.ndarray:
        """Foot positions in the trunk frame at the 'home' keyframe (numpy FK)."""
        m = env._mj_model
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe("home").qpos
        mujoco.mj_forward(m, d)
        body_pos = d.xpos[env._torso_body_id]
        R = d.xmat[env._torso_body_id].reshape(3, 3)
        feet_w = d.site_xpos[self._foot_site_ids]       # (4,3) world
        return (feet_w - body_pos) @ R                  # rows: Rᵀ·(foot−trunk)

    def _feet_in_body(self, env, data: mjx.Data) -> jax.Array:
        """Current foot positions expressed in the trunk frame, (4,3)."""
        body_pos = data.xpos[env._torso_body_id]
        R = data.xmat[env._torso_body_id].reshape(3, 3)
        feet_w = data.site_xpos[self._foot_site_ids]
        return (feet_w - body_pos) @ R

    def _run_osc(self, env, data, deltas, kp, kd, info=None, accel_force=None):
        """Decimation loop: drive feet to (nominal + deltas) at gains kp/kd.

        kp/kd are (3,) shared across legs (fixed impedance) or (4,3) per-foot
        (variable impedance). The OSC and VarImpedance controllers both call this.
        ``accel_force`` (4,3) = the virtual-mass A·ẍ task force, constant across
        the substep loop (computed once per control step); None for plain OSC.

        STABILITY-CRITICAL: the impedance torque is recomputed every physics
        substep (250 Hz), NOT once per 50 Hz control step. The unit-mass loop
        q̈ = kp·err − kd·ẋ with kp up to 4000 has ω_n ≈ 63 rad/s; held over a
        50 Hz step (h=0.02) the discrete loop diverges (ρ≈2.8), but at the
        250 Hz substep (h=0.004) ρ≈0.8 (stable, audited). Do NOT hoist the
        torque computation out of this scan.
        """
        dynamic = self._target_mode == "delta_current"
        base_targets = self._nominal_foot_body + deltas             # used if static
        model = env.mjx_model
        a2j = env._act_to_joint

        def substep(data, _):
            data = env._force_field.apply_substep(env, data, info)  # mud &c; NoField = identity
            if dynamic:
                # Chase a moving anchor: target = current foot + delta, in
                # trunk frame, recomputed each substep.
                targets = self._feet_in_body(env, data) + deltas
            else:
                targets = base_targets
            tau_joint = go2_osc.compute_leg_impedance_torque(
                model, data,
                self._foot_site_ids, self._leg_dof_ids, env._torso_body_id,
                targets, kp, kd, self._torque_limit,
                use_op_space_inertia=self._use_lambda, ridge=self._ridge,
                accel_force=accel_force,
            )
            tau_joint = env._apply_torque_speed_limit(tau_joint, data.qvel[6:])
            tau_act = tau_joint[a2j]     # joint order → actuator order
            data = data.replace(ctrl=tau_act)
            return mjx.step(model, data), None

        return jax.lax.scan(substep, data, (), env.n_substeps)[0]

    def apply(self, env, data, action, info=None):
        deltas = action.reshape(4, 3) * env._config.action_scale   # (4,3) metres
        return self._run_osc(env, data, deltas, self._kp, self._kd, info)


class VarImpedance(OSC):
    """OSC where the policy also commands stiffness (per-foot or per-axis) and,
    optionally, damping ratio. Action tail decoded by :func:`impedance_gains`."""

    def action_size(self, env) -> int:
        osc = env._config.osc
        return var_action_size(
            str(getattr(osc, "stiffness_granularity", "per_foot")),
            bool(getattr(osc, "damping_action", False)),
        )

    def setup(self, env) -> None:
        super().setup(env)
        osc = env._config.osc
        self._s_min = float(osc.var_s_min)
        self._s_max = float(osc.var_s_max)
        gran = str(getattr(osc, "stiffness_granularity", "per_foot"))
        if gran not in _N_STIFFNESS:
            raise ValueError(
                f"stiffness_granularity={gran!r} not in {list(_N_STIFFNESS)}"
            )
        self._granularity = gran
        self._n_stiffness = _N_STIFFNESS[gran]
        self._damping_action = bool(getattr(osc, "damping_action", False))
        self._z_min = float(getattr(osc, "var_zeta_min", 0.5))
        self._z_max = float(getattr(osc, "var_zeta_max", 2.0))

    def apply(self, env, data, action, info=None):
        deltas = action[:12].reshape(4, 3) * env._config.action_scale   # (4,3)
        kp, kd = impedance_gains(
            action, self._kp, self._kd,
            granularity=self._granularity,
            s_min=self._s_min, s_max=self._s_max,
            damping_action=self._damping_action,
            z_min=self._z_min, z_max=self._z_max,
        )
        return self._run_osc(env, data, deltas, kp, kd, info)


class VarImpedanceMass(VarImpedance):
    """VarImpedance + a policy-commanded virtual MASS A (per foot/axis).

    Adds the acceleration-feedback term to the bare-impedance force:

        F = A·ẍ + K·err + D·ẋ          (use_op_space_inertia=False)

    ẍ is the foot acceleration, sensed as a CONTROL-STEP finite difference of the
    foot velocity (info["last_foot_vel"] vs the current foot-linvel sensor), held
    constant across the substep decimation loop. A is decoded LINEARLY from the
    action tail to [var_a_min, var_a_max]. See the virtual-mass spec.
    """

    def action_size(self, env) -> int:
        osc = env._config.osc
        return var_action_size(
            str(getattr(osc, "stiffness_granularity", "per_foot")),
            bool(getattr(osc, "damping_action", False)),
            mass_action=True,
        )

    def setup(self, env) -> None:
        super().setup(env)
        osc = env._config.osc
        self._a_min = float(getattr(osc, "var_a_min", 0.0))
        self._a_max = float(getattr(osc, "var_a_max", 2.0))
        # ẍ EMA low-pass coeff α (0<α≤1; 1 = no smoothing). Tames the contact-
        # impact ẍ spikes that otherwise drive A·ẍ chatter / catapult.
        self._xdd_ema = float(getattr(osc, "var_xdd_ema", 1.0))
        # Mass block starts after deltas(12) + stiffness(n) + damping(n if on).
        self._mass_start = 12 + self._n_stiffness * (1 + int(self._damping_action))

    def apply(self, env, data, action, info=None):
        deltas = action[:12].reshape(4, 3) * env._config.action_scale   # (4,3)
        kp, kd = impedance_gains(
            action, self._kp, self._kd,
            granularity=self._granularity,
            s_min=self._s_min, s_max=self._s_max,
            damping_action=self._damping_action,
            z_min=self._z_min, z_max=self._z_max,
        )
        n = self._n_stiffness
        A = lin_action_scale(
            action[self._mass_start:self._mass_start + n], self._a_min, self._a_max
        ).reshape(4, 3)
        # ẍ = control-step finite diff of foot WORLD velocity. CRITICAL: update
        # last_foot_vel HERE (start of step) so consecutive reads are one control
        # step apart — storing it post-control (in env.step) made v_now==last and
        # ẍ≡0 (dead mass term). EMA-smooth to tame contact-impact spikes. info is
        # None only outside the env loop (unit probes) → no accel feedback.
        if info is not None:
            v_now = data.sensordata[env._foot_linvel_sensor_adr].reshape(4, 3)
            acc_raw = (v_now - info["last_foot_vel"]) / env.dt
            a = self._xdd_ema
            acc = (1.0 - a) * info["xdd_ema"] + a * acc_raw
            info["last_foot_vel"] = v_now      # start-of-step vel → next step's prev
            info["xdd_ema"] = acc              # carry smoothed ẍ
        else:
            acc = jp.zeros((4, 3))
        accel_force = A * acc                                            # (4,3)
        return self._run_osc(env, data, deltas, kp, kd, info, accel_force=accel_force)


def controller_from_config(config) -> Controller:
    """Pick the controller from the config shape (legacy bridge): no ``osc``
    block → JointPD; ``osc`` without ``stiffness_granularity`` → OSC; with it →
    VarImpedance."""
    osc = getattr(config, "osc", None)
    if osc is None:
        return JointPD()
    if getattr(osc, "stiffness_granularity", None) is None:
        return OSC()
    if getattr(osc, "mass_action", False):
        return VarImpedanceMass()
    return VarImpedance()
