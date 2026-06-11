"""Pluggable env components for the Go2 Warp env — the three orthogonal axes of
variation, each a small strategy object the host (`Go2WarpEnv`) delegates to:

  - Actuation : how commanded joint torque becomes applied torque (+ per-joint
                reflected inertia). TorqueOnly | MotorModel.
  - Terrain   : floor shape — MjSpec mutation (structure) + post-compile hfield
                data. Flat | RoughHF.
  - Controller: action -> joint torque each substep, incl. the OSC mechanics
                and controller state. JointPD | OSC | VarImpedance.

Design: ``.superpowers/specs/2026-06-09-go2-env-composition.md``. Logic is
extracted VERBATIM from the former subclasses/flags so there is no behaviour
change — only where the code lives.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from scipy import ndimage

from jax_rl.envs.locomotion import go2_osc
from jax_rl.envs.locomotion.go2_warp_base import torque_speed_clip, physical_armature


# ── Actuation ────────────────────────────────────────────────────────────────
class Actuation:
    """Maps commanded joint torque -> applied torque, and sets joint armature.

    `customize_model` runs once at build (before mjx.put_model); `clip_torque`
    runs every physics substep on the controller's output torque (joint order).
    """

    def customize_model(self, mj_model) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def clip_torque(self, tau, dq, stall, velocity_limit, effort_limit):  # pragma: no cover
        raise NotImplementedError


class TorqueOnly(Actuation):
    """Ideal torque source: commanded torque applied as-is (clamped only by the
    MJCF ctrlrange/forcerange, as before). Uniform MJCF armature untouched."""

    def customize_model(self, mj_model) -> None:
        pass

    def clip_torque(self, tau, dq, stall, velocity_limit, effort_limit):
        return tau


class MotorModel(Actuation):
    """Physical DC-motor model: per-joint reflected armature + a torque-speed
    clip (driving torque -> 0 at the no-load speed). mjlab-matched. See
    `go2_warp_base.physical_armature` / `torque_speed_clip`."""

    def __init__(self, armature: bool = True, torque_speed: bool = True):
        self.armature = armature
        self.torque_speed = torque_speed

    def customize_model(self, mj_model) -> None:
        if self.armature:
            mj_model.dof_armature[:] = physical_armature(mj_model)

    def clip_torque(self, tau, dq, stall, velocity_limit, effort_limit):
        if not self.torque_speed:
            return tau
        return torque_speed_clip(tau, dq, stall, velocity_limit, effort_limit)


def actuation_from_config(config) -> Actuation:
    """Build the Actuation component from the legacy config flags
    (`torque_speed_model`, `physical_armature`). Bridges the old config-driven
    construction to the component model during the staged migration."""
    ts = bool(getattr(config, "torque_speed_model", False))
    arm = bool(getattr(config, "physical_armature", False))
    if not ts and not arm:
        return TorqueOnly()
    return MotorModel(armature=arm, torque_speed=ts)


# ── Terrain ──────────────────────────────────────────────────────────────────
# Heightfield noise generators (moved verbatim from the deleted rough-config
# module's private helpers).
def _norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def _fractal_perlin_noise_2d(nx, ny, rng, octaves=4, persistence=0.5,
                             lacunarity=2.0, scale=14.0):
    """Borrowed from mjlab terrains/heightfield_terrains.py (pure numpy)."""
    def lerp(a, b, x): return a + x * (b - a)
    def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
    def grad(h, x, y):
        h = h % 4
        return np.where(h == 0, x + y, np.where(h == 1, x - y,
                        np.where(h == 2, -x + y, -x - y)))
    def perlin(x, y, p):
        xi = x.astype(int) % 256; yi = y.astype(int) % 256
        xf = x - x.astype(int); yf = y - y.astype(int)
        u = fade(xf); v = fade(yf)
        n00 = grad(p[p[xi] + yi], xf, yf)
        n01 = grad(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = grad(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = grad(p[p[xi + 1] + yi], xf - 1, yf)
        return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
    p = np.arange(256, dtype=int); rng.shuffle(p); p = np.stack([p, p]).flatten()
    noise = np.zeros((nx, ny)); amp = 1.0; freq = scale; tot = 0.0
    xx, yy = np.meshgrid(np.linspace(0, nx, nx, endpoint=False),
                         np.linspace(0, ny, ny, endpoint=False), indexing="ij")
    for _ in range(octaves):
        noise += amp * perlin(xx * freq / nx, yy * freq / ny, p)
        tot += amp; amp *= persistence; freq *= lacunarity
    return noise / tot


def _make_heightfield(profile, nrow, ncol, seed):
    rng = np.random.default_rng(seed)
    if profile == "perlin_hf":
        return _norm01(_fractal_perlin_noise_2d(nrow, ncol, np.random.default_rng(seed)))
    if profile == "uniform":
        return rng.uniform(0.0, 1.0, (nrow, ncol))
    if profile == "smooth_uniform":
        return _norm01(ndimage.gaussian_filter(rng.uniform(0, 1, (nrow, ncol)), 1.5))
    raise ValueError(f"unknown rough_profile {profile!r}")


class Terrain:
    """Shapes the scene floor. `apply(spec)` mutates the MjSpec before compile
    (structural: geom type + hfield asset); `customize_model(mj_model)` runs
    after compile (data: hfield elevation). Either may be a no-op."""

    def apply(self, spec) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def customize_model(self, mj_model) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class Flat(Terrain):
    """Flat plane floor — already declared by the base scene; nothing to do."""

    def apply(self, spec) -> None:
        pass

    def customize_model(self, mj_model) -> None:
        pass


class RoughHF(Terrain):
    """Procedural rough heightfield floor.

    `apply` adds the hfield asset and switches the base scene's ``floor`` geom to
    it IN THE SPEC — so the second scene XML (``go2_warp_scene_rough.xml``) is no
    longer needed. MjSpec requires non-empty ``userdata`` to compile an hfield AND
    renormalizes it to [0,1] (rescaling the amplitude). So `apply` sets the
    elevation as a placeholder (lets compile succeed) and `customize_model`
    re-writes the EXACT elevation into ``hfield_data`` AFTER compile — overwriting
    the renormalized values, giving a surface BIT-IDENTICAL to the former
    ``_RoughHFMixin`` post-compile poke. nrow/ncol/size match the old rough XML.
    """

    NROW = NCOL = 160                # matches go2_warp_scene_rough.xml <hfield>
    SIZE = (5.0, 5.0, 0.10, 0.5)     # x/y half-extent, max height, base thick (m)

    def __init__(self, profile: str = "uniform", amplitude: float = 0.07,
                 seed: int = 0):
        self.profile = profile
        self.amplitude = amplitude
        self.seed = seed

    def _elevation(self, nrow, ncol, zmax) -> np.ndarray:
        noise01 = _make_heightfield(self.profile, nrow, ncol, self.seed)   # [0,1]
        return (noise01 * (self.amplitude / zmax)).astype(np.float32)      # scale to amp

    def apply(self, spec) -> None:
        hf = spec.add_hfield()
        hf.name = "rough"
        hf.nrow, hf.ncol = self.NROW, self.NCOL
        hf.size = list(self.SIZE)
        # Placeholder elevation so compile succeeds; customize_model overwrites it
        # post-compile with the exact (un-renormalized) values.
        data = self._elevation(self.NROW, self.NCOL, self.SIZE[2])
        hf.userdata = data.flatten().tolist()
        floor = next(g for g in spec.worldbody.geoms if g.name == "floor")
        floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
        floor.hfieldname = "rough"

    def customize_model(self, mj_model) -> None:
        m = mj_model
        hid = m.hfield("rough").id
        nrow = int(m.hfield_nrow[hid]); ncol = int(m.hfield_ncol[hid])
        zmax = float(m.hfield_size[hid, 2])
        data = self._elevation(nrow, ncol, zmax)
        adr = int(m.hfield_adr[hid])
        m.hfield_data[adr:adr + nrow * ncol] = data.flatten()


def terrain_from_config(config) -> Terrain:
    """Build the Terrain from the legacy config keys (`rough_profile`,
    `rough_amplitude`, `rough_seed`). No `rough_profile` set → Flat. Bridges the
    old config-driven construction to the component model during migration."""
    profile = getattr(config, "rough_profile", None)
    if profile is None:
        return Flat()
    return RoughHF(
        profile=str(profile),
        amplitude=float(getattr(config, "rough_amplitude", 0.07)),
        seed=int(getattr(config, "rough_seed", 0)),
    )


# ── Controller ───────────────────────────────────────────────────────────────
# Variable-impedance action decoding (moved verbatim from the former
# go2_warp_osc_var_impedance).
_N_STIFFNESS = {"per_foot": 4, "per_axis": 12}


def log_action_scale(a: jax.Array, lo: float, hi: float) -> jax.Array:
    """Map action a∈[-1,1] → [lo,hi] log-spaced (a=0 → geometric mean √(lo·hi))."""
    u = 0.5 * (jp.clip(a, -1.0, 1.0) + 1.0)
    return lo * (hi / lo) ** u


def var_action_size(granularity: str, damping_action: bool) -> int:
    """12 foot targets + stiffness (+ damping when enabled), per granularity."""
    n = _N_STIFFNESS[granularity]
    return 12 + n * (2 if damping_action else 1)


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

    def apply(self, env, data, action):
        motor_targets = env._default_pose + action * env._config.action_scale
        kp = env._kp
        kd = env._kd
        model = env.mjx_model
        a2j = env._act_to_joint

        def substep(data, _):
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

    def _run_osc(self, env, data, deltas, kp, kd):
        """Decimation loop: drive feet to (nominal + deltas) at gains kp/kd.

        kp/kd are (3,) shared across legs (fixed impedance) or (4,3) per-foot
        (variable impedance). The OSC and VarImpedance controllers both call this.

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
            )
            tau_joint = env._apply_torque_speed_limit(tau_joint, data.qvel[6:])
            tau_act = tau_joint[a2j]     # joint order → actuator order
            data = data.replace(ctrl=tau_act)
            return mjx.step(model, data), None

        return jax.lax.scan(substep, data, (), env.n_substeps)[0]

    def apply(self, env, data, action):
        deltas = action.reshape(4, 3) * env._config.action_scale   # (4,3) metres
        return self._run_osc(env, data, deltas, self._kp, self._kd)


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

    def apply(self, env, data, action):
        deltas = action[:12].reshape(4, 3) * env._config.action_scale   # (4,3)
        kp, kd = impedance_gains(
            action, self._kp, self._kd,
            granularity=self._granularity,
            s_min=self._s_min, s_max=self._s_max,
            damping_action=self._damping_action,
            z_min=self._z_min, z_max=self._z_max,
        )
        return self._run_osc(env, data, deltas, kp, kd)


def controller_from_config(config) -> Controller:
    """Pick the controller from the config shape (legacy bridge): no ``osc``
    block → JointPD; ``osc`` without ``stiffness_granularity`` → OSC; with it →
    VarImpedance."""
    osc = getattr(config, "osc", None)
    if osc is None:
        return JointPD()
    if getattr(osc, "stiffness_granularity", None) is None:
        return OSC()
    return VarImpedance()
