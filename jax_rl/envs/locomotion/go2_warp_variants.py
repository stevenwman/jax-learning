"""Go2 Warp env variants as DATA: one declaration per named env.

Each named Go2 Warp env is an :class:`EnvVariant` row in :data:`GO2_WARP_VARIANTS`
— a config callable, the host class name, and (later tasks) train/algo preset
overrides. `mjx_backend` registers Go2 envs by looping this table;
`env_presets` resolves Go2 presets from it.

IMPORT-LIGHT BY CONTRACT: module-level imports are stdlib +
``ml_collections.config_dict`` ONLY (enforced by
``tests/test_go2_warp_variants.py::test_variants_file_is_import_light``).
Variants whose config lives in an env module (curriculum, PosTrack) import it
function-locally inside their config callable.
"""

import dataclasses
import functools
from typing import Callable, Union

from ml_collections import config_dict

# Λ-OSC base gains, shared by the builder defaults and the Kp sweep
# (mirrors the legacy factories' _OSC_BASE_KP/KD sharing).
_OSC_BASE_KP = [3000.0, 3000.0, 4000.0]
_OSC_BASE_KD = [110.0, 110.0, 130.0]


def go2_config(
    *,
    controller: str = "joint_pd",          # "joint_pd" | "osc" | "var_impedance"
    osc_kp=None, osc_kd=None,              # (3,) lists; required for cartesian controllers
    use_op_space_inertia: bool = True,     # False = Jt-impedance ablation
    target_mode: str = "abs_body",
    stiffness_granularity: str = "per_foot",   # var_impedance only
    damping_action: bool = False,              # var_impedance only
    var_s=(0.25, 2.0), var_zeta=(0.5, 2.0),
    motor: str = "ideal",                  # "ideal" | "torque_speed" | "physical"
    terrain: Union[str, tuple] = "flat",   # "flat" | (profile, amplitude)
    push=(0.75, 0.75),
    action_scale: Union[float, None] = None,  # default 0.5 joint_pd / 0.12 cartesian
    mud=None,                              # None | dict of mud DR ranges (analytic foot force field)
) -> config_dict.ConfigDict:
    """Build a complete Go2 Warp joystick config from knobs.

    Reproduces the legacy per-env config factories exactly (pinned by the
    config-equality / snapshot test). The controller is selected FROM the
    config by ``controller_from_config`` — "osc"/"var_impedance" here just
    means "emit the osc block (+ var keys)".
    """
    if controller not in ("joint_pd", "osc", "var_impedance"):
        raise ValueError(f"unknown controller: {controller!r}")
    if motor not in ("ideal", "torque_speed", "physical"):
        raise ValueError(f"unknown motor: {motor!r}")
    if controller == "joint_pd" and (
        osc_kp is not None 
        or osc_kd is not None
        or use_op_space_inertia is not True 
        or target_mode != "abs_body"
    ):
        raise ValueError(
            "osc_kp/osc_kd/use_op_space_inertia/target_mode require a "
            f"cartesian controller, got controller={controller!r}")
    if controller != "var_impedance" and (
        damping_action or stiffness_granularity != "per_foot"
        or tuple(var_s) != (0.25, 2.0) or tuple(var_zeta) != (0.5, 2.0)
    ):
        raise ValueError(
            "damping_action/stiffness_granularity/var_s/var_zeta require "
            f"controller='var_impedance', got controller={controller!r}")

    if action_scale is None:
        # Joint PD: action is a joint-target delta in rad. Cartesian: action
        # maps to a ±action_scale metre box around the nominal foot.
        action_scale = 0.5 if controller == "joint_pd" else 0.12

    cfg = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        Kp=20.0,
        Kd=0.5,
        torque_speed_model=False,
        physical_armature=False,
        action_repeat=1,
        action_scale=action_scale,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
                accelerometer=0.1,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=10.0,
                tracking_ang_vel=5.0,
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
                dof_pos_limits=-1.0,
                feet_air_time=0.1,
                feet_slip=-0.1,
                feet_clearance=-2.0,
                feet_height=-0.2,
                termination=-1.0,
                stand_still=-1.0,
                pose=0.5,
                base_height=-5.0,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        command_config=config_dict.create(
            a=[1.5, 0.8, 1.2],
            b=[0.9, 0.25, 0.5],
        ),
        push_config=config_dict.create(
            interval=350,        # steps between base-velocity kicks (~7s @ 50Hz)
            vel_min=push[0],     # per-episode kick bound ~ U[vel_min, vel_max];
            vel_max=push[1],     # vel_min==vel_max reproduces the old fixed bound
        ),
        impl="warp",
        contact_mode="training",
        naconmax=4 * 8192,
        naccdmax=4000,
        njmax=100,
    )

    if controller in ("osc", "var_impedance"):
        kp = list(osc_kp) if osc_kp is not None else list(_OSC_BASE_KP)
        kd = list(osc_kd) if osc_kd is not None else list(_OSC_BASE_KD)
        cfg.osc = config_dict.create(
            target_mode=target_mode,       # {abs_body, delta_current}
            use_op_space_inertia=use_op_space_inertia,  # True=Khatib OSC (Λ); False=Jᵀ
            gravity_ff="none",             # pure impedance, spring bears load
            ridge=1e-4,                    # Λ inversion ridge
            kp=kp,
            kd=kd,
        )
        if controller == "var_impedance":
            cfg.osc.var_s_min = var_s[0]
            cfg.osc.var_s_max = var_s[1]
            cfg.osc.stiffness_granularity = stiffness_granularity  # {per_foot, per_axis}
            cfg.osc.damping_action = damping_action
            cfg.osc.var_zeta_min = var_zeta[0]
            cfg.osc.var_zeta_max = var_zeta[1]

    if motor == "torque_speed":
        cfg.torque_speed_model = True
    elif motor == "physical":
        cfg.torque_speed_model = True
        cfg.physical_armature = True

    if terrain != "flat":
        profile, amplitude = terrain
        cfg.rough_profile = profile
        cfg.rough_amplitude = amplitude
        cfg.rough_seed = 0

    if mud is not None:
        # Analytic mud foot-force field (MudField). Per-episode DR ranges; a
        # fixed depth is depth_range=(d, d). Coeff ranges default to the Isaac
        # `mud` source values; c1/c2 now ACTUALLY randomize across the range
        # (the source quirk read only the first tuple element — fixed here).
        cfg.mud = config_dict.create(
            depth_range=tuple(mud.get("depth_range", (0.10, 0.10))),
            f_range=tuple(mud.get("f_range", (14.0, 15.0))),       # suction/resist coeff
            c1_range=tuple(mud.get("c1_range", (9.0, 10.0))),      # shear viscous
            c2_range=tuple(mud.get("c2_range", (6.0, 7.0))),       # shear yield offset
            area_range=tuple(mud.get("area_range", (0.1, 0.14))),  # leg circumference
        )

    return cfg


@dataclasses.dataclass(frozen=True)
class EnvVariant:
    config: Callable[[], config_dict.ConfigDict]
    cls: str = "WarpJoystick"
    train: dict[str, object] = dataclasses.field(default_factory=dict)
    algo: dict[str, dict] = dataclasses.field(default_factory=dict)   # {algo_name: {field: val}}
    notes: str = ""


# ── RMA-style reward profile (minimal "natural constraints", arXiv 2107.04034) ──
def _rma_feet(cfg):
    """Strip the gait-shaping feet terms RMA does NOT use (the tripod suspects):
    feet_air_time REWARDS lifting a foot, which + heavy contact-gated feet_slip
    both pay to keep a foot up → tripod. RMA (A1) uses ONLY a slip penalty (0.8)
    on the feet, no air-time/clearance/height/pose/stand-still. Zero those; raise
    slip to RMA's 0.8. Keep base_height + orientation as posture anchors (RMA
    leans on its own term set; dropping ours risks a crouch). Mutates + returns cfg."""
    s = cfg.reward_config.scales
    s.feet_air_time = 0.0      # RMA: none (this REWARDED lift → tripod)
    s.feet_clearance = 0.0     # RMA: none
    s.feet_height = 0.0        # RMA: none
    s.pose = 0.0               # RMA: none
    s.stand_still = 0.0        # RMA: none
    s.feet_slip = -0.8         # RMA's slip coefficient
    return cfg


def _var_muddr4x_slowfirm_rma_config():
    """slow+firm recipe + RMA-minimal feet profile — test if dropping the
    gait-shaping feet terms cleans the tripod gait while keeping mud traversal."""
    return _rma_feet(_var_muddr4x_slowfirm_config())


# ── Reward-shaped mud config (R2: firm-planting hypothesis test) ────────────
def _var_muddr4x_firmplant_config():
    """var-impedance + 1→4× mud DR + reward shaping toward FIRM FOOT PLANTING:
    heavier feet_slip penalty (penalize contact-phase foot sliding) + heavier
    orientation penalty (discourage the forward lean the lean-and-drive policy
    uses). Tests whether 'slow + firm' beats 'lean + drive' on Newton mud.
    Velocity tracking left at default so the policy still chooses to advance."""
    cfg = go2_config(controller="var_impedance", stiffness_granularity="per_axis",
                     damping_action=True, motor="physical",
                     mud=dict(depth_range=(0.03, 0.22), f_range=(14.0, 60.0),
                              c1_range=(9.0, 40.0), c2_range=(6.0, 28.0)))
    cfg.reward_config.scales.feet_slip = -0.6      # 6× default (-0.1): firm planting
    cfg.reward_config.scales.orientation = -8.0    # 1.6× default (-5.0): less lean
    return cfg


def _var_muddr4x_slowfirm_config():
    """R4: 'slow + firm' — firm-planting shaping PLUS reduced velocity-tracking
    pressure (tracking_lin_vel 10→4) so the policy isn't punished for slowing in
    mud. Addresses the firm-plant failure mode (over-lunging in deep thick mud →
    fall). Completes the user's 'slower AND firmly plant' intuition. ROBUST: both
    seeds clear the full Newton mud gradient upright."""
    cfg = _var_muddr4x_firmplant_config()
    cfg.reward_config.scales.tracking_lin_vel = 4.0   # 0.4× default (10.0): less lunge
    return cfg


def _var_muddr4x_slowslip_config():
    """R6 ablation: slow + feet_slip ONLY (no orientation change) — isolates
    whether the firm-planting (feet_slip) term is the load-bearing reward lever
    vs the orientation penalty. var + 4× DR + feet_slip -0.6 + tracking 4."""
    cfg = go2_config(controller="var_impedance", stiffness_granularity="per_axis",
                     damping_action=True, motor="physical",
                     mud=dict(depth_range=(0.03, 0.22), f_range=(14.0, 60.0),
                              c1_range=(9.0, 40.0), c2_range=(6.0, 28.0)))
    cfg.reward_config.scales.feet_slip = -0.6        # firm planting (only reward change)
    cfg.reward_config.scales.tracking_lin_vel = 4.0  # slow
    return cfg


def _var_muddr1x_slowfirm_config():
    """#5 ablation: slow+firm reward but only 1× Isaac mud DR (f 14-15, c1 9-10,
    c2 6-7) — does the strong 4× DR still matter once you have firm-planting +
    reduced velocity? If this also clears Newton, 4× DR is dispensable (cheaper)."""
    cfg = go2_config(controller="var_impedance", stiffness_granularity="per_axis",
                     damping_action=True, motor="physical",
                     mud=dict(depth_range=(0.03, 0.22)))   # default coeffs = 1× Isaac
    cfg.reward_config.scales.feet_slip = -0.6
    cfg.reward_config.scales.orientation = -8.0
    cfg.reward_config.scales.tracking_lin_vel = 4.0
    return cfg


def _jointpd_muddr4x_slowfirm_config():
    """R5: the slow+firm recipe on JOINT-PD control — tests whether variable
    impedance is ESSENTIAL or whether the DR+reward recipe rescues fixed-gain
    control too (joint-PD 4× alone REGRESSED to y2.78)."""
    cfg = go2_config(motor="physical",
                     mud=dict(depth_range=(0.03, 0.22), f_range=(14.0, 60.0),
                              c1_range=(9.0, 40.0), c2_range=(6.0, 28.0)))
    cfg.reward_config.scales.feet_slip = -0.6
    cfg.reward_config.scales.orientation = -8.0
    cfg.reward_config.scales.tracking_lin_vel = 4.0
    return cfg


# ── Lazy config callables (env-module configs; import inside the call) ──────
def _curriculum_config():
    from jax_rl.envs.locomotion.go2_warp_curriculum import default_config
    return default_config()


def _curriculum_torque_speed_config():
    from jax_rl.envs.locomotion.go2_warp_curriculum import default_config
    cfg = default_config()
    cfg.torque_speed_model = True
    return cfg


def _flat_postrack_config():
    from jax_rl.envs.locomotion.go2_warp_flat_postrack import default_config
    return default_config()


def _kp_sweep_gains(scale):
    """Λ-OSC stiffness-sweep gains: kp×s, kd×√s (ζ stays ~critical).

    Scaling kd as √s keeps the damping ratio ~critical (kd ≈ 2·√kp) — the
    sweep varies the natural frequency / stiffness while holding ζ≈1. Tests
    how the trained-policy gait (bounce, tracking, effort) varies with
    stiffness, and whether spring stiffness drives the pogo (stiffer = more
    stored spring energy). The open-loop hold-probe showed a static
    weight-bearing floor near s≈0.5 (no gravity FF); a trained policy may
    stand below it via active stance.

    MUST stay computed expressions (not rounded literals) to match the legacy
    factory bit-for-bit — e.g. Kp05 kd ≈ [77.78, 77.78, 91.92], which is NOT
    the SoftPhysical literals [78, 78, 92]."""
    return dict(
        osc_kp=[k * scale for k in _OSC_BASE_KP],
        osc_kd=[d * scale**0.5 for d in _OSC_BASE_KD],
    )


_SOFT_OSC_GAINS = dict(            # s=0.5 sweep gains ROUNDED — literal by design
    osc_kp=[1500.0, 1500.0, 2000.0],
    osc_kd=[78.0, 78.0, 92.0],
)

# Train defaults for the OSC/physical/rough experiment family: per-step domain
# randomization + frequent eval (500 episodes ≈ every ~500k steps @ 1k envs).
# ONE decision shared BY REFERENCE across all entries below — the frozen
# EnvVariant only holds it, never mutates it. If a variant ever needs to
# differ, give it its own dict(_DR_TRAIN)-copy instead of editing this one.
_DR_TRAIN = {"reset_mode": "per_step", "eval_every_n_episodes": 500}

def _cfg(**knobs):
    """Bind go2_config knobs into a zero-arg config callable.

    Every knob is BAKED into the factory (no config_overrides): Playground's
    registry.load passes config_overrides=None by default, which would clobber
    a partial(..., config_overrides=...) — so per-variant flags must live in a
    dedicated default_config factory."""
    return functools.partial(go2_config, **knobs)


GO2_WARP_VARIANTS = {
    # ── Joint-PD family ──────────────────────────────────────────────────
    "Go2WarpJoystickFlat": EnvVariant(config=_cfg()),
    # Linear torque-speed actuator limit (approximates motor saturation).
    "Go2WarpJoystickFlatTorqueSpeed": EnvVariant(config=_cfg(motor="torque_speed")),
    "Go2WarpJoystickFlatNoAccel": EnvVariant(
        config=_cfg(), cls="WarpJoystickNoAccel",
        notes="actor obs without accelerometer (state 45d, priv 119d) — an "
              "obs variant (not a controller swap), so it keeps its own class"),
    # Hardware-conservative variant, named "Unitree" for the parts partially
    # aligned with unitree_rl_lab's Go2 deploy contract:
    #   - matched: action_scale 0.25, no accelerometer in actor obs,
    #     Kp=20/Kd=0.5 (already shared)
    #   - NOT matched: explicit per-term obs scales (gyro×0.2, jvel×0.05);
    #     we rely on running obs_norm for whitening instead. Reward scaling,
    #     command sampling, and event randomization also differ.
    # Closer-than-default to the working Unitree stack; not bitwise parity.
    "Go2WarpJoystickUnitree": EnvVariant(
        config=_cfg(action_scale=0.25), cls="WarpJoystickNoAccel",
        notes="hardware-conservative: no-accel obs + action_scale 0.25"),
    # ── Fixed-gain OSC family ────────────────────────────────────────────
    # Cartesian impedance / OSC: the 12-d action is four foot-position targets
    # (trunk frame) driven by a per-leg operational-space controller instead of
    # joint PD. Same task / obs / reward as the joint-PD joystick — the
    # controller is picked FROM the config (osc block present), not the class.
    # Design notes: .superpowers/specs/2026-06-08-go2-osc-impedance-design.md
    "Go2WarpOscJoystickFlat": EnvVariant(
        config=_cfg(controller="osc"), train=_DR_TRAIN),
    # Jᵀ Cartesian-impedance ablation: use_op_space_inertia=False — no Λ
    # unit-mass normalization, real N/m gains, feet keep their natural
    # anisotropic inertia (heavy along the leg). Tests whether Λ's unit-mass
    # feet are what drive the bounding/pogo gait of the Λ-OSC variant. Gains
    # hold-probed (N/m, not the acceleration-gains of the Λ variant).
    "Go2WarpOscJoystickFlatJt": EnvVariant(
        config=_cfg(controller="osc", use_op_space_inertia=False,
                    osc_kp=[1500.0, 1500.0, 2500.0], osc_kd=[60.0, 60.0, 80.0]),
        train=_DR_TRAIN,
        notes="Jᵀ-impedance ablation: real N/m gains, no Λ"),
    # Stiffness sweep (Λ-OSC): see _kp_sweep_gains for the design rationale.
    "Go2WarpOscJoystickFlatKp025": EnvVariant(
        config=_cfg(controller="osc", **_kp_sweep_gains(0.25)), train=_DR_TRAIN),
    "Go2WarpOscJoystickFlatKp05": EnvVariant(
        config=_cfg(controller="osc", **_kp_sweep_gains(0.5)), train=_DR_TRAIN),
    "Go2WarpOscJoystickFlatKp2": EnvVariant(
        config=_cfg(controller="osc", **_kp_sweep_gains(2.0)), train=_DR_TRAIN),
    "Go2WarpOscJoystickFlatKp4": EnvVariant(
        config=_cfg(controller="osc", **_kp_sweep_gains(4.0)), train=_DR_TRAIN),
    # ── Variable-impedance family ────────────────────────────────────────
    # Action grows to 16-d (12 foot targets + 4 per-foot stiffness scalars);
    # each maps log-spaced to s∈[0.25,2] scaling that foot's baseline Cartesian
    # gains (kd∝√s). Policy learns to stiffen stance / soften swing legs.
    "Go2WarpOscVarImpedanceFlat": EnvVariant(
        config=_cfg(controller="var_impedance"), train=_DR_TRAIN),
    "Go2WarpOscVarImpedanceAxisFlat": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis"),
        train=_DR_TRAIN,
        notes="per-foot-per-axis stiffness (+12 → action 24): policy picks "
              "vertical-stiff / tangential-soft per leg"),
    # ── Flat + PHYSICAL motor model (zero-shot-from-flat experiment) ─────
    # Same controllers/gains as the rough-physical eval envs, but on FLAT
    # ground. Policies train WITH the physical motor model (DC-motor
    # torque-speed curve + mjlab per-joint armature) then zero-shot transfer
    # onto rough-physical (Go2Warp*RoughUni). Mirrors the original zero-shot
    # protocol, now with the motor model held consistent across train + eval.
    "Go2WarpJoystickFlatPhysical": EnvVariant(
        config=_cfg(motor="physical"), train=_DR_TRAIN, notes="joint-PD"),
    "Go2WarpOscFlatSoftPhysical": EnvVariant(
        config=_cfg(controller="osc", motor="physical", **_SOFT_OSC_GAINS),
        train=_DR_TRAIN,
        notes="fixed-soft OSC (kp/kd mirror the rough soft gains, s=0.5)"),
    "Go2WarpOscVarFlatPhysical": EnvVariant(
        config=_cfg(controller="var_impedance", motor="physical"),
        train=_DR_TRAIN,
        notes="variable per-foot (locked critical)"),
    "Go2WarpOscVarAxisFlatPhysical": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    motor="physical"),
        train=_DR_TRAIN,
        notes="variable per-axis"),
    "Go2WarpOscVarDampingFlatPhysical": EnvVariant(
        config=_cfg(controller="var_impedance", damping_action=True,
                    motor="physical"),
        train=_DR_TRAIN,
        notes="decoupled K+D (per-foot)"),
    "Go2WarpOscVarDampingAxisFlatPhysical": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    damping_action=True, motor="physical"),
        train=_DR_TRAIN,
        notes="decoupled K+D (per-axis)"),
    # ── Mud force-field variants (analytic foot-wrench OOD probe) ─────────
    # Same controller/motor as Go2WarpOscVarDampingAxisFlatPhysical + an analytic
    # mud foot-force field (MudField; see go2_warp_components.mud_foot_force and
    # .context/references/mud_force_port_handoff.md). The MudNN previews pin a
    # fixed mud surface height for eval recording; MudDR randomizes depth per
    # episode for TRAINING. Newton MPM (projects/mud_eval/) stays the held-out
    # transfer test — never tune these coeffs against it.
    "Go2WarpOscVarDampingAxisFlatPhysicalMud05": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    damping_action=True, motor="physical",
                    mud=dict(depth_range=(0.05, 0.05))),
        train=_DR_TRAIN, notes="fixed 0.05 m mud preview (eval OOD probe)"),
    "Go2WarpOscVarDampingAxisFlatPhysicalMud10": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    damping_action=True, motor="physical",
                    mud=dict(depth_range=(0.10, 0.10))),
        train=_DR_TRAIN, notes="fixed 0.10 m mud preview (eval OOD probe)"),
    "Go2WarpOscVarDampingAxisFlatPhysicalMud22": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    damping_action=True, motor="physical",
                    mud=dict(depth_range=(0.22, 0.22))),
        train=_DR_TRAIN, notes="fixed 0.22 m mud preview (Isaac default; calves submerged)"),
    "Go2WarpOscVarDampingAxisFlatPhysicalMudDR": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    damping_action=True, motor="physical",
                    mud=dict(depth_range=(0.03, 0.22))),
        train=_DR_TRAIN,
        notes="randomized mud depth U[0.03,0.22] — the trainable variant; "
              "train-on-analytic, Newton MPM held out"),
    "Go2WarpOscVarDampingAxisFlatPhysicalMud22Heavy": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    damping_action=True, motor="physical",
                    mud=dict(depth_range=(0.22, 0.22),
                             f_range=(116.0, 116.0),    # 8× Isaac default suction/resist
                             c1_range=(76.0, 76.0),     # 8× shear viscous
                             c2_range=(52.0, 52.0))),   # 8× shear yield
        train=_DR_TRAIN,
        notes="EVAL PROBE (not a calibrated training target): 8× coeffs — the "
              "regime where the frozen DR ckpt BOGS (upright, ~0 net progress, "
              "lin-track err ≈ commanded speed). Coeff sweep 2026-06-11: upright "
              "through 8×, falls at 12×+. Magnitudes uncalibrated vs Newton MPM."),
    # Joint-PD CONTROL for the mud-training comparison: same DR (Isaac coeffs +
    # depth U[0.03,0.22]) and physical motor as ...MudDR, but conventional joint
    # PD instead of variable Cartesian impedance. Tests whether var-impedance's
    # deep-mud edge (the mud_eval headline) survives mud training. 12-d action.
    "Go2WarpJoystickFlatPhysicalMudDR": EnvVariant(
        config=_cfg(motor="physical", mud=dict(depth_range=(0.03, 0.22))),
        train=_DR_TRAIN,
        notes="joint-PD control + mud DR — the joint-PD arm of the "
              "var-impedance-vs-joint-PD mud-training comparison"),
    # R1 of the autonomous mud-training iteration: EXPAND the mud-coeff DR to
    # span 1×→4× Isaac (f 14→60, c1 9→40, c2 6→28), depth unchanged. R0 (1×)
    # bogged both arms in medium mud (var y1.29, joint-PD y1.62, never thick).
    # Stronger/wider mud DR pushes into the "biting" regime (coeff sweep: 4× =
    # labored). area_range left at Isaac default.
    "Go2WarpOscVarDampingAxisFlatPhysicalMudDR4x": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    damping_action=True, motor="physical",
                    mud=dict(depth_range=(0.03, 0.22), f_range=(14.0, 60.0),
                             c1_range=(9.0, 40.0), c2_range=(6.0, 28.0))),
        train=_DR_TRAIN,
        notes="var-impedance + mud DR spanning 1→4× Isaac coeffs (R1)"),
    "Go2WarpJoystickFlatPhysicalMudDR4x": EnvVariant(
        config=_cfg(motor="physical",
                    mud=dict(depth_range=(0.03, 0.22), f_range=(14.0, 60.0),
                             c1_range=(9.0, 40.0), c2_range=(6.0, 28.0))),
        train=_DR_TRAIN,
        notes="joint-PD + mud DR spanning 1→4× Isaac coeffs (R1)"),
    # R2: reward-shaped toward firm foot planting (heavier feet_slip + orientation
    # penalties) on the var-impedance 4× mud DR — tests the "slow+firm beats
    # lean+drive" hypothesis. Config callable sets reward_config overrides.
    "Go2WarpOscVarDampingAxisFlatPhysicalMudDR4xFirm": EnvVariant(
        config=_var_muddr4x_firmplant_config,
        train=_DR_TRAIN,
        notes="var-impedance 4× mud DR + firm-planting reward shaping (R2): seed0 "
              "CLEARED the mud (y-1.4); seed1 reached deep thick then fell (lunge)"),
    "Go2WarpOscVarDampingAxisFlatPhysicalMudDR4xSlowFirm": EnvVariant(
        config=_var_muddr4x_slowfirm_config,
        train=_DR_TRAIN,
        notes="R4 ROBUST WINNER: firm-planting + reduced velocity pressure — both "
              "seeds clear the full Newton mud gradient upright (s0 y-0.36, s1 y-2.20)"),
    "Go2WarpJoystickFlatPhysicalMudDR4xSlowFirm": EnvVariant(
        config=_jointpd_muddr4x_slowfirm_config,
        train=_DR_TRAIN,
        notes="R5: slow+firm recipe on joint-PD — FROZE at spawn; var-impedance essential"),
    "Go2WarpOscVarDampingAxisFlatPhysicalMudDR4xSlowSlip": EnvVariant(
        config=_var_muddr4x_slowslip_config,
        train=_DR_TRAIN,
        notes="R6 ablation: slow + feet_slip only (no orient) — both terms matter (y0.64)"),
    "Go2WarpOscVarDampingAxisFlatPhysicalMudDR1xSlowFirm": EnvVariant(
        config=_var_muddr1x_slowfirm_config,
        train=_DR_TRAIN,
        notes="#5 ablation: slow+firm reward at 1× Isaac mud DR — is 4× DR dispensable?"),
    "Go2WarpOscVarDampingAxisFlatPhysicalMudDR4xSlowFirmRMA": EnvVariant(
        config=_var_muddr4x_slowfirm_rma_config,
        train=_DR_TRAIN,
        notes="slow+firm + RMA-minimal feet (drop air_time/clearance/height/pose/"
              "stand_still, slip→0.8) — fix the tripod gait (arXiv 2107.04034)"),
    # ── Hard-kick comparison ladder ──────────────────────────────────────
    # DOMAIN-RANDOMIZED kick strength: per-episode kick bound ~ U[0.5, 2.5] m/s
    # (into the ≥2 m/s pure-impedance failure regime), vs the default fixed
    # ±0.75. Identical DR'd kick applied to conventional joint-PD, fixed-soft
    # OSC, scalar variable impedance, and per-axis variable impedance. Each
    # step isolates one factor — (a) does Cartesian impedance help disturbance
    # rejection at all (vs joint-PD), (b) does stiffness modulation help (vs
    # fixed), (c) does per-axis beat scalar — i.e. does the policy learn to
    # stiffen on demand to reject hard disturbances.
    "Go2WarpJoystickFlatHardKick": EnvVariant(
        config=_cfg(push=(0.5, 2.5)), notes="joint-PD control"),
    "Go2WarpOscJoystickFlatKp05HardKick": EnvVariant(
        config=_cfg(controller="osc", push=(0.5, 2.5), **_kp_sweep_gains(0.5)),
        train=_DR_TRAIN,
        notes="fixed-soft OSC control"),
    "Go2WarpOscVarImpedanceHardKickFlat": EnvVariant(
        config=_cfg(controller="var_impedance", push=(0.5, 2.5)),
        train=_DR_TRAIN,
        notes="scalar +4"),
    "Go2WarpOscVarImpedanceAxisHardKickFlat": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    push=(0.5, 2.5)),
        train=_DR_TRAIN,
        notes="per-axis +12"),
    # ── Rough HEIGHTFIELD floor ──────────────────────────────────────────
    # Real continuous rough, borrowed from mjlab's noise recipe. Uni = uniform
    # (jagged ~7 cm foot-scale; perlin "A" dropped — too smooth). Gains match
    # the flat runs so flat-trained policies zero-shot transfer (the headline).
    # Terrain is config-driven (RoughHF via terrain_from_config reads the
    # rough_* keys), so the rough envs are just the plain controller class + a
    # rough config — no rough mixin/subclass. (The earlier box-terrain "rough
    # curriculum" envs were removed 2026-06-09: they spawned the robot on the
    # flat border so it never actually saw rough; superseded by these.)
    "Go2WarpJointRoughUni": EnvVariant(
        config=_cfg(motor="physical", terrain=("uniform", 0.07)),
        train=_DR_TRAIN),
    "Go2WarpOscRoughUni": EnvVariant(
        config=_cfg(controller="osc", motor="physical",
                    terrain=("uniform", 0.07), **_SOFT_OSC_GAINS),
        train=_DR_TRAIN),
    "Go2WarpOscVarRoughUni": EnvVariant(
        config=_cfg(controller="var_impedance", motor="physical",
                    terrain=("uniform", 0.07)),
        train=_DR_TRAIN),
    "Go2WarpOscVarAxisRoughUni": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    motor="physical", terrain=("uniform", 0.07)),
        train=_DR_TRAIN),
    "Go2WarpOscVarDampingAxisRoughUni": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    damping_action=True, motor="physical",
                    terrain=("uniform", 0.07)),
        train=_DR_TRAIN,
        notes="decoupled K+D per-axis on rough — zero-shot eval target for the "
              "VarDampingAxisFlatPhysical DR-era ckpts (added 2026-06-11)"),
    # ── Env-module configs (lazy imports) ────────────────────────────────
    "Go2WarpJoystickCurriculum": EnvVariant(
        config=_curriculum_config, cls="WarpJoystickCurriculum",
        train={"reset_mode": "per_step"}),
    "Go2WarpJoystickCurriculumTorqueSpeed": EnvVariant(
        config=_curriculum_torque_speed_config, cls="WarpJoystickCurriculum",
        train={"reset_mode": "per_step"}),
    "Go2WarpFlatPosTrackProto": EnvVariant(
        config=_flat_postrack_config, cls="WarpFlatPosTrack",
        notes="prototype flat-ground PosTrack — delta_xy_yaw obs + Lorentzian "
              "reward; shares parameterization with Go2WarpSplitbeltPosTrack "
              "so cross-deploy is direct"),
}
