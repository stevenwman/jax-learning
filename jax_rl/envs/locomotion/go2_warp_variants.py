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
from typing import Union

from ml_collections import config_dict


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
        kp = list(osc_kp) if osc_kp is not None else [3000.0, 3000.0, 4000.0]
        kd = list(osc_kd) if osc_kd is not None else [110.0, 110.0, 130.0]
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

    return cfg


@dataclasses.dataclass(frozen=True)
class EnvVariant:
    config: object                    # Callable[[], ConfigDict]
    cls: str = "WarpJoystick"
    train: dict = dataclasses.field(default_factory=dict)
    algo: dict = dataclasses.field(default_factory=dict)   # {algo_name: {field: val}}
    notes: str = ""


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

    MUST stay computed expressions (not rounded literals) to match the legacy
    factory bit-for-bit — e.g. Kp05 kd ≈ [77.78, 77.78, 91.92], which is NOT
    the SoftPhysical literals [78, 78, 92]."""
    return dict(
        osc_kp=[k * scale for k in (3000.0, 3000.0, 4000.0)],
        osc_kd=[d * scale**0.5 for d in (110.0, 110.0, 130.0)],
    )


_SOFT_OSC_GAINS = dict(            # s=0.5 sweep gains ROUNDED — literal by design
    osc_kp=[1500.0, 1500.0, 2000.0],
    osc_kd=[78.0, 78.0, 92.0],
)

def _cfg(**knobs):
    """Bind go2_config knobs into a zero-arg config callable."""
    return functools.partial(go2_config, **knobs)


GO2_WARP_VARIANTS = {
    # ── Joint-PD family ──────────────────────────────────────────────────
    "Go2WarpJoystickFlat": EnvVariant(config=_cfg()),
    "Go2WarpJoystickFlatTorqueSpeed": EnvVariant(config=_cfg(motor="torque_speed")),
    "Go2WarpJoystickFlatNoAccel": EnvVariant(
        config=_cfg(), cls="WarpJoystickNoAccel",
        notes="actor obs without accelerometer (state 45d)"),
    "Go2WarpJoystickUnitree": EnvVariant(
        config=_cfg(action_scale=0.25), cls="WarpJoystickNoAccel",
        notes="hardware-conservative: no-accel obs + action_scale 0.25"),
    "Go2WarpJoystickFlatPhysical": EnvVariant(config=_cfg(motor="physical")),
    "Go2WarpJoystickFlatHardKick": EnvVariant(config=_cfg(push=(0.5, 2.5))),
    # ── Fixed-gain OSC family ────────────────────────────────────────────
    "Go2WarpOscJoystickFlat": EnvVariant(config=_cfg(controller="osc")),
    "Go2WarpOscJoystickFlatJt": EnvVariant(
        config=_cfg(controller="osc", use_op_space_inertia=False,
                    osc_kp=[1500.0, 1500.0, 2500.0], osc_kd=[60.0, 60.0, 80.0]),
        notes="Jᵀ-impedance ablation: real N/m gains, no Λ"),
    "Go2WarpOscJoystickFlatKp025": EnvVariant(
        config=_cfg(controller="osc", **_kp_sweep_gains(0.25))),
    "Go2WarpOscJoystickFlatKp05": EnvVariant(
        config=_cfg(controller="osc", **_kp_sweep_gains(0.5))),
    "Go2WarpOscJoystickFlatKp2": EnvVariant(
        config=_cfg(controller="osc", **_kp_sweep_gains(2.0))),
    "Go2WarpOscJoystickFlatKp4": EnvVariant(
        config=_cfg(controller="osc", **_kp_sweep_gains(4.0))),
    "Go2WarpOscJoystickFlatKp05HardKick": EnvVariant(
        config=_cfg(controller="osc", push=(0.5, 2.5), **_kp_sweep_gains(0.5))),
    "Go2WarpOscFlatSoftPhysical": EnvVariant(
        config=_cfg(controller="osc", motor="physical", **_SOFT_OSC_GAINS)),
    # ── Variable-impedance family ────────────────────────────────────────
    "Go2WarpOscVarImpedanceFlat": EnvVariant(config=_cfg(controller="var_impedance")),
    "Go2WarpOscVarImpedanceAxisFlat": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis")),
    "Go2WarpOscVarFlatPhysical": EnvVariant(
        config=_cfg(controller="var_impedance", motor="physical")),
    "Go2WarpOscVarAxisFlatPhysical": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    motor="physical")),
    "Go2WarpOscVarDampingFlatPhysical": EnvVariant(
        config=_cfg(controller="var_impedance", damping_action=True,
                    motor="physical")),
    "Go2WarpOscVarDampingAxisFlatPhysical": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    damping_action=True, motor="physical")),
    "Go2WarpOscVarImpedanceHardKickFlat": EnvVariant(
        config=_cfg(controller="var_impedance", push=(0.5, 2.5))),
    "Go2WarpOscVarImpedanceAxisHardKickFlat": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    push=(0.5, 2.5))),
    # ── Rough heightfield (physical motor; gains match flat for zero-shot) ──
    "Go2WarpJointRoughUni": EnvVariant(
        config=_cfg(motor="physical", terrain=("uniform", 0.07))),
    "Go2WarpOscRoughUni": EnvVariant(
        config=_cfg(controller="osc", motor="physical",
                    terrain=("uniform", 0.07), **_SOFT_OSC_GAINS)),
    "Go2WarpOscVarRoughUni": EnvVariant(
        config=_cfg(controller="var_impedance", motor="physical",
                    terrain=("uniform", 0.07))),
    "Go2WarpOscVarAxisRoughUni": EnvVariant(
        config=_cfg(controller="var_impedance", stiffness_granularity="per_axis",
                    motor="physical", terrain=("uniform", 0.07))),
    # ── Env-module configs (lazy imports) ────────────────────────────────
    "Go2WarpJoystickCurriculum": EnvVariant(
        config=_curriculum_config, cls="WarpJoystickCurriculum",
        train={"reset_mode": "per_step"}),
    "Go2WarpJoystickCurriculumTorqueSpeed": EnvVariant(
        config=_curriculum_torque_speed_config, cls="WarpJoystickCurriculum",
        train={"reset_mode": "per_step"}),
    "Go2WarpFlatPosTrackProto": EnvVariant(
        config=_flat_postrack_config, cls="WarpFlatPosTrack"),
}
