"""Rough-heightfield CONFIG FACTORIES for the Go2 Warp joystick task.

Terrain is now a pluggable component (`go2_warp_components.RoughHF`, selected by
``terrain_from_config`` from the ``rough_*`` config keys) — there is no longer a
separate rough scene XML or a ``_RoughHFMixin``. These factories just stamp the
``rough_profile``/``rough_amplitude``/``rough_seed`` keys (which the Terrain
component reads) plus the physical motor model onto the base controller configs,
so the registry binds ``Go2Warp{Joint,Osc,OscVar,OscVarAxis}RoughUni`` to the
plain controller class + a rough config.

Profiles (``rough_profile``): "uniform" (jagged ~7 cm foot-scale, the headline),
"perlin_hf" (smooth rounded bumps), "smooth_uniform" (rounded random). MJX-Warp
heightfield collision is confirmed working (Go2 rests on the bumps; hfield CCD
kernels take a one-time ~9 s JIT compile).
"""
from jax_rl.envs.locomotion.go2_warp_joystick import (
    default_config as joystick_default_config,
)
from jax_rl.envs.locomotion.go2_warp_osc_joystick import (
    default_config as osc_default_config,
)
from jax_rl.envs.locomotion.go2_warp_osc_var_impedance import (
    default_config as var_default_config,
    default_config_per_axis as var_axis_default_config,
)


def _add_rough(cfg, profile, amplitude, seed=0):
    cfg.rough_profile = profile
    cfg.rough_amplitude = amplitude
    cfg.rough_seed = seed
    # Physical motor model ON for rough/physical envs only (flat envs stay off
    # for reproducibility). DC-motor torque-speed curve caps the calf at its
    # 20.07 rad/s no-load speed (torque_speed_model); mjlab per-joint rotor
    # inertia replaces the uniform 0.01 armature (physical_armature). See
    # go2_warp_base.torque_speed_clip / physical_armature.
    cfg.torque_speed_model = True
    cfg.physical_armature = True
    return cfg


# ── Config factories (gains matched to the flat runs for zero-shot) ──────────
def joint_rough_config(profile="uniform", amp=0.07):
    return _add_rough(joystick_default_config(), profile, amp)


def osc_soft_rough_config(profile="uniform", amp=0.07):
    cfg = osc_default_config()
    cfg.osc.kp = [1500.0, 1500.0, 2000.0]   # s=0.5 (matches flat fixed-soft)
    cfg.osc.kd = [78.0, 78.0, 92.0]
    return _add_rough(cfg, profile, amp)


def var_rough_config(profile="uniform", amp=0.07):
    return _add_rough(var_default_config(), profile, amp)        # per_foot, baseline kp


def var_axis_rough_config(profile="uniform", amp=0.07):
    return _add_rough(var_axis_default_config(), profile, amp)
