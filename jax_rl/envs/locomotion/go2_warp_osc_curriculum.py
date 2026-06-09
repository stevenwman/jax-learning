"""OSC / variable-impedance Go2 on a ROUGH-only terrain curriculum (Warp).

Composes the per-leg Cartesian-impedance controller (:mod:`go2_warp_osc_joystick`,
:mod:`go2_warp_osc_var_impedance`) onto the procedural terrain curriculum
(:class:`WarpJoystickCurriculum`) via multiple inheritance — the curriculum
provides terrain / spawn / goal-directed ``step`` / ``reset``; the OSC mixin
provides ``_apply_control``. No refactor of either side.

Rough terrain is where foot compliance should matter: feet land at unexpected
heights, so a soft / adaptive impedance can absorb the contact where a stiff one
transmits shock and loses footing. The grid is rough-only (4 rough columns × 6
difficulty rows) to isolate that signal from slope/stair walking.

Envs:
  - Go2WarpRoughCurriculum            joint-PD (control)
  - Go2WarpOscRoughCurriculum         fixed-soft OSC (s=0.5)
  - Go2WarpOscVarRoughCurriculum      variable per-foot stiffness (action 16)
  - Go2WarpOscVarAxisRoughCurriculum  variable per-foot-per-axis (action 24)

MRO note: for e.g. WarpOscCurriculum(WarpOscJoystick, WarpJoystickCurriculum),
__init__ resolves to WarpOscJoystick.__init__ → super() → WarpJoystickCurriculum
.__init__ (terrain + self._post_init()); _post_init resolves to the OSC chain
(WarpJoystick._post_init then OSC setup); step/reset resolve to the curriculum.
"""
from ml_collections import config_dict

from jax_rl.envs.terrains import RoughTerrainCfg, TerrainGridCfg
from jax_rl.envs.locomotion.go2_warp_curriculum import (
    WarpJoystickCurriculum,
    default_config as curriculum_default_config,
)
from jax_rl.envs.locomotion.go2_warp_osc_joystick import WarpOscJoystick
from jax_rl.envs.locomotion.go2_warp_osc_var_impedance import WarpOscVarImpedance

# Rough-only grid: 4 rough columns × 6 difficulty rows (row 0 ~flat → row 5
# bumpy). Isolates rough-contact compliance from slope/stair walking.
ROUGH_ONLY_CFG = TerrainGridCfg(
    num_rows=6,
    tile_size=(9.6, 9.6),
    border_width=20.0,
    terrain_types=[RoughTerrainCfg(), RoughTerrainCfg(),
                   RoughTerrainCfg(), RoughTerrainCfg()],
)


# ── Config factories (curriculum config + OSC block, gains matched to the flat
#    runs so flat-trained policies can be zero-shot transferred) ──────────────

def _osc_block(kp, kd, **extra):
    return config_dict.create(
        target_mode="abs_body",
        use_op_space_inertia=True,
        gravity_ff="none",
        ridge=1e-4,
        kp=list(kp),
        kd=list(kd),
        **extra,
    )


def rough_curriculum_config() -> config_dict.ConfigDict:
    """Joint-PD on rough (control). Inherits Kp=20/Kd=0.5 joint-space PD."""
    return curriculum_default_config()


def osc_rough_soft_config() -> config_dict.ConfigDict:
    cfg = curriculum_default_config()
    cfg.action_scale = 0.12
    cfg.osc = _osc_block([1500.0, 1500.0, 2000.0], [78.0, 78.0, 92.0])  # s=0.5
    return cfg


def var_rough_config(granularity: str = "per_foot") -> config_dict.ConfigDict:
    cfg = curriculum_default_config()
    cfg.action_scale = 0.12
    cfg.osc = _osc_block(
        [3000.0, 3000.0, 4000.0], [110.0, 110.0, 130.0],   # baseline (s=1 ref)
        var_s_min=0.25, var_s_max=2.0, stiffness_granularity=granularity,
    )
    return cfg


def var_axis_rough_config() -> config_dict.ConfigDict:
    return var_rough_config("per_axis")


# ── Envs ────────────────────────────────────────────────────────────────────

class RoughCurriculum(WarpJoystickCurriculum):
    """Joint-PD on the rough-only curriculum."""
    _terrain_grid_cfg = ROUGH_ONLY_CFG


class WarpOscCurriculum(WarpOscJoystick, WarpJoystickCurriculum):
    """Fixed-impedance OSC on the rough-only curriculum."""
    _terrain_grid_cfg = ROUGH_ONLY_CFG


class WarpOscVarImpedanceCurriculum(WarpOscVarImpedance, WarpJoystickCurriculum):
    """Variable-impedance OSC on the rough-only curriculum (per_foot or per_axis)."""
    _terrain_grid_cfg = ROUGH_ONLY_CFG
