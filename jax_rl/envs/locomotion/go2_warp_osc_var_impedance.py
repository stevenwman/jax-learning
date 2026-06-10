"""Variable-impedance Cartesian-OSC joystick env for Go2 (Warp backend).

The policy ALSO commands stiffness. Two granularities (config
`osc.stiffness_granularity`):

  - ``per_foot``  (+4 → action 16): one stiffness scalar per foot.
  - ``per_axis``  (+12 → action 24): one stiffness per foot AND axis (x,y,z).

Each stiffness scalar a ∈ [-1,1] maps log-spaced to s ∈ [s_min, s_max], scaling
the corresponding baseline Cartesian gain (kp = s·kp_base, kd = ζ·√s·kd_base) so
the damping ratio stays ~critical unless the policy also commands ζ (decoupled
damping). Range default s ∈ [0.25, 2] comes from the fixed sweep.

After the env-composition refactor this is a THIN config preset over
:class:`WarpOscJoystick`: the variable-impedance default config selects the
VarImpedance controller (via ``stiffness_granularity``); the action-tail decode
(:func:`impedance_gains`) and the control loop live in
``go2_warp_components.VarImpedance`` + the host. The decode helpers are
re-exported here so existing import paths (tests, record) keep working.

Design notes: ``.superpowers/specs/2026-06-08-go2-osc-impedance-design.md``.
"""
from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from jax_rl.envs.locomotion.go2_warp_osc_joystick import (
    WarpOscJoystick,
    default_config as _osc_default_config,
)
# Decode helpers now live in the components module; re-exported for back-compat.
from jax_rl.envs.locomotion.go2_warp_components import (  # noqa: F401
    _N_STIFFNESS, log_action_scale, var_action_size, impedance_gains,
)


def default_config() -> config_dict.ConfigDict:
    cfg = _osc_default_config()
    cfg.osc.var_s_min = 0.25
    cfg.osc.var_s_max = 2.0
    cfg.osc.stiffness_granularity = "per_foot"   # {per_foot (+4), per_axis (+12)}
    # Decoupled damping: when True the policy ALSO commands ζ (damping ratio,
    # 1=critical) per the same granularity. Off by default → kd locked critical.
    cfg.osc.damping_action = False
    cfg.osc.var_zeta_min = 0.5
    cfg.osc.var_zeta_max = 2.0
    return cfg


def default_config_per_axis() -> config_dict.ConfigDict:
    cfg = default_config()
    cfg.osc.stiffness_granularity = "per_axis"
    return cfg


def default_config_damping() -> config_dict.ConfigDict:
    """Per-foot stiffness AND damping (action 20) — decoupled K and D."""
    cfg = default_config()
    cfg.osc.damping_action = True
    return cfg


def default_config_damping_axis() -> config_dict.ConfigDict:
    """Per-axis stiffness AND damping (action 36) — decoupled K and D, per foot+axis."""
    cfg = default_config_per_axis()
    cfg.osc.damping_action = True
    return cfg


class WarpOscVarImpedance(WarpOscJoystick):
    """OSC joystick where the policy also commands stiffness (per-foot or -axis).

    Thin preset: supplies the variable-impedance default config (which selects
    the VarImpedance controller). Control logic lives in the host +
    components.VarImpedance; action_size comes from the controller.
    """

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            task=task, config=config, config_overrides=config_overrides
        )
