"""Variable-impedance Cartesian-OSC joystick env for Go2 (Warp backend).

Extends :class:`WarpOscJoystick`: the policy ALSO commands stiffness. Two
granularities (config `osc.stiffness_granularity`):

  - ``per_foot``  (+4 → action 16): one stiffness scalar per foot.
  - ``per_axis``  (+12 → action 24): one stiffness per foot AND axis (x,y,z).

Each stiffness scalar a ∈ [-1,1] maps log-spaced to s ∈ [s_min, s_max], scaling
the corresponding baseline Cartesian gain:

    kp = s · kp_base ,   kd = √s · kd_base

so the damping ratio stays ~critical as s varies (exactly the relationship the
fixed-stiffness sweep used). `per_foot` lets the policy stiffen a stance leg and
soften a swing/contact leg; `per_axis` additionally lets it pick, per leg,
vertical-stiff (bear load) vs tangential-soft (contact compliance).

Range default s ∈ [0.25, 2] comes from the fixed sweep (s≤1 calmest, s≥2
degrades), bracketing the useful band for the policy to choose within.

Design notes: ``.superpowers/specs/2026-06-08-go2-osc-impedance-design.md``.
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict

from jax_rl.envs.locomotion.go2_warp_osc_joystick import (
    WarpOscJoystick,
    default_config as _osc_default_config,
)

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
    """OSC joystick where the policy also commands stiffness (per-foot or -axis)."""

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
        self._var_s_min = float(self._config.osc.var_s_min)
        self._var_s_max = float(self._config.osc.var_s_max)
        gran = str(getattr(self._config.osc, "stiffness_granularity", "per_foot"))
        if gran not in _N_STIFFNESS:
            raise ValueError(
                f"stiffness_granularity={gran!r} not in {list(_N_STIFFNESS)}"
            )
        self._stiffness_granularity = gran
        self._n_stiffness = _N_STIFFNESS[gran]
        self._damping_action = bool(getattr(self._config.osc, "damping_action", False))
        self._var_z_min = float(getattr(self._config.osc, "var_zeta_min", 0.5))
        self._var_z_max = float(getattr(self._config.osc, "var_zeta_max", 2.0))

    @property
    def action_size(self) -> int:
        # 12 foot targets + stiffness (+ damping when enabled), per granularity.
        return var_action_size(self._stiffness_granularity, self._damping_action)

    def _apply_control(self, data, action: jax.Array):
        deltas = action[:12].reshape(4, 3) * self._config.action_scale   # (4,3)
        kp, kd = impedance_gains(
            action, self._osc_kp, self._osc_kd,
            granularity=self._stiffness_granularity,
            s_min=self._var_s_min, s_max=self._var_s_max,
            damping_action=self._damping_action,
            z_min=self._var_z_min, z_max=self._var_z_max,
        )
        return self._run_osc(data, deltas, kp, kd)
