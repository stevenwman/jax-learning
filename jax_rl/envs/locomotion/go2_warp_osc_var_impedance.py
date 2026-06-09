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


def default_config() -> config_dict.ConfigDict:
    cfg = _osc_default_config()
    cfg.osc.var_s_min = 0.25
    cfg.osc.var_s_max = 2.0
    cfg.osc.stiffness_granularity = "per_foot"   # {per_foot (+4), per_axis (+12)}
    return cfg


def default_config_per_axis() -> config_dict.ConfigDict:
    cfg = default_config()
    cfg.osc.stiffness_granularity = "per_axis"
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

    @property
    def action_size(self) -> int:
        # 12 foot-position targets + per-foot (4) or per-foot-per-axis (12).
        return 12 + self._n_stiffness

    def _stiffness_scale(self, a: jax.Array) -> jax.Array:
        """Map action a∈[-1,1] → stiffness scale s∈[s_min,s_max] (log), elementwise."""
        u = 0.5 * (jp.clip(a, -1.0, 1.0) + 1.0)
        return self._var_s_min * (self._var_s_max / self._var_s_min) ** u

    def _apply_control(self, data, action: jax.Array):
        deltas = action[:12].reshape(4, 3) * self._config.action_scale   # (4,3)
        s = self._stiffness_scale(action[12:12 + self._n_stiffness])
        if self._stiffness_granularity == "per_foot":
            s = s[:, None]              # (4,1) — one scale per foot, all axes
        else:                           # per_axis
            s = s.reshape(4, 3)         # (4,3) — per foot AND axis
        kp = s * self._osc_kp           # (4,3): broadcasts (4,1|4,3)*(3,)
        kd = jp.sqrt(s) * self._osc_kd  # (4,3)
        return self._run_osc(data, deltas, kp, kd)
