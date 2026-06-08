"""Variable-impedance Cartesian-OSC joystick env for Go2 (Warp backend).

Extends :class:`WarpOscJoystick`: the policy now ALSO commands stiffness. The
12-d foot-target action grows to 16-d = 12 foot position targets (trunk frame)
+ 4 per-foot stiffness scalars (this is the "+4 per-foot scalar" variant; a
"+12 per-foot-per-axis" variant follows).

Each stiffness scalar a ∈ [-1,1] maps log-spaced to s ∈ [s_min, s_max], scaling
that foot's baseline Cartesian gains:

    kp_foot = s · kp_base ,   kd_foot = √s · kd_base

so the damping ratio stays ~critical as s varies (exactly the relationship the
fixed-stiffness sweep used). This lets the policy stiffen a stance leg to bear
load and soften a swing/contact leg for compliance — the variable-impedance
behaviour we want to study.

Range default s ∈ [0.25, 2] comes from the fixed sweep: s≤1 gave the calmest
gait, s≥2 degraded tracking and doubled effort, so the policy is handed a band
that brackets "useful" and lets it pick per-foot.

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


def default_config() -> config_dict.ConfigDict:
    cfg = _osc_default_config()
    # Stiffness-action range the per-foot scalar maps onto (log-spaced).
    cfg.osc.var_s_min = 0.25
    cfg.osc.var_s_max = 2.0
    return cfg


class WarpOscVarImpedance(WarpOscJoystick):
    """OSC joystick where the policy also commands per-foot stiffness."""

    N_STIFFNESS = 4   # one scalar per foot (FL, FR, RL, RR)

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

    @property
    def action_size(self) -> int:
        # 12 foot-position targets + N_STIFFNESS per-foot stiffness scalars.
        return 12 + self.N_STIFFNESS

    def _stiffness_scale(self, stiff_action: jax.Array) -> jax.Array:
        """Map per-foot action a∈[-1,1] → stiffness scale s∈[s_min,s_max] (log)."""
        u = 0.5 * (jp.clip(stiff_action, -1.0, 1.0) + 1.0)          # [0,1]
        return self._var_s_min * (self._var_s_max / self._var_s_min) ** u

    def _apply_control(self, data, action: jax.Array):
        deltas = action[:12].reshape(4, 3) * self._config.action_scale   # (4,3)
        s = self._stiffness_scale(action[12:12 + self.N_STIFFNESS])      # (4,)
        kp = s[:, None] * self._osc_kp                                    # (4,3)
        kd = jp.sqrt(s)[:, None] * self._osc_kd                           # (4,3)
        return self._run_osc(data, deltas, kp, kd)
