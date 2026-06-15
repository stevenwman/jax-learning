"""Pluggable env components for the Go2 Warp env — the four orthogonal axes of
variation, each a small strategy object the host (`Go2WarpEnv`/`WarpJoystick`)
delegates to:

  - Actuation   : commanded joint torque → applied torque (+ reflected inertia).
                  ``actuation`` — TorqueOnly | MotorModel.
  - Terrain     : floor shape — MjSpec mutation (structure) + post-compile hfield
                  data. ``terrain`` — Flat | RoughHF.
  - Controller  : action → joint torque each substep, incl. the OSC mechanics and
                  controller state. ``controllers`` — JointPD | OSC | VarImpedance
                  | VarImpedanceMass.
  - ForceField  : environmental wrench on the feet each substep (analytic mud),
                  controller-independent. ``force_fields`` — NoField | MudField.

Design: ``.superpowers/specs/2026-06-09-go2-env-composition.md`` (composition) +
``.superpowers/specs/2026-06-15-components-package-split.md`` (this package
split). Logic is extracted VERBATIM from the former subclasses/flags and the
former single ``go2_warp_components`` module, so there is no behaviour change —
only where the code lives. ``go2_warp_components`` remains as a back-compat
re-export shim.
"""
from __future__ import annotations

from jax_rl.envs.locomotion.components.actuation import (
    Actuation, TorqueOnly, MotorModel, actuation_from_config,
)
from jax_rl.envs.locomotion.components.terrain import (
    Terrain, Flat, RoughHF, terrain_from_config,
    _norm01, _fractal_perlin_noise_2d, _make_heightfield,
)
from jax_rl.envs.locomotion.components.controllers import (
    Controller, JointPD, OSC, VarImpedance, VarImpedanceMass, controller_from_config,
    log_action_scale, lin_action_scale, var_action_size, impedance_gains,
    _N_STIFFNESS,
)
from jax_rl.envs.locomotion.components.force_fields import (
    ForceField, NoField, MudField, field_from_config, mud_foot_force,
)

__all__ = [
    # actuation
    "Actuation", "TorqueOnly", "MotorModel", "actuation_from_config",
    # terrain
    "Terrain", "Flat", "RoughHF", "terrain_from_config", "_make_heightfield",
    # controllers
    "Controller", "JointPD", "OSC", "VarImpedance", "VarImpedanceMass",
    "controller_from_config", "log_action_scale", "lin_action_scale",
    "var_action_size", "impedance_gains",
    # force fields
    "ForceField", "NoField", "MudField", "field_from_config", "mud_foot_force",
]
