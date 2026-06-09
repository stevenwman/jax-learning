"""Pluggable env components for the Go2 Warp env — the three orthogonal axes of
variation, each a small strategy object the host (`Go2WarpEnv`) delegates to:

  - Actuation : how commanded joint torque becomes applied torque (+ per-joint
                reflected inertia). TorqueOnly | MotorModel.
  - Terrain   : which scene + post-load model mutation. (added in a later stage)
  - Controller: action -> joint torque each substep.            (later stage)

Design: ``.superpowers/specs/2026-06-09-go2-env-composition.md``. Logic is
extracted VERBATIM from the former subclasses/flags so there is no behaviour
change — only where the code lives.
"""
from __future__ import annotations

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
