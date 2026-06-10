"""Cartesian impedance / OSC joystick env for Go2 (Warp backend).

Same task, observations, rewards, command sampling, termination and domain
randomization as :class:`WarpJoystick` — the ONLY change is the low-level
controller. Instead of joint-space PD, the policy's 12-d action is interpreted
as four foot position targets in the trunk frame, and a per-leg Cartesian
impedance / OSC controller (:class:`go2_warp_components.OSC`) drives the feet
there.

After the env-composition refactor this is a THIN config preset over
:class:`WarpJoystick`: the OSC default config selects the OSC controller via
``controller_from_config``; all control logic lives in the host (`_run_osc` &c)
and `go2_warp_components.OSC`. Kept as a named constructor for tests / record /
back-compatible imports.

Design notes live in
``.superpowers/specs/2026-06-08-go2-osc-impedance-design.md``.
"""
from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from jax_rl.envs.locomotion.go2_warp_joystick import (
    WarpJoystick,
    default_config as _joystick_default_config,
)


def default_config() -> config_dict.ConfigDict:
    cfg = _joystick_default_config()
    # Foot-target action scale: action ∈ ~[-1,1] maps to a ±action_scale metre
    # box around the nominal foot (trunk frame). 0.5 rad suited joint targets;
    # for Cartesian foot targets that would be absurd — feet only reach
    # ~0.10–0.15 m from nominal, so 0.12 m gives a generous-but-sane box.
    cfg.action_scale = 0.12
    cfg.osc = config_dict.create(
        # target_mode:
        #   abs_body      — target = nominal_foot + action·scale (fixed trunk
        #                   anchor). Default. Holds a stand under zero action.
        #   delta_current — target = current_foot + action·scale (integrative,
        #                   moving anchor). WARNING: has NO restoring force vs
        #                   gravity under zero action (target tracks the sagging
        #                   foot), so a passive/zero-action policy COLLAPSES and
        #                   terminates in ~0.25 s (audited). A policy must
        #                   actively command an upward bias to stand. Provided
        #                   as an exploration-dynamics ablation, not a drop-in.
        target_mode="abs_body",        # {abs_body, delta_current}
        use_op_space_inertia=True,     # True = Khatib OSC (Λ); False = Jᵀ impedance
        gravity_ff="none",             # MVP: pure impedance, spring bears load
        ridge=1e-4,                    # Λ inversion ridge
        # Cartesian gains. Tuned via a zero-action hold probe (no gravity FF, so
        # the spring alone bears body weight): kp=[800,800,1000] sagged the base
        # to the 0.18 m termination floor; [3000,3000,4000] holds ~0.27 m with
        # mild compliance. kd ≈ 2·sqrt(kp) ≈ critical damping of the unit-mass
        # OSC closed loop (q̈_foot = kp·err − kd·ẋ). Re-tune if walking is harsh.
        kp=[3000.0, 3000.0, 4000.0],
        kd=[110.0, 110.0, 130.0],
    )
    return cfg


class WarpOscJoystick(WarpJoystick):
    """Joystick velocity tracking with a per-leg Cartesian impedance controller.

    Thin preset: supplies the OSC default config (which selects the OSC
    controller). All control logic is inherited from the host + components.OSC.
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
