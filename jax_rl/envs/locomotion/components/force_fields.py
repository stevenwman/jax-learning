"""Force-field component — environmental wrench on the feet, applied each physics
substep INSIDE the controller's decimation scan (it depends on per-substep foot
velocity), summed with the controller torque. Controller-independent. One of the
four orthogonal axes the Go2 Warp env delegates to (see the ``components``
package docstring).

This is research-side physics (the analytic mud reduced-order model) and is the
one component the redesign wants OUT of the env library eventually — for now it
is isolated in its own module (Bleed 1, step 1). Logic moved VERBATIM from the
former ``go2_warp_components`` — no behaviour change. See
`.context/references/mud_force_port_handoff.md` for the mud physics provenance.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np


def mud_foot_force(z_foot, v, mud_height, coeff, m, b, circ):
    """Analytic mud wrench on ONE foot — world-frame (3,). Ported from the Isaac
    `mud` reduced-order model: a depth²-gated suction/resistance on world-Z plus a
    Herschel–Bulkley shear opposing the foot's velocity. Traceable (masks, no
    Python branches → vmap/jit-safe).

      depth = clip(mud_height − z_foot, 0)          in_mud = depth > 0
      ratio = (depth / mud_height)²                 # ≥0, quadratic in submersion
      sign  = −1 if v_z>0 (pulling out → SUCTION, force DOWN)
              +1 if v_z≤0 (sinking    → RESIST,  force UP)
      z     = coeff · ratio · sign
      shear = −clip(circ·depth·(log(|v|+1e-3) + m·|v|) + b·[|v_xy|>0.3], 0) · v̂
      F     = (z·ẑ + shear) · in_mud

    coeff = per-foot lumped suction/resistance stiffness; m = viscous coeff (c1);
    b = yield-stress offset (c2), only while sliding; circ = leg circumference.
    """
    depth = jp.clip(mud_height - z_foot, 0.0, None)
    in_mud = depth > 0.0
    ratio = jp.square(depth / mud_height)
    sign = jp.where(v[2] > 0.0, -1.0, 1.0)
    suction_z = coeff * ratio * sign
    speed = jp.linalg.norm(v)
    speed_xy = jp.linalg.norm(v[:2])
    stress = circ * depth * (jp.log(speed + 1e-3) + m * speed)
    stress = stress + b * (speed_xy > 0.3)
    shear = -jp.clip(stress, 0.0, None) * v / (speed + 1e-8)
    F = suction_z * jp.array([0.0, 0.0, 1.0]) + shear
    return jp.where(in_mud, F, 0.0)


class ForceField:
    """Environmental force field applied to the feet each physics substep,
    independent of the controller. Same READ-ONLY env contract as Controller.

    ``sample(env, rng)`` → per-episode info dict (called at reset); ``setup(env)``
    caches geometry; ``apply_substep(env, data, info)`` writes ``xfrc_applied``
    and returns the new data. NoField is the identity (bit-identical to the
    pre-force-field env)."""

    def setup(self, env) -> None:
        pass

    def sample(self, env, rng) -> dict:
        return {}

    def apply_substep(self, env, data, info):
        return data


class NoField(ForceField):
    """No environmental force — the default for every non-mud env."""


class MudField(ForceField):
    """Analytic mud field (see :func:`mud_foot_force`). Per-episode DR over the
    mud surface height + per-foot lumped coeff/circumference + shear viscous/yield
    constants — ranges from ``config.mud``. Applies force ONLY to submerged feet
    (``z_foot < mud_height``); torque rows left 0; trunk xfrc untouched (privileged
    obs reads it)."""

    def setup(self, env) -> None:
        feet = np.asarray(env._feet_site_id)               # (4,) FL,FR,RL,RR sites
        self._foot_site_ids = feet
        self._foot_body_ids = np.asarray(env._mj_model.site_bodyid)[feet]
        self._foot_linvel_adr = env._foot_linvel_sensor_adr  # (4,3) sensordata idx

    def sample(self, env, rng) -> dict:
        mud = env._config.mud
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

        def u(key, lohi, shape=()):
            return jax.random.uniform(key, shape,
                                      minval=float(lohi[0]), maxval=float(lohi[1]))
        return {
            "mud_height": u(k1, mud.depth_range),      # scalar mud surface height
            "mud_coeff": u(k2, mud.f_range, (4,)),     # per-foot suction/resist coeff
            "mud_m": u(k3, mud.c1_range),              # shear viscous coeff
            "mud_b": u(k4, mud.c2_range),              # shear yield offset
            "mud_circ": u(k5, mud.area_range, (4,)),   # per-foot leg circumference
        }

    def apply_substep(self, env, data, info):
        z = data.site_xpos[self._foot_site_ids, 2]          # (4,) foot world z
        v = data.sensordata[self._foot_linvel_adr]          # (4,3) foot world vel
        F = jax.vmap(mud_foot_force, in_axes=(0, 0, None, 0, None, None, 0))(
            z, v, info["mud_height"], info["mud_coeff"],
            info["mud_m"], info["mud_b"], info["mud_circ"])
        xfrc = data.xfrc_applied.at[self._foot_body_ids, :3].set(F)
        return data.replace(xfrc_applied=xfrc)


def field_from_config(config) -> ForceField:
    """No ``mud`` block → NoField; a ``mud`` block → MudField (mirrors the
    ``*_from_config`` pattern)."""
    return MudField() if getattr(config, "mud", None) is not None else NoField()
