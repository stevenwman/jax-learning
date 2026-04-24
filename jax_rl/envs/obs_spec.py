"""Composable observation specification.

Obs terms are functions that receive kwargs (data, info, etc.) and return
an array. Terms are grouped into "state" (policy obs, with noise) and
"privileged_state" (critic obs, clean). Each term has an optional noise_scale.

compute_obs() iterates groups, evaluates terms, applies noise, and hstacks
per group. Returns the same dict structure as _get_obs().

For DIAYN: append ObsTerm("skill_z", lambda info, **kw: info["skill_z"])
to the "state" group. One line, no env surgery.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jp


@dataclass
class ObsTerm:
    """A single observation component."""
    name: str
    fn: Callable[..., Any]       # (**kwargs) -> jax.Array
    noise_scale: float = 0.0    # 0.0 = no noise


@dataclass
class IncludeGroup:
    """Include another group's computed output (e.g., privileged includes state)."""
    group_name: str


def schema_from_obs_groups(
    groups: dict[str, list[ObsTerm | IncludeGroup]],
) -> dict[str, list[str]]:
    """Extract a flat per-group list of term names, resolving IncludeGroup.

    Returns a JSON-serialisable dict suitable for meta.json. Used by deploy
    code to reconstruct the obs layout without hardcoding it. The deploy-side
    sensor registry maps each term name to a sensor-fetcher fn, so adding/
    removing/reordering terms in sim auto-syncs to deploy.

    Example:
        groups = {"state": [ObsTerm("gyro", ...), ObsTerm("accel", ...)],
                  "privileged_state": [IncludeGroup("state"), ObsTerm("xfrc", ...)]}
        → {"state": ["gyro", "accel"],
           "privileged_state": ["gyro", "accel", "xfrc"]}
    """
    out: dict[str, list[str]] = {}
    for group_name, terms in groups.items():
        names: list[str] = []
        for term in terms:
            if isinstance(term, IncludeGroup):
                names.extend(out[term.group_name])
            else:
                names.append(term.name)
        out[group_name] = names
    return out


def compute_obs(
    groups: dict[str, list[ObsTerm | IncludeGroup]],
    noise_level: float,
    rng: jax.Array,
    **kwargs: Any,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Compute all obs groups with noise.

    Groups are processed in dict insertion order. IncludeGroup references
    a previously computed group's output.

    Args:
        groups: Dict of {group_name: [ObsTerm | IncludeGroup, ...]}.
        noise_level: Global noise multiplier (0.0 = no noise).
        rng: JAX PRNG key. Advanced for each noisy term.
        **kwargs: Passed to each term's fn (data, info, etc.)

    Returns:
        (obs_dict, new_rng) -- obs_dict maps group names to hstacked arrays.
    """
    computed = {}

    for group_name, terms in groups.items():
        parts = []
        for term in terms:
            if isinstance(term, IncludeGroup):
                parts.append(computed[term.group_name])
            else:
                val = term.fn(**kwargs)
                if term.noise_scale > 0.0 and noise_level > 0.0:
                    rng, noise_rng = jax.random.split(rng)
                    noise = (2 * jax.random.uniform(noise_rng, shape=val.shape) - 1) \
                            * noise_level * term.noise_scale
                    val = val + noise
                parts.append(val)

        computed[group_name] = jp.concatenate(parts) if parts else jp.array([])

    return computed, rng
