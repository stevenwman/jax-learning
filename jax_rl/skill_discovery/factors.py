"""Factor extractor registry for skill discovery.

A factor extractor is a named function (batch_dict) -> jax.Array that pulls
factor inputs out of a replay batch. Each FactorConfig references one by name.

Why a registry: decouples policy obs layout from auxiliary reward inputs.
Extractors can read sim_data / info that aren't in deploy actor obs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax

from jax_rl.skill_discovery.config import FactorConfig


@dataclass
class FactorExtractor:
    name: str
    source: str
    dim: int  # -1 sentinel = "any dim, skip extractor-side check"
    fn: Callable[[dict], jax.Array]


_REGISTRY: dict[str, FactorExtractor] = {}


def register_extractor(*, name: str, source: str, dim: int):
    """Decorator to register a factor extractor.

    Last-write-wins (silent overwrite) for pytest-xdist re-import safety.
    """

    def _decorator(fn: Callable[[dict], jax.Array]) -> Callable[[dict], jax.Array]:
        _REGISTRY[name] = FactorExtractor(name=name, source=source, dim=dim, fn=fn)
        return fn

    return _decorator


def get_extractor(name: str) -> FactorExtractor:
    if name not in _REGISTRY:
        raise KeyError(f"unknown extractor: {name}")
    return _REGISTRY[name]


def resolve_factor(factor: FactorConfig, batch: dict) -> jax.Array:
    """Look up factor.extractor, run it on batch, validate against factor.dim.

    Validation uses factor.dim (config = source of truth), not ext.dim.
    """
    ext = get_extractor(factor.extractor)
    out = ext.fn(batch)
    if out.shape[-1] != factor.dim:
        raise ValueError(
            f"dim mismatch: extractor {factor.extractor!r} returned "
            f"shape[-1]={out.shape[-1]}, factor.dim={factor.dim}"
        )
    return out


@register_extractor(name="actor_obs_full", source="actor_obs", dim=-1)
def _actor_obs_full(batch: dict) -> jax.Array:
    """Built-in: full actor_obs passthrough. dim=-1 sentinel handles any width."""
    return batch["obs"]
