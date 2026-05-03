"""Per-episode belt-speed schedule samplers for SplitbeltTreadmill envs.

Each factory: (rng, T, **cfg) -> jax.Array of shape (T, 2) with columns (vL, vR).
Pure functions; no env knowledge. See spec S§5.4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def tied(rng: jax.Array, T: int, *, v: float = 0.5) -> jax.Array:
    """Both belts at constant speed v for the whole episode."""
    del rng  # deterministic
    return jnp.full((T, 2), v, dtype=jnp.float32)


def split_constant(
    rng: jax.Array, T: int, *, vL: float = 0.5, vR: float = 1.0
) -> jax.Array:
    """Asymmetric belts at constant (vL, vR) for the whole episode (A2 fixed-context)."""
    del rng
    row = jnp.array([vL, vR], dtype=jnp.float32)
    return jnp.tile(row, (T, 1))


def tied_split_tied(
    rng: jax.Array,
    T: int,
    *,
    v_warm: float = 0.5,
    vL_split: float = 0.5,
    vR_split: float = 1.0,
    t1: int = 200,
    t2: int = 600,
) -> jax.Array:
    """Three-phase schedule for A1 (within-episode adaptation) protocol.

    Phase 1: 0 <= t < t1, tied at v_warm.
    Phase 2: t1 <= t < t1+t2, split at (vL_split, vR_split).
    Phase 3: t1+t2 <= t < T, tied at v_warm.
    """
    del rng
    idx = jnp.arange(T)
    in_split = (idx >= t1) & (idx < t1 + t2)
    vL = jnp.where(in_split, vL_split, v_warm).astype(jnp.float32)
    vR = jnp.where(in_split, vR_split, v_warm).astype(jnp.float32)
    return jnp.stack([vL, vR], axis=-1)


def random_per_episode(
    rng: jax.Array,
    T: int,
    *,
    v_range: tuple[float, float] = (0.3, 1.5),
    ratio_range: tuple[float, float] = (0.5, 2.0),
) -> jax.Array:
    """Sample (vL, vR) once at episode start; constant for whole episode (A2/A3 training)."""
    k1, k2 = jax.random.split(rng)
    v = jax.random.uniform(k1, (), minval=v_range[0], maxval=v_range[1])
    ratio = jax.random.uniform(k2, (), minval=ratio_range[0], maxval=ratio_range[1])
    vL = v
    vR = v * ratio
    row = jnp.stack([vL, vR]).astype(jnp.float32)
    return jnp.tile(row, (T, 1))


def continual_phase(
    rng: jax.Array,
    T: int,
    *,
    phase_id: int,
    v_warm: float = 0.5,
    vL_split: float = 0.5,
    vR_split: float = 1.0,
) -> jax.Array:
    """A4 continual-learning schedule: phase 0 = tied warmup, phase 1 = split fine-tune."""
    if phase_id == 0:
        return tied(rng, T, v=v_warm)
    if phase_id == 1:
        return split_constant(rng, T, vL=vL_split, vR=vR_split)
    raise ValueError(f"continual_phase: phase_id must be 0 or 1, got {phase_id}")


_FACTORIES = {
    "tied": tied,
    "split_constant": split_constant,
    "tied_split_tied": tied_split_tied,
    "random_per_episode": random_per_episode,
    "continual_phase": continual_phase,
}


def sample_schedule(rng: jax.Array, T: int, *, kind: str, params: dict) -> jax.Array:
    """Dispatch to the appropriate sampler by string kind."""
    if kind not in _FACTORIES:
        raise ValueError(
            f"unknown schedule kind {kind!r}; valid: {sorted(_FACTORIES)}"
        )
    return _FACTORIES[kind](rng, T, **params)
