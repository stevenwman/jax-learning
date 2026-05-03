"""Robot-agnostic geometry helpers for splitbelt env (foot-to-belt assignment)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BeltLayout(NamedTuple):
    """Y-range of left and right belts. X-range is unbounded (long slabs)."""
    left_y_min: float
    left_y_max: float
    right_y_min: float
    right_y_max: float


def foot_belt_id(foot_xy: jax.Array, layout: BeltLayout) -> jax.Array:
    """Map foot (x, y) positions to belt id.

    Returns int32 array of shape foot_xy.shape[:-1]: 0 = left, 1 = right, -1 = neither.
    """
    y = foot_xy[..., 1]
    in_left = (y >= layout.left_y_min) & (y <= layout.left_y_max)
    in_right = (y >= layout.right_y_min) & (y <= layout.right_y_max)
    # left=0, right=1, neither=-1; gap layout enforces mutual exclusion.
    return jnp.where(in_left, 0, jnp.where(in_right, 1, -1)).astype(jnp.int32)
