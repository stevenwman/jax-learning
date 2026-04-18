"""Base types for terrain generators.

A terrain generator is a dataclass subclass of SubTerrainCfg that implements
``generate(difficulty, size, rng) -> TerrainOutput``.  Each generator produces
tile-local MJCF geometry — the caller is responsible for assembling tiles into
a full scene.

Coordinate convention
---------------------
- Tile origin (0, 0, 0) is at the **centre** of the tile at ground level.
- ``size`` is the *full* tile extent, e.g. ``(8.0, 8.0)`` spans −4…+4 in x/y.
- Box geom ``size`` uses MuJoCo half-extents (hx, hy, hz).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class TerrainOutput:
    """The result of one terrain generator call.

    Attributes
    ----------
    geoms:
        List of dicts, each describing one MJCF ``<geom>``.  Required keys per
        dict:

        * ``"type"`` (str)  — always ``"box"`` for now.
        * ``"size"`` (tuple of 3 floats) — MuJoCo half-extents (hx, hy, hz).
        * ``"pos"``  (tuple of 3 floats) — tile-local centre position.

        Optional keys:

        * ``"quat"``     (tuple of 4 floats, wxyz) — rotation quaternion.
        * ``"rgba"``     (tuple of 4 floats)       — colour + alpha.
        * ``"friction"`` (tuple of 3 floats)       — slide / roll / spin.
        * ``"name"``     (str)                     — geom name tag.

    spawn_origin:
        Shape (3,) ndarray — tile-local position where the robot should be
        placed at episode reset (typically slightly above ground level).
    """

    geoms: list[dict[str, Any]]
    spawn_origin: np.ndarray

    def __post_init__(self) -> None:
        self.spawn_origin = np.asarray(self.spawn_origin, dtype=float)
        assert self.spawn_origin.shape == (3,), (
            f"spawn_origin must be shape (3,), got {self.spawn_origin.shape}"
        )


# ---------------------------------------------------------------------------
# Abstract base config
# ---------------------------------------------------------------------------

@dataclass
class SubTerrainCfg(ABC):
    """Abstract base class for all sub-terrain configuration objects.

    Subclass this, add your config fields as dataclass fields, and implement
    ``generate``.
    """

    @property
    def name(self) -> str:
        """Short type name for metrics/logging (e.g. 'flat', 'rough')."""
        return type(self).__name__.removesuffix("TerrainCfg").lower()

    @abstractmethod
    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
        *,
        grid_idx: tuple[int, int] = (0, 0),
    ) -> TerrainOutput:
        """Generate tile-local geometry.

        Parameters
        ----------
        difficulty:
            Float in [0, 1].  0 = easiest, 1 = hardest.
        size:
            Full tile extent ``(width_x, width_y)`` in metres.
        rng:
            NumPy random Generator for reproducible stochasticity.
        grid_idx:
            ``(row, col)`` position of this tile in the grid.  Primitives that
            vary based on neighbours (e.g., alternating slope) use this; others
            ignore it.  Defaults to ``(0, 0)`` so standalone calls still work.

        Returns
        -------
        TerrainOutput
        """
