"""TerrainGridCfg — configuration for a full multi-row terrain grid.

Composes SubTerrainCfg instances (columns) with a difficulty progression
across rows (row 0 = easiest, row num_rows-1 = hardest).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .primitives import (
    DiscreteObstaclesTerrainCfg,
    FlatTerrainCfg,
    InvertedPyramidStairsTerrainCfg,
    PyramidStairsTerrainCfg,
    RoughTerrainCfg,
    SlopeTerrainCfg,
    SteppingStonesTerrainCfg,
    TiltedGridTerrainCfg,
)


@dataclass
class TerrainGridCfg:
    """Configuration for a rectangular terrain grid.

    Each column is one terrain type; each row is one difficulty level.
    Row 0 is easiest (difficulty=0), row num_rows-1 is hardest (difficulty=1).

    Attributes
    ----------
    num_rows:
        Number of difficulty levels.
    tile_size:
        (width_x, width_y) of each tile in metres.
    border_width:
        Extra flat border around the grid to prevent fall-off.
    terrain_types:
        List of SubTerrainCfg instances — one per column.
    """

    num_rows: int = 10
    tile_size: tuple[float, float] = (8.0, 8.0)
    border_width: float = 20.0
    terrain_types: list = field(default_factory=list)

    @property
    def num_cols(self) -> int:
        return len(self.terrain_types)


# ---------------------------------------------------------------------------
# Default config for Go2 — all 8 terrain types
# ---------------------------------------------------------------------------

GO2_DEFAULT_CFG = TerrainGridCfg(
    num_rows=10,
    tile_size=(8.0, 8.0),
    border_width=20.0,
    terrain_types=[
        FlatTerrainCfg(),
        RoughTerrainCfg(),
        SlopeTerrainCfg(),
        PyramidStairsTerrainCfg(),
        InvertedPyramidStairsTerrainCfg(),
        DiscreteObstaclesTerrainCfg(),
        SteppingStonesTerrainCfg(),
        TiltedGridTerrainCfg(),
    ],
)
