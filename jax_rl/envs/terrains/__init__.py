"""Procedural terrain generators for JAX RL locomotion environments.

Public API
----------
- ``TerrainOutput``                   — geometry container returned by generators
- ``SubTerrainCfg``                   — abstract base class for all generators
- All 8 concrete generators from ``primitives``
"""

from .base import SubTerrainCfg, TerrainOutput
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

__all__ = [
    "SubTerrainCfg",
    "TerrainOutput",
    "FlatTerrainCfg",
    "RoughTerrainCfg",
    "SlopeTerrainCfg",
    "PyramidStairsTerrainCfg",
    "InvertedPyramidStairsTerrainCfg",
    "DiscreteObstaclesTerrainCfg",
    "SteppingStonesTerrainCfg",
    "TiltedGridTerrainCfg",
]
