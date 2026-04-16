"""Procedural terrain generators for JAX RL locomotion environments.

Public API
----------
- ``TerrainOutput``                   — geometry container returned by generators
- ``SubTerrainCfg``                   — abstract base class for all generators
- All 8 concrete generators from ``primitives``
- ``TerrainGridCfg``                  — grid layout configuration
- ``GO2_DEFAULT_CFG``                 — default 10×8 grid for Go2
- ``TerrainGenerator``                — assembles grid into MJCF + spawn origins
"""

from .base import SubTerrainCfg, TerrainOutput
from .config import GO2_DEFAULT_CFG, TerrainGridCfg
from .generator import TerrainGenerator
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
    "TerrainGridCfg",
    "GO2_DEFAULT_CFG",
    "TerrainGenerator",
]
