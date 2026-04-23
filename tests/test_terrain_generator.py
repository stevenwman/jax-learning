"""Tests for TerrainGenerator and TerrainGridCfg."""

from __future__ import annotations

import numpy as np
import pytest

from jax_rl.envs.terrains import GO2_DEFAULT_CFG, TerrainGenerator, TerrainGridCfg
from jax_rl.envs.terrains.primitives import FlatTerrainCfg, RoughTerrainCfg


def test_generator_produces_mjcf_string():
    """GO2_DEFAULT_CFG generates a valid XML fragment with correct origins shape."""
    gen = TerrainGenerator(GO2_DEFAULT_CFG)
    mjcf, origins = gen.generate(seed=0)

    assert isinstance(mjcf, str)
    assert "<body" in mjcf
    assert "<geom" in mjcf
    assert origins.shape == (6, 5, 3)


def test_generator_origins_reflect_grid_layout():
    """Row 0 tiles share y; column 0 tiles share x.

    Uses a uniform FlatTerrainCfg grid so all tile-local spawn_origins are
    (0, 0, 0.3) — only the world-frame offset differs between tiles.
    """
    cfg = TerrainGridCfg(
        num_rows=4,
        tile_size=(8.0, 8.0),
        border_width=5.0,
        terrain_types=[FlatTerrainCfg(), FlatTerrainCfg(), FlatTerrainCfg()],
    )
    gen = TerrainGenerator(cfg)
    _, origins = gen.generate(seed=0)

    # All tiles in row 0 should have the same y coordinate
    row0_y = origins[0, :, 1]
    assert np.allclose(row0_y, row0_y[0]), "Row 0 tiles should all have the same y"

    # All tiles in column 0 should have the same x coordinate
    col0_x = origins[:, 0, 0]
    assert np.allclose(col0_x, col0_x[0]), "Column 0 tiles should all have the same x"


def test_generator_is_seed_reproducible():
    """Same seed produces identical MJCF output."""
    gen = TerrainGenerator(GO2_DEFAULT_CFG)
    mjcf_a, origins_a = gen.generate(seed=42)
    mjcf_b, origins_b = gen.generate(seed=42)

    assert mjcf_a == mjcf_b
    np.testing.assert_array_equal(origins_a, origins_b)


def test_different_seeds_differ():
    """Different seeds produce different MJCF output (for non-trivial terrain)."""
    gen = TerrainGenerator(GO2_DEFAULT_CFG)
    mjcf_a, _ = gen.generate(seed=0)
    mjcf_b, _ = gen.generate(seed=1)
    # RoughTerrainCfg uses rng so output must differ
    assert mjcf_a != mjcf_b


def test_generated_mjcf_parses_via_mujoco():
    """Wrap fragment in minimal MuJoCo XML and verify it loads."""
    mujoco = pytest.importorskip("mujoco")

    gen = TerrainGenerator(GO2_DEFAULT_CFG)
    mjcf_fragment, _ = gen.generate(seed=0)

    full_xml = f"""
<mujoco>
  <worldbody>
    {mjcf_fragment}
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(full_xml)
    assert model.ngeom > 0


def test_grid_layout_tile_spacing():
    """Tiles are spaced exactly tile_size apart in x and y."""
    cfg = TerrainGridCfg(
        num_rows=3,
        tile_size=(8.0, 8.0),
        border_width=5.0,
        terrain_types=[FlatTerrainCfg(), FlatTerrainCfg()],
    )
    gen = TerrainGenerator(cfg)
    _, origins = gen.generate(seed=0)

    # Adjacent columns in the same row: delta-x == tile_size_x
    dx = origins[0, 1, 0] - origins[0, 0, 0]
    assert np.isclose(dx, cfg.tile_size[0]), f"Expected dx={cfg.tile_size[0]}, got {dx}"

    # Adjacent rows in the same column: delta-y == tile_size_y
    dy = origins[1, 0, 1] - origins[0, 0, 1]
    assert np.isclose(dy, cfg.tile_size[1]), f"Expected dy={cfg.tile_size[1]}, got {dy}"


def test_num_cols_property():
    """TerrainGridCfg.num_cols reflects terrain_types length."""
    cfg = TerrainGridCfg(
        num_rows=5,
        terrain_types=[FlatTerrainCfg(), RoughTerrainCfg(), FlatTerrainCfg()],
    )
    assert cfg.num_cols == 3
