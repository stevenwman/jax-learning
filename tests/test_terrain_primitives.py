"""Tests for terrain primitive generators.

One test class per terrain type.  Tests are written to the spec first; the
implementation is expected to make them pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from jax_rl.envs.terrains.base import TerrainOutput
from jax_rl.envs.terrains.primitives import (
    DiscreteObstaclesTerrainCfg,
    FlatTerrainCfg,
    InvertedPyramidStairsTerrainCfg,
    PyramidStairsTerrainCfg,
    RoughTerrainCfg,
    SlopeTerrainCfg,
    SteppingStonesTerrainCfg,
    TiltedGridTerrainCfg,
)

RNG = np.random.default_rng(42)
SIZE = (8.0, 8.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _is_identity_quat(quat: tuple | None) -> bool:
    """Return True if quat is None or (1,0,0,0)."""
    if quat is None:
        return True
    q = np.asarray(quat, dtype=float)
    return np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# 1. FlatTerrainCfg
# ---------------------------------------------------------------------------


class TestFlatTerrain:
    def test_generates_exactly_one_geom(self):
        out = FlatTerrainCfg().generate(0.0, SIZE, _rng())
        assert isinstance(out, TerrainOutput)
        assert len(out.geoms) == 1

    def test_difficulty_ignored(self):
        """Both difficulties produce identical single-geom output."""
        out0 = FlatTerrainCfg().generate(0.0, SIZE, _rng())
        out1 = FlatTerrainCfg().generate(1.0, SIZE, _rng())
        assert len(out0.geoms) == len(out1.geoms) == 1
        # Positions should be the same
        assert np.allclose(out0.geoms[0]["pos"], out1.geoms[0]["pos"])

    def test_spawn_at_correct_height(self):
        out = FlatTerrainCfg().generate(0.5, SIZE, _rng())
        assert np.isclose(out.spawn_origin[2], 0.3)

    def test_spawn_at_tile_centre(self):
        out = FlatTerrainCfg().generate(0.5, SIZE, _rng())
        assert np.isclose(out.spawn_origin[0], 0.0)
        assert np.isclose(out.spawn_origin[1], 0.0)

    def test_ground_rgba(self):
        out = FlatTerrainCfg().generate(0.0, SIZE, _rng())
        g = out.geoms[0]
        assert "rgba" in g
        assert np.allclose(g["rgba"], [0.5, 0.5, 0.5, 1.0])


# ---------------------------------------------------------------------------
# 2. RoughTerrainCfg
# ---------------------------------------------------------------------------


class TestRoughTerrain:
    def test_generates_64_geoms(self):
        out = RoughTerrainCfg().generate(0.5, SIZE, _rng())
        assert len(out.geoms) == 64

    def test_zero_difficulty_no_height_variation(self):
        out = RoughTerrainCfg().generate(0.0, SIZE, _rng())
        heights = [g["pos"][2] for g in out.geoms]
        assert np.std(heights) == pytest.approx(0.0, abs=1e-6)

    def test_full_difficulty_has_height_variation(self):
        out = RoughTerrainCfg().generate(1.0, SIZE, _rng())
        heights = [g["pos"][2] for g in out.geoms]
        assert np.std(heights) > 0.0

    def test_all_geoms_type_box(self):
        out = RoughTerrainCfg().generate(0.5, SIZE, _rng())
        for g in out.geoms:
            assert g["type"] == "box"


# ---------------------------------------------------------------------------
# 3. SlopeTerrainCfg
# ---------------------------------------------------------------------------


def _slope_surface(out):
    """Return the tilted surface geom (the one with quat or named 'slope_surface')."""
    for g in out.geoms:
        if g.get("name") == "slope_surface":
            return g
    raise AssertionError("no slope_surface geom found")


class TestSlopeTerrain:
    def test_generates_surface_and_base(self):
        out = SlopeTerrainCfg().generate(0.5, SIZE, _rng())
        # Solid base + tilted surface
        assert len(out.geoms) == 2
        names = {g.get("name") for g in out.geoms}
        assert names == {"slope_base", "slope_surface"}

    def test_zero_difficulty_identity_quat(self):
        out = SlopeTerrainCfg().generate(0.0, SIZE, _rng())
        g = _slope_surface(out)
        assert _is_identity_quat(g.get("quat"))

    def test_full_difficulty_nonidentity_quat(self):
        out = SlopeTerrainCfg().generate(1.0, SIZE, _rng())
        g = _slope_surface(out)
        quat = g.get("quat")
        assert quat is not None
        assert not _is_identity_quat(quat), f"Expected non-identity quat, got {quat}"

    def test_quat_is_unit(self):
        out = SlopeTerrainCfg().generate(1.0, SIZE, _rng())
        q = np.asarray(_slope_surface(out)["quat"])
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)

    def test_alternating_rows_have_opposite_tilt(self):
        """Even row: +tilt; odd row: -tilt. Alternating gives visually continuous surface."""
        cfg = SlopeTerrainCfg()
        even = cfg.generate(1.0, SIZE, _rng(), grid_idx=(2, 0))
        odd = cfg.generate(1.0, SIZE, _rng(), grid_idx=(3, 0))
        q_even = np.asarray(_slope_surface(even)["quat"])
        q_odd = np.asarray(_slope_surface(odd)["quat"])
        # Quaternion for rotation -angle about x equals conjugate (flip x component)
        # For rotation about x: q = (cos(a/2), sin(a/2), 0, 0), opposite: (cos(a/2), -sin(a/2), 0, 0)
        assert np.isclose(q_even[0], q_odd[0], atol=1e-6)  # same w
        assert np.isclose(q_even[1], -q_odd[1], atol=1e-6)  # opposite x component


# ---------------------------------------------------------------------------
# 4. PyramidStairsTerrainCfg
# ---------------------------------------------------------------------------


class TestPyramidStairs:
    def test_generates_at_least_5_geoms(self):
        out = PyramidStairsTerrainCfg().generate(1.0, SIZE, _rng())
        assert len(out.geoms) >= 5

    def test_zero_difficulty_all_same_z(self):
        out = PyramidStairsTerrainCfg().generate(0.0, SIZE, _rng())
        tops = [g["pos"][2] + g["size"][2] for g in out.geoms]
        # All top surfaces at same height when difficulty=0
        assert np.std(tops) == pytest.approx(0.0, abs=1e-6)

    def test_full_difficulty_z_range_nonzero(self):
        out = PyramidStairsTerrainCfg().generate(1.0, SIZE, _rng())
        tops = [g["pos"][2] + g["size"][2] for g in out.geoms]
        assert max(tops) - min(tops) > 0.01

    def test_spawn_at_outer_edge_z_near_zero(self):
        out = PyramidStairsTerrainCfg().generate(1.0, SIZE, _rng())
        # spawn_origin z should be at/near ground (outer edge)
        assert out.spawn_origin[2] < 0.5


# ---------------------------------------------------------------------------
# 5. InvertedPyramidStairsTerrainCfg
# ---------------------------------------------------------------------------


class TestInvertedPyramidStairs:
    def test_generates_at_least_5_geoms(self):
        out = InvertedPyramidStairsTerrainCfg().generate(1.0, SIZE, _rng())
        assert len(out.geoms) >= 5

    def test_spawn_at_outer_rim(self):
        """Spawn at rim (outer edge, at ground level) — robot walks down inward."""
        out = InvertedPyramidStairsTerrainCfg().generate(1.0, SIZE, _rng())
        # Spawn at outer edge along y-axis (like pyramid stairs)
        assert abs(out.spawn_origin[1]) > SIZE[1] * 0.3
        # Spawn above ground (robot stands on outer rim)
        assert out.spawn_origin[2] > 0.0

    def test_outer_ring_z_greater_than_inner_at_full_difficulty(self):
        """Bowl shape — outer ring top (rim) should be higher than inner rings."""
        cfg = InvertedPyramidStairsTerrainCfg(num_steps=5, max_step_height=0.2)
        out = cfg.generate(1.0, SIZE, _rng())
        geoms = out.geoms
        # Outer geoms have larger |pos x| or |pos y| values
        outer = [g for g in geoms if max(abs(g["pos"][0]), abs(g["pos"][1])) > SIZE[0] * 0.3]
        inner = [g for g in geoms if max(abs(g["pos"][0]), abs(g["pos"][1])) < SIZE[0] * 0.2]
        if outer and inner:
            outer_tops = [g["pos"][2] + g["size"][2] for g in outer]
            inner_tops = [g["pos"][2] + g["size"][2] for g in inner]
            assert np.mean(outer_tops) > np.mean(inner_tops)

    def test_has_pit_at_bottom(self):
        """Pit geom at the deepest point (below ground)."""
        out = InvertedPyramidStairsTerrainCfg().generate(1.0, SIZE, _rng())
        # At least one geom should be well below ground (the pit floor at center)
        bottom_z = [g["pos"][2] for g in out.geoms]
        assert min(bottom_z) < 0.0


# ---------------------------------------------------------------------------
# 6. DiscreteObstaclesTerrainCfg
# ---------------------------------------------------------------------------


class TestDiscreteObstacles:
    def test_zero_difficulty_no_obstacles(self):
        out = DiscreteObstaclesTerrainCfg().generate(0.0, SIZE, _rng())
        # Only the floor geom, no obstacles
        obstacle_geoms = [g for g in out.geoms if g.get("name", "").startswith("obstacle")]
        assert len(obstacle_geoms) == 0

    def test_full_difficulty_max_count_obstacles(self):
        cfg = DiscreteObstaclesTerrainCfg(max_count=20)
        out = cfg.generate(1.0, SIZE, _rng())
        obstacle_geoms = [g for g in out.geoms if g.get("name", "").startswith("obstacle")]
        assert len(obstacle_geoms) == 20

    def test_spawn_clearance_respected(self):
        cfg = DiscreteObstaclesTerrainCfg(max_count=20, spawn_clearance=1.5)
        out = cfg.generate(1.0, SIZE, _rng())
        ox, oy = out.spawn_origin[0], out.spawn_origin[1]
        for g in out.geoms:
            if g.get("name", "").startswith("obstacle"):
                px, py = g["pos"][0], g["pos"][1]
                dist = np.sqrt((px - ox) ** 2 + (py - oy) ** 2)
                assert dist >= 1.5, f"Obstacle at distance {dist:.3f} < clearance 1.5"

    def test_obstacle_count_scales_with_difficulty(self):
        cfg = DiscreteObstaclesTerrainCfg(max_count=10)
        out_mid = cfg.generate(0.5, SIZE, _rng())
        obstacle_mid = [g for g in out_mid.geoms if g.get("name", "").startswith("obstacle")]
        assert len(obstacle_mid) == 5


# ---------------------------------------------------------------------------
# 7. SteppingStonesTerrainCfg
# ---------------------------------------------------------------------------


class TestSteppingStones:
    def test_generates_at_least_65_geoms(self):
        """64 stones + 1 pit floor."""
        out = SteppingStonesTerrainCfg().generate(0.5, SIZE, _rng())
        assert len(out.geoms) >= 65

    def test_pit_floor_exists(self):
        out = SteppingStonesTerrainCfg().generate(0.5, SIZE, _rng())
        z_vals = [g["pos"][2] for g in out.geoms]
        assert min(z_vals) < -0.1

    def test_stone_xy_size_decreases_with_difficulty(self):
        """Higher difficulty → larger gaps → smaller stones."""
        cfg = SteppingStonesTerrainCfg()
        out0 = cfg.generate(0.0, SIZE, _rng())
        out1 = cfg.generate(1.0, SIZE, _rng())
        # Filter out pit floor (the one at very negative z)
        stones0 = [g for g in out0.geoms if g["pos"][2] > -0.4]
        stones1 = [g for g in out1.geoms if g["pos"][2] > -0.4]
        mean_size0 = np.mean([g["size"][0] for g in stones0])
        mean_size1 = np.mean([g["size"][0] for g in stones1])
        assert mean_size0 >= mean_size1

    def test_all_stone_geoms_elevated(self):
        """Stone top surfaces should be at or near ground level (z >= -0.01)."""
        out = SteppingStonesTerrainCfg().generate(0.0, SIZE, _rng())
        stones = [g for g in out.geoms if g["pos"][2] > -0.4]
        for g in stones:
            top_z = g["pos"][2] + g["size"][2]
            assert top_z >= -0.01, f"Stone top surface z={top_z:.3f} unexpectedly low"


# ---------------------------------------------------------------------------
# 8. TiltedGridTerrainCfg
# ---------------------------------------------------------------------------


class TestTiltedGrid:
    def test_generates_at_least_9_geoms(self):
        out = TiltedGridTerrainCfg().generate(0.5, SIZE, _rng())
        assert len(out.geoms) >= 9  # 3×3 grid + base

    def test_zero_difficulty_all_identity_quats(self):
        out = TiltedGridTerrainCfg().generate(0.0, SIZE, _rng())
        for g in out.geoms:
            assert _is_identity_quat(g.get("quat")), (
                f"Expected identity quat at difficulty=0, got {g.get('quat')}"
            )

    def test_full_difficulty_has_nonidentity_quats(self):
        out = TiltedGridTerrainCfg().generate(1.0, SIZE, _rng())
        non_identity = [
            g for g in out.geoms if not _is_identity_quat(g.get("quat"))
        ]
        assert len(non_identity) > 0

    def test_central_platform_always_flat(self):
        """The central tile should always have identity quat."""
        cfg = TiltedGridTerrainCfg(grid_size=(6, 6))
        out = cfg.generate(1.0, SIZE, _rng())
        # Central platform is near (0,0)
        central = [
            g for g in out.geoms
            if abs(g["pos"][0]) < 0.1 and abs(g["pos"][1]) < 0.1
        ]
        for g in central:
            assert _is_identity_quat(g.get("quat")), (
                f"Central platform should be flat, got quat {g.get('quat')}"
            )

    def test_spawn_at_tile_centre(self):
        out = TiltedGridTerrainCfg().generate(0.5, SIZE, _rng())
        assert np.isclose(out.spawn_origin[0], 0.0, atol=0.1)
        assert np.isclose(out.spawn_origin[1], 0.0, atol=0.1)
