"""Concrete terrain generator implementations.

Eight procedural terrain types, all subclassing SubTerrainCfg.

All coordinates are tile-local with the origin at the tile centre at ground
level.  ``size`` is the full tile extent in metres.  Box geom ``size`` uses
MuJoCo half-extents (hx, hy, hz).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .base import SubTerrainCfg, TerrainOutput

# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

_DEFAULT_FRICTION = (0.6, 0.005, 0.0001)
_GROUND_RGBA = (0.5, 0.5, 0.5, 1.0)
_STEP_RGBA = (0.6, 0.55, 0.5, 1.0)
_OBSTACLE_RGBA = (0.7, 0.4, 0.3, 1.0)
_STONE_RGBA = (0.5, 0.6, 0.5, 1.0)
_TILE_RGBA = (0.55, 0.55, 0.6, 1.0)
_PIT_RGBA = (0.3, 0.3, 0.3, 1.0)


def _box(
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    quat: tuple[float, float, float, float] | None = None,
    rgba: tuple[float, float, float, float] = _GROUND_RGBA,
    friction: tuple[float, float, float] = _DEFAULT_FRICTION,
    name: str | None = None,
) -> dict[str, Any]:
    """Convenience constructor for a box geom dict."""
    g: dict[str, Any] = {
        "type": "box",
        "pos": pos,
        "size": size,
        "rgba": rgba,
        "friction": friction,
    }
    if quat is not None:
        g["quat"] = quat
    if name is not None:
        g["name"] = name
    return g


def _identity_quat() -> tuple[float, float, float, float]:
    return (1.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 1. FlatTerrainCfg
# ---------------------------------------------------------------------------


@dataclass
class FlatTerrainCfg(SubTerrainCfg):
    """Single flat ground plane.  Difficulty is ignored."""

    thickness: float = 0.5  # half-height of the ground box

    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
        *,
        grid_idx: tuple[int, int] = (0, 0),
    ) -> TerrainOutput:
        hx, hy = size[0] / 2.0, size[1] / 2.0
        geoms = [
            _box(
                pos=(0.0, 0.0, -self.thickness),
                size=(hx, hy, self.thickness),
                rgba=_GROUND_RGBA,
            )
        ]
        return TerrainOutput(geoms=geoms, spawn_origin=np.array([0.0, 0.0, 0.3]))


# ---------------------------------------------------------------------------
# 2. RoughTerrainCfg
# ---------------------------------------------------------------------------


@dataclass
class RoughTerrainCfg(SubTerrainCfg):
    """8×8 grid of boxes with per-cell random height offsets."""

    grid: tuple[int, int] = (8, 8)
    max_height: float = 0.22
    thickness: float = 0.5

    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
        *,
        grid_idx: tuple[int, int] = (0, 0),
    ) -> TerrainOutput:
        nx, ny = self.grid
        cell_w = size[0] / nx
        cell_h = size[1] / ny
        half_w, half_h = cell_w / 2.0, cell_h / 2.0

        amp = difficulty * self.max_height
        geoms: list[dict[str, Any]] = []

        for ix in range(nx):
            for iy in range(ny):
                cx = -size[0] / 2.0 + cell_w * ix + half_w
                cy = -size[1] / 2.0 + cell_h * iy + half_h
                dz = float(rng.uniform(-amp, amp)) if amp > 0 else 0.0
                # Box top is at dz, box bottom is at dz - thickness - 0.5 margin
                hz = self.thickness / 2.0
                pos_z = dz - hz
                geoms.append(
                    _box(
                        pos=(cx, cy, pos_z),
                        size=(half_w, half_h, hz),
                        rgba=_GROUND_RGBA,
                    )
                )

        return TerrainOutput(geoms=geoms, spawn_origin=np.array([0.0, 0.0, 0.3]))


# ---------------------------------------------------------------------------
# 3. SlopeTerrainCfg
# ---------------------------------------------------------------------------


@dataclass
class SlopeTerrainCfg(SubTerrainCfg):
    """Tilted plane that alternates tilt direction between adjacent rows.

    Row-parity determines sign: even rows tilt +x (rising toward +y), odd rows
    tilt -x (rising toward -y). Adjacent rows' high edges meet at row
    boundaries, so the surface is roughly continuous.

    Surface box is scaled by ``surface_overhang`` so rotation-induced edge
    pull-in doesn't leave flat gaps between adjacent tilted tiles.

    A solid base sits below the tile to back-fill any remaining gap — robot
    can't fall through.
    """

    max_angle_deg: float = 22.0
    thickness: float = 0.2
    base_depth: float = 0.3  # solid base below the tilted surface
    surface_overhang: float = 1.15  # surface xy extent = tile_half * overhang

    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
        *,
        grid_idx: tuple[int, int] = (0, 0),
    ) -> TerrainOutput:
        hx, hy = size[0] / 2.0, size[1] / 2.0
        sign = -1.0 if grid_idx[0] % 2 == 1 else 1.0
        angle_deg = sign * difficulty * self.max_angle_deg

        if abs(angle_deg) < 1e-6:
            quat = None
        else:
            rot = Rotation.from_euler("x", angle_deg, degrees=True)
            q = rot.as_quat()  # scipy: (x, y, z, w)
            quat = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))  # wxyz

        # Solid base underneath — prevents fall-through at tile edges and gaps.
        base_hz = self.base_depth / 2.0
        # Surface is oversized so post-rotation corners still cover the tile.
        sur_hx = hx * self.surface_overhang
        sur_hy = hy * self.surface_overhang
        geoms = [
            _box(
                pos=(0.0, 0.0, -self.thickness - base_hz),
                size=(hx, hy, base_hz),
                rgba=_GROUND_RGBA,
                name="slope_base",
            ),
            _box(
                pos=(0.0, 0.0, -self.thickness),
                size=(sur_hx, sur_hy, self.thickness),
                quat=quat,
                rgba=_GROUND_RGBA,
                name="slope_surface",
            ),
        ]
        return TerrainOutput(geoms=geoms, spawn_origin=np.array([0.0, 0.0, 0.3]))


# ---------------------------------------------------------------------------
# 4. PyramidStairsTerrainCfg
# ---------------------------------------------------------------------------


@dataclass
class PyramidStairsTerrainCfg(SubTerrainCfg):
    """Concentric square step rings rising toward the centre.

    Ring 0 is the outermost (ground level), ring num_steps-1 is the centre
    platform at the top.
    """

    num_steps: int = 5
    max_step_height: float = 0.4
    thickness: float = 0.5  # extra depth below the surface

    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
        *,
        grid_idx: tuple[int, int] = (0, 0),
    ) -> TerrainOutput:
        n = self.num_steps
        step_h = difficulty * self.max_step_height
        step_w = min(size[0], size[1]) / (2.0 * n)  # radial width of each ring
        slab_depth = self.thickness

        geoms: list[dict[str, Any]] = []

        for ring in range(n):
            # Outer boundary of this ring
            outer = size[0] / 2.0 - ring * step_w
            inner = outer - step_w
            top_z = ring * step_h  # top surface of this ring
            hz = (top_z + slab_depth) / 2.0
            pos_z = top_z / 2.0 - slab_depth / 2.0

            # Four strips: north, south, east, west
            # North (y > 0)
            geoms.append(
                _box(
                    pos=(0.0, (outer + inner) / 2.0, pos_z),
                    size=(outer, step_w / 2.0, hz),
                    rgba=_STEP_RGBA,
                    name=f"stair_ring{ring}_north",
                )
            )
            # South (y < 0)
            geoms.append(
                _box(
                    pos=(0.0, -(outer + inner) / 2.0, pos_z),
                    size=(outer, step_w / 2.0, hz),
                    rgba=_STEP_RGBA,
                    name=f"stair_ring{ring}_south",
                )
            )
            # East (x > 0)
            geoms.append(
                _box(
                    pos=((outer + inner) / 2.0, 0.0, pos_z),
                    size=(step_w / 2.0, inner, hz),
                    rgba=_STEP_RGBA,
                    name=f"stair_ring{ring}_east",
                )
            )
            # West (x < 0)
            geoms.append(
                _box(
                    pos=(-(outer + inner) / 2.0, 0.0, pos_z),
                    size=(step_w / 2.0, inner, hz),
                    rgba=_STEP_RGBA,
                    name=f"stair_ring{ring}_west",
                )
            )

        # Centre platform (top)
        platform_half = step_w / 2.0
        top_z = n * step_h
        hz = (top_z + slab_depth) / 2.0
        pos_z = top_z / 2.0 - slab_depth / 2.0
        geoms.append(
            _box(
                pos=(0.0, 0.0, pos_z),
                size=(platform_half, platform_half, hz),
                rgba=_STEP_RGBA,
                name="stair_platform",
            )
        )

        return TerrainOutput(geoms=geoms, spawn_origin=np.array([0.0, size[1] / 2.0 * 0.9, 0.3]))


# ---------------------------------------------------------------------------
# 5. InvertedPyramidStairsTerrainCfg
# ---------------------------------------------------------------------------


@dataclass
class InvertedPyramidStairsTerrainCfg(SubTerrainCfg):
    """Bowl-shaped descent: outer rim at ground level, steps descend inward
    to a central pit.

    Ring 0 is the outermost (at z=0, where the robot spawns).  Each subsequent
    ring descends by ``step_h`` toward the central pit at the bottom.  Robot
    walks DOWN from rim to centre.
    """

    num_steps: int = 5
    max_step_height: float = 0.4
    thickness: float = 0.5  # extra depth below each step surface

    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
        *,
        grid_idx: tuple[int, int] = (0, 0),
    ) -> TerrainOutput:
        n = self.num_steps
        step_h = difficulty * self.max_step_height
        step_w = min(size[0], size[1]) / (2.0 * n)
        slab_depth = self.thickness

        geoms: list[dict[str, Any]] = []

        for ring in range(n):
            outer = size[0] / 2.0 - ring * step_w
            inner = outer - step_w
            # Ring 0 at z=0 (rim/ground), each inner ring descends by step_h.
            top_z = -ring * step_h
            # Slab thickness reaches from top_z down — ensure positive extent.
            hz = (slab_depth - top_z + step_h) / 2.0 if step_h > 0 else slab_depth / 2.0
            pos_z = top_z - hz

            # Four strips: north, south, east, west
            geoms.append(
                _box(
                    pos=(0.0, (outer + inner) / 2.0, pos_z),
                    size=(outer, step_w / 2.0, hz),
                    rgba=_STEP_RGBA,
                    name=f"inv_stair_ring{ring}_north",
                )
            )
            geoms.append(
                _box(
                    pos=(0.0, -(outer + inner) / 2.0, pos_z),
                    size=(outer, step_w / 2.0, hz),
                    rgba=_STEP_RGBA,
                    name=f"inv_stair_ring{ring}_south",
                )
            )
            geoms.append(
                _box(
                    pos=((outer + inner) / 2.0, 0.0, pos_z),
                    size=(step_w / 2.0, inner, hz),
                    rgba=_STEP_RGBA,
                    name=f"inv_stair_ring{ring}_east",
                )
            )
            geoms.append(
                _box(
                    pos=(-(outer + inner) / 2.0, 0.0, pos_z),
                    size=(step_w / 2.0, inner, hz),
                    rgba=_STEP_RGBA,
                    name=f"inv_stair_ring{ring}_west",
                )
            )

        # Central pit at the deepest point.
        platform_half = step_w / 2.0
        bottom_z = -n * step_h
        hz = (slab_depth - bottom_z) / 2.0
        pos_z = bottom_z - hz
        geoms.append(
            _box(
                pos=(0.0, 0.0, pos_z),
                size=(platform_half, platform_half, hz),
                rgba=_PIT_RGBA,
                name="inv_stair_floor",
            )
        )

        # Robot spawns at the outer rim, ready to walk down.
        return TerrainOutput(
            geoms=geoms,
            spawn_origin=np.array([0.0, size[1] / 2.0 * 0.9, 0.3]),
        )


# ---------------------------------------------------------------------------
# 6. DiscreteObstaclesTerrainCfg
# ---------------------------------------------------------------------------


@dataclass
class DiscreteObstaclesTerrainCfg(SubTerrainCfg):
    """Flat floor with randomly placed box obstacles."""

    max_count: int = 20
    min_size: float = 0.2   # min half-extent in x/y
    max_size: float = 0.5   # max half-extent in x/y
    max_height: float = 0.5  # max half-height
    spawn_clearance: float = 1.5
    floor_thickness: float = 0.5

    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
        *,
        grid_idx: tuple[int, int] = (0, 0),
    ) -> TerrainOutput:
        count = round(difficulty * self.max_count)
        hx, hy = size[0] / 2.0, size[1] / 2.0
        spawn_xy = (0.0, 0.0)

        geoms: list[dict[str, Any]] = [
            _box(
                pos=(0.0, 0.0, -self.floor_thickness),
                size=(hx, hy, self.floor_thickness),
                rgba=_GROUND_RGBA,
                name="floor",
            )
        ]

        placed = 0
        max_attempts = count * 100
        attempts = 0
        while placed < count and attempts < max_attempts:
            attempts += 1
            ox = float(rng.uniform(-hx + self.max_size, hx - self.max_size))
            oy = float(rng.uniform(-hy + self.max_size, hy - self.max_size))
            dist = math.sqrt((ox - spawn_xy[0]) ** 2 + (oy - spawn_xy[1]) ** 2)
            if dist < self.spawn_clearance:
                continue
            bx = float(rng.uniform(self.min_size, self.max_size))
            by_ = float(rng.uniform(self.min_size, self.max_size))
            bz = float(rng.uniform(0.05, self.max_height))
            geoms.append(
                _box(
                    pos=(ox, oy, bz),
                    size=(bx, by_, bz),
                    rgba=_OBSTACLE_RGBA,
                    name=f"obstacle_{placed}",
                )
            )
            placed += 1

        return TerrainOutput(geoms=geoms, spawn_origin=np.array([0.0, 0.0, 0.3]))


# ---------------------------------------------------------------------------
# 7. SteppingStonesTerrainCfg
# ---------------------------------------------------------------------------


@dataclass
class SteppingStonesTerrainCfg(SubTerrainCfg):
    """Grid of raised stones over a pit floor."""

    stone_count: tuple[int, int] = (8, 8)
    max_gap: float = 0.3         # fraction of cell withheld as gap
    max_height_variation: float = 0.08
    pit_depth: float = 0.5
    stone_thickness: float = 0.3

    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
        *,
        grid_idx: tuple[int, int] = (0, 0),
    ) -> TerrainOutput:
        nx, ny = self.stone_count
        cell_w = size[0] / nx
        cell_h = size[1] / ny
        gap_frac = difficulty * self.max_gap
        stone_w = cell_w * (1.0 - gap_frac)
        stone_h = cell_h * (1.0 - gap_frac)

        geoms: list[dict[str, Any]] = []

        # Pit floor
        pit_z = -self.pit_depth
        geoms.append(
            _box(
                pos=(0.0, 0.0, pit_z - self.stone_thickness / 2.0),
                size=(size[0] / 2.0, size[1] / 2.0, self.stone_thickness / 2.0),
                rgba=_PIT_RGBA,
                name="pit_floor",
            )
        )

        amp = difficulty * self.max_height_variation
        for ix in range(nx):
            for iy in range(ny):
                cx = -size[0] / 2.0 + cell_w * ix + cell_w / 2.0
                cy = -size[1] / 2.0 + cell_h * iy + cell_h / 2.0
                dz = float(rng.uniform(-amp, amp)) if amp > 0 else 0.0
                hz = self.stone_thickness / 2.0
                geoms.append(
                    _box(
                        pos=(cx, cy, dz - hz),
                        size=(stone_w / 2.0, stone_h / 2.0, hz),
                        rgba=_STONE_RGBA,
                        name=f"stone_{ix}_{iy}",
                    )
                )

        return TerrainOutput(geoms=geoms, spawn_origin=np.array([0.0, 0.0, 0.3]))


# ---------------------------------------------------------------------------
# 8. TiltedGridTerrainCfg
# ---------------------------------------------------------------------------


@dataclass
class TiltedGridTerrainCfg(SubTerrainCfg):
    """Grid of individually tilted square tiles.

    The central cell is always flat (identity quat) to provide a safe spawn
    platform.
    """

    grid_size: tuple[int, int] = (6, 6)
    max_tilt_deg: float = 25.0
    tile_thickness: float = 0.1
    base_depth: float = 0.3  # solid base below tiles — prevents fall-through gaps

    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
        *,
        grid_idx: tuple[int, int] = (0, 0),
    ) -> TerrainOutput:
        nx, ny = self.grid_size
        cell_w = size[0] / nx
        cell_h = size[1] / ny
        half_w, half_h = cell_w / 2.0, cell_h / 2.0
        hz = self.tile_thickness / 2.0

        # Central cell indices (for even grids this is the cell just above/right of centre)
        cx_idx = nx // 2
        cy_idx = ny // 2

        geoms: list[dict[str, Any]] = []

        # Solid base underneath all tiles — plugs any gaps left by tile tilt.
        # Top of base sits just below the nominal tile bottom so any fall-through
        # gap between tilted tiles lands on the base (no real hole).
        base_hz = self.base_depth / 2.0
        base_top = -hz  # directly under tiles
        geoms.append(
            _box(
                pos=(0.0, 0.0, base_top - base_hz),
                size=(size[0] / 2.0, size[1] / 2.0, base_hz),
                rgba=_TILE_RGBA,
                name="tilt_base",
            )
        )

        for ix in range(nx):
            for iy in range(ny):
                px = -size[0] / 2.0 + cell_w * ix + half_w
                py = -size[1] / 2.0 + cell_h * iy + half_h

                is_central = (ix == cx_idx or ix == cx_idx - 1) and (
                    iy == cy_idx or iy == cy_idx - 1
                )

                if is_central or difficulty < 1e-9:
                    quat = None
                else:
                    max_rad = math.radians(self.max_tilt_deg) * difficulty
                    ax = float(rng.uniform(-max_rad, max_rad))
                    ay = float(rng.uniform(-max_rad, max_rad))
                    rot = Rotation.from_euler("xy", [ax, ay])
                    q = rot.as_quat()  # scipy: (x, y, z, w)
                    quat = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))  # wxyz
                    # If effectively identity, keep None for cleaner output
                    if _is_identity_quat_tuple(quat):
                        quat = None

                geoms.append(
                    _box(
                        pos=(px, py, 0.0),
                        size=(half_w, half_h, hz),
                        quat=quat,
                        rgba=_TILE_RGBA,
                        name=f"tile_{ix}_{iy}",
                    )
                )

        return TerrainOutput(geoms=geoms, spawn_origin=np.array([0.0, 0.0, 0.3]))


def _is_identity_quat_tuple(quat: tuple) -> bool:
    q = np.asarray(quat, dtype=float)
    return bool(np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6))
