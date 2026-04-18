"""TerrainGenerator — assembles a grid of terrain tiles into a single MJCF fragment.

Usage
-----
    from jax_rl.envs.terrains import TerrainGenerator, GO2_DEFAULT_CFG

    gen = TerrainGenerator(GO2_DEFAULT_CFG)
    mjcf, origins = gen.generate(seed=42)
    # mjcf: XML string (a <body name="terrain"> element)
    # origins: np.ndarray, shape (num_rows, num_cols, 3)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import TerrainGridCfg


class TerrainGenerator:
    """Generates a full terrain grid as an MJCF string plus spawn origins.

    Parameters
    ----------
    cfg:
        TerrainGridCfg describing the grid layout and terrain types.
    """

    def __init__(self, cfg: TerrainGridCfg) -> None:
        self.cfg = cfg

    def generate(self, seed: int = 0) -> tuple[str, np.ndarray]:
        """Build the full grid.

        Parameters
        ----------
        seed:
            Integer seed for reproducible RNG.

        Returns
        -------
        mjcf:
            XML string — a ``<body name="terrain">`` element containing all
            tile geoms, ready to inject inside ``<worldbody>``.
        origins:
            Shape ``(num_rows, num_cols, 3)`` — world-frame spawn positions.
        """
        cfg = self.cfg
        rng = np.random.default_rng(seed)

        num_rows = cfg.num_rows
        num_cols = cfg.num_cols
        tile_sx, tile_sy = cfg.tile_size

        grid_w = num_cols * tile_sx
        grid_h = num_rows * tile_sy

        # Centre of the first tile (col 0, row 0)
        x0 = -grid_w / 2.0 + tile_sx / 2.0
        y0 = -grid_h / 2.0 + tile_sy / 2.0

        origins = np.zeros((num_rows, num_cols, 3), dtype=float)
        geom_lines: list[str] = []
        geom_idx = 0  # global counter for unique geom names

        for r in range(num_rows):
            difficulty = r / max(num_rows - 1, 1)
            tile_y = y0 + r * tile_sy

            for c in range(num_cols):
                tile_x = x0 + c * tile_sx
                terrain_type = cfg.terrain_types[c]

                output = terrain_type.generate(
                    difficulty, cfg.tile_size, rng, grid_idx=(r, c)
                )

                # Offset tile-local geoms to world coordinates
                for geom in output.geoms:
                    world_geom = dict(geom)
                    local_pos = geom["pos"]
                    world_geom["pos"] = (
                        local_pos[0] + tile_x,
                        local_pos[1] + tile_y,
                        local_pos[2],
                    )
                    # Override any name from the tile generator with a unique name
                    world_geom["name"] = f"t{geom_idx}"
                    geom_idx += 1
                    geom_lines.append(_geom_to_xml(world_geom))

                # Store world-frame spawn origin
                so = output.spawn_origin
                origins[r, c] = [so[0] + tile_x, so[1] + tile_y, so[2]]

        # Border: four flat strips AROUND the grid (not underneath it) — avoids
        # covering up pits/descending terrain beneath tile ground level.
        bw = cfg.border_width
        outer_hx = grid_w / 2.0 + bw
        outer_hy = grid_h / 2.0 + bw
        inner_hx = grid_w / 2.0
        inner_hy = grid_h / 2.0
        border_strips = [
            # North strip: +y side, full outer x extent
            {"pos": (0.0, (outer_hy + inner_hy) / 2.0, -0.05),
             "size": (outer_hx, bw / 2.0, 0.05)},
            # South strip
            {"pos": (0.0, -(outer_hy + inner_hy) / 2.0, -0.05),
             "size": (outer_hx, bw / 2.0, 0.05)},
            # East strip: +x side, inner y extent (avoid overlap with N/S)
            {"pos": ((outer_hx + inner_hx) / 2.0, 0.0, -0.05),
             "size": (bw / 2.0, inner_hy, 0.05)},
            # West strip
            {"pos": (-(outer_hx + inner_hx) / 2.0, 0.0, -0.05),
             "size": (bw / 2.0, inner_hy, 0.05)},
        ]
        for i, strip in enumerate(border_strips):
            geom: dict[str, Any] = {
                "name": f"t{geom_idx + i}",
                "type": "box",
                "pos": strip["pos"],
                "size": strip["size"],
                "rgba": (0.4, 0.4, 0.4, 1.0),
            }
            geom_lines.append(_geom_to_xml(geom))

        inner = "\n    ".join(geom_lines)
        mjcf = f'<body name="terrain">\n    {inner}\n  </body>'

        return mjcf, origins


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


_MIN_GEOM_HALF_EXTENT = 1e-4  # MuJoCo requires strictly positive half-extents


def _fmt_floats(vals: tuple | list) -> str:
    """Format a sequence of floats as space-separated string."""
    return " ".join(f"{v:.6g}" for v in vals)


def _clamp_size(size: tuple | list) -> tuple:
    """Clamp all half-extents to a minimum positive value.

    Terrain primitives can produce near-zero sizes due to floating-point
    cancellation (e.g. PyramidStairs innermost ring).  MuJoCo requires
    strictly positive half-extents, so we clamp here.
    """
    return tuple(max(v, _MIN_GEOM_HALF_EXTENT) for v in size)


def _geom_to_xml(g: dict[str, Any]) -> str:
    """Serialize a geom dict to an MJCF ``<geom .../>`` string."""
    parts: list[str] = []

    parts.append(f'name="{g["name"]}"')
    parts.append(f'type="{g["type"]}"')
    parts.append(f'size="{_fmt_floats(_clamp_size(g["size"]))}"')
    parts.append(f'pos="{_fmt_floats(g["pos"])}"')

    if "quat" in g and g["quat"] is not None:
        parts.append(f'quat="{_fmt_floats(g["quat"])}"')

    if "rgba" in g:
        parts.append(f'rgba="{_fmt_floats(g["rgba"])}"')

    if "friction" in g:
        parts.append(f'friction="{_fmt_floats(g["friction"])}"')

    attrs = " ".join(parts)
    return f"<geom {attrs}/>"
