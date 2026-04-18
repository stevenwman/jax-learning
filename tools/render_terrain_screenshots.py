"""Render screenshots of the terrain grid headlessly.

Usage:
    MUJOCO_GL=egl uv run python tools/render_terrain_screenshots.py

Writes PNG files to .temp/terrain_*.png — top-down, iso, and per-column side views.
Go2-scale reference boxes placed on each tile for size reference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import imageio
import mujoco
import numpy as np

from jax_rl.envs.terrains.config import GO2_DEFAULT_CFG
from jax_rl.envs.terrains.generator import TerrainGenerator


# Go2 approximate bounding box (L × W × H standing)
GO2_HALF = (0.325, 0.15, 0.15)  # half-extents


MJCF_WRAPPER = """<mujoco model="terrain_preview">
  <compiler angle="radian" autolimits="true" />
  <option timestep="0.004" />
  <visual>
    <global offwidth="1920" offheight="1800" />
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.0 0.0 0.0" />
  </visual>
  <worldbody>
    <light pos="0 0 30" dir="0 0 -1" diffuse="0.9 0.9 0.9" />
    <camera name="topdown" pos="0 0 80" xyaxes="1 0 0 0 1 0" />
    <camera name="iso" pos="40 -40 35" xyaxes="1 1 0 -0.5 0.5 1" />
{per_col_cams}
{terrain}
{scale_refs}
  </worldbody>
</mujoco>
"""


def render(model_data, cam_name: str, out: str, w: int = 1920, h: int = 1080) -> None:
    model, data = model_data
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam_id = model.camera(cam_name).id
    renderer.update_scene(data, camera=cam_id)
    pixels = renderer.render()
    imageio.imwrite(out, pixels)
    print(f"  wrote {out}")


def main() -> None:
    Path(".temp").mkdir(exist_ok=True)
    gen = TerrainGenerator(GO2_DEFAULT_CFG)
    terrain_xml, origins = gen.generate(seed=0)

    sx, sy = GO2_DEFAULT_CFG.tile_size
    nr = GO2_DEFAULT_CFG.num_rows
    nc = GO2_DEFAULT_CFG.num_cols
    grid_h = nr * sy  # y-extent of grid

    # Per-column close-up cameras.
    # Column extends in Y from -grid_h/2 to +grid_h/2, at world X = origins[0, c, 0].
    # Camera sits to the +X side of the column, looking at column center.
    cams = []
    for c, tcfg in enumerate(GO2_DEFAULT_CFG.terrain_types):
        tile_x = float(origins[0, c, 0])
        cams.append(
            f'    <camera name="col_{c}_{tcfg.name}" '
            f'pos="{tile_x + sx * 0.6} 0 {sy * 0.75}" '
            f'xyaxes="0 -1 0 0.3 0 1" fovy="50" />'
        )

    # Scale reference: place a Go2-sized red box at each tile's spawn origin,
    # so every row/col of every terrain has a size reference.
    hx, hy, hz = GO2_HALF
    scale_refs = []
    for r in range(nr):
        for c in range(nc):
            ox, oy, oz = origins[r, c]
            # Place box so its bottom sits near the spawn z
            box_z = float(oz) + hz
            scale_refs.append(
                f'    <geom type="box" name="scale_r{r}_c{c}" '
                f'size="{hx} {hy} {hz}" pos="{float(ox)} {float(oy)} {box_z}" '
                f'rgba="0.85 0.1 0.1 1.0" />'
            )

    full = MJCF_WRAPPER.format(
        terrain=terrain_xml,
        per_col_cams="\n".join(cams),
        scale_refs="\n".join(scale_refs),
    )
    scene_path = Path(".temp/terrain_screenshots_scene.xml")
    scene_path.write_text(full)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    print(f"Loaded: {model.ngeom} geoms, {model.ncam} cameras")

    # Renders: topdown + iso. Scale boxes on every tile provide size reference.
    render((model, data), "topdown", ".temp/terrain_topdown.png")
    render((model, data), "iso", ".temp/terrain_iso.png")


if __name__ == "__main__":
    main()
