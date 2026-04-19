"""Render spawn + goal markers for every (terrain_type, level) tile.

Writes to .temp/curriculum_spawns_topdown.png and .temp/curriculum_spawns_iso.png.

Green sphere = spawn position. Red sphere = goal position. Both floating above
the terrain so they're visible over pyramid apices and tilted surfaces.

Catches bugs where goal_xy lands on the wrong tile or at the wrong offset.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from jax_rl.envs.terrains.config import GO2_DEFAULT_CFG
from jax_rl.envs.terrains.generator import TerrainGenerator
from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum


MJCF_WRAPPER = """<mujoco model="curriculum_spawns">
  <compiler angle="radian" autolimits="true" />
  <option timestep="0.004" />
  <visual>
    <global offwidth="2400" offheight="2400" />
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.0 0.0 0.0" />
  </visual>
  <worldbody>
    <light pos="0 0 60" dir="0 0 -1" diffuse="0.9 0.9 0.9" />
    <camera name="topdown" pos="0 0 150" xyaxes="1 0 0 0 1 0" />
    <camera name="iso" pos="80 -80 80" xyaxes="1 1 0 -0.5 0.5 1" />
{terrain}
{markers}
  </worldbody>
</mujoco>
"""


def sphere_geom(name: str, pos, rgba) -> str:
    return (f'    <geom type="sphere" name="{name}" '
            f'size="0.4" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" '
            f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" contype="0" conaffinity="0" />')


def render(scene_path: str, cam_name: str, out: str, w: int = 2400, h: int = 2400) -> None:
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam_id = model.camera(cam_name).id
    renderer.update_scene(data, camera=cam_id)
    pixels = renderer.render()
    imageio.imwrite(out, pixels)
    print(f"  wrote {out}")


def main() -> None:
    Path(".temp").mkdir(exist_ok=True)
    env = WarpJoystickCurriculum()
    origins = np.asarray(env._terrain_origins)  # (rows, cols, 3) — tile centers
    num_rows = env._num_rows
    num_cols = env._num_cols
    terrain_names = ["rough", "pyramid_up", "pyramid_down", "tilted"]
    print(f"Grid: {num_rows} rows × {num_cols} cols = {num_rows*num_cols} tiles")

    # Generate terrain (independent of env — same cfg)
    gen = TerrainGenerator(GO2_DEFAULT_CFG)
    terrain_xml, _ = gen.generate(seed=0)

    # For each (level, type), compute spawn_world + goal_world using env helper.
    marker_lines = []
    for r in range(num_rows):
        for c in range(num_cols):
            rng = jax.random.PRNGKey(r * 100 + c)
            spawn_rng, yaw_rng = jax.random.split(rng)
            spawn_local, goal_local, _ = env._sample_spawn_goal(
                jnp.int32(c), env._tile_size, spawn_rng, yaw_rng
            )
            tile_origin = origins[r, c]
            spawn_world = np.array([
                float(spawn_local[0]) + tile_origin[0],
                float(spawn_local[1]) + tile_origin[1],
                2.5,  # elevated above apex
            ])
            goal_world = np.array([
                float(goal_local[0]) + tile_origin[0],
                float(goal_local[1]) + tile_origin[1],
                2.5,
            ])
            marker_lines.append(sphere_geom(f"spawn_r{r}_c{c}", spawn_world, (0.15, 1.0, 0.15, 0.85)))
            marker_lines.append(sphere_geom(f"goal_r{r}_c{c}",  goal_world,  (1.0, 0.15, 0.15, 0.85)))

    scene = MJCF_WRAPPER.format(terrain=terrain_xml, markers="\n".join(marker_lines))
    scene_path = Path(".temp/curriculum_spawns_scene.xml")
    scene_path.write_text(scene)

    render(str(scene_path), "topdown", ".temp/curriculum_spawns_topdown.png")
    render(str(scene_path), "iso",     ".temp/curriculum_spawns_iso.png")
    print(f"Done. {len(marker_lines) // 2} (spawn, goal) pairs rendered.")


if __name__ == "__main__":
    main()
