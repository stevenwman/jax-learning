"""Generate a terrain MJCF and write to file. Open in MuJoCo viewer to inspect.

Usage:
    uv run python tools/generate_terrain_preview.py [--seed N] [--out PATH]

After generating, open with:
    MUJOCO_GL=glfw uv run python -m mujoco.viewer --mjcf /tmp/terrain_preview.xml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_rl.envs.terrains.generator import TerrainGenerator
from jax_rl.envs.terrains.config import GO2_DEFAULT_CFG


MJCF_WRAPPER = """<mujoco model="terrain_preview">
  <compiler angle="radian" autolimits="true" />
  <option timestep="0.004" />
  <visual>
    <global offwidth="1920" offheight="1080" />
  </visual>
  <worldbody>
    <light pos="0 0 30" dir="0 0 -1" diffuse="0.7 0.7 0.7" />
{terrain}
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="/tmp/terrain_preview.xml")
    args = parser.parse_args()

    gen = TerrainGenerator(GO2_DEFAULT_CFG)
    terrain_xml, origins = gen.generate(seed=args.seed)
    full_xml = MJCF_WRAPPER.format(terrain=terrain_xml)
    out_path = Path(args.out)
    out_path.write_text(full_xml)

    print(f"Wrote {args.out}")
    print(f"Grid: {origins.shape[0]} rows x {origins.shape[1]} cols")
    print(f"Terrain types: {[t.name for t in GO2_DEFAULT_CFG.terrain_types]}")
    print(f"Open with: MUJOCO_GL=glfw uv run python -m mujoco.viewer --mjcf {args.out}")


if __name__ == "__main__":
    main()
