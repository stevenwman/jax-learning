"""Generate the 32 bore-wall tile prisms used by the PegInsert env.

Extracted verbatim from `spike_drop.py:ensure_bore_tiles` so we can drop the
Phase 0 spike scripts without losing the ability to regen the bore wall.

Output:
  extracted/bore_tiles/bore_00.obj ... bore_31.obj

Usage:
  uv run python jax_rl/envs/manipulation/factory/assets/peg_insert/generate_bore_tiles.py
"""
from pathlib import Path
import numpy as np
import trimesh

HERE = Path(__file__).parent
BORE_TILES = HERE / "extracted" / "bore_tiles"


def generate(force: bool = False):
    if BORE_TILES.exists() and len(list(BORE_TILES.glob("bore_*.obj"))) == 32 and not force:
        print(f"  {BORE_TILES} already has 32 tiles. Pass force=True to regen.")
        return

    BORE_TILES.mkdir(parents=True, exist_ok=True)
    for old in BORE_TILES.glob("bore_*.obj"):
        old.unlink()

    N = 32
    r_bore = 0.00405   # 4.05mm bore inner radius
    thickness = 0.001  # 1mm wall thickness
    z_start = -0.005   # 5mm below bore bottom
    z_end = 0.035      # 10mm above bore opening

    for i in range(N):
        angle = 2 * np.pi * i / N
        next_angle = 2 * np.pi * (i + 1) / N
        r_out = r_bore + thickness
        verts = []
        for z in (z_start, z_end):
            for r, ang in [
                (r_bore, angle), (r_bore, next_angle),
                (r_out, angle), (r_out, next_angle),
            ]:
                verts.append([r * np.cos(ang), r * np.sin(ang), z])
        mesh = trimesh.convex.convex_hull(np.array(verts))
        mesh.export(str(BORE_TILES / f"bore_{i:02d}.obj"))

    print(f"  Generated {N} bore tiles in {BORE_TILES}")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    generate(force=force)
