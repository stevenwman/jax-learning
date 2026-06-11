"""Decisive: put the robot's feet on the plane and ask CPU mujoco (solver.mj_data)
whether it generates plane<->foot contacts. If d.ncon>0 with a plane contact, the
MODEL is collidable (so the GPU mujoco_warp path is what's dropping it). If ncon=0
/ the pair is excluded, the model has a collision FILTER we can fix.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np                # noqa: E402
import warp as wp                 # noqa: E402
import torch                      # noqa: E402
import mujoco                     # noqa: E402
import newton.examples            # noqa: E402
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex  # noqa: E402
import mud_model                  # noqa: E402
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(meta["control"]["default_pose_policy"])
mud_model.enable()
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_ic.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"),
                     spawn_xyz=(0.0, -1.0, 0.30), yaw_pi_mult=0.0)
sys.argv = ["ic", "--viewer", "null", "--num-frames", "1", "--policy-path", CKPT,
            "--config", cfg, "--voxel-size", "0.06", "--max-iterations", "4"]
parser = newton.examples.create_parser()
for a, t in [("--config", str), ("--voxel-size", float), ("--max-iterations", int),
             ("--tolerance", float), ("--policy-path", str)]:
    parser.add_argument(a, type=t, default=None)
parser.add_argument("--precompute-frames", type=int, default=0)
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--video-fps", type=int, default=0)
parser.add_argument("--debug-forces", action="store_true")
parser.add_argument("--plot-actions", type=str, default=None)
parser.add_argument("--plot-forces", type=str, default=None)
parser.add_argument("--plot-forces-foot", type=str, default="FL_calf")
parser.add_argument("--plot-forces-mode", choices=["magnitude", "xyz"], default="magnitude")
viewer, args = newton.examples.init(parser)
example = ex.Example(viewer, args)
m, d = example.solver.mj_model, example.solver.mj_data

# place the base so the feet penetrate the plane (z=0). sweep a few base heights.
print(f"[IC] nexclude={m.nexclude}  ngeom={m.ngeom}  (plane=geom0, type={int(m.geom_type[0])})")
gt = {int(v): k.replace('mjGEOM_', '') for k, v in vars(mujoco.mjtGeom).items() if isinstance(v, int) and k.startswith('mjGEOM')}
for base_z in (0.30, 0.22, 0.15):
    d.qpos[:] = 0
    d.qpos[2] = base_z
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:19] = [0, 0.9, -1.8] * 4
    mujoco.mj_forward(m, d)
    foot_zs = [float(d.geom_xpos[g][2]) for g in range(m.ngeom) if int(m.geom_bodyid[g]) in (4, 7, 10, 13)]
    print(f"[IC] base_z={base_z}: ncon={d.ncon}  min foot/leg geom z={min(foot_zs):+.3f}")
    for c in range(min(d.ncon, 6)):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        print(f"     contact {gt.get(int(m.geom_type[g1]),'?')}(g{g1},body{int(m.geom_bodyid[g1])})"
              f" <-> {gt.get(int(m.geom_type[g2]),'?')}(g{g2},body{int(m.geom_bodyid[g2])})  dist={d.contact[c].dist:+.4f}")
# excludes involving the world body (plane is on body 0)
if m.nexclude:
    sig = m.exclude_signature
    print(f"[IC] exclude_signature (body pairs): {[ (int(s>>16), int(s&0xFFFF)) for s in sig]}")
print("[IC] => if a PLANE<->sphere/box contact appears, the MODEL collides (GPU path drops it);"
      " if ncon stays 0 with feet below 0, the model FILTERS plane<->robot.")
