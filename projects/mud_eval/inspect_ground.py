"""Why does the robot fall through the ground plane? Dump solver.mj_model geoms
(type, contype, conaffinity, body) — is there a plane, and do its collision masks
overlap the robot's foot geoms? (mujoco collides geom a,b iff
contype_a & conaffinity_b or contype_b & conaffinity_a.)
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
                     "/tmp/mud_cfg_ig.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"),
                     spawn_xyz=(0.0, -1.0, 0.45), yaw_pi_mult=0.5)
sys.argv = ["ig", "--viewer", "null", "--num-frames", "1", "--policy-path", CKPT,
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
m = example.solver.mj_model

gt = {int(v): k for k, v in vars(mujoco.mjtGeom).items() if isinstance(v, int) and k.startswith("mjGEOM")}
print(f"[IG] ngeom={m.ngeom}  (looking for a PLANE + the robot feet)")
plane_ids = []
for g in range(m.ngeom):
    t = int(m.geom_type[g]); ct = int(m.geom_contype[g]); ca = int(m.geom_conaffinity[g])
    name = gt.get(t, str(t)).replace("mjGEOM_", "")
    if name == "PLANE":
        plane_ids.append(g)
    if name == "PLANE" or g < 6 or g > m.ngeom - 4:
        print(f"  geom{g:3d} {name:8s} body={int(m.geom_bodyid[g]):2d} contype={ct} conaffinity={ca}")
# collision mask overlap: plane vs a foot geom
if plane_ids:
    p = plane_ids[0]
    # a robot foot geom: pick the last few (feet are added last on the legs)
    foot = m.ngeom - 1
    pc, pa = int(m.geom_contype[p]), int(m.geom_conaffinity[p])
    fc, fa = int(m.geom_contype[foot]), int(m.geom_conaffinity[foot])
    collides = bool((pc & fa) or (fc & pa))
    print(f"[IG] plane(geom{p}) contype={pc}/conaff={pa}  vs  geom{foot} contype={fc}/conaff={fa}"
          f"  -> mujoco-collides={collides}")
else:
    print("[IG] NO PLANE geom in the mujoco model — add_ground_plane didn't convert into SolverMuJoCo")
