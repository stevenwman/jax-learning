"""GATE: does the robot stand on the flat GROUND plane (off the mud)? Spawn at
y=-1.0 (before the thick mud at y=0), facing +Y, hold pose, co-stepped. If it
settles at a steady z (feet on ground) the ground plane catches it and a
flat-ground -> mud traversal is possible; if z -> negative it falls through.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np                # noqa: E402
import warp as wp                 # noqa: E402
import torch                      # noqa: E402
import newton.examples            # noqa: E402
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex  # noqa: E402
import mud_model                  # noqa: E402
import mud_costep                 # noqa: E402
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SPAWN = (0.0, -1.0, 0.45)         # flat ground, 1 m before the mud (y0); facing +Y

meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(meta["control"]["default_pose_policy"])
mud_model.enable()
mud_costep.enable(sim_substeps=5)
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_ground.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"),
                     spawn_xyz=SPAWN, yaw_pi_mult=0.5)
sys.argv = ["gate", "--viewer", "null", "--num-frames", str(NF), "--policy-path", CKPT,
            "--config", cfg, "--voxel-size", "0.05", "--max-iterations", "8"]
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
mud_costep.apply(example)
example.policy.hold = True              # hold pose — isolate ground contact
example._auto_forward = False

jq = np.asarray(example.state_0.joint_q.numpy())
print(f"[GATE] spawn base joint_q xyz={jq[:3].round(3)} | facing +Y (yaw +90deg)")
z0 = float(jq[2])
for f in range(NF):
    example.step()
    if f % 4 == 0 or f == NF - 1:
        q = np.asarray(example.state_0.joint_q.numpy())
        print(f"  f{f:3d} base z={q[2]:+.3f} y={q[1]:+.3f}", flush=True)
example.viewer.close()
q = np.asarray(example.state_0.joint_q.numpy())
z1, y1 = float(q[2]), float(q[1])
stood = np.isfinite(q).all() and z1 > 0.0
print(f"[GATE] base z {z0:.3f}->{z1:.3f}  y->{y1:.3f}  "
      f"{'PASS — stands on flat ground (traversal possible)' if stood else 'FAIL — fell through (no ground collision)'}")
sys.exit(0 if stood else 1)
