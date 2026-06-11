"""Does the ORIGINAL URDF robot (go2_description.urdf, no add_mjcf seam) stand on
the flat ground off the mud — where go2.xml fell through? Same spawn / co-step /
hold; only the robot model differs. Answers: mujoco_warp plane limitation (both
fall) vs a go2.xml-conversion issue (URDF stands, go2.xml doesn't = fixable).
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
import mud_costep                 # noqa: E402  (keep co-step identical to the go2.xml test)
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 28
SPAWN = (-2.0, 1.5, 0.45)         # LEFT of the mud, same as the go2.xml fall test

# NOTE: NO mud_model.enable() -> the example loads its bundled go2_description.urdf.
mud_costep.enable(sim_substeps=5)
ex.Go2Policy = MudJaxPolicy
# mjcf_model=None -> keep the example's urdf_relative_path; empty initial_joint_q to
# avoid the URDF joint-name posing risk (robot spawns at URDF default pose — fine
# for a ground-contact test).
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_gurdf.yaml", mjcf_model=None,
                     spawn_xyz=SPAWN, yaw_pi_mult=0.0)
import yaml  # blank the posing dict (URDF names may not match policy_joint_names)
_c = yaml.safe_load(open(cfg)); _c["policy"]["initial_joint_q"] = {}
yaml.safe_dump(_c, open(cfg, "w"))

sys.argv = ["gurdf", "--viewer", "null", "--num-frames", str(NF), "--policy-path", CKPT,
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
example.policy.hold = True
example._auto_forward = False

jq = np.asarray(example.state_0.joint_q.numpy())
print(f"[URDF] model=go2_description.urdf  spawn base xyz={jq[:3].round(3)}")
z0 = float(jq[2])
for f in range(NF):
    example.step()
    if f % 4 == 0 or f == NF - 1:
        z = float(np.asarray(example.state_0.joint_q.numpy())[2])
        print(f"  f{f:3d} base z={z:+.3f}", flush=True)
example.viewer.close()
z1 = float(np.asarray(example.state_0.joint_q.numpy())[2])
stood = np.isfinite(z1) and z1 > 0.0
print(f"[URDF] base z {z0:.3f}->{z1:.3f}  "
      f"{'STANDS on ground -> go2.xml conversion is the issue (FIXABLE)' if stood else 'ALSO falls through -> mujoco_warp plane limitation'}")
sys.exit(0 if stood else 1)
