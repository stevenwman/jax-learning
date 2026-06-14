"""GATE: does the trained go2.xml (loaded via add_mjcf) couple with the MPM mud?
Hold-pose smoke — zero policy action, just the PD holding the home pose. If the
robot stands on the mud (torso z stays > 0) the MJCF colliders register with the
MPM collider; if it free-falls (z -> negative) coupling is broken.

    PYTHONPATH=<worktree> .venv/bin/python gate_mjcf_coupling.py <ckpt> [frames]
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
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 60
GO2XML = HERE / "models/unitree_go2/go2.xml"

# 1) loader seam: home pose from meta + enable add_mjcf dispatch
meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(meta["control"]["default_pose_policy"])
mud_model.enable()
print(f"[GATE] add_mjcf seam ON, home_pose={np.round(meta['control']['default_pose_policy'], 2)}")

# 2) drop-in policy (held) + config pointed at go2.xml
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_mjcf.yaml", mjcf_model=str(GO2XML))

sys.argv = ["gate", "--viewer", "null", "--num-frames", str(NF), "--policy-path", CKPT,
            "--config", cfg, "--voxel-size", "0.05", "--max-iterations", "10"]
parser = newton.examples.create_parser()
for a, t in [("--config", str), ("--voxel-size", float), ("--max-iterations", int),
             ("--tolerance", float), ("--policy-path", str)]:
    parser.add_argument(a, type=t, default=None)
parser.add_argument("--precompute-frames", type=int, default=0)
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--video-fps", type=int, default=50)
parser.add_argument("--debug-forces", action="store_true")
parser.add_argument("--plot-actions", type=str, default=None)
parser.add_argument("--plot-forces", type=str, default=None)
parser.add_argument("--plot-forces-foot", type=str, default="FL_calf")
parser.add_argument("--plot-forces-mode", choices=["magnitude", "xyz"], default="magnitude")
viewer, args = newton.examples.init(parser)

example = ex.Example(viewer, args)
example.policy.hold = True            # zero action — isolate coupling from the policy
example._auto_forward = False

# model sanity: shapes + joint coords
m = example.model
print(f"[GATE] model: shape_count={int(m.shape_count)} joint_q_len={len(np.asarray(example.state_0.joint_q.numpy()))}")
pq = np.asarray(example.state_0.particle_q.numpy())
rb = np.asarray(example.state_0.body_q.numpy())[0]
print(f"[GATE] robot xyz={rb[:3].round(2)} | mud x[{pq[:,0].min():.1f},{pq[:,0].max():.1f}] "
      f"y[{pq[:,1].min():.1f},{pq[:,1].max():.1f}] z[{pq[:,2].min():.2f},{pq[:,2].max():.2f}]")

z0 = float(np.asarray(example.state_0.body_q.numpy())[0][2])
for f in range(NF):
    example.step()
    if f % 6 == 0 or f == NF - 1:
        z = float(np.asarray(example.state_0.body_q.numpy())[0][2])
        print(f"  f{f:3d} z={z:+.3f}", flush=True)
        if not np.isfinite(z):
            print("  non-finite — stop"); break
example.viewer.close()
z1 = float(np.asarray(example.state_0.body_q.numpy())[0][2])
finite = np.isfinite(np.asarray(example.state_0.body_q.numpy())).all()
stood = finite and z1 > 0.0
print(f"[GATE] torso z {z0:.3f}->{z1:.3f}  finite={finite}  "
      f"{'PASS — go2.xml couples with mud (stands)' if stood else 'FAIL — fell through / non-finite'}")
sys.exit(0 if stood else 1)
