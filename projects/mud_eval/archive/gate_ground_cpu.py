"""Test the fix: force SolverMuJoCo(use_mujoco_cpu=True) — the CPU mujoco path
that DOES generate plane<->robot contacts (proven by inspect_contacts). Spawn off
the mud (y=-1) on flat ground, hold pose. If it now STANDS, the GPU mujoco_warp
plane gap was the whole problem and we have walkable ground.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np                # noqa: E402
import warp as wp                 # noqa: E402
import torch                      # noqa: E402
import newton                     # noqa: E402
import newton.examples            # noqa: E402
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex  # noqa: E402
import mud_model                  # noqa: E402
import mud_costep                 # noqa: E402  (no-ops capture() — the CPU path can't be graph-captured)
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SPAWN_Y = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0   # -1 = flat ground, 1.5 = on mud

# force the CPU mujoco backend for the rigid solver
_orig_init = newton.solvers.SolverMuJoCo.__init__
def _cpu_init(self, model, *a, **kw):
    kw["use_mujoco_cpu"] = True
    return _orig_init(self, model, *a, **kw)
newton.solvers.SolverMuJoCo.__init__ = _cpu_init

meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(meta["control"]["default_pose_policy"])
mud_model.enable()
mud_costep.enable(sim_substeps=5)
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_gcpu.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"),
                     spawn_xyz=(0.0, SPAWN_Y, 0.45), yaw_pi_mult=0.5)
sys.argv = ["gcpu", "--viewer", "null", "--num-frames", str(NF), "--policy-path", CKPT,
            "--config", cfg, "--voxel-size", "0.06", "--max-iterations", "6"]
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
print(f"[CPU] use_mujoco_cpu={getattr(example.solver, 'use_mujoco_cpu', '?')}")

jq = np.asarray(example.state_0.joint_q.numpy())
z0 = float(jq[2])
for f in range(NF):
    example.step()
    if f % 4 == 0 or f == NF - 1:
        z = float(np.asarray(example.state_0.joint_q.numpy())[2])
        print(f"  f{f:3d} base z={z:+.3f}", flush=True)
example.viewer.close()
z1 = float(np.asarray(example.state_0.joint_q.numpy())[2])
stood = np.isfinite(z1) and z1 > 0.05
print(f"[CPU] base z {z0:.3f}->{z1:.3f}  "
      f"{'PASS — stands on flat ground! (use_mujoco_cpu fixes it)' if stood else 'still falls'}")
sys.exit(0 if stood else 1)
