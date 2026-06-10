"""GATE: prove OSC's J + M come straight from Newton's OWN cpu mujoco model
(solver.mj_model / solver.mj_data) — no shadow. Build the example (go2.xml seam),
sync live qpos/qvel into solver.mj_data, and pull mj_jacSite (foot Jacobian) +
mj_fullM (mass matrix). If the foot sites resolve and J/M are finite, M2's data
path is confirmed.
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
GO2XML = HERE / "models/unitree_go2/go2.xml"

meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(meta["control"]["default_pose_policy"])
mud_model.enable()
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_osc.yaml", mjcf_model=str(GO2XML))
sys.argv = ["gate", "--viewer", "null", "--num-frames", "1", "--policy-path", CKPT,
            "--config", cfg, "--voxel-size", "0.06", "--max-iterations", "4"]
parser = newton.examples.create_parser()
for a, t in [("--config", str), ("--voxel-size", float), ("--max-iterations", int),
             ("--tolerance", float), ("--policy-path", str)]:
    parser.add_argument(a, type=t, default=None)
for a in ["--precompute-frames", "--video-fps"]:
    parser.add_argument(a, type=int, default=0)
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--debug-forces", action="store_true")
parser.add_argument("--plot-actions", type=str, default=None)
parser.add_argument("--plot-forces", type=str, default=None)
parser.add_argument("--plot-forces-foot", type=str, default="FL_calf")
parser.add_argument("--plot-forces-mode", choices=["magnitude", "xyz"], default="magnitude")
viewer, args = newton.examples.init(parser)
example = ex.Example(viewer, args)

# --- reach the solver's OWN cpu mujoco model/data ---
sol = example.solver
m, d = sol.mj_model, sol.mj_data
print(f"[GATE] solver.mj_model: nq={m.nq} nv={m.nv} nsite={m.nsite}")

feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in feet]
print(f"[GATE] foot site ids {dict(zip(feet, sids))}  (-1 = MISSING)")
if any(s < 0 for s in sids):
    print("[GATE] FAIL — foot sites did not survive the Newton->mjModel conversion")
    sys.exit(1)

# sync live state: newton joint_q [pos3, quat xyzw, 12], mujoco qpos [pos3, quat wxyz, 12]
jq = np.asarray(example.state_0.joint_q.numpy())
jqd = np.asarray(example.state_0.joint_qd.numpy())
d.qpos[0:3] = jq[0:3]
d.qpos[3:7] = [jq[6], jq[3], jq[4], jq[5]]   # xyzw -> wxyz
d.qpos[7:7 + 12] = jq[7:19]
d.qvel[0:6] = jqd[0:6]
d.qvel[6:6 + 12] = jqd[6:18]
mujoco.mj_forward(m, d)

# foot Jacobian (linear) for FL, leg dofs [6,7,8]
jacp = np.zeros((3, m.nv))
mujoco.mj_jacSite(m, d, jacp, None, sids[0])
J_FL = jacp[:, 6:9]
# mass matrix
M = np.zeros((m.nv, m.nv))
mujoco.mj_fullM(m, M, d.qM)

print(f"[GATE] FL_foot site_xpos={d.site_xpos[sids[0]].round(3)}")
print(f"[GATE] J(FL foot, leg dofs)=\n{np.round(J_FL, 4)}")
print(f"[GATE] M leg-block (dofs 6:9)=\n{np.round(M[6:9, 6:9], 4)}")
print(f"[GATE] finite J={np.isfinite(J_FL).all()} M={np.isfinite(M).all()} | "
      f"M nv={m.nv} (expect 18 = 6 free + 12 joints)")
ok = np.isfinite(J_FL).all() and np.isfinite(M).all() and m.nv == 18
print("[GATE] PASS — J + M pulled from solver's own mujoco (no shadow needed)" if ok
      else "[GATE] check above")
sys.exit(0 if ok else 1)
