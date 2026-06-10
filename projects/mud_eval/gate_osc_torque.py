"""GATE: verify the OSC torque port + foot-site mapping in isolation, before
wiring into the co-step loop. At the home pose with zero deltas (target = current
foot) the impedance error is 0 -> tau ~ 0. A downward foot delta -> finite tau
that extends the legs. Confirms frames, site mapping, J/M, and the math.
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
import mud_osc                    # noqa: E402
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
KP = np.array([1500.0, 1500.0, 2000.0]); KD = np.array([78.0, 78.0, 92.0])
TLIM = np.array([23.7, 23.7, 45.43] * 4)   # go2.xml motor ctrlrange (hip,thigh,calf)

meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(meta["control"]["default_pose_policy"])
mud_model.enable()
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_osctau.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"))
sys.argv = ["gate", "--viewer", "null", "--num-frames", "1", "--policy-path", CKPT,
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

sol = example.solver
m, d = sol.mj_model, sol.mj_data
foot_sites, trunk, leg_dofs = mud_osc.find_legs(m)
print(f"[GATE] trunk_body={trunk}  foot_site_ids={foot_sites}  "
      f"site_bodies={[int(m.site_bodyid[s]) for s in foot_sites]}")
print(f"[GATE] leg_dof_ids=\n{leg_dofs}")

# sync home spawn state, forward, read current foot-in-trunk = nominal
jq = np.asarray(example.state_0.joint_q.numpy()); jqd = np.asarray(example.state_0.joint_qd.numpy())
mud_osc.sync_mjdata(d, jq, jqd)
mujoco.mj_forward(m, d)
R = d.xmat[trunk].reshape(3, 3); bp = d.xpos[trunk]
nominal = (d.site_xpos[foot_sites] - bp) @ R
print(f"[GATE] nominal foot-in-trunk (m)=\n{np.round(nominal, 3)}")

# (1) zero delta -> tau ~ 0
tau0 = mud_osc.osc_torque(m, d, foot_sites, leg_dofs, trunk, nominal, KP, KD, TLIM)
# (2) push feet down 2cm -> finite, nonzero, legs extend (calf torque responds)
tgt = nominal + np.array([0.0, 0.0, -0.02])
tau1 = mud_osc.osc_torque(m, d, foot_sites, leg_dofs, trunk, tgt, KP, KD, TLIM)
print(f"[GATE] tau(delta=0)  max|.|={np.abs(tau0).max():.4f}  (expect ~0)")
print(f"[GATE] tau(foot -2cm) =\n{np.round(tau1.reshape(4,3), 2)}  (FL,FR,RL,RR x hip,thigh,calf)")
ok = (np.abs(tau0).max() < 1e-3 and np.isfinite(tau1).all()
      and np.abs(tau1).max() > 0.1 and len(foot_sites) == 4
      and len(set(int(m.site_bodyid[s]) for s in foot_sites)) == 4)
print("[GATE] PASS — OSC port + site map verified" if ok else "[GATE] CHECK above")
sys.exit(0 if ok else 1)
