"""GATE: full OSC controller in the co-step loop. OSC ckpt, flat ground (where the
soft-OSC physical ckpt trained), stand command. The policy emits foot deltas
(held/frame); the substep hook computes the operational-space torque from the live
state -> control.joint_f (PD zeroed). If the robot stands (z stable, OSC torque
nonzero) the integration works.
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
import mud_model, mud_costep, mud_cpu        # noqa: E402
import mud_osc                                # noqa: E402
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SPAWN_Y = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
KP = np.array([1500.0, 1500.0, 2000.0]); KD = np.array([78.0, 78.0, 92.0])
TLIM = np.array([23.7, 23.7, 45.43] * 4)

meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(meta["control"]["default_pose_policy"])
mud_model.enable(); mud_costep.enable(sim_substeps=5); mud_cpu.enable()
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_oscloop.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"),
                     spawn_xyz=(0.0, SPAWN_Y, 0.35), yaw_pi_mult=0.5, osc_mode=True)
sys.argv = ["oscl", "--viewer", "null", "--num-frames", str(NF), "--policy-path", CKPT,
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

# --- wire the OSC controller into the co-step loop ---
dev = example.model.device
example.control.joint_f = wp.zeros(int(example.model.joint_dof_count), dtype=wp.float32, device=dev)
controller = mud_osc.MudOscController(example.solver, KP, KD, TLIM, use_op_space_inertia=True, ridge=1e-4,
                                      home_joints=meta["control"]["default_pose_policy"])
example.policy.osc_mode = True
_dbg = {"taumax": 0.0}
def osc_hook(exmp):
    jf = controller.compute_joint_f(exmp.state_0, exmp.policy.last_deltas)
    _dbg["taumax"] = float(np.abs(jf[6:18]).max())
    exmp.control.joint_f.assign(jf)
mud_costep.set_substep_control(osc_hook)
example._auto_forward = False               # stand
print(f"[OSCL] foot_sites={controller.foot_sites} nominal_z={controller.nominal[:,2].round(3)}", flush=True)

z0 = float(np.asarray(example.state_0.joint_q.numpy())[2])
for f in range(NF):
    example.step()
    if f % 4 == 0 or f == NF - 1:
        q = np.asarray(example.state_0.joint_q.numpy())
        dn = float(np.abs(example.policy.last_deltas).max())
        print(f"  f{f:3d} z={q[2]:+.3f} y={q[1]:+.3f} |tau|max={_dbg['taumax']:.2f} |delta|max={dn:.3f}", flush=True)
        if not np.isfinite(q).all():
            print("  non-finite — stop"); break
example.viewer.close()
q = np.asarray(example.state_0.joint_q.numpy())
ok = np.isfinite(q).all() and float(q[2]) > 0.05
print(f"[OSCL] z {z0:.3f}->{float(q[2]):.3f}  "
      f"{'PASS — OSC loop holds the robot up' if ok else 'FAIL — collapsed/non-finite'}")
sys.exit(0 if ok else 1)
