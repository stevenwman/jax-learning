"""Diagnose the M1 obs adapter + control wiring at frame 0 (no full sim).

Checks the things that make a policy fall: joint ORDER (Newton vs policy),
initial OBS sanity (gravity ~[0,0,-1], joint_pos_offset ~0 at home pose), the
first ACTION, and the PD gains the Newton model actually uses vs the policy's
training Kp/Kd.
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np
import warp as wp
import torch  # noqa
import newton.examples
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex
from mud_jax_policy import MudJaxPolicy, patched_config

CKPT = sys.argv[1]
ex.Go2Policy = MudJaxPolicy
_cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                      "/tmp/mud_cfg_patched.yaml")
sys.argv = ["diag", "--viewer", "null", "--num-frames", "1", "--policy-path", CKPT,
            "--config", _cfg, "--voxel-size", "0.06", "--max-iterations", "6"]
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

m = example.model
print("=== JOINT ORDER ===")
jk = list(m.joint_key) if hasattr(m, "joint_key") else "n/a"
print("model.joint_key:", jk)
meta = json.load(open(Path(CKPT) / "meta.json"))
print("policy joint names:", meta["control"]["policy_joint_names"])

print("\n=== PD GAINS ===")
ke = np.asarray(m.joint_target_ke.numpy()); kd = np.asarray(m.joint_target_kd.numpy())
print(f"newton joint_target_ke (uniq): {np.unique(ke)}  kd: {np.unique(kd)}")
print(f"policy training Kp={meta['control']['Kp']} Kd={meta['control']['Kd']}")

print("\n=== RAW FREE-JOINT STATE + QUAT CONVENTION ===")
jq = np.asarray(example.state_0.joint_q.numpy(), np.float32)
print("joint_q[0:7] (pos+quat):", np.round(jq[0:7], 4))
print("|joint_q[3:7]| =", round(float(np.linalg.norm(jq[3:7])), 4))
def grav_xyzw(q4):  # q=(x,y,z,w)
    from mud_jax_policy import _quat_rotate_inverse as r
    return r(q4, np.array([0,0,-1.], np.float32))
def grav_wxyz(q4):  # q=(w,x,y,z) -> reorder to xyzw
    from mud_jax_policy import _quat_rotate_inverse as r
    return r(np.array([q4[1], q4[2], q4[3], q4[0]], np.float32), np.array([0,0,-1.], np.float32))
print("gravity if xyzw:", np.round(grav_xyzw(jq[3:7]), 3), " |.|=", round(float(np.linalg.norm(grav_xyzw(jq[3:7]))),3))
print("gravity if wxyz:", np.round(grav_wxyz(jq[3:7]), 3), " |.|=", round(float(np.linalg.norm(grav_wxyz(jq[3:7]))),3))

print("\n=== INITIAL OBS (at home pose, cmd=[1,0,0]) ===")
pol = example.policy
obs = pol._build_obs(example.state_0, np.array([1.0, 0.0, 0.0], np.float32))
labels = [("gyro", 0, 3), ("accel", 3, 6), ("gravity", 6, 9),
          ("jpos_off", 9, 21), ("jvel", 21, 33), ("last_act", 33, 45), ("cmd", 45, 48)]
for name, a, b in labels:
    print(f"  {name:9s}: {np.round(obs[a:b], 3)}")
action = np.asarray(pol._select(pol._params, __import__('jax.numpy', fromlist=['asarray']).asarray(obs)))
print("  first action:", np.round(action, 3))
print("  => targets:", np.round(pol.default_pose + action * pol.action_scale, 3))
print("\nSANITY: gravity should be ~[0,0,-1] upright; jpos_off ~0 at home; jvel ~0.")
