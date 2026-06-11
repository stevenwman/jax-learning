"""Newton MPM force probe — measure the mud->body force the trained policy feels,
to compare against the analytic foot-force field (jax_rl/.../mud_foot_force).

Same thick-first traversal as record_traverse.py (robot walks flat ground -> thick
mud and bogs), running the VarDampingAxis DR ckpt in OSC mode. ADDS: per-frame
logging of `example.body_sand_forces` (the MPM->rigid coupling force, force=impulse
/sim_dt) on the foot + calf bodies, then prints magnitude stats and writes a
gating video. Physics UNCHANGED — read-only force tap.

    PYTHONPATH=<wt> .venv/bin/python probe_forces.py <ckpt> [frames] [spawn_y] [spawn_z]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np            # noqa: E402
import warp as wp             # noqa: E402
import torch                  # noqa: E402  (vendored scene-build interop)
import imageio.v2 as iio      # noqa: E402
import newton.examples        # noqa: E402
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex  # noqa: E402
import mud_model              # noqa: E402
import mud_costep             # noqa: E402
import mud_cpu                # noqa: E402
import mud_osc                # noqa: E402
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 150
SPAWN_Y = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
SPAWN_Z = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
YAW = 0.5                                    # face +Y (thick-first, the bog scenario)
TAG = "probe_forces"
OUT = HERE / "recordings"; OUT.mkdir(exist_ok=True)

_meta = json.load(open(Path(CKPT) / "meta.json"))
_ad = int(_meta["action_dim"])
mud_model.set_home_pose(_meta["control"]["default_pose_policy"])
mud_model.enable()
mud_costep.enable(sim_substeps=5)
mud_cpu.enable()
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_probe.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"),
                     spawn_xyz=(0.0, SPAWN_Y, SPAWN_Z), yaw_pi_mult=YAW, osc_mode=True)
sys.argv = ["probe", "--viewer", "gl", "--headless", "--num-frames", str(NF),
            "--policy-path", CKPT, "--config", cfg, "--voxel-size", "0.05", "--max-iterations", "8"]
parser = newton.examples.create_parser()
for a, t in [("--config", str), ("--voxel-size", float), ("--max-iterations", int),
             ("--tolerance", float), ("--policy-path", str)]:
    parser.add_argument(a, type=t, default=None)
parser.add_argument("--precompute-frames", type=int, default=0)
viewer, args = newton.examples.init(parser)
example = ex.Example(viewer, args)
mud_costep.apply(example)

# OSC wiring (VarDampingAxis = per-axis stiffness + damping, 36-d action).
OSC_KP = np.array([1500.0, 1500.0, 2000.0]); OSC_KD = np.array([78.0, 78.0, 92.0])
OSC_TLIM = np.array([23.7, 23.7, 45.43] * 4)
_gran = "per_foot" if _ad in (16, 20) else "per_axis"
_var = dict(granularity=_gran, damping_action=(_ad in (20, 36)),
            s_min=0.25, s_max=2.0, z_min=0.5, z_max=2.0,
            kp_base=[3000.0, 3000.0, 4000.0], kd_base=[110.0, 110.0, 130.0]) if _ad > 12 else None
example.control.joint_f = wp.zeros(int(example.model.joint_dof_count), dtype=wp.float32,
                                   device=example.model.device)
_ctrl = mud_osc.MudOscController(example.solver, OSC_KP, OSC_KD, OSC_TLIM,
                                 use_op_space_inertia=True, ridge=1e-4,
                                 home_joints=_meta["control"]["default_pose_policy"], var=_var)
example.policy.osc_mode = True
mud_costep.set_substep_control(lambda exmp: exmp.control.joint_f.assign(
    _ctrl.compute_joint_f(exmp.state_0, exmp.policy.last_deltas, exmp.policy.last_act)))
example._auto_forward = True
viewer.set_camera(wp.vec3(5.5, 1.5, 2.1), -26.0, 180.0)

# ── Force tap: identify foot + calf bodies ───────────────────────────────────
body_key = list(example.model.body_key)
W = float(example.model.body_mass.numpy().sum() * 9.81)
probe_ids = [i for i, n in enumerate(body_key)
             if ("foot" in n.lower() or "calf" in n.lower())]
print(f"[PROBE] action_dim={_ad} ({_gran}+damp) | bodyweight={W:.0f} N", flush=True)
print(f"[PROBE] probing bodies: {[body_key[i] for i in probe_ids]}", flush=True)


def read_force():
    """Return (nbody,3) linear mud force from body_sand_forces. Newton spatial
    force is [angular(3), linear(3)] — linear part is the last 3."""
    bf = np.asarray(example.body_sand_forces.numpy()).reshape(len(body_key), -1)
    return bf[:, 3:6] if bf.shape[1] == 6 else bf[:, :3]


hist = {body_key[i]: [] for i in probe_ids}
frames = []
for f in range(NF):
    example.step()
    F = read_force()
    for i in probe_ids:
        hist[body_key[i]].append(float(np.linalg.norm(F[i])))
    example.render()
    img = np.asarray(example.viewer.get_frame().numpy())
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    frames.append(img)
    if f % 15 == 0 or f == NF - 1:
        q = np.asarray(example.state_0.joint_q.numpy())
        tot = sum(hist[body_key[i]][-1] for i in probe_ids)
        print(f"  f{f:3d} y={q[1]:+.3f} z={q[2]:+.3f} | mud|F| total={tot:6.1f} N", flush=True)

# ── Stats ────────────────────────────────────────────────────────────────────
# only count frames where the robot is actually IN the mud (y > 0)
ys = []  # recompute coarse y per frame not stored; use last-half as in-mud proxy
print("\n[PROBE] per-body mud force (N), full run:")
peak_tot = 0.0
half = NF // 3  # skip the flat-ground approach (first third), measure in/at mud
for name in hist:
    arr = np.array(hist[name][half:])
    if arr.size:
        print(f"  {name:12s} mean {arr.mean():6.1f}  peak {arr.max():6.1f}")
tot_series = np.array([sum(hist[body_key[i]][t] for i in probe_ids)
                       for t in range(half, NF)])
print(f"\n[PROBE] TOTAL mud force on legs (in-mud frames): "
      f"mean {tot_series.mean():.1f} N  peak {tot_series.max():.1f} N "
      f"= {tot_series.mean()/W*100:.0f}% / {tot_series.max()/W*100:.0f}% of bodyweight ({W:.0f} N)")

keys = sorted(set(range(0, NF, max(1, NF // 8))) | {NF - 1})
for i in keys:
    iio.imwrite(OUT / f"{TAG}_f{i:03d}.png", frames[i])
try:
    iio.mimwrite(OUT / f"{TAG}.mp4", frames, fps=20, macro_block_size=2)
    print(f"[PROBE] gating video: recordings/{TAG}.mp4 ({NF} frames)")
except Exception as e:  # noqa: BLE001
    print("[PROBE] mp4 skipped:", str(e)[:80])
np.savez(OUT / f"{TAG}_forces.npz", **{k: np.array(v) for k, v in hist.items()})
print(f"[PROBE] force trace: recordings/{TAG}_forces.npz")
