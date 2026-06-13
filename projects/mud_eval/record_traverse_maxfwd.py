"""Record the trained policy walking from flat GROUND into the graded mud:
spawn at y<0 on the ground plane (use_mujoco_cpu so it actually collides), face
+Y, forward command, walk into thick(y0-1) -> medium(y1-2) -> thin(y2-3) mud.

    PYTHONPATH=<wt> .venv/bin/python record_traverse.py <ckpt> [frames] [spawn_y] [spawn_z]
PNG keyframes + mp4 in recordings/.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np            # noqa: E402
import warp as wp             # noqa: E402
import torch                  # noqa: E402
import imageio.v2 as iio      # noqa: E402
import newton.examples        # noqa: E402
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex  # noqa: E402
import mud_model              # noqa: E402
import mud_costep             # noqa: E402
import mud_cpu                # noqa: E402
import mud_osc                # noqa: E402
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 120
SPAWN_Y = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
SPAWN_Z = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
YAW = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5   # 0.5=face +Y (thick-first); -0.5=face -Y (thin-first)
TAG = sys.argv[6] if len(sys.argv) > 6 else "traverse"
OSC = (len(sys.argv) > 7 and sys.argv[7].lower() == "osc")   # OSC controller vs joint-PD
# Forward command vx (read BEFORE sys.argv is reassigned for the Newton parser below).
# Lower it (e.g. 0.5) to test whether pitch-forward is a max-command lunge artifact.
VX = float(sys.argv[8]) if len(sys.argv) > 8 else 1.5
# soft-OSC ckpt gains (mjx_backend _osc_soft_physical); other OSC ckpts differ
OSC_KP = np.array([1500.0, 1500.0, 2000.0]); OSC_KD = np.array([78.0, 78.0, 92.0])
OSC_TLIM = np.array([23.7, 23.7, 45.43] * 4)
OUT = HERE / "recordings"; OUT.mkdir(exist_ok=True)

_meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(_meta["control"]["default_pose_policy"])
mud_model.enable()
mud_costep.enable(sim_substeps=5)
mud_cpu.enable()                          # CPU backend -> walkable ground
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_trav.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"),
                     spawn_xyz=(0.0, SPAWN_Y, SPAWN_Z), yaw_pi_mult=YAW, osc_mode=OSC)
sys.argv = ["trav", "--viewer", "gl", "--headless", "--num-frames", str(NF),
            "--policy-path", CKPT, "--config", cfg, "--voxel-size", "0.05", "--max-iterations", "8"]
parser = newton.examples.create_parser()
for a, t in [("--config", str), ("--voxel-size", float), ("--max-iterations", int),
             ("--tolerance", float), ("--policy-path", str)]:
    parser.add_argument(a, type=t, default=None)
parser.add_argument("--precompute-frames", type=int, default=0)
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--debug-forces", action="store_true")
parser.add_argument("--plot-actions", type=str, default=None)
parser.add_argument("--plot-forces", type=str, default=None)
parser.add_argument("--plot-forces-foot", type=str, default="FL_calf")
parser.add_argument("--plot-forces-mode", choices=["magnitude", "xyz"], default="magnitude")
viewer, args = newton.examples.init(parser)
example = ex.Example(viewer, args)
mud_costep.apply(example)
if OSC:                                     # wire the operational-space controller (PD zeroed in config)
    _ad = int(_meta["action_dim"])
    # variable impedance (M3) when action > 12: tail decodes per-foot/per-axis
    # stiffness (+ damping). base gains [3000,3000,4000]/[110,110,130], s in [0.25,2].
    _var = None
    import os
    # PREFER the controller config saved in the ckpt meta (self-describing,
    # added 2026-06-13). Falls back to action_dim inference + env-var overrides
    # for older ckpts whose meta predates the osc block.
    _oscm = _meta["control"].get("osc")
    if _oscm is not None:
        _mass = bool(_oscm.get("mass_action", False))
        _use_lambda = bool(_oscm["use_op_space_inertia"])
        if _ad > 12:
            _var = dict(granularity=_oscm["stiffness_granularity"],
                        damping_action=bool(_oscm["damping_action"]),
                        s_min=_oscm["var_s_min"], s_max=_oscm["var_s_max"],
                        z_min=_oscm["var_zeta_min"], z_max=_oscm["var_zeta_max"],
                        kp_base=_oscm["kp"], kd_base=_oscm["kd"])
            if _mass:
                _var.update(mass_action=True, a_min=_oscm["var_a_min"],
                            a_max=_oscm["var_a_max"], xdd_ema=_oscm["var_xdd_ema"])
    else:
        # Legacy ckpt (no osc in meta): infer from action_dim + env-var overrides.
        # 48-d = per_axis deltas(12)+stiffness(12)+damping(12)+MASS(12).
        _mass = (_ad >= 48)
        if _ad > 12:
            _gran = "per_axis" if (_mass or _ad not in (16, 20)) else "per_foot"
            _damp = _mass or (_ad in (20, 36))
            _var = dict(granularity=_gran, damping_action=_damp,
                        s_min=0.25, s_max=2.0, z_min=0.5, z_max=2.0,
                        kp_base=[3000.0, 3000.0, 4000.0], kd_base=[110.0, 110.0, 130.0])
            if _mass:
                _var.update(mass_action=True, a_min=0.0,
                            a_max=float(os.environ.get("MASS_A_MAX", "2.0")),
                            xdd_ema=float(os.environ.get("MASS_XDD_EMA", "1.0")))
        _use_lambda = (os.environ.get("MASS_USE_LAMBDA", "0") == "1") if _mass else True
    example.control.joint_f = wp.zeros(int(example.model.joint_dof_count), dtype=wp.float32,
                                       device=example.model.device)
    _ctrl = mud_osc.MudOscController(example.solver, OSC_KP, OSC_KD, OSC_TLIM,
                                     use_op_space_inertia=_use_lambda, ridge=1e-4,
                                     home_joints=_meta["control"]["default_pose_policy"],
                                     var=_var, ctrl_dt=0.02)
    example.policy.osc_mode = True
    mud_costep.set_substep_control(lambda exmp: exmp.control.joint_f.assign(
        _ctrl.compute_joint_f(exmp.state_0, exmp.policy.last_deltas, exmp.policy.last_act)))
    _massinfo = (f", MASS A·ẍ a_max={_var['a_max']} ema={_var['xdd_ema']}"
                 if _mass else "")
    print(f"[TRAV] OSC wired ({'var-' + _var['granularity'] if _var else 'fixed-soft'}"
          f"{_massinfo}, {'Λ' if _use_lambda else 'bare'}, PD off)", flush=True)
example._auto_forward = True               # forward command (body +X)
# MAX forward velocity (vx=1.5 = cmd_a[0] upper bound). Wrap apply_control so the
# command is set immediately before it's consumed — the headless keyboard block in
# example.step() zeros self.command each frame, so setting it earlier doesn't stick.
_orig_apply_control = example.apply_control
def _maxfwd_apply_control():
    example.command[0, 0] = VX
    example.command[0, 1] = 0.0
    example.command[0, 2] = 0.0
    _orig_apply_control()
example.apply_control = _maxfwd_apply_control
print(f"[TRAV] forward command vx={VX}", flush=True)
# wide side camera framing the whole strip (mud y[0,3], runway either end); robot walks along Y
viewer.set_camera(wp.vec3(5.5, 1.5, 2.1), -26.0, 180.0)

# ── PARITY with warp/MJX training spawn ──────────────────────────────────────
# The vendored example spawns the splayed INITIAL_Q pose (hip ±0.1, thigh 0.8-1.0,
# calf -1.5) at base z~0.54 and lets it DROP (mud_model.set_home_pose silently does
# NOT override it). That start state is out-of-distribution: warp training ALWAYS
# starts SETTLED at the home pose (0, 0.9, -1.8), base z 0.27. Overwrite state_0 to
# the home pose + standing height, then re-run FK so body_q agrees. Leaves the
# initial yaw (joint_q[3:7]) and xy spawn (joint_q[0:2]) untouched. Home is uniform
# across the 4 legs, so the policy_joint_names<->Newton joint_key leg-order mismatch
# is moot (every leg gets [hip=0, thigh=0.9, calf=-1.8]).
_home_pose = np.asarray(_meta["control"]["default_pose_policy"], np.float32)
_jq = example.state_0.joint_q.numpy()
_jq[7:7 + _home_pose.shape[0]] = _home_pose   # 12 leg joints -> warp home
_jq[2] = 0.27                                 # base z -> warp standing height (no drop-in)
example.state_0.joint_q.assign(_jq)
newton.eval_fk(example.model, example.state_0.joint_q, example.state_0.joint_qd, example.state_0)

jq = np.asarray(example.state_0.joint_q.numpy())
print(f"[TRAV] spawn base xyz={jq[:3].round(2)} joints={jq[7:19].round(2)} (PARITY: warp home) "
      f"facing -Y(thin-first if yaw<0) | walking into mud", flush=True)
def _pitch_deg(quat_xyzw):
    """Body pitch (deg) about the lateral axis from the free-joint quat. >0 =
    nose-down (pitch forward). Newton free-joint quat is (x,y,z,w)."""
    x, y, z, w = [float(v) for v in quat_xyzw]
    # standard pitch (rotation about body Y): asin(2(wy - zx))
    s = 2.0 * (w * y - z * x)
    s = max(-1.0, min(1.0, s))
    return np.degrees(np.arcsin(s))

frames = []
qpos_hist = []
for f in range(NF):
    example.step()
    example.render()
    img = np.asarray(example.viewer.get_frame().numpy())
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    frames.append(img)
    q = np.asarray(example.state_0.joint_q.numpy())
    qpos_hist.append(q.copy())
    if f % 10 == 0 or f == NF - 1:
        print(f"  f{f:3d} y={q[1]:+.3f} z={q[2]:+.3f} pitch={_pitch_deg(q[3:7]):+.1f}deg", flush=True)
qpos_hist = np.array(qpos_hist)
np.savez(OUT / f"{TAG}_pose.npz", qpos=qpos_hist,
         pitch_deg=np.array([_pitch_deg(q[3:7]) for q in qpos_hist]))
print(f"  pose: {TAG}_pose.npz (qpos + pitch_deg, {NF} frames)")
keys = sorted(set(range(0, NF, max(1, NF // 8))) | {NF - 1})
for i in keys:
    p = OUT / f"{TAG}_f{i:03d}.png"
    iio.imwrite(p, frames[i])
print("  PNGs:", ", ".join(f"{TAG}_f{i:03d}" for i in keys))
try:
    iio.mimwrite(OUT / f"{TAG}.mp4", frames, fps=20, macro_block_size=2)
    print(f"  mp4 {TAG}.mp4")
except Exception as e:  # noqa: BLE001
    print("  mp4 skipped:", str(e)[:80])
