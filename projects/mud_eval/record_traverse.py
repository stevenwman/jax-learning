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
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 120
SPAWN_Y = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
SPAWN_Z = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
YAW = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5   # 0.5=face +Y (thick-first); -0.5=face -Y (thin-first)
TAG = sys.argv[6] if len(sys.argv) > 6 else "traverse"
OUT = HERE / "recordings"; OUT.mkdir(exist_ok=True)

_meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(_meta["control"]["default_pose_policy"])
mud_model.enable()
mud_costep.enable(sim_substeps=5)
mud_cpu.enable()                          # CPU backend -> walkable ground
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_trav.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"),
                     spawn_xyz=(0.0, SPAWN_Y, SPAWN_Z), yaw_pi_mult=YAW)
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
example._auto_forward = True               # forward command (body +X)
# wide side camera framing the whole strip (mud y[0,3], runway either end); robot walks along Y
viewer.set_camera(wp.vec3(5.5, 1.5, 2.1), -26.0, 180.0)

jq = np.asarray(example.state_0.joint_q.numpy())
print(f"[TRAV] spawn base xyz={jq[:3].round(2)} facing +Y | walking into mud y0->y3", flush=True)
frames = []
for f in range(NF):
    example.step()
    example.render()
    img = np.asarray(example.viewer.get_frame().numpy())
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    frames.append(img)
    if f % 10 == 0 or f == NF - 1:
        q = np.asarray(example.state_0.joint_q.numpy())
        print(f"  f{f:3d} y={q[1]:+.3f} z={q[2]:+.3f}", flush=True)
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
