"""Visual gate — render a mud_eval rollout to PNG frames (+ optional mp4).

ViewerGL headless (offscreen EGL). We drive a BOUNDED step()+render() loop
(NOT newton.examples.run, which never terminates headless) and pull each
framebuffer via viewer.get_frame() -> PNG (PNG avoids the imageio-ffmpeg
subprocess-fork-vs-JAX-threads deadlock that corrupts the mp4 path).

    PYTHONPATH=<worktree> .venv/bin/python record_mud.py <ckpt> <tag> [frames] [fwd|stand]
writes recordings/<tag>_fNN.png (a few) + recordings/<tag>.mp4 (best effort).
"""
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
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT, TAG = sys.argv[1], sys.argv[2]
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 40
CMD = sys.argv[4] if len(sys.argv) > 4 else "fwd"
OUTDIR = HERE / "recordings"; OUTDIR.mkdir(exist_ok=True)

ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_patched.yaml")
sys.argv = ["record", "--viewer", "gl", "--headless", "--num-frames", str(NF),
            "--policy-path", CKPT, "--config", cfg,
            "--voxel-size", "0.05", "--max-iterations", "8"]
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
example._auto_forward = (CMD == "fwd")
print(f"[REC] {Path(CKPT).name} cmd={CMD} frames={NF}", flush=True)

frames = []
for f in range(NF):
    example.step()
    example.render()                       # begin/log_state/log_points/end_frame
    fr = example.viewer.get_frame()        # wp.array, framebuffer pixels
    img = np.asarray(fr.numpy())
    if img.dtype != np.uint8:              # float [0,1] -> uint8
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    frames.append(img)
    z = float(np.asarray(example.state_0.body_q.numpy())[0][2])
    if f % 6 == 0 or f == NF - 1:
        print(f"  f{f:3d} z={z:+.3f}", flush=True)

keys = sorted(set([0, NF // 4, NF // 2, 3 * NF // 4, NF - 1]))
for i in keys:
    p = OUTDIR / f"{TAG}_f{i:02d}.png"
    iio.imwrite(p, frames[i]); print("  PNG", p.name)
try:
    iio.mimwrite(OUTDIR / f"{TAG}.mp4", frames, fps=12, macro_block_size=2)
    print("  mp4", f"{TAG}.mp4")
except Exception as e:  # noqa: BLE001
    print("  mp4 skipped:", str(e)[:80])
print("[REC] done")
