"""Visual proof: spawn the robot to the LEFT of the mud (x=-2, off it), hold pose,
co-stepped, and record it falling through the floor — the ground plane doesn't
catch the rigid body (only the MPM mud does). Side camera so the floor line shows.
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
from mud_jax_policy import MudJaxPolicy, patched_config  # noqa: E402

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 24
OUT = HERE / "recordings"; OUT.mkdir(exist_ok=True)
SPAWN = (-2.0, 1.5, 0.45)        # LEFT of the mud (mud is x[-1,1]); on flat ground

_meta = json.load(open(Path(CKPT) / "meta.json"))
mud_model.set_home_pose(_meta["control"]["default_pose_policy"])
mud_model.enable()
mud_costep.enable(sim_substeps=5)
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_fall.yaml", mjcf_model=str(HERE / "models/unitree_go2/go2.xml"),
                     spawn_xyz=SPAWN, yaw_pi_mult=0.0)
sys.argv = ["fall", "--viewer", "gl", "--headless", "--num-frames", str(NF),
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
example.policy.hold = True
example._auto_forward = False
# side camera: stand off in +x, look back toward -x along a near-level line so the
# floor reads as a horizon and the robot visibly drops below it. mud at x[-1,1].
viewer.set_camera(wp.vec3(2.5, 1.5, 0.45), -4.0, 180.0)

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
    z = float(np.asarray(example.state_0.joint_q.numpy())[2])
    if f % 3 == 0 or f == NF - 1:
        print(f"  f{f:3d} base z={z:+.3f}", flush=True)
keys = sorted(set([0, NF // 4, NF // 2, 3 * NF // 4, NF - 1]))
for i in keys:
    p = OUT / f"falloff_f{i:02d}.png"
    iio.imwrite(p, frames[i]); print("  PNG", p.name)
try:
    iio.mimwrite(OUT / "falloff.mp4", frames, fps=10, macro_block_size=2)
    print("  mp4 falloff.mp4")
except Exception as e:  # noqa: BLE001
    print("  mp4 skipped:", str(e)[:80])
