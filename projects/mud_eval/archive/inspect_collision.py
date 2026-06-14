"""Show the CURRENT robot's collision geometry — numeric summary + a close-up
render of only the collider shapes (vs the visual mesh). Helps decide whether
to redirect from the example's go2_description.urdf to the trained go2.xml.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np
import warp as wp
import torch  # noqa
import imageio.v2 as iio
import newton, newton.examples
import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as ex
from mud_jax_policy import MudJaxPolicy, patched_config

CKPT = sys.argv[1]
ex.Go2Policy = MudJaxPolicy
cfg = patched_config(CKPT, HERE / "vendor/newton/examples/mpm/mpm_go2_multi/config.yaml",
                     "/tmp/mud_cfg_patched.yaml")
sys.argv = ["insp", "--viewer", "gl", "--headless", "--num-frames", "1",
            "--policy-path", CKPT, "--config", cfg, "--voxel-size", "0.06", "--max-iterations", "4"]
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
m = example.model


def arr(a):
    return np.asarray(a.numpy()) if hasattr(a, "numpy") else np.asarray(a)


st = arr(m.shape_type); flags = arr(m.shape_flags); sb = arr(m.shape_body); ss = arr(m.shape_scale)
bk = list(m.body_key)
COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)
print(f"=== COLLISION SHAPES (of {m.shape_count} total shapes) ===")
ncol = 0
from collections import Counter
by_type = Counter()
for i in range(int(m.shape_count)):
    if int(flags[i]) & COLLIDE:
        ncol += 1
        name = newton.GeoType(int(st[i])).name
        by_type[name] += 1
        body = bk[int(sb[i])] if int(sb[i]) >= 0 else "world"
        print(f"  {body:18s} {name:11s} scale={np.round(ss[i], 3)}")
print(f"--> {ncol} collider shapes; by type: {dict(by_type)}")

# render: robot at ~[0,1.5,0.4]; put camera close, look at it
OUT = HERE / "recordings"; OUT.mkdir(exist_ok=True)
viewer.set_camera(wp.vec3(1.4, 1.5, 0.7), -18.0, 180.0)
for show_col, tag in [(False, "visual"), (True, "collision")]:
    viewer.show_collision = show_col
    viewer.show_visual = not show_col
    viewer.begin_frame(0.0)
    viewer.log_state(example.state_0)        # robot only (skip mud particles)
    viewer.end_frame()
    fr = viewer.get_frame()
    img = np.asarray(fr.numpy())
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    p = OUT / f"collision_{tag}.png"
    iio.imwrite(p, img)
    print("wrote", p.name)
