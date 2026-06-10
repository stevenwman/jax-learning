"""Load the TRAINED go2.xml standalone via Newton add_mjcf, dump its collision
geometry + render colliders — to compare vs the example's URDF and decide the
URDF->go2.xml migration. No mud scene; robot only.
"""
import os
import sys
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np
import warp as wp
import imageio.v2 as iio
import newton

GO2 = Path("/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/"
           "go2-osc-impedance/jax_rl/envs/locomotion/xmls/unitree_go2/go2.xml")

wp.init()
builder = newton.ModelBuilder(up_axis="Z")
builder.add_mjcf(
    str(GO2),
    xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
    floating=None,                               # auto: free joint -> floating base
    force_show_colliders=True,
    collider_classes=("collision", "foot"),      # go2.xml feet are class="foot" (else missed!)
    parse_meshes=True,
    ignore_inertial_definitions=True,
)
print("[ok] add_mjcf parsed go2.xml")
print("joint_key:", list(builder.joint_key))

model = builder.finalize()
state = model.state()
# home pose direct on joint_q: free joint = 7 coords (pos3 + quat xyzw), then 12
# actuated (FL,FR,RL,RR × hip,thigh,calf; the 0-dof *_foot_joint add no coords).
jq = np.asarray(state.joint_q.numpy()).copy()
print(f"joint_q len={len(jq)} (expect 19 = 7 free + 12 actuated)")
if len(jq) == 19:
    jq[0:3] = [0.0, 0.0, 0.45]; jq[3:7] = [0.0, 0.0, 0.0, 1.0]
    for l in range(4):
        jq[7 + 3 * l + 1] = 0.9     # thigh
        jq[7 + 3 * l + 2] = -1.8    # calf
    state.joint_q.assign(wp.array(jq, dtype=wp.float32, device=state.joint_q.device))
newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def arr(a):
    return np.asarray(a.numpy()) if hasattr(a, "numpy") else np.asarray(a)


st = arr(model.shape_type); flags = arr(model.shape_flags)
sb = arr(model.shape_body); ss = arr(model.shape_scale)
bk = list(model.body_key)
COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)
print(f"\n=== go2.xml COLLISION SHAPES (of {model.shape_count} total) ===")
ncol = 0; by_type = Counter()
for i in range(int(model.shape_count)):
    if int(flags[i]) & COLLIDE:
        ncol += 1
        name = newton.GeoType(int(st[i])).name
        by_type[name] += 1
        body = bk[int(sb[i])] if int(sb[i]) >= 0 else "world"
        print(f"  {body:18s} {name:11s} scale={np.round(ss[i], 3)}")
print(f"--> {ncol} collider shapes; by type: {dict(by_type)}")

# render colliders
OUT = HERE / "recordings"; OUT.mkdir(exist_ok=True)
try:
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    try:
        from newton.viewer import ViewerGL
    except Exception:
        from newton._src.viewer.viewer_gl import ViewerGL
    viewer = ViewerGL(width=600, height=400, headless=True)
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(1.3, 0.0, 0.30), -6.0, 180.0)
    for show_col, tag in [(False, "visual"), (True, "collision")]:
        viewer.show_collision = show_col
        viewer.show_visual = not show_col
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
        img = np.asarray(viewer.get_frame().numpy())
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[-1] == 4:
            img = img[..., :3]
        p = OUT / f"go2xml_{tag}.png"
        iio.imwrite(p, img); print("wrote", p.name)
except Exception as e:  # noqa: BLE001
    import traceback; traceback.print_exc()
    print("RENDER FAILED (numeric dump above still valid):", str(e)[:120])
