"""Robot-loader seam: let the vendored example load an MJCF (our trained go2.xml)
instead of its bundled URDF, so the Newton SolverMuJoCo builds the mjModel from
OUR model — matching the in-house OSC (foot sites + Jacobian + mass matrix) and
opening the door to other morphologies (a new robot = a new XML, not a new port).

How: monkeypatch newton.ModelBuilder.add_urdf to dispatch `.xml` paths to
add_mjcf. The example's downstream posing loop (`joint_key.index(key)+6`) overflows
on go2.xml's 0-dof `*_foot_joint`, so patched_config(mjcf_model=...) EMPTIES
initial_joint_q and we set the home pose here, directly on joint_q, right after
add_mjcf (free joint = 7 coords, then 12 actuated FL,FR,RL,RR hip/thigh/calf).

Usage (before constructing the example):
    import mud_model
    mud_model.set_home_pose(default_pose_12)   # from ckpt meta default_pose_policy
    mud_model.enable()
"""
from __future__ import annotations

import newton

_orig_add_urdf = newton.ModelBuilder.add_urdf
_HOME = {"pose": None}
# go2.xml contact geoms are class "collision" + "foot"; the default
# collider_classes=("collision",) would MISS the feet (the mud-contact shape).
COLLIDER_CLASSES = ("collision", "foot")


def set_home_pose(pose_12):
    """12 actuated joint angles in FL,FR,RL,RR × (hip,thigh,calf) order — the
    policy's default_pose_policy. Used to pose the floating base at spawn."""
    _HOME["pose"] = [float(v) for v in pose_12]


def _dispatch_add_urdf(self, path, **kw):
    if not str(path).endswith(".xml"):
        return _orig_add_urdf(self, path, **kw)
    self.add_mjcf(
        str(path),
        xform=kw.get("xform"),
        floating=kw.get("floating"),                       # None/True -> floating (free joint)
        enable_self_collisions=kw.get("enable_self_collisions", False),
        collapse_fixed_joints=kw.get("collapse_fixed_joints", False),
        ignore_inertial_definitions=kw.get("ignore_inertial_definitions", True),
        collider_classes=COLLIDER_CLASSES,
        parse_meshes=True,
    )
    hp = _HOME["pose"]
    if hp is not None and len(self.joint_q) >= 7 + len(hp):
        for i, v in enumerate(hp):                          # joint_q[0:7]=free, [7:]=actuated
            self.joint_q[7 + i] = v


def enable():
    newton.ModelBuilder.add_urdf = _dispatch_add_urdf


def disable():
    newton.ModelBuilder.add_urdf = _orig_add_urdf
