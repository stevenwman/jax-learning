"""Factory reward — v15 phased design.

Encodes the physical phase structure of insertion:
  Phase A — peg tip above bore opening: reward xy + tilt alignment only,
            no descent incentive (prevents diving while off-axis).
  Phase B — peg tip below bore opening: reward descent IF aligned;
            penalty otherwise (dead-end where peg sidewall jams against bolt
            body).
  Terminal — +5.0 success bonus at full insertion + xy + tilt.

Per-step max ≈ 6.0 at seated success state (r_B_desc ≈ 1.0 + r_success = 5.0).
"""
from typing import Tuple

import jax
import jax.numpy as jp


def squashing_fn(x: jp.ndarray, a: float, b: float) -> jp.ndarray:
    """Smooth bump: 1 / (exp(a·x) + b + exp(-a·x)). Peaks at x=0 → 1/(2+b)."""
    return 1.0 / (jp.exp(a * x) + b + jp.exp(-a * x))


def keypoint_offsets(num_keypoints: int, scale: float) -> jp.ndarray:
    """Return (N, 3) offsets along z, evenly spaced in [-0.5, 0.5] * scale."""
    z = jp.linspace(-0.5, 0.5, num_keypoints) * scale
    out = jp.zeros((num_keypoints, 3))
    return out.at[:, 2].set(z)


def quat_rotate_vec(q: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
    """Rotate vector v by quaternion q = (w, x, y, z)."""
    w = q[0]
    xyz = q[1:]
    t = 2.0 * jp.cross(xyz, v)
    return v + w * t + jp.cross(xyz, t)


def keypoint_distance(
    held_pos: jp.ndarray, held_quat: jp.ndarray,
    target_pos: jp.ndarray, target_quat: jp.ndarray,
    offsets: jp.ndarray,
) -> jp.ndarray:
    """Mean L2 distance across N keypoints between held and target poses."""
    def transform(p, q):
        return p + jp.stack([quat_rotate_vec(q, o) for o in offsets])
    held_kps = transform(held_pos, held_quat)
    target_kps = transform(target_pos, target_quat)
    return jp.linalg.norm(held_kps - target_kps, axis=-1).mean()


_XY_PROXIMITY_THRESHOLD = 0.005     # 5 mm bore-radial slop for the binary gate
_TILT_COS_THRESHOLD = 0.9994        # cos(2°) — vertical tilt tolerance
_ENTRY_Z = 0.096                    # peg body z at which tip enters bore
                                    # (hole top z=0.075 + peg half-length ≈0.021)


def _peg_cos_dot(peg_quat: jp.ndarray) -> jp.ndarray:
    """cos(angle) between peg body z-axis and world -z (gripper-down).

    Closed form from quat (w, x, y, z):
      peg_z_axis_world_z = 1 − 2(x² + y²)
      cos_dot = -peg_z_axis_world_z       (gripper-down: aligned ↔ cos_dot=+1)
    """
    x, y = peg_quat[1], peg_quat[2]
    z_axis_world_z = 1.0 - 2.0 * (x * x + y * y)
    return -z_axis_world_z


def _peg_aligned(peg_quat: jp.ndarray,
                 cos_threshold: float = _TILT_COS_THRESHOLD) -> jp.ndarray:
    """True if peg z-axis within acos(cos_threshold) of world -z."""
    return _peg_cos_dot(peg_quat) > cos_threshold


def _physically_alignable(
    peg_xy: jp.ndarray, hole_xy: jp.ndarray, peg_quat: jp.ndarray,
    xy_threshold: float = _XY_PROXIMITY_THRESHOLD,
    tilt_cos_threshold: float = _TILT_COS_THRESHOLD,
) -> jp.ndarray:
    """Shared xy + tilt hard gate for both engaged and success bonuses."""
    xy_dist = jp.linalg.norm(peg_xy - hole_xy)
    return (xy_dist < xy_threshold) & _peg_aligned(peg_quat, tilt_cos_threshold)


def is_engaged(peg_xy: jp.ndarray, peg_z: jp.ndarray, peg_quat: jp.ndarray,
               hole_xy: jp.ndarray, hole_top_z: jp.ndarray,
               asset_height: float, engage_threshold: float,
               xy_threshold: float = _XY_PROXIMITY_THRESHOLD,
               tilt_cos_threshold: float = _TILT_COS_THRESHOLD) -> jp.ndarray:
    """Engaged ↔ peg physically alignable AND z past engage_threshold."""
    z_engaged = hole_top_z - peg_z > asset_height * engage_threshold
    return jp.asarray(_physically_alignable(
        peg_xy, hole_xy, peg_quat, xy_threshold, tilt_cos_threshold
    ) & z_engaged)


def is_success(peg_xy: jp.ndarray, peg_z: jp.ndarray, peg_quat: jp.ndarray,
               hole_xy: jp.ndarray, hole_top_z: jp.ndarray,
               asset_height: float, success_threshold: float,
               xy_threshold: float = _XY_PROXIMITY_THRESHOLD,
               tilt_cos_threshold: float = _TILT_COS_THRESHOLD) -> jp.ndarray:
    """Success ↔ peg physically alignable AND z past success_threshold."""
    z_success = hole_top_z - peg_z > asset_height * success_threshold
    return jp.asarray(_physically_alignable(
        peg_xy, hole_xy, peg_quat, xy_threshold, tilt_cos_threshold
    ) & z_success)


def compute_reward(
    held_pos: jp.ndarray, held_quat: jp.ndarray,
    target_pos: jp.ndarray, target_quat: jp.ndarray,
    peg_z: jp.ndarray, hole_top_z: jp.ndarray,
    keypoint_coef_baseline: Tuple[float, float] = (100.0, 2.0),
    keypoint_coef_coarse: Tuple[float, float] = (500.0, 2.0),
    keypoint_coef_fine: Tuple[float, float] = (1500.0, 0.0),
    asset_height: float = 0.025,
    engage_threshold: float = 0.5,
    success_threshold: float = 0.04,
    keypoint_scale: float = 0.05,
    engage_xy_threshold: float = _XY_PROXIMITY_THRESHOLD,
    success_xy_threshold: float = _XY_PROXIMITY_THRESHOLD,
    tilt_cos_threshold: float = _TILT_COS_THRESHOLD,
    num_keypoints: int = 4,
    entry_z: float = _ENTRY_Z,
) -> jp.ndarray:
    """Phased reward — Phase A align + Phase B aligned-descent / dead-end + success.

    Keypoint, target_quat, engage_threshold, engage_xy_threshold args are
    accepted for signature compatibility with the env caller but unused in
    v15. v14's multi-term shaping stack (3 keypoint bells + xy_proximity +
    aligned_descent) was abandoned for the phased structure that encodes
    the physical "align above, descend below" split directly.
    """
    del target_quat
    del keypoint_coef_baseline, keypoint_coef_coarse, keypoint_coef_fine
    del keypoint_scale, num_keypoints
    del engage_threshold, engage_xy_threshold

    peg_xy = held_pos[:2]
    hole_xy = target_pos[:2]
    xy_dist = jp.linalg.norm(peg_xy - hole_xy)
    cos_dot = _peg_cos_dot(held_quat)
    tilt_rad = jp.arccos(jp.clip(cos_dot, -1.0, 1.0 - 1e-7))

    # v18: altitude-weighted alignment attractor.
    # v15.6 (3-DOF, no DR) reached Return 6545 because pos-only exploration
    # occasionally hit (aligned ∧ low_z) → critic learned descent pays.
    # 6-DOF runs (v1-v10) plateau ~880 because rot exploration breaks the
    # aligned half of that conjunction → critic never sees high-Q at low z →
    # Q surface is flat in z → policy mean drifts upward → hover trap.
    # Fix: scale r_align by altitude_bonus so aligned-at-low-z is strictly
    # higher Q than aligned-at-high-z. Critic learns descent direction from
    # alignment alone, no aligned-∧-low-z conjunction needed.
    r_xy = squashing_fn(xy_dist, 100.0, 0.0)            # max 0.5 at xy=0
    r_tilt = squashing_fn(tilt_rad, 50.0, 0.0)          # max 0.5 at tilt=0
    # altitude_bonus: 0 at z = entry_z + 10cm, 1 at z = entry_z.
    altitude_bonus = jp.clip((entry_z + 0.10 - peg_z) / 0.10, 0.0, 1.0)
    r_align = 2.0 * (r_xy + r_tilt) * altitude_bonus    # max 2.0 (at z≤entry)

    # Phase factor: ~1 above entry, ~0 below. Smooth so the critic sees a
    # clean gradient through the boundary.
    phase_above = jax.nn.sigmoid((peg_z - entry_z) * 200.0)
    phase_below = 1.0 - phase_above

    # Phase B — descent rewarded only when aligned; dead-end penalty otherwise.
    # Soft `aligned` gate centered OUTSIDE the success thresholds (10mm xy,
    # cos(3.6°) tilt) so success state gives `aligned ≈ 1` rather than the
    # 0.5 you'd get from centering the sigmoid on the binary threshold itself.
    aligned_xy = jax.nn.sigmoid(
        (2.0 * _XY_PROXIMITY_THRESHOLD - xy_dist) * 1000.0)
    aligned_tilt = jax.nn.sigmoid((cos_dot - 0.998) * 5000.0)
    aligned = aligned_xy * aligned_tilt
    # z_progress anchored at entry_z (peg tip crossing bore opening), NOT
    # hole_top_z (peg body crossing bore opening — fires too late, leaves
    # a 2cm dead zone where tip is descending into the bore but body still
    # above the opening so z_progress=0). Anchoring at entry gives smooth
    # 0→1 gradient throughout the actual insertion path. Physical bore tile
    # blocks z from growing if peg is misaligned (xy > bore clearance), so
    # r_B_desc can't fire without genuine insertion — the gate is enforced
    # by physics, not just by the soft `aligned` mask.
    z_progress = jp.clip((entry_z - peg_z) / asset_height, 0.0, 1.0)
    r_B_desc = phase_below * aligned * z_progress       # max 1.0
    r_B_pen = phase_below * (1.0 - aligned) * (-0.5)    # min -0.5

    # Floor attractor: SHARP Gaussian bell at target_pos[2] (seated centroid).
    # σ = 1/sqrt(2·100000) ≈ 2.2mm. Coefficient 1000 (σ=22mm, prior round) was
    # too soft — policy farmed near-saturated 1.9/step at current depth without
    # closing the 6.7mm gap. At coef 100000: peak +2.0, drops to ~0 at 5mm off.
    target_z = target_pos[2]
    r_floor = 2.0 * jp.exp(-100000.0 * (peg_z - target_z) ** 2)

    # Terminal success: +5.0 at fully seated + xy + tilt (hard gate).
    r_success = is_success(
        peg_xy, peg_z, held_quat, hole_xy, hole_top_z,
        asset_height, success_threshold,
        xy_threshold=success_xy_threshold,
        tilt_cos_threshold=tilt_cos_threshold,
    ).astype(jp.float32) * 5.0

    return r_align + r_B_desc + r_B_pen + r_floor + r_success
