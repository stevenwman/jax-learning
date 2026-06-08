"""Task 2.2 — hermetic test: compute_pose_error sign + axis-angle correctness.

OSC consumes (pos_err, rot_err) where:
  - pos_err = target - current (points TOWARD target)
  - rot_err = axis-angle of (target ⊗ conj(current))
              i.e. the rotation that takes current → target.

A 90° rotation around +z must produce rot_err ≈ (0, 0, π/2).
"""
import jax.numpy as jp
import numpy as np
import pytest


def test_pos_error_points_to_target():
    from jax_rl.envs.manipulation.factory.controller.osc import compute_pose_error
    cur_pos = jp.array([0.0, 0.0, 0.0])
    tgt_pos = jp.array([0.10, 0.0, 0.0])
    cur_q = jp.array([1.0, 0.0, 0.0, 0.0])
    tgt_q = cur_q
    pos_err, rot_err = compute_pose_error(cur_pos, cur_q, tgt_pos, tgt_q)
    np.testing.assert_allclose(pos_err, jp.array([0.10, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(rot_err, jp.zeros(3), atol=1e-6)


def test_rot_error_axis_angle_z():
    """90° about +z: rot_err = (0, 0, π/2)."""
    from jax_rl.envs.manipulation.factory.controller.osc import compute_pose_error
    cur_q = jp.array([1.0, 0.0, 0.0, 0.0])
    tgt_q = jp.array([jp.cos(jp.pi / 4), 0.0, 0.0, jp.sin(jp.pi / 4)])
    _, rot_err = compute_pose_error(jp.zeros(3), cur_q, jp.zeros(3), tgt_q)
    assert rot_err[2] == pytest.approx(np.pi / 2, rel=1e-5)
    assert abs(float(rot_err[0])) < 1e-5
    assert abs(float(rot_err[1])) < 1e-5


def test_rot_error_axis_angle_x():
    """45° about +x: rot_err = (π/4, 0, 0)."""
    from jax_rl.envs.manipulation.factory.controller.osc import compute_pose_error
    cur_q = jp.array([1.0, 0.0, 0.0, 0.0])
    tgt_q = jp.array([jp.cos(jp.pi / 8), jp.sin(jp.pi / 8), 0.0, 0.0])
    _, rot_err = compute_pose_error(jp.zeros(3), cur_q, jp.zeros(3), tgt_q)
    assert rot_err[0] == pytest.approx(np.pi / 4, rel=1e-5)
    assert abs(float(rot_err[1])) < 1e-5
    assert abs(float(rot_err[2])) < 1e-5


def test_rot_error_zero_for_identical_quats():
    """Identical orientation → zero rotation error (no NaN at sin_half=0)."""
    from jax_rl.envs.manipulation.factory.controller.osc import compute_pose_error
    q = jp.array([1.0, 0.0, 0.0, 0.0])
    _, rot_err = compute_pose_error(jp.zeros(3), q, jp.zeros(3), q)
    np.testing.assert_allclose(rot_err, jp.zeros(3), atol=1e-7)
    assert np.all(np.isfinite(np.asarray(rot_err))), "rot_err must be finite at zero rotation"


def test_rot_error_drives_current_toward_target():
    """Direction sign convention: applying rot_err to current rotates it toward target.

    For pure-z rotations: cur_q = identity, tgt_q = R_z(θ). rot_err should be
    +z·θ (rotate identity by +z·θ to reach target). Verify sign with small angle.
    """
    from jax_rl.envs.manipulation.factory.controller.osc import compute_pose_error
    theta = 0.2  # radians
    cur_q = jp.array([1.0, 0.0, 0.0, 0.0])
    tgt_q = jp.array([jp.cos(theta / 2), 0.0, 0.0, jp.sin(theta / 2)])
    _, rot_err = compute_pose_error(jp.zeros(3), cur_q, jp.zeros(3), tgt_q)
    assert float(rot_err[2]) > 0, f"rot_err z-component must be positive, got {rot_err}"
    assert rot_err[2] == pytest.approx(theta, rel=1e-5)
