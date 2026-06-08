"""Hermetic tests for Factory reward (squashing_fn + keypoint_distance + binaries)."""
import jax.numpy as jp
import numpy as np
import pytest

from jax_rl.envs.manipulation.factory.reward import (
    squashing_fn, keypoint_offsets, keypoint_distance,
    is_engaged, is_success, compute_reward,
)


def test_squashing_fn_peaks_at_zero():
    """squashing_fn(0, a, b) = 1 / (1 + b + 1) = 1 / (2 + b)."""
    assert squashing_fn(jp.array(0.0), a=100.0, b=2.0) == pytest.approx(1/4, rel=1e-5)
    assert squashing_fn(jp.array(0.0), a=500.0, b=0.0) == pytest.approx(1/2, rel=1e-5)


def test_squashing_fn_decays_with_distance():
    near = float(squashing_fn(jp.array(0.001), a=100.0, b=2.0))
    far = float(squashing_fn(jp.array(0.050), a=100.0, b=2.0))
    assert near > far
    assert far < 0.01


def test_keypoint_offsets_evenly_spaced_along_z():
    """4 keypoints, z evenly spaced in [-0.5, 0.5] * scale."""
    offsets = keypoint_offsets(num_keypoints=4, scale=0.05)
    assert offsets.shape == (4, 3)
    np.testing.assert_allclose(offsets[:, :2], 0.0)
    expected_z = jp.linspace(-0.025, 0.025, 4)
    np.testing.assert_allclose(offsets[:, 2], expected_z, rtol=1e-5)


def test_keypoint_distance_zero_at_perfect_match():
    """Same held + target pose → zero keypoint distance."""
    held_pos = jp.array([0.6, 0.0, 0.0])
    held_quat = jp.array([1.0, 0.0, 0.0, 0.0])
    offsets = keypoint_offsets(4, 0.05)
    d = keypoint_distance(held_pos, held_quat, held_pos, held_quat, offsets)
    assert float(d) == pytest.approx(0.0, abs=1e-6)


def test_keypoint_distance_grows_with_offset():
    held_pos = jp.array([0.6, 0.0, 0.0])
    quat = jp.array([1.0, 0.0, 0.0, 0.0])
    target_far = held_pos + jp.array([0.10, 0.0, 0.0])
    offsets = keypoint_offsets(4, 0.05)
    d_near = float(keypoint_distance(held_pos, quat, held_pos, quat, offsets))
    d_far = float(keypoint_distance(held_pos, quat, target_far, quat, offsets))
    assert d_far > d_near
    assert d_far == pytest.approx(0.10, rel=1e-4)


def test_is_engaged_xy_z_tilt_gates():
    """Engaged requires xy near bolt AND z below threshold AND tilt < 2°.

    The peg's body z-axis after gripper-down convention is rotated π about
    x relative to identity, so the 'upright' peg_quat for our test is
    quat_from_axis_angle(π about x) = (0, 1, 0, 0).
    """
    centered = jp.array([0.0, 0.0])
    offset_2cm = jp.array([0.02, 0.0])
    hole_xy = jp.array([0.0, 0.0])
    aligned = jp.array([0.0, 1.0, 0.0, 0.0])   # π about x — gripper-down
    tilted = jp.array([0.9961947, 0.0871557, 0.0, 0.0])  # 10° about x — too tilted

    # Centered + low + aligned → engaged ✓
    assert bool(is_engaged(peg_xy=centered, peg_z=jp.array(-0.020),
                           peg_quat=aligned,
                           hole_xy=hole_xy, hole_top_z=jp.array(0.0),
                           asset_height=0.025, engage_threshold=0.5))
    # Centered + low + TILTED → not engaged (closes the tilt exploit)
    assert not bool(is_engaged(peg_xy=centered, peg_z=jp.array(-0.020),
                               peg_quat=tilted,
                               hole_xy=hole_xy, hole_top_z=jp.array(0.0),
                               asset_height=0.025, engage_threshold=0.5))
    # Centered but not low enough → not engaged
    assert not bool(is_engaged(peg_xy=centered, peg_z=jp.array(0.005),
                               peg_quat=aligned,
                               hole_xy=hole_xy, hole_top_z=jp.array(0.0),
                               asset_height=0.025, engage_threshold=0.5))
    # Low + aligned but xy offset → not engaged
    assert not bool(is_engaged(peg_xy=offset_2cm, peg_z=jp.array(-0.020),
                               peg_quat=aligned,
                               hole_xy=hole_xy, hole_top_z=jp.array(0.0),
                               asset_height=0.025, engage_threshold=0.5))


def test_compute_reward_higher_near_target():
    """Reward higher when held asset is at target pose than far from it.

    v15 phased reward: tilt aligned (gripper-down) for both so the tilt gate
    fires; near gets r_B_desc + r_success, far gets r_A only with weak tilt
    bonus + dead-end penalty.
    """
    held = jp.array([0.6, 0.0, 0.0])
    peg_quat = jp.array([0.0, 1.0, 0.0, 0.0])   # gripper-down (π about x)
    r_near = float(compute_reward(
        held_pos=held, held_quat=peg_quat,
        target_pos=held, target_quat=peg_quat,
        peg_z=jp.array(-0.020), hole_top_z=jp.array(0.0),
        asset_height=0.025, engage_threshold=0.5, success_threshold=0.04,
    ))
    r_far = float(compute_reward(
        held_pos=held + jp.array([0.10, 0.0, 0.10]), held_quat=peg_quat,
        target_pos=held, target_quat=peg_quat,
        peg_z=jp.array(0.10), hole_top_z=jp.array(0.0),
        asset_height=0.025, engage_threshold=0.5, success_threshold=0.04,
    ))
    assert r_near > r_far


def test_reward_per_step_max_bound():
    """Per-step max ≈ 8.0 at fully seated + aligned + below entry.

    Decomposition under v15.1 (r_align untied from phase to remove cliff):
      r_align   = 2 * (r_xy(=0.5) + r_tilt(=0.5)) = 2.0      (always on)
      r_B_desc  = phase_below(≈1) * aligned(≈1) * z_progress(=1.0) = 1.0
      r_B_pen   = phase_below(≈1) * (1 - aligned)(≈0) * (-0.5) ≈ 0
      r_success = 5.0       (hard gate: xy<5mm AND tilt<2° AND z seated)
      ──────────────────
      Total ≈ 8.0
    """
    held = jp.array([0.6, 0.0, 0.0])
    peg_quat = jp.array([0.0, 1.0, 0.0, 0.0])   # gripper-down (π about x)
    r = float(compute_reward(
        held_pos=held, held_quat=peg_quat,
        target_pos=held, target_quat=peg_quat,
        peg_z=jp.array(-0.030), hole_top_z=jp.array(0.0),
        asset_height=0.025, engage_threshold=0.5, success_threshold=0.5,
    ))
    assert r == pytest.approx(8.0, abs=0.05)
