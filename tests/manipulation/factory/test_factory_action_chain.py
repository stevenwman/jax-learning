"""Hermetic tests for Factory action chain (EMA + denorm + clip + reset)."""
import jax.numpy as jp
import numpy as np
import pytest

from jax_rl.envs.manipulation.factory.controller.action_chain import (
    apply_ema, denormalize, clip_to_bounds, reset_on_done,
)

POS_THRESHOLD = jp.array([0.02, 0.02, 0.02])
ROT_THRESHOLD = jp.array([0.097, 0.097, 0.097])
POS_BOUNDS = jp.array([0.05, 0.05, 0.05])


def test_ema_smoothing():
    prev = jp.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    raw = jp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    out = apply_ema(raw, prev, ema_factor=0.2)
    expected = 0.2 * raw + 0.8 * prev
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_denormalize_pos_and_rot():
    ema = jp.array([0.5, 1.0, -0.5, 0.5, 1.0, -0.5])
    pos_delta, rot_delta = denormalize(ema, POS_THRESHOLD, ROT_THRESHOLD,
                                       unidirectional_rot=False)
    np.testing.assert_allclose(pos_delta, jp.array([0.010, 0.020, -0.010]), rtol=1e-6)
    np.testing.assert_allclose(rot_delta, jp.array([0.0485, 0.097, -0.0485]), rtol=1e-6)


def test_denormalize_unidirectional_rot_yaw_only_negative():
    """When unidirectional_rot=True, yaw (rot[2]) is constrained to [-rot_thresh, 0]."""
    ema = jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    _, rot_delta = denormalize(ema, POS_THRESHOLD, ROT_THRESHOLD,
                               unidirectional_rot=True)
    assert rot_delta[2] == pytest.approx(-0.097, rel=1e-5)
    ema2 = jp.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    _, rot_delta2 = denormalize(ema2, POS_THRESHOLD, ROT_THRESHOLD,
                                unidirectional_rot=True)
    assert rot_delta2[2] == pytest.approx(0.0, abs=1e-6)


def test_clip_to_bounds_clamps_relative_to_anchor():
    target = jp.array([0.7, 0.1, 0.05])
    anchor = jp.array([0.6, 0.0, 0.0])
    clipped = clip_to_bounds(target, anchor, POS_BOUNDS)
    np.testing.assert_allclose(clipped, anchor + POS_BOUNDS, rtol=1e-6)


def test_reset_on_done_zeros_terminated_episodes():
    """EMA actions must zero out on done=True per env, persist on done=False."""
    ema = jp.array([[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                    [0.3, 0.3, 0.3, 0.0, 0.0, 0.0]])
    done = jp.array([True, False])
    reset = reset_on_done(ema, done)
    np.testing.assert_allclose(reset[0], jp.zeros(6))
    np.testing.assert_allclose(reset[1], ema[1])


def test_reset_on_done_single_env():
    """reset_on_done works on un-batched (6,) input with scalar done."""
    ema = jp.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    reset_true = reset_on_done(ema, jp.array(True))
    np.testing.assert_allclose(reset_true, jp.zeros(6))
    reset_false = reset_on_done(ema, jp.array(False))
    np.testing.assert_allclose(reset_false, ema)
