"""Tests for frame stacking utility."""

import os
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.utils.frame_stack import (
    init_frame_stack, update_frame_stack, get_stacked_obs, reset_frame_stack
)

OBS_DIM = 31
NUM_ENVS = 4
N_FRAMES = 3


def test_init_shape():
    """Initial stack should have shape (num_envs, obs_dim * n_frames)."""
    stack = init_frame_stack(OBS_DIM, N_FRAMES, NUM_ENVS)
    assert stack.shape == (NUM_ENVS, OBS_DIM * N_FRAMES)


def test_init_zeros():
    """Initial stack should be all zeros."""
    stack = init_frame_stack(OBS_DIM, N_FRAMES, NUM_ENVS)
    assert jnp.allclose(stack, 0.0)


def test_update_inserts_at_front():
    """After update, new obs should be at the front of the stack."""
    stack = init_frame_stack(OBS_DIM, N_FRAMES, NUM_ENVS)
    new_obs = jnp.ones((NUM_ENVS, OBS_DIM)) * 5.0
    stack = update_frame_stack(stack, new_obs, OBS_DIM)

    # Front frame should be 5.0
    assert jnp.allclose(stack[:, :OBS_DIM], 5.0)
    # Remaining frames should still be 0.0 (shifted from init)
    assert jnp.allclose(stack[:, OBS_DIM:], 0.0)


def test_update_shifts_frames():
    """Updating should shift old frames back and insert new one at front."""
    stack = init_frame_stack(OBS_DIM, N_FRAMES, NUM_ENVS)

    obs1 = jnp.ones((NUM_ENVS, OBS_DIM)) * 1.0
    obs2 = jnp.ones((NUM_ENVS, OBS_DIM)) * 2.0
    obs3 = jnp.ones((NUM_ENVS, OBS_DIM)) * 3.0

    stack = update_frame_stack(stack, obs1, OBS_DIM)
    stack = update_frame_stack(stack, obs2, OBS_DIM)
    stack = update_frame_stack(stack, obs3, OBS_DIM)

    # Most recent frame (front) should be obs3
    assert jnp.allclose(stack[:, :OBS_DIM], 3.0)
    # Second frame should be obs2
    assert jnp.allclose(stack[:, OBS_DIM:2*OBS_DIM], 2.0)
    # Oldest frame should be obs1
    assert jnp.allclose(stack[:, 2*OBS_DIM:], 1.0)


def test_reset_fills_all_frames():
    """reset_frame_stack should fill all frames with new_obs for done envs."""
    stack = init_frame_stack(OBS_DIM, N_FRAMES, NUM_ENVS)
    obs1 = jnp.ones((NUM_ENVS, OBS_DIM)) * 1.0
    stack = update_frame_stack(stack, obs1, OBS_DIM)

    new_obs = jnp.ones((NUM_ENVS, OBS_DIM)) * 9.0
    done = jnp.array([1.0, 0.0, 1.0, 0.0])
    stack = reset_frame_stack(stack, new_obs, OBS_DIM, done)

    # Env 0 (done): all frames should be 9.0
    assert jnp.allclose(stack[0], 9.0)
    # Env 1 (not done): unchanged (front=1.0, rest=0.0)
    assert jnp.allclose(stack[1, :OBS_DIM], 1.0)
    assert jnp.allclose(stack[1, OBS_DIM:], 0.0)
    # Env 2 (done): all frames should be 9.0
    assert jnp.allclose(stack[2], 9.0)


def test_get_stacked_obs_identity():
    """get_stacked_obs should just return the stack as-is."""
    stack = jnp.ones((NUM_ENVS, OBS_DIM * N_FRAMES))
    assert jnp.array_equal(get_stacked_obs(stack), stack)


def test_single_frame():
    """Frame stack with n_frames=1 should just be obs_dim wide."""
    stack = init_frame_stack(OBS_DIM, 1, NUM_ENVS)
    assert stack.shape == (NUM_ENVS, OBS_DIM)

    obs = jnp.ones((NUM_ENVS, OBS_DIM)) * 3.0
    stack = update_frame_stack(stack, obs, OBS_DIM)
    assert jnp.allclose(stack, 3.0)
