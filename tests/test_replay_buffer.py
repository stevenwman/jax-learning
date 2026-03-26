"""Tests for JaxReplayBuffer — GPU-resident replay buffer."""

import os
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer

OBS_DIM = 17
ACTION_DIM = 6
KEY = jax.random.PRNGKey(0)


def test_init_empty():
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    assert len(buf) == 0


def test_add_batch():
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    buf.add_batch(
        obs=jnp.zeros((10, OBS_DIM)),
        action=jnp.zeros((10, ACTION_DIM)),
        reward=jnp.zeros(10),
        next_obs=jnp.zeros((10, OBS_DIM)),
        done=jnp.zeros(10),
        truncation=jnp.zeros(10),
    )
    assert len(buf) == 10


def test_add_multiple_batches():
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    for i in range(5):
        buf.add_batch(
            obs=jnp.ones((10, OBS_DIM)) * i,
            action=jnp.zeros((10, ACTION_DIM)),
            reward=jnp.ones(10) * i,
            next_obs=jnp.ones((10, OBS_DIM)) * i,
            done=jnp.zeros(10),
            truncation=jnp.zeros(10),
        )
    assert len(buf) == 50


def test_wrap_around():
    """Buffer should wrap when exceeding max_size."""
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=20)
    for i in range(5):
        buf.add_batch(
            obs=jnp.ones((10, OBS_DIM)) * i,
            action=jnp.zeros((10, ACTION_DIM)),
            reward=jnp.ones(10) * i,
            next_obs=jnp.ones((10, OBS_DIM)) * i,
            done=jnp.zeros(10),
            truncation=jnp.zeros(10),
        )
    # Should cap at max_size
    assert len(buf) == 20


def test_sample_shape():
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    buf.add_batch(
        obs=jax.random.normal(KEY, (50, OBS_DIM)),
        action=jax.random.normal(KEY, (50, ACTION_DIM)),
        reward=jnp.zeros(50),
        next_obs=jax.random.normal(KEY, (50, OBS_DIM)),
        done=jnp.zeros(50),
        truncation=jnp.zeros(50),
    )

    batch = buf.sample(32, key=KEY)
    assert batch["obs"].shape == (32, OBS_DIM)
    assert batch["action"].shape == (32, ACTION_DIM)
    assert batch["reward"].shape == (32, 1)
    assert batch["next_obs"].shape == (32, OBS_DIM)
    assert batch["done"].shape == (32, 1)
    assert batch["truncation"].shape == (32, 1)


def test_sample_values_from_buffer():
    """Sampled values should come from data that was actually added."""
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    buf.add_batch(
        obs=jnp.ones((50, OBS_DIM)) * 7.0,
        action=jnp.ones((50, ACTION_DIM)) * 3.0,
        reward=jnp.ones(50) * 2.0,
        next_obs=jnp.ones((50, OBS_DIM)) * 7.0,
        done=jnp.zeros(50),
        truncation=jnp.zeros(50),
    )

    batch = buf.sample(16, key=KEY)
    assert jnp.allclose(batch["obs"], 7.0)
    assert jnp.allclose(batch["action"], 3.0)
    assert jnp.allclose(batch["reward"], 2.0)


def test_sample_different_keys_give_different_batches():
    """Different random keys should produce different samples."""
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=1000)
    buf.add_batch(
        obs=jax.random.normal(KEY, (500, OBS_DIM)),
        action=jax.random.normal(KEY, (500, ACTION_DIM)),
        reward=jax.random.normal(KEY, (500,)),
        next_obs=jax.random.normal(KEY, (500, OBS_DIM)),
        done=jnp.zeros(500),
        truncation=jnp.zeros(500),
    )

    b1 = buf.sample(32, key=jax.random.PRNGKey(1))
    b2 = buf.sample(32, key=jax.random.PRNGKey(2))
    # Very unlikely to be identical with different keys
    assert not jnp.allclose(b1["obs"], b2["obs"])
