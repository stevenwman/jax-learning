"""Tests for network builders (Actor, DeterministicActor, VCritic).

Verifies that composed modules produce correct output shapes and that
the encoder is swappable (the key property builders provide).
"""

import os
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig, ValueHeadConfig
from jax_rl.networks.builders import Actor, DeterministicActor, VCritic

OBS_DIM = 17
ACTION_DIM = 6
BATCH = 8
KEY = jax.random.PRNGKey(0)


def test_actor_output_shape():
    """Actor should output (mean, log_std) with shape (batch, action_dim)."""
    enc = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64))
    head = PolicyHeadConfig(action_dim=ACTION_DIM, squash=True, state_dependent_std=True)
    actor = Actor(enc, head)

    params = actor.init(KEY, jnp.zeros((1, OBS_DIM)))
    obs = jax.random.normal(KEY, (BATCH, OBS_DIM))
    mean, log_std = actor.apply(params, obs)

    assert mean.shape == (BATCH, ACTION_DIM)
    assert log_std.shape == (BATCH, ACTION_DIM)


def test_actor_state_independent_std():
    """Actor with state_independent_std should still output correct shapes."""
    enc = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64))
    head = PolicyHeadConfig(action_dim=ACTION_DIM, squash=False, state_dependent_std=False)
    actor = Actor(enc, head)

    params = actor.init(KEY, jnp.zeros((1, OBS_DIM)))
    obs = jax.random.normal(KEY, (BATCH, OBS_DIM))
    mean, log_std = actor.apply(params, obs)

    assert mean.shape == (BATCH, ACTION_DIM)
    assert log_std.shape == (BATCH, ACTION_DIM)


def test_deterministic_actor_output_shape():
    """DeterministicActor should output actions with shape (batch, action_dim)."""
    enc = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64))
    actor = DeterministicActor(enc, ACTION_DIM)

    params = actor.init(KEY, jnp.zeros((1, OBS_DIM)))
    obs = jax.random.normal(KEY, (BATCH, OBS_DIM))
    action = actor.apply(params, obs)

    assert action.shape == (BATCH, ACTION_DIM)
    # Deterministic actor uses tanh, output should be in [-1, 1]
    assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)


def test_vcritic_output_shape():
    """VCritic should output scalar values with shape (batch,)."""
    enc = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64))
    head = ValueHeadConfig()
    critic = VCritic(enc, head)

    params = critic.init(KEY, jnp.zeros((1, OBS_DIM)))
    obs = jax.random.normal(KEY, (BATCH, OBS_DIM))
    values = critic.apply(params, obs)

    assert values.shape == (BATCH,)


def test_different_encoder_dims():
    """Builders should work with any hidden_dim configuration."""
    for dims in [(32,), (128, 128), (256, 128, 64)]:
        enc = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=dims)
        head = PolicyHeadConfig(action_dim=ACTION_DIM, squash=True)
        actor = Actor(enc, head)

        params = actor.init(KEY, jnp.zeros((1, OBS_DIM)))
        obs = jax.random.normal(KEY, (BATCH, OBS_DIM))
        mean, log_std = actor.apply(params, obs)

        assert mean.shape == (BATCH, ACTION_DIM)


def test_different_activations():
    """Builders should work with all registered activations."""
    for act in ["relu", "swish", "elu", "gelu", "tanh"]:
        enc = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(32, 32), activation=act)
        head = PolicyHeadConfig(action_dim=ACTION_DIM, squash=True)
        actor = Actor(enc, head)

        params = actor.init(KEY, jnp.zeros((1, OBS_DIM)))
        obs = jax.random.normal(KEY, (BATCH, OBS_DIM))
        mean, _log_std = actor.apply(params, obs)

        assert mean.shape == (BATCH, ACTION_DIM)
        assert not jnp.any(jnp.isnan(mean)), f"NaN with activation={act}"
