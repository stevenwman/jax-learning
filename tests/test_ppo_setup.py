"""Tests for PPO initialization, action selection, update, and buffer/GAE."""

import os
import sys

import jax
import jax.numpy as jnp
import optax
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.algos import PPO
from jax_rl.buffers import RolloutBatch, RolloutBuffer


OBS_DIM = 17
ACTION_DIM = 6
NUM_ENVS = 4
NUM_STEPS = 32
KEY = jax.random.PRNGKey(0)


def _make_ppo():
    """Create a PPO instance with test config."""
    encoder_config = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64))
    policy_config = PolicyHeadConfig(action_dim=ACTION_DIM, squash=True)

    config = PPOConfig(
        encoder=encoder_config,
        critic_encoder=EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64)),
        policy_head=policy_config,
        num_envs=NUM_ENVS,
        gamma=0.99,
        minibatch_size=NUM_ENVS * NUM_STEPS // 4,  # 4 minibatches
    )

    actor_opt = optax.adam(3e-4)
    critic_opt = optax.adam(3e-4)
    ppo = PPO(config, OBS_DIM, ACTION_DIM, actor_opt, critic_opt)
    state = ppo.init(KEY)
    return ppo, config, state


def test_ppo_initialization():
    """PPO initializes and produces valid TrainingState."""
    ppo, config, state = _make_ppo()
    assert state.actor_params is not None
    assert state.critic_params is not None
    actor_count = sum(x.size for x in jax.tree.leaves(state.actor_params))
    assert actor_count > 0


def test_action_selection():
    """select_action returns correct shapes."""
    ppo, _, state = _make_ppo()
    obs = jax.random.normal(KEY, (NUM_ENVS, OBS_DIM))
    key = jax.random.PRNGKey(42)
    action, log_prob, value = ppo.select_action(state, obs, key, deterministic=False)
    assert action.shape == (NUM_ENVS, ACTION_DIM)
    assert log_prob.shape == (NUM_ENVS,)
    assert value.shape == (NUM_ENVS,)
    # Actions should be in [-1, 1] with squash=True
    assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)


def test_deterministic_action():
    """Deterministic actions are consistent across calls."""
    ppo, _, state = _make_ppo()
    obs = jax.random.normal(KEY, (NUM_ENVS, OBS_DIM))
    key = jax.random.PRNGKey(42)
    a1, _, _ = ppo.select_action(state, obs, key, deterministic=True)
    a2, _, _ = ppo.select_action(state, obs, key, deterministic=True)
    assert jnp.allclose(a1, a2)


def test_ppo_update():
    """PPO update runs and produces valid metrics."""
    ppo, config, state = _make_ppo()

    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 7)

    batch = RolloutBatch(
        obs=jax.random.normal(keys[0], (NUM_STEPS, NUM_ENVS, OBS_DIM)),
        actions=jax.random.normal(keys[1], (NUM_STEPS, NUM_ENVS, ACTION_DIM)),
        rewards=jax.random.normal(keys[2], (NUM_STEPS, NUM_ENVS)),
        dones=jnp.zeros((NUM_STEPS, NUM_ENVS)),
        truncations=jnp.zeros((NUM_STEPS, NUM_ENVS)),
        log_probs=jax.random.normal(keys[3], (NUM_STEPS, NUM_ENVS)),
        values=jax.random.normal(keys[4], (NUM_STEPS, NUM_ENVS)),
        advantages=jax.random.normal(keys[5], (NUM_STEPS, NUM_ENVS)),
        returns=jax.random.normal(keys[6], (NUM_STEPS, NUM_ENVS)),
    )

    next_obs = jax.random.normal(key, (NUM_ENVS, OBS_DIM))
    new_state, metrics = ppo.update(state, batch, key, next_obs=next_obs)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert not jnp.isnan(metrics["policy_loss"])
    assert not jnp.isnan(metrics["value_loss"])


def test_buffer_and_gae():
    """RolloutBuffer fills and computes GAE correctly."""
    buffer = RolloutBuffer(NUM_STEPS, NUM_ENVS, OBS_DIM, ACTION_DIM)

    key = jax.random.PRNGKey(456)
    for step in range(NUM_STEPS):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, 5)
        buffer.add(
            obs=jax.random.normal(keys[0], (NUM_ENVS, OBS_DIM)),
            action=jax.random.normal(keys[1], (NUM_ENVS, ACTION_DIM)),
            reward=jax.random.normal(keys[2], (NUM_ENVS,)),
            done=jnp.zeros((NUM_ENVS,)),
            truncation=jnp.zeros((NUM_ENVS,)),
            log_prob=jax.random.normal(keys[3], (NUM_ENVS,)),
            value=jax.random.normal(keys[4], (NUM_ENVS,)),
        )

    key, subkey = jax.random.split(key)
    next_value = jax.random.normal(subkey, (NUM_ENVS,))
    batch = buffer.get(next_value, gamma=0.99, gae_lambda=0.95)

    assert batch.advantages.shape == (NUM_STEPS, NUM_ENVS)
    assert batch.returns.shape == (NUM_STEPS, NUM_ENVS)
    assert not jnp.any(jnp.isnan(batch.advantages))
