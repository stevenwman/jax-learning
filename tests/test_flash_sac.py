"""Tests for FlashSAC algorithm."""

import jax
import jax.numpy as jnp
import optax
import pytest
from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.algos.flash_sac import FlashSAC

OBS_DIM = 12
ACTION_DIM = 4
BATCH_SIZE = 16
KEY = jax.random.PRNGKey(42)


def _make_batch(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    obs = jax.random.normal(k1, (BATCH_SIZE, OBS_DIM))
    next_obs = jax.random.normal(k4, (BATCH_SIZE, OBS_DIM))
    return {
        "obs": obs,
        "action": jax.random.uniform(k2, (BATCH_SIZE, ACTION_DIM), minval=-1, maxval=1),
        "reward": jnp.zeros((BATCH_SIZE, 1)),
        "next_obs": next_obs,
        "done": jnp.zeros((BATCH_SIZE, 1)),
        "truncation": jnp.zeros((BATCH_SIZE, 1)),
        "critic_obs": obs,
        "critic_next_obs": next_obs,
    }


def _make_flash_sac():
    cfg = FlashSACConfig(
        num_blocks=1, actor_hidden_dim=32, critic_hidden_dim=32,
        num_atoms=21, batch_size=BATCH_SIZE,
    )
    opt = optax.adam(3e-4)
    alpha_opt = optax.adam(3e-4)
    return FlashSAC(config=cfg, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                    optimizer=opt, alpha_optimizer=alpha_opt)


def test_init_produces_valid_state():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    assert state.actor_params is not None
    assert state.q1_params is not None
    assert state.q2_params is not None
    assert state.actor_batch_stats is not None
    assert state.q1_batch_stats is not None
    assert state.target_q1_batch_stats is not None
    assert state.noise_state is not None


def test_select_action_shape():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    obs = jnp.ones((1, OBS_DIM))
    action = algo.select_action(state.actor_params, obs, KEY)
    assert action.shape == (1, ACTION_DIM)
    assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)


def test_update_returns_metrics():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    batch = _make_batch(KEY)
    new_state, metrics = algo.update(state, batch)
    assert "q1_loss" in metrics or "critic_loss" in metrics
    assert new_state.update_count == 1


def test_update_modifies_params():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    batch = _make_batch(KEY)
    new_state, _ = algo.update(state, batch)
    q1_diff = jax.tree_util.tree_map(lambda a, b: jnp.sum(jnp.abs(a - b)),
                                      state.q1_params, new_state.q1_params)
    total_diff = sum(jax.tree_util.tree_leaves(q1_diff))
    assert total_diff > 0, "Critic params should change after update"


def test_deterministic_action_consistency():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    obs = jnp.ones((1, OBS_DIM))
    a1 = algo.select_action(state.actor_params, obs, KEY, deterministic=True)
    a2 = algo.select_action(state.actor_params, obs, KEY, deterministic=True)
    assert jnp.allclose(a1, a2)


def test_get_q_value():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    obs = jnp.ones((1, OBS_DIM))
    action = jnp.zeros((1, ACTION_DIM))
    q_val = algo.get_q_value(state, obs, action)
    assert q_val.shape == (1,)
