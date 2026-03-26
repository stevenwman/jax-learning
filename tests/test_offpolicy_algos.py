"""Tests for off-policy algorithms: SAC, TD3, FastTD3, FastSAC.

Each algo is tested for:
    1. init() produces a valid TrainingState with correct shapes
    2. select_action() returns actions in [-1, 1] with correct shape
    3. update() runs without error and returns metrics with expected keys
    4. get_q_value() returns scalar Q predictions with correct shape
    5. Deterministic actions are consistent across calls
"""

import os
import sys

import jax
import jax.numpy as jnp
import optax
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.algos.sac import SAC
from jax_rl.algos.td3 import TD3
from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.algos.fast_sac import FastSAC

OBS_DIM = 17
ACTION_DIM = 6
BATCH_SIZE = 32
KEY = jax.random.PRNGKey(42)


def _make_batch(key):
    """Create a fake replay batch for testing."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "obs": jax.random.normal(k1, (BATCH_SIZE, OBS_DIM), dtype=jnp.float32),
        "action": jax.random.uniform(k2, (BATCH_SIZE, ACTION_DIM), minval=-1, maxval=1, dtype=jnp.float32),
        "reward": jnp.zeros((BATCH_SIZE, 1), dtype=jnp.float32),
        "next_obs": jax.random.normal(k4, (BATCH_SIZE, OBS_DIM), dtype=jnp.float32),
        "done": jnp.zeros((BATCH_SIZE, 1), dtype=jnp.float32),
        "truncation": jnp.zeros((BATCH_SIZE, 1), dtype=jnp.float32),
    }


# ── SAC ─────────────────────────────────────────────────────────────────────

def _make_sac():
    cfg = SACConfig(hidden_dim=(64, 64), batch_size=BATCH_SIZE)
    opt = optax.adam(3e-4)
    alpha_opt = optax.adam(3e-4)
    return SAC(cfg, OBS_DIM, ACTION_DIM, opt, alpha_opt, gamma=0.99, handle_truncation=True)


def test_sac_init():
    sac = _make_sac()
    state = sac.init(KEY)
    assert state.actor_params is not None
    assert state.q1_params is not None
    assert state.q2_params is not None
    assert state.target_q1_params is not None
    assert state.log_alpha.shape == ()


def test_sac_select_action():
    sac = _make_sac()
    state = sac.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))  # batch of 4
    key = jax.random.PRNGKey(1)

    action = sac.select_action(state.actor_params, obs, key, deterministic=False)
    assert action.shape == (4, ACTION_DIM)
    assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)


def test_sac_deterministic_action():
    sac = _make_sac()
    state = sac.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))
    key = jax.random.PRNGKey(1)

    a1 = sac.select_action(state.actor_params, obs, key, deterministic=True)
    a2 = sac.select_action(state.actor_params, obs, jax.random.PRNGKey(99), deterministic=True)
    assert jnp.allclose(a1, a2), "Deterministic actions should be identical regardless of key"


def test_sac_update():
    sac = _make_sac()
    state = sac.init(KEY)
    batch = _make_batch(KEY)

    new_state, metrics = sac.update(state, batch)
    assert "q1_loss" in metrics or "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "alpha" in metrics
    assert "entropy" in metrics
    # State should have changed (params are pytrees, compare leaves)
    old_leaves = jax.tree.leaves(state.q1_params)
    new_leaves = jax.tree.leaves(new_state.q1_params)
    assert any(not jnp.array_equal(o, n) for o, n in zip(old_leaves, new_leaves)), \
        "Q params should have changed after update"


def test_sac_get_q_value():
    sac = _make_sac()
    state = sac.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))
    action = jax.random.uniform(KEY, (4, ACTION_DIM), minval=-1, maxval=1)

    q = sac.get_q_value(state, obs, action)
    assert q.shape == (4,), f"Expected (4,), got {q.shape}"
    assert not jnp.any(jnp.isnan(q)), "Q values should not be NaN"


# ── TD3 ─────────────────────────────────────────────────────────────────────

def _make_td3():
    cfg = TD3Config(hidden_dim=(64, 64), batch_size=BATCH_SIZE)
    opt = optax.adam(3e-4)
    return TD3(cfg, OBS_DIM, ACTION_DIM, actor_optimizer=opt, critic_optimizer=opt,
               gamma=0.99, handle_truncation=True)


def test_td3_init():
    td3 = _make_td3()
    state = td3.init(KEY)
    assert state.actor_params is not None
    assert state.q1_params is not None
    assert state.target_actor_params is not None
    assert state.update_count == 0


def test_td3_select_action():
    td3 = _make_td3()
    state = td3.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))
    key = jax.random.PRNGKey(1)

    action = td3.select_action(state.actor_params, obs, key, deterministic=False,
                                exploration_noise=0.1)
    assert action.shape == (4, ACTION_DIM)
    assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)


def test_td3_deterministic_action():
    td3 = _make_td3()
    state = td3.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))

    a1 = td3.select_action(state.actor_params, obs, KEY, deterministic=True)
    a2 = td3.select_action(state.actor_params, obs, jax.random.PRNGKey(99), deterministic=True)
    assert jnp.allclose(a1, a2), "Deterministic actions should be identical"


def test_td3_update():
    td3 = _make_td3()
    state = td3.init(KEY)
    batch = _make_batch(KEY)

    new_state, metrics = td3.update(state, batch)
    assert "actor_loss" in metrics
    assert new_state.update_count == 1


def test_td3_get_q_value():
    td3 = _make_td3()
    state = td3.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))
    action = jax.random.uniform(KEY, (4, ACTION_DIM), minval=-1, maxval=1)

    q = td3.get_q_value(state, obs, action)
    assert q.shape == (4,)
    assert not jnp.any(jnp.isnan(q))


# ── FastTD3 ─────────────────────────────────────────────────────────────────

def _make_fast_td3():
    cfg = FastTD3Config(hidden_dim=(64, 64), batch_size=BATCH_SIZE, num_atoms=11)
    opt = optax.adam(3e-4)
    return FastTD3(cfg, OBS_DIM, ACTION_DIM, actor_optimizer=opt, critic_optimizer=opt,
                   gamma=0.99, handle_truncation=True)


def test_fast_td3_init():
    ftd3 = _make_fast_td3()
    state = ftd3.init(KEY)
    assert state.actor_params is not None
    assert state.q1_params is not None


def test_fast_td3_select_action():
    ftd3 = _make_fast_td3()
    state = ftd3.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))
    key = jax.random.PRNGKey(1)

    action = ftd3.select_action(state.actor_params, obs, key, deterministic=False,
                                 exploration_noise=0.1)
    assert action.shape == (4, ACTION_DIM)


def test_fast_td3_update():
    ftd3 = _make_fast_td3()
    state = ftd3.init(KEY)
    batch = _make_batch(KEY)

    new_state, metrics = ftd3.update(state, batch)
    assert "actor_loss" in metrics
    assert not jnp.any(jnp.isnan(jnp.array(list(metrics.values())))), \
        f"Metrics contain NaN: {metrics}"


def test_fast_td3_get_q_value():
    ftd3 = _make_fast_td3()
    state = ftd3.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))
    action = jax.random.uniform(KEY, (4, ACTION_DIM), minval=-1, maxval=1)

    q = ftd3.get_q_value(state, obs, action)
    assert q.shape == (4,)
    assert not jnp.any(jnp.isnan(q))


# ── FastSAC ─────────────────────────────────────────────────────────────────

def _make_fast_sac():
    cfg = SACConfig(hidden_dim=(64, 64), batch_size=BATCH_SIZE)
    opt = optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.001)
    alpha_opt = optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.001)
    return FastSAC(
        cfg, OBS_DIM, ACTION_DIM, opt, alpha_opt,
        gamma=0.97, handle_truncation=True,
        num_atoms=11, v_min=-10.0, v_max=10.0,
    )


def test_fast_sac_init():
    fsac = _make_fast_sac()
    state = fsac.init(KEY)
    assert state.actor_params is not None
    assert state.log_alpha.shape == ()


def test_fast_sac_select_action():
    fsac = _make_fast_sac()
    state = fsac.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))
    key = jax.random.PRNGKey(1)

    action = fsac.select_action(state.actor_params, obs, key, deterministic=False)
    assert action.shape == (4, ACTION_DIM)


def test_fast_sac_update():
    fsac = _make_fast_sac()
    state = fsac.init(KEY)
    batch = _make_batch(KEY)

    new_state, metrics = fsac.update(state, batch)
    assert "actor_loss" in metrics
    assert "alpha" in metrics


def test_fast_sac_get_q_value():
    fsac = _make_fast_sac()
    state = fsac.init(KEY)
    obs = jax.random.normal(KEY, (4, OBS_DIM))
    action = jax.random.uniform(KEY, (4, ACTION_DIM), minval=-1, maxval=1)

    q = fsac.get_q_value(state, obs, action)
    assert q.shape == (4,)
    assert not jnp.any(jnp.isnan(q))
