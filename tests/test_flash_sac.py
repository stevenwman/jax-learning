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


# ── reward_norm_state persistence ─────────────────────────────────────────


def test_init_sets_reward_norm_state():
    """TrainingState should ship with fresh RewardNormState matching num_envs."""
    from jax_rl.utils.reward_scaling import RewardNormState
    algo = _make_flash_sac()
    state = algo.init(KEY)
    assert isinstance(state.reward_norm_state, RewardNormState)
    # Default num_envs=1 in _make_flash_sac. G_r shape = (num_envs,).
    assert state.reward_norm_state.G_r.shape == (1,)
    assert float(state.reward_norm_state.G_r_max) == 0.0
    assert float(state.reward_norm_state.G_count) == 0.0


def test_reward_norm_state_roundtrips_via_orbax(tmp_path):
    """Save a TrainingState with updated reward_norm_state; orbax restore preserves it."""
    import orbax.checkpoint as ocp
    from jax_rl.utils.reward_scaling import update_reward_stats
    algo = _make_flash_sac()
    state = algo.init(KEY)

    # Mutate reward_norm_state via the update function.
    dummy_reward = jnp.array([1.5])
    dummy_done = jnp.array([False])
    dummy_trunc = jnp.array([False])
    new_rns = update_reward_stats(
        state.reward_norm_state, dummy_reward, dummy_done, dummy_trunc, gamma=0.99,
    )
    state = state.replace(reward_norm_state=new_rns)
    assert float(state.reward_norm_state.G_r[0]) != 0.0  # sanity

    # Save via orbax (simulates ckpt save_checkpoint path)
    ckpt_dir = str(tmp_path / "orbax")
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {"training_state": state}
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.abspath(ckpt_dir), ckpt, force=True)
    checkpointer.wait_until_finished()

    # Restore against a fresh init target
    fresh = algo.init(jax.random.PRNGKey(0))
    target = {"training_state": fresh}
    restored = ocp.StandardCheckpointer().restore(os.path.abspath(ckpt_dir), target=target)
    restored_rns = restored["training_state"].reward_norm_state

    # Must match what we saved, not the fresh target.
    assert jnp.allclose(restored_rns.G_r, state.reward_norm_state.G_r)
    assert jnp.allclose(restored_rns.G_r_max, state.reward_norm_state.G_r_max)
    assert jnp.allclose(restored_rns.G_var, state.reward_norm_state.G_var)
    assert float(restored_rns.G_count) == float(state.reward_norm_state.G_count)
