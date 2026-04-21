"""Tests for PPOContraction algorithm (Task 6)."""

import jax
import jax.numpy as jnp
import optax
import pytest

from jax_rl.algos import PPOContraction
from jax_rl.buffers import RolloutBatch
from jax_rl.configs import (
    ContractionConfig,
    EncoderConfig,
    PolicyHeadConfig,
    PPOConfig,
)


OBS_DIM = 17
ACTION_DIM = 6
CONSTRAINT_DIM = 3
NUM_ENVS = 4
NUM_STEPS = 32
KEY = jax.random.PRNGKey(0)


def _make_ppo(enable_contraction=True):
    encoder = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64))
    critic_enc = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64))
    policy = PolicyHeadConfig(action_dim=ACTION_DIM, squash=True)
    cc = None
    if enable_contraction:
        cc = ContractionConfig(constraint_dim=CONSTRAINT_DIM, hidden_dims=(32,))
    config = PPOConfig(
        encoder=encoder, critic_encoder=critic_enc, policy_head=policy,
        num_envs=NUM_ENVS, gamma=0.99,
        minibatch_size=NUM_ENVS * NUM_STEPS // 4,
        contraction=cc,
    )
    ppo = PPOContraction(
        config, OBS_DIM, ACTION_DIM,
        optax.adam(3e-4), optax.adam(3e-4),
    )
    return ppo, config, ppo.init(KEY)


def _make_batch(with_contraction=True):
    keys = jax.random.split(KEY, 9)
    batch_kwargs = dict(
        obs=jax.random.normal(keys[0], (NUM_STEPS, NUM_ENVS, OBS_DIM)),
        actions=jax.random.normal(keys[1], (NUM_STEPS, NUM_ENVS, ACTION_DIM)),
        rewards=jax.random.normal(keys[2], (NUM_STEPS, NUM_ENVS)),
        dones=jnp.zeros((NUM_STEPS, NUM_ENVS)),
        truncations=jnp.zeros((NUM_STEPS, NUM_ENVS)),
        log_probs=jax.random.normal(keys[3], (NUM_STEPS, NUM_ENVS)),
        values=jax.random.normal(keys[4], (NUM_STEPS, NUM_ENVS)),
    )
    if with_contraction:
        batch_kwargs["contraction_c"] = jax.random.normal(keys[5], (NUM_STEPS, NUM_ENVS, CONSTRAINT_DIM)) * 0.2
        batch_kwargs["contraction_c_dot"] = jax.random.normal(keys[6], (NUM_STEPS, NUM_ENVS, CONSTRAINT_DIM)) * 0.2
    return RolloutBatch(**batch_kwargs)


def test_init_without_contraction_has_none_metric():
    ppo, _, state = _make_ppo(enable_contraction=False)
    assert state.metric_params is None
    assert state.metric_opt_state is None


def test_init_with_contraction_has_metric():
    ppo, _, state = _make_ppo(enable_contraction=True)
    assert state.metric_params is not None
    assert state.metric_opt_state is not None
    # params should be a non-empty pytree
    n = sum(x.size for x in jax.tree.leaves(state.metric_params))
    assert n > 0


def test_init_validate_raises_on_constraint_dim_zero():
    encoder = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64,))
    policy = PolicyHeadConfig(action_dim=ACTION_DIM)
    cc = ContractionConfig()  # constraint_dim=0 default
    config = PPOConfig(
        encoder=encoder, critic_encoder=encoder, policy_head=policy,
        num_envs=NUM_ENVS, gamma=0.99, minibatch_size=8, contraction=cc,
    )
    with pytest.raises(ValueError, match="constraint_dim"):
        PPOContraction(config, OBS_DIM, ACTION_DIM, optax.adam(3e-4), optax.adam(3e-4))


def test_update_with_contraction_produces_metric_metrics():
    ppo, _, state = _make_ppo(enable_contraction=True)
    batch = _make_batch(with_contraction=True)
    next_obs = jax.random.normal(KEY, (NUM_ENVS, OBS_DIM))
    new_state, metrics = ppo.update(state, batch, KEY, next_obs=next_obs)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    # new metrics from contraction path
    assert "contraction_penalty" in metrics
    assert "V_mean" in metrics
    assert "V_dot_mean" in metrics
    for v in metrics.values():
        assert not jnp.isnan(v), f"NaN in {list(metrics)}"


def test_update_changes_metric_params():
    ppo, _, state = _make_ppo(enable_contraction=True)
    batch = _make_batch(with_contraction=True)
    next_obs = jax.random.normal(KEY, (NUM_ENVS, OBS_DIM))
    new_state, _ = ppo.update(state, batch, KEY, next_obs=next_obs)

    # metric_params should have moved
    old_flat = jax.tree.leaves(state.metric_params)
    new_flat = jax.tree.leaves(new_state.metric_params)
    any_diff = any(
        not jnp.allclose(a, b) for a, b in zip(old_flat, new_flat)
    )
    assert any_diff, "metric_params did not change after update"


def test_update_without_contraction_matches_ppo_baseline_actor():
    """PPOContraction with contraction=None must produce identical actor/critic
    updates to baseline PPO on the same seed + batch."""
    from jax_rl.algos import PPO

    def _make_both():
        encoder = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64))
        critic_enc = EncoderConfig(obs_dim=OBS_DIM, hidden_dim=(64, 64))
        policy = PolicyHeadConfig(action_dim=ACTION_DIM, squash=True)
        config = PPOConfig(
            encoder=encoder, critic_encoder=critic_enc, policy_head=policy,
            num_envs=NUM_ENVS, gamma=0.99,
            minibatch_size=NUM_ENVS * NUM_STEPS // 4,
            contraction=None,
        )
        ppo_a = PPO(config, OBS_DIM, ACTION_DIM, optax.adam(3e-4), optax.adam(3e-4))
        ppo_b = PPOContraction(config, OBS_DIM, ACTION_DIM, optax.adam(3e-4), optax.adam(3e-4))
        return ppo_a, ppo_b, config

    ppo_a, ppo_b, _ = _make_both()
    sa = ppo_a.init(KEY)
    sb = ppo_b.init(KEY)
    batch = _make_batch(with_contraction=False)
    next_obs = jax.random.normal(KEY, (NUM_ENVS, OBS_DIM))

    na, _ = ppo_a.update(sa, batch, KEY, next_obs=next_obs)
    nb, _ = ppo_b.update(sb, batch, KEY, next_obs=next_obs)

    a_leaves = jax.tree.leaves(na.actor_params)
    b_leaves = jax.tree.leaves(nb.actor_params)
    for a, b in zip(a_leaves, b_leaves):
        assert jnp.allclose(a, b, atol=1e-6), "actor params diverged"
    c_leaves = jax.tree.leaves(na.critic_params)
    d_leaves = jax.tree.leaves(nb.critic_params)
    for a, b in zip(c_leaves, d_leaves):
        assert jnp.allclose(a, b, atol=1e-6), "critic params diverged"
