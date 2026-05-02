"""Tests for DIAYN auxiliary module."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import optax

from jax_rl.skill_discovery.diayn import Discriminator, diayn_reward, diayn_update

KEY = jax.random.PRNGKey(42)
BATCH = 32


def test_discriminator_init_and_forward():
    disc = Discriminator(hidden_dim=(64, 64), num_skills=10)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    logits = disc.apply(params, jnp.zeros((BATCH, 5)))
    assert logits.shape == (BATCH, 10)


def test_diayn_reward_finite_and_correct_shape():
    disc = Discriminator(hidden_dim=(64, 64), num_skills=10)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    obs = jax.random.normal(KEY, (BATCH, 5))
    z = jax.nn.one_hot(jnp.zeros(BATCH, dtype=jnp.int32), 10)
    r = diayn_reward(params, disc, obs, z, num_skills=10)
    assert r.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(r))


def test_diayn_reward_deterministic_for_fixed_params_and_batch():
    """Sample-time reward must be deterministic — required by SD-A acceptance."""
    disc = Discriminator(hidden_dim=(64, 64), num_skills=4)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, 5))
    z = jax.nn.one_hot(jnp.arange(BATCH) % 4, 4)
    r1 = diayn_reward(params, disc, obs, z, num_skills=4)
    r2 = diayn_reward(params, disc, obs, z, num_skills=4)
    assert jnp.array_equal(r1, r2)


def test_diayn_update_reduces_loss():
    disc = Discriminator(hidden_dim=(64, 64), num_skills=4)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    # Synthetic separable data: class i has obs = e_i (orthogonal one-hot direction)
    class_means = jnp.eye(4, 5)  # shape (4, 5) — class i's mean is the i-th 5-dim one-hot
    obs = jnp.concatenate([jnp.tile(class_means[i], (8, 1)) for i in range(4)])
    z_idx = jnp.concatenate([jnp.full(8, i, dtype=jnp.int32) for i in range(4)])

    for _ in range(50):
        params, opt_state, m = diayn_update(params, opt_state, disc, opt, obs, z_idx)

    logits = disc.apply(params, obs)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == z_idx)
    assert acc > 0.5, f"discriminator acc {acc:.2f} should exceed 0.5 after 50 updates"


def test_diayn_update_metrics_keys():
    disc = Discriminator(hidden_dim=(32,), num_skills=4)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    obs = jax.random.normal(KEY, (16, 5))
    z_idx = jax.random.randint(KEY, (16,), 0, 4)
    _, _, metrics = diayn_update(params, opt_state, disc, opt, obs, z_idx)
    assert "disc_loss" in metrics
    assert "disc_accuracy" in metrics
