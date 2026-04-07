import jax
import jax.numpy as jnp
import pytest
from jax_rl.utils.reward_scaling import (
    RewardNormState, init_reward_norm, update_reward_stats, scale_reward,
)

def test_init_creates_correct_shapes():
    state = init_reward_norm(num_envs=4)
    assert state.G_r.shape == (4,)
    assert state.G_r_max.shape == ()
    assert state.G_mean.shape == ()
    assert state.G_var.shape == ()
    assert state.G_count.shape == ()
    assert state.G_var == 1.0

def test_update_resets_on_done():
    state = init_reward_norm(num_envs=2)
    reward = jnp.array([1.0, 2.0])
    terminated = jnp.array([0.0, 1.0])
    truncated = jnp.array([0.0, 0.0])
    new_state = update_reward_stats(state, reward, terminated, truncated, gamma=0.99)
    assert jnp.allclose(new_state.G_r, jnp.array([1.0, 2.0]))

def test_scale_reward_bounds_output():
    state = init_reward_norm(num_envs=1)
    state = state.replace(G_var=jnp.array(100.0), G_r_max=jnp.array(50.0))
    reward = jnp.array([10.0])
    scaled = scale_reward(state, reward, G_max=5.0)
    assert jnp.allclose(scaled, jnp.array([1.0]))

def test_scale_reward_uses_G_r_max_floor():
    state = init_reward_norm(num_envs=1)
    state = state.replace(G_var=jnp.array(0.01), G_r_max=jnp.array(25.0))
    reward = jnp.array([5.0])
    scaled = scale_reward(state, reward, G_max=5.0)
    assert jnp.allclose(scaled, jnp.array([1.0]))

def test_welford_update_accumulates():
    state = init_reward_norm(num_envs=4)
    key = jax.random.PRNGKey(0)
    for i in range(10):
        reward = jax.random.normal(key, (4,))
        key, _ = jax.random.split(key)
        terminated = jnp.zeros(4)
        truncated = jnp.zeros(4)
        state = update_reward_stats(state, reward, terminated, truncated, gamma=0.99)
    assert state.G_count > 0
    assert state.G_var > 0
