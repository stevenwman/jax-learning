"""Tests for RewardSpec — composable reward terms."""
import jax.numpy as jnp
import pytest

from jax_rl.envs.reward_spec import RewardTerm, compute_rewards


def test_compute_rewards_returns_dict():
    terms = [
        RewardTerm("a", lambda **kw: jnp.float32(1.0)),
        RewardTerm("b", lambda **kw: jnp.float32(2.0)),
    ]
    result = compute_rewards(terms)
    assert "a" in result
    assert "b" in result
    assert jnp.allclose(result["a"], 1.0)
    assert jnp.allclose(result["b"], 2.0)


def test_compute_rewards_passes_kwargs():
    def my_reward(data=None, **kw):
        return data * 2.0

    terms = [RewardTerm("x", my_reward)]
    result = compute_rewards(terms, data=jnp.float32(5.0))
    assert jnp.allclose(result["x"], 10.0)


def test_empty_spec():
    result = compute_rewards([])
    assert result == {}


def test_reward_term_name():
    fn = lambda **kw: jnp.float32(0.0)
    t = RewardTerm("test", fn)
    assert t.name == "test"
    assert t.fn is fn


def test_multiple_kwargs():
    def needs_both(action=None, info=None, **kw):
        return action + info

    terms = [RewardTerm("r", needs_both)]
    result = compute_rewards(terms, action=jnp.float32(3.0), info=jnp.float32(4.0))
    assert jnp.allclose(result["r"], 7.0)
