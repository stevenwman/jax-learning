"""Tests for ObsSpec -- composable observation groups."""
import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs


def test_single_group_no_noise():
    groups = {
        "state": [
            ObsTerm("a", lambda **kw: jnp.array([1.0, 2.0])),
            ObsTerm("b", lambda **kw: jnp.array([3.0])),
        ],
    }
    rng = jax.random.PRNGKey(0)
    obs, new_rng = compute_obs(groups, noise_level=0.0, rng=rng)
    assert "state" in obs
    assert obs["state"].shape == (3,)
    assert jnp.allclose(obs["state"], jnp.array([1.0, 2.0, 3.0]))


def test_noise_applied():
    groups = {
        "state": [
            ObsTerm("x", lambda **kw: jnp.zeros(4), noise_scale=1.0),
        ],
    }
    rng = jax.random.PRNGKey(0)
    obs, _ = compute_obs(groups, noise_level=1.0, rng=rng)
    assert not jnp.allclose(obs["state"], jnp.zeros(4))


def test_no_noise_when_scale_zero():
    groups = {
        "state": [
            ObsTerm("x", lambda **kw: jnp.ones(3), noise_scale=0.0),
        ],
    }
    rng = jax.random.PRNGKey(0)
    obs, _ = compute_obs(groups, noise_level=1.0, rng=rng)
    assert jnp.allclose(obs["state"], jnp.ones(3))


def test_kwargs_passed_to_terms():
    def needs_data(data=None, **kw):
        return data * 2

    groups = {"state": [ObsTerm("x", needs_data)]}
    rng = jax.random.PRNGKey(0)
    obs, _ = compute_obs(groups, noise_level=0.0, rng=rng, data=jnp.array([5.0]))
    assert jnp.allclose(obs["state"], jnp.array([10.0]))


def test_include_group():
    groups = {
        "state": [
            ObsTerm("a", lambda **kw: jnp.array([1.0, 2.0])),
        ],
        "privileged_state": [
            IncludeGroup("state"),
            ObsTerm("extra", lambda **kw: jnp.array([3.0])),
        ],
    }
    rng = jax.random.PRNGKey(0)
    obs, _ = compute_obs(groups, noise_level=0.0, rng=rng)
    assert obs["state"].shape == (2,)
    assert obs["privileged_state"].shape == (3,)
    assert jnp.allclose(obs["privileged_state"], jnp.array([1.0, 2.0, 3.0]))


def test_multiple_groups():
    groups = {
        "state": [ObsTerm("a", lambda **kw: jnp.ones(2))],
        "other": [ObsTerm("b", lambda **kw: jnp.zeros(3))],
    }
    rng = jax.random.PRNGKey(0)
    obs, _ = compute_obs(groups, noise_level=0.0, rng=rng)
    assert "state" in obs
    assert "other" in obs


def test_rng_advances():
    groups = {
        "state": [
            ObsTerm("a", lambda **kw: jnp.zeros(2), noise_scale=1.0),
            ObsTerm("b", lambda **kw: jnp.zeros(2), noise_scale=1.0),
        ],
    }
    rng = jax.random.PRNGKey(0)
    _, new_rng = compute_obs(groups, noise_level=1.0, rng=rng)
    assert not jnp.array_equal(rng, new_rng)


def test_empty_groups():
    obs, rng = compute_obs({}, noise_level=0.0, rng=jax.random.PRNGKey(0))
    assert obs == {}


def test_determinism():
    """Same RNG should produce same obs."""
    groups = {
        "state": [
            ObsTerm("x", lambda **kw: jnp.ones(4), noise_scale=0.5),
        ],
    }
    rng = jax.random.PRNGKey(42)
    obs1, _ = compute_obs(groups, noise_level=1.0, rng=rng)
    obs2, _ = compute_obs(groups, noise_level=1.0, rng=rng)
    assert jnp.allclose(obs1["state"], obs2["state"])
