"""Tests for skill priors."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import pytest

from jax_rl.skill_discovery.prior import sample_skill, validate_skill

KEY = jax.random.PRNGKey(0)


def test_one_hot_shape_and_values():
    z = sample_skill(KEY, prior="one_hot", num_envs=16, skill_dim=8)
    assert z.shape == (16, 8)
    # one-hot: each row sums to 1, all entries 0 or 1
    assert jnp.allclose(z.sum(axis=-1), 1.0)
    assert jnp.all((z == 0) | (z == 1))


def test_one_hot_uniform_distribution_over_skills():
    """Sampling many envs should approximate uniform over skill indices."""
    z = sample_skill(KEY, prior="one_hot", num_envs=10_000, skill_dim=4)
    counts = z.sum(axis=0)
    # each skill should get ~2500 ± 200 (3-sigma loose bound)
    assert jnp.all(counts > 2000)
    assert jnp.all(counts < 3000)


def test_one_hot_deterministic_under_same_key():
    z1 = sample_skill(KEY, prior="one_hot", num_envs=8, skill_dim=4)
    z2 = sample_skill(KEY, prior="one_hot", num_envs=8, skill_dim=4)
    assert jnp.array_equal(z1, z2)


def test_dirichlet_not_implemented_in_sd_a():
    with pytest.raises(NotImplementedError, match="SD-E"):
        sample_skill(KEY, prior="dirichlet", num_envs=4, skill_dim=4)


def test_unit_sphere_prior_shape_and_norm():
    """METRA continuous z prior: rows sampled from N(0,I) and normalized."""
    z = sample_skill(KEY, prior="unit_sphere", num_envs=32, skill_dim=4)
    assert z.shape == (32, 4)
    norms = jnp.linalg.norm(z, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)


def test_unit_sphere_prior_deterministic_under_same_key():
    z1 = sample_skill(KEY, prior="unit_sphere", num_envs=8, skill_dim=4)
    z2 = sample_skill(KEY, prior="unit_sphere", num_envs=8, skill_dim=4)
    assert jnp.array_equal(z1, z2)


def test_validate_skill_one_hot():
    z = jax.nn.one_hot(jnp.arange(4), 4)
    validate_skill(z, prior="one_hot", skill_dim=4)  # no raise

    bad = jnp.ones((4, 4)) * 0.5  # not one-hot
    with pytest.raises(ValueError, match="one-hot"):
        validate_skill(bad, prior="one_hot", skill_dim=4)


def test_validate_skill_unit_sphere():
    """validate_skill accepts unit-norm rows, rejects non-unit rows."""
    rng = jax.random.PRNGKey(7)
    v = jax.random.normal(rng, (16, 4))
    z = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    validate_skill(z, prior="unit_sphere", skill_dim=4)  # no raise

    bad = jnp.ones((4, 4))  # rows have norm=2, not 1
    with pytest.raises(ValueError, match="unit_sphere"):
        validate_skill(bad, prior="unit_sphere", skill_dim=4)
