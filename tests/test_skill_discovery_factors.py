"""Tests for factor extractor registry."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import pytest

from jax_rl.skill_discovery.factors import (
    FactorExtractor, register_extractor, get_extractor, resolve_factor,
)
from jax_rl.skill_discovery.config import FactorConfig

KEY = jax.random.PRNGKey(0)


def test_register_and_get_extractor():
    @register_extractor(name="test_full_actor", source="actor_obs", dim=48)
    def _extract(batch):
        return batch["obs"]

    ext = get_extractor("test_full_actor")
    assert ext.name == "test_full_actor"
    assert ext.source == "actor_obs"
    assert ext.dim == 48


def test_get_unknown_extractor_raises():
    with pytest.raises(KeyError, match="unknown extractor"):
        get_extractor("definitely_not_registered_xyz")


def test_resolve_factor_pulls_named_extractor():
    @register_extractor(name="test_resolve_xy", source="sim_data", dim=2)
    def _extract(batch):
        return batch["sim_data"][:, :2]

    fc = FactorConfig(name="pos", method="metra", skill_dim=2,
                      source="sim_data", extractor="test_resolve_xy", dim=2)
    batch = {"sim_data": jax.random.normal(KEY, (4, 6))}
    out = resolve_factor(fc, batch)
    assert out.shape == (4, 2)
    assert jnp.array_equal(out, batch["sim_data"][:, :2])


def test_resolve_factor_dim_mismatch_raises():
    @register_extractor(name="test_dim_mismatch", source="sim_data", dim=2)
    def _extract(batch):
        return batch["sim_data"][:, :3]  # returns 3, declared 2

    fc = FactorConfig(name="bad", method="metra", skill_dim=2,
                      source="sim_data", extractor="test_dim_mismatch", dim=2)
    batch = {"sim_data": jax.random.normal(KEY, (4, 6))}
    with pytest.raises(ValueError, match="dim mismatch"):
        resolve_factor(fc, batch)


def test_builtin_extractors_actor_obs_full():
    """Built-in: full actor_obs passthrough."""
    fc = FactorConfig(name="all", method="diayn", skill_dim=4,
                      source="actor_obs", extractor="actor_obs_full", dim=48)
    batch = {"obs": jax.random.normal(KEY, (4, 48))}
    out = resolve_factor(fc, batch)
    assert jnp.array_equal(out, batch["obs"])
