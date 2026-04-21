"""Tests for ContractionConfig validation."""

import pytest

from jax_rl.configs import ContractionConfig, PPOConfig


def test_default_constraint_dim_is_zero():
    """Default is 0 — must be populated at runtime from env."""
    cfg = ContractionConfig()
    assert cfg.constraint_dim == 0


def test_validate_raises_when_constraint_dim_unset():
    cfg = ContractionConfig()
    with pytest.raises(ValueError, match="constraint_dim"):
        cfg.validate()


def test_validate_passes_when_constraint_dim_set():
    cfg = ContractionConfig()
    cfg.constraint_dim = 3
    cfg.validate()  # should not raise


def test_ppo_config_accepts_contraction():
    """PPOConfig.contraction is optional, default None."""
    assert PPOConfig().contraction is None
    c = ContractionConfig()
    c.constraint_dim = 3
    pcfg = PPOConfig(contraction=c)
    assert pcfg.contraction is c
