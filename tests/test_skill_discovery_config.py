"""Tests for skill discovery config dataclasses."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.skill_discovery.config import (
    SkillDiscoveryConfig, FactorConfig, SkillDeployConfig,
    config_to_dict, config_from_dict,
)


def test_factor_config_minimal():
    fc = FactorConfig(
        name="full_state", method="diayn", skill_dim=8,
        source="actor_obs", extractor="full", dim=48,
    )
    assert fc.name == "full_state"
    assert fc.method == "diayn"


def test_skill_discovery_config_defaults():
    cfg = SkillDiscoveryConfig()
    assert cfg.enabled is True
    assert cfg.mode == "diayn"
    assert cfg.prior == "one_hot"
    assert cfg.resample == "episode"
    assert cfg.reward_mode == "sample_time"
    assert cfg.intrinsic_weight == 1.0
    assert cfg.task_reward_weight == 0.0
    assert cfg.style_reward_weight == 0.0
    assert cfg.safety_penalty_weight == 0.0
    assert cfg.factors == ()
    assert isinstance(cfg.deploy, SkillDeployConfig)


def test_skill_deploy_config_defaults():
    d = SkillDeployConfig()
    assert d.skill_input_mode == "fixed"
    assert d.default_skill is None  # set at deploy time


def test_config_json_round_trip():
    cfg = SkillDiscoveryConfig(
        mode="diayn",
        total_skill_dim=8,
        factors=(
            FactorConfig(name="full_state", method="diayn", skill_dim=8,
                         source="actor_obs", extractor="full", dim=48),
        ),
        deploy=SkillDeployConfig(
            skill_input_mode="fixed",
            default_skill=[1, 0, 0, 0, 0, 0, 0, 0],
        ),
    )
    d = config_to_dict(cfg)
    s = json.dumps(d)
    cfg2 = config_from_dict(json.loads(s))
    assert cfg2 == cfg


def test_config_validation_total_skill_dim_matches_factors():
    """If factors are set, total_skill_dim must match sum(skill_dim)."""
    import pytest
    with pytest.raises(ValueError, match="total_skill_dim"):
        SkillDiscoveryConfig(
            total_skill_dim=4,  # wrong — factors sum to 8
            factors=(
                FactorConfig(name="a", method="diayn", skill_dim=8,
                             source="actor_obs", extractor="full", dim=48),
            ),
        )


def test_config_resample_steps_required_when_fixed():
    import pytest
    with pytest.raises(ValueError, match="resample_steps"):
        SkillDiscoveryConfig(resample="fixed_steps", resample_steps=None)
