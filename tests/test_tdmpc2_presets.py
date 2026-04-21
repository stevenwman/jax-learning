"""Tests for TD-MPC2 DMC presets."""
import pytest

from jax_rl.configs.env_presets import get_tdmpc2_preset, TDMPC2_PRESETS
from jax_rl.configs.tdmpc2_config import TDMPC2Config


def test_tdmpc2_presets_cheetah_run():
    cfg = get_tdmpc2_preset("CheetahRun")
    assert isinstance(cfg, TDMPC2Config)
    assert cfg.action_dim == 6
    assert cfg.episode_lengths == (1000,)
    assert cfg.discount == 0.995

def test_tdmpc2_presets_humanoid_run():
    cfg = get_tdmpc2_preset("HumanoidRun")
    assert cfg.action_dim == 21
    assert cfg.discount == 0.995

def test_tdmpc2_presets_acrobot_swingup():
    cfg = get_tdmpc2_preset("AcrobatSwingup")
    assert cfg.action_dim == 1
    assert cfg.episode_lengths == (1000,)

def test_tdmpc2_presets_dict_has_expected_envs():
    assert "CheetahRun" in TDMPC2_PRESETS
    assert "HumanoidRun" in TDMPC2_PRESETS
    assert "AcrobatSwingup" in TDMPC2_PRESETS

def test_tdmpc2_preset_fallback_for_unknown_env():
    """Unknown env → raises, since action_dim is required and we can't infer."""
    with pytest.raises(KeyError):
        get_tdmpc2_preset("NonexistentEnv")
