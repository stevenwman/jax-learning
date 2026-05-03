"""Hermetic tests for env_presets: lock in env_name + key-field shape per registered preset.

Default-lane (no markers) — preset lookup is pure dict access; no env construction.
"""

from __future__ import annotations


def test_splitbelt_ppo_preset_shape():
    from jax_rl.configs.env_presets import PRESETS
    cfg = PRESETS["Go2WarpSplitbelt"]
    assert cfg.env_name == "Go2WarpSplitbelt"
    assert cfg.episode_length == 1250  # matches splitbelt env default (25 s)
    assert cfg.reset_mode == "per_step"  # DomainRandWrapper auto-reset path


def test_splitbelt_fast_sac_preset_shape():
    from jax_rl.configs.env_presets import FAST_SAC_PRESETS
    train_cfg, algo_cfg = FAST_SAC_PRESETS["Go2WarpSplitbelt"]
    assert train_cfg.env_name == "Go2WarpSplitbelt"
    assert train_cfg.episode_length == 1250
    assert train_cfg.reset_mode == "per_step"
    assert algo_cfg is not None  # uses _FAST_SAC_BASE_ALGO
