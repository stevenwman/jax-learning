"""Tests for EnvBundle + make_env_bundle wrapper."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.configs.env_presets import get_sac_preset
from jax_rl.training import make_env_bundle, EnvBundle


def test_env_bundle_flat_obs_cheetahrun():
    """CheetahRun has flat array obs — bundle should have dict_obs=False."""
    cfg, _ = get_sac_preset("CheetahRun")
    cfg.num_envs = 4  # keep test fast
    bundle = make_env_bundle(cfg, seed=0)

    assert isinstance(bundle, EnvBundle)
    assert bundle.dict_obs is False
    assert bundle.has_privileged is False
    assert bundle.critic_obs_dim is None
    assert bundle.obs_dim > 0  # don't hardcode — Playground spec could change
    assert bundle.action_dim > 0
    assert bundle.key is not None


@pytest.mark.slow
def test_env_bundle_dict_obs_go2warp():
    """Go2WarpJoystickFlat has dict obs with privileged_state — bundle should reflect that."""
    cfg, _ = get_sac_preset("Go2WarpJoystickFlat")
    cfg.num_envs = 4
    bundle = make_env_bundle(cfg, seed=0)

    assert bundle.dict_obs is True
    assert bundle.has_privileged is True
    assert bundle.critic_obs_dim is not None and bundle.critic_obs_dim > bundle.obs_dim
    assert bundle.obs_dim > 0  # don't hardcode — env spec evolves over time
