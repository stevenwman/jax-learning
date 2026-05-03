"""Bundle test for SplitbeltTreadmill env (lesson §7.5).

Marked [gpu, warp, go2] because make_env_bundle("Go2WarpSplitbelt", ...)
constructs the real Warp env (lesson §1).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]


@pytest.fixture
def cfg():
    # TrainConfig lives in jax_rl.configs.train_config (NOT jax_rl.training.train_config).
    from jax_rl.configs.train_config import TrainConfig
    return TrainConfig(env_name="Go2WarpSplitbelt", num_envs=4, total_timesteps=100)


def test_make_env_bundle_returns_populated(cfg):
    from jax_rl.training import make_env_bundle
    bundle = make_env_bundle(cfg, seed=0)
    assert bundle.backend_kind == "mjx"
    assert bundle.has_privileged is True
    assert bundle.dict_obs is True
    assert bundle.action_dim == 12
    assert bundle.obs_dim > 0
    assert bundle.critic_obs_dim is not None
    assert bundle.num_envs == 4   # mjx doesn't cap; should equal cfg.num_envs


def test_make_env_bundle_obs_schema_matches_name_layout(cfg):
    """Locks in deploy-side contract: schema_from_obs_groups output names ==
    obs_term_names() output names. Catches IncludeGroup expansion drift."""
    from jax_rl.training import make_env_bundle
    from jax_rl.envs.obs_spec import schema_from_obs_groups
    from jax_rl.envs.locomotion.go2_warp_splitbelt import obs_term_names
    bundle = make_env_bundle(cfg, seed=0)
    # bundle.env exposes the inner env via the wrapper chain.
    inner = bundle.env
    while hasattr(inner, "env"):
        inner = inner.env
    schema = schema_from_obs_groups(inner._obs_groups)
    layout = obs_term_names(inner._config.obs_mode)
    assert schema["state"] == layout["state"]
    state_set = set(layout["state"])
    expected_priv = layout["state"] + [n for n in layout["privileged_state"] if n not in state_set]
    assert schema["privileged_state"] == expected_priv
