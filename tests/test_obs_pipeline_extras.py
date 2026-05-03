"""Non-regression + generic extras tests for ObsPipeline.make_buffer."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from jax_rl.training.obs_pipeline import ObsPipeline


def _make_pipeline(has_privileged=False, n_frame_stack=1):
    return ObsPipeline(
        dict_obs=False,
        has_privileged=has_privileged,
        use_obs_norm=False,
        n_frame_stack=n_frame_stack,
    )


def test_make_buffer_no_extras():
    """Default case — no critic, no extras. extra_obs_dims should be None."""
    pipe = _make_pipeline(has_privileged=False)
    buf = pipe.make_buffer(obs_dim=18, action_dim=6, buffer_size=1000)
    # Buffer should expose no extras
    assert buf._extra_obs_dims == {}


def test_make_buffer_critic_only_unchanged():
    """Asymmetric critic path — same behavior as before SD-B."""
    pipe = _make_pipeline(has_privileged=True)
    buf = pipe.make_buffer(
        obs_dim=18, action_dim=6, buffer_size=1000, critic_obs_dim=64
    )
    assert buf._extra_obs_dims == {"critic_obs": 64}


def test_make_buffer_extras_only():
    """New generic extras — no critic. Only skill_z + factor_obs."""
    pipe = _make_pipeline(has_privileged=False)
    buf = pipe.make_buffer(
        obs_dim=18, action_dim=6, buffer_size=1000,
        extra_obs_dims={"skill_z": 8, "factor_obs": 18},
    )
    assert buf._extra_obs_dims == {"skill_z": 8, "factor_obs": 18}


def test_make_buffer_critic_plus_extras_merged():
    """Both critic_obs and generic extras — should merge into one dict."""
    pipe = _make_pipeline(has_privileged=True)
    buf = pipe.make_buffer(
        obs_dim=18, action_dim=6, buffer_size=1000,
        critic_obs_dim=64,
        extra_obs_dims={"skill_z": 8, "factor_obs": 18},
    )
    assert buf._extra_obs_dims == {"critic_obs": 64, "skill_z": 8, "factor_obs": 18}


def test_make_buffer_critic_obs_in_extras_raises():
    """Caller should not pass critic_obs in extra_obs_dims when has_privileged=True
    — that would conflict with the auto-injected one. Raise to prevent silent overwrite."""
    pipe = _make_pipeline(has_privileged=True)
    with pytest.raises(ValueError, match="critic_obs"):
        pipe.make_buffer(
            obs_dim=18, action_dim=6, buffer_size=1000,
            critic_obs_dim=64,
            extra_obs_dims={"critic_obs": 999},  # would conflict
        )


def test_make_buffer_critic_required_when_privileged():
    """Existing constraint preserved — critic_obs_dim required when has_privileged."""
    pipe = _make_pipeline(has_privileged=True)
    with pytest.raises(ValueError, match="critic_obs_dim"):
        pipe.make_buffer(obs_dim=18, action_dim=6, buffer_size=1000)
