"""Hermetic obs name-layout tests for SplitbeltTreadmill env (S§10.1, lesson §7.1)."""

from __future__ import annotations

import pytest

from jax_rl.envs.locomotion.go2_warp_splitbelt import obs_term_names


_VALID_MODES = ("blind", "informed", "error", "history", "pose_track")


@pytest.mark.parametrize("obs_mode", _VALID_MODES)
def test_obs_term_names_has_state_and_privileged(obs_mode):
    layout = obs_term_names(obs_mode)
    assert "state" in layout
    assert "privileged_state" in layout
    assert isinstance(layout["state"], list)
    assert isinstance(layout["privileged_state"], list)


def test_blind_state_excludes_belt_speeds():
    layout = obs_term_names("blind")
    assert "belt_vel" not in layout["state"]


def test_informed_state_includes_belt_speeds():
    layout = obs_term_names("informed")
    assert "belt_vel" in layout["state"]


def test_error_state_includes_tracking_error_not_belt():
    layout = obs_term_names("error")
    assert "cmd_track_error" in layout["state"]
    assert "drift_xy" in layout["state"]
    assert "belt_vel" not in layout["state"]


def test_privileged_always_full_info():
    for mode in _VALID_MODES:
        priv = obs_term_names(mode)["privileged_state"]
        for required in ("belt_vel", "cmd_track_error", "drift_xy"):
            assert required in priv, f"mode={mode}: privileged missing {required}"


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="obs_mode"):
        obs_term_names("not_a_real_mode")


@pytest.mark.parametrize("obs_mode", _VALID_MODES)
def test_build_obs_groups_matches_obs_term_names(obs_mode):
    """Structural contract: schema-expanded names in build_obs_groups match
    obs_term_names. Hermetic — uses a fake env with the minimal interface
    build_obs_groups touches (just `_config.obs_mode` + `_config.noise_config.scales`).
    Catches IncludeGroup expansion drift in the default lane (no GPU needed).
    """
    from types import SimpleNamespace
    from jax_rl.envs.locomotion.go2_warp_splitbelt import build_obs_groups
    from jax_rl.envs.obs_spec import schema_from_obs_groups

    fake_noise = SimpleNamespace(
        joint_pos=0.0, joint_vel=0.0, gyro=0.0, gravity=0.0,
        linvel=0.0, accelerometer=0.0,
    )
    fake_env = SimpleNamespace(
        _config=SimpleNamespace(
            obs_mode=obs_mode,
            noise_config=SimpleNamespace(scales=fake_noise),
        ),
        # Methods that term_factory lambdas might capture — never invoked
        # by schema_from_obs_groups (it only walks names), so any callable suffices.
        get_gravity=lambda data: None,
        get_gyro=lambda data: None,
        get_local_linvel=lambda data: None,
        get_global_angvel=lambda data: None,
        _default_pose=None,
    )
    groups = build_obs_groups(fake_env)
    schema = schema_from_obs_groups(groups)
    layout = obs_term_names(obs_mode)
    assert schema["state"] == layout["state"]
    state_set = set(layout["state"])
    expected_priv = layout["state"] + [n for n in layout["privileged_state"] if n not in state_set]
    assert schema["privileged_state"] == expected_priv
