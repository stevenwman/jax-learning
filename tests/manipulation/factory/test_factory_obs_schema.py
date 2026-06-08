"""Hermetic tests for FactoryPegInsert obs schema + body/site/joint ID resolution.

These tests run on CPU (no Warp). mjx_model is lazy-constructed only when
physics stepping is needed, so env instantiation + schema queries are CPU-safe.
"""
import jax
import jax.numpy as jp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def env():
    from jax_rl.envs.manipulation.factory.factory_peg_insert import FactoryPegInsert
    return FactoryPegInsert()


def test_env_constructs(env):
    """Env builds without error on CPU (mjx_model lazy)."""
    assert env._mj_model is not None
    assert env._hand_body_id >= 0
    assert env._peg_body_id >= 0
    assert env._hole_body_id >= 0
    assert env._fingertip_site_id >= 0


def test_action_size_6(env):
    assert env.action_size == 6


def test_obs_groups_present(env):
    assert "state" in env._obs_groups
    assert "privileged_state" in env._obs_groups


def test_arm_joints_resolved(env):
    """7 panda arm joints, qpos addresses sorted ascending."""
    assert len(env._arm_jnt_ids) == 7
    assert len(env._arm_qposadr) == 7
    assert all(np.diff(env._arm_qposadr) >= 0), "qpos addresses should be monotonic"


def test_state_dim_25(env):
    """state obs should sum to 25 dims:
        fingertip_pos_rel_fixed (3) + fingertip_quat (4) + ee_linvel (3)
        + ee_angvel (3) + actions (6) + prev_actions (6) = 25
    """
    from jax_rl.envs.obs_spec import compute_obs
    import mujoco
    data = mujoco.MjData(env._mj_model)
    mujoco.mj_forward(env._mj_model, data)
    info = {
        "fixed_pos": jp.array([0.6, 0.0, 0.0]),
        "fixed_quat": jp.array([1.0, 0.0, 0.0, 0.0]),
        "actions": jp.zeros(6),
        "prev_actions": jp.zeros(6),
        "prev_fingertip_pos": jp.array(data.site_xpos[env._fingertip_site_id]),
        "prev_fingertip_quat": jp.array([1.0, 0.0, 0.0, 0.0]),
    }
    # Use mj_data directly (compute_obs reads .site_xpos[i], .qpos[i] etc.)
    obs, _ = compute_obs(
        env._obs_groups, noise_level=0.0, rng=jax.random.PRNGKey(0),
        data=data, info=info,
    )
    assert obs["state"].shape == (25,), f"state shape {obs['state'].shape} != (25,)"


def test_privileged_state_strictly_larger(env):
    """privileged_state includes state via IncludeGroup, plus extra fields."""
    from jax_rl.envs.obs_spec import compute_obs
    import mujoco
    data = mujoco.MjData(env._mj_model)
    mujoco.mj_forward(env._mj_model, data)
    info = {
        "fixed_pos": jp.array([0.6, 0.0, 0.0]),
        "fixed_quat": jp.array([1.0, 0.0, 0.0, 0.0]),
        "actions": jp.zeros(6),
        "prev_actions": jp.zeros(6),
        "prev_fingertip_pos": jp.array(data.site_xpos[env._fingertip_site_id]),
        "prev_fingertip_quat": jp.array([1.0, 0.0, 0.0, 0.0]),
    }
    obs, _ = compute_obs(
        env._obs_groups, noise_level=0.0, rng=jax.random.PRNGKey(0),
        data=data, info=info,
    )
    state_dim = obs["state"].shape[0]
    priv_dim = obs["privileged_state"].shape[0]
    assert priv_dim > state_dim
    # Privileged state expected ≈ 64-72d (state 25 + fingertip_pos 3 + joint_pos 7
    # + joint_vel 7 + held_pos 3 + held_pos_rel 3 + held_quat 4 + fixed_pos 3
    # + fixed_quat 4 + task_prop_gains 6 + ema_factor 1 + pos_threshold 3
    # + rot_threshold 3 = 72)
    assert 60 <= priv_dim <= 80, f"privileged_state dim {priv_dim} outside expected range"
