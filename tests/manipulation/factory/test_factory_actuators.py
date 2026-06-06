"""Tests for FactoryPegInsert actuator mode swap (position-PD ↔ motor).

Phase 2 prereq: OSC needs raw torque control, not position-PD. The env exposes
an `actuator_mode` config that, when set to "motor", post-mutates the panda's
arm actuators (joints 1-7) to fixed-gain motor type so ctrl[:7] is interpreted
as raw torque.

Sanity gate: with identical ctrl=0, position-PD trajectory must diverge wildly
from motor trajectory (position-PD yanks toward q_target=0, motor lets gravity
take over).
"""
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────
# Hermetic (CPU) — actuator mutation correctness
# ─────────────────────────────────────────────────────────────────────


def test_default_actuator_mode_is_motor():
    """Default config must be 'motor' so SAC training engages the OSC."""
    from jax_rl.envs.manipulation.factory.factory_peg_insert import (
        FactoryPegInsert,
        default_config,
    )
    cfg = default_config()
    assert cfg.actuator_mode == "motor"


def test_position_pd_mode_keeps_menagerie_pd_schema():
    """Opting in to 'position_pd' preserves the menagerie panda schema."""
    from jax_rl.envs.manipulation.factory.factory_peg_insert import (
        FactoryPegInsert,
        default_config,
    )
    cfg = default_config()
    cfg.actuator_mode = "position_pd"
    env = FactoryPegInsert(cfg)
    # Position-PD (menagerie panda): gain=fixed (scalar 4500), bias=affine
    # → torque = gain*ctrl + biasprm[1]*q + biasprm[2]*qdot.
    assert env._mj_model.actuator_biastype[0] == int(mujoco.mjtBias.mjBIAS_AFFINE)
    assert env._mj_model.actuator_gaintype[0] == int(mujoco.mjtGain.mjGAIN_FIXED)
    # gainprm[0] = kp = 4500 for joints 1,2; 3500 for 3,4; 2000 for 5,6,7.
    assert env._mj_model.actuator_gainprm[0, 0] == pytest.approx(4500.0)
    # biasprm[1] = -kp (the position term in affine bias).
    assert env._mj_model.actuator_biasprm[0, 1] == pytest.approx(-4500.0)


def test_motor_mode_zeros_arm_bias():
    """actuator_mode='motor' must set arm actuators 0..6 to gain=fixed(1), bias=none."""
    from jax_rl.envs.manipulation.factory.factory_peg_insert import (
        FactoryPegInsert,
        default_config,
    )
    cfg = default_config()
    cfg.actuator_mode = "motor"
    env = FactoryPegInsert(cfg)
    m = env._mj_model
    for i in range(7):
        assert m.actuator_gaintype[i] == int(mujoco.mjtGain.mjGAIN_FIXED), (
            f"arm act {i} gaintype != FIXED"
        )
        assert m.actuator_biastype[i] == int(mujoco.mjtBias.mjBIAS_NONE), (
            f"arm act {i} biastype != NONE"
        )
        np.testing.assert_allclose(m.actuator_gainprm[i, 0], 1.0)
        np.testing.assert_allclose(m.actuator_gainprm[i, 1:], 0.0)
        np.testing.assert_allclose(m.actuator_biasprm[i, :], 0.0)
    # Gripper actuator (idx 7) must be untouched — still position-PD on tendon.
    assert m.actuator_biastype[7] == int(mujoco.mjtBias.mjBIAS_AFFINE)


def test_motor_mode_preserves_forcerange():
    """Forcerange (torque clip) must survive the swap."""
    from jax_rl.envs.manipulation.factory.factory_peg_insert import (
        FactoryPegInsert,
        default_config,
    )
    cfg = default_config()
    cfg.actuator_mode = "motor"
    env = FactoryPegInsert(cfg)
    m = env._mj_model
    # joints 1-4: ±87, joints 5-7: ±12 (panda menagerie).
    for i in range(4):
        np.testing.assert_allclose(m.actuator_forcerange[i], [-87, 87])
    for i in range(4, 7):
        np.testing.assert_allclose(m.actuator_forcerange[i], [-12, 12])


def test_invalid_actuator_mode_raises():
    from jax_rl.envs.manipulation.factory.factory_peg_insert import (
        FactoryPegInsert,
        default_config,
    )
    cfg = default_config()
    cfg.actuator_mode = "bogus"
    with pytest.raises(ValueError, match="actuator_mode"):
        FactoryPegInsert(cfg)


# ─────────────────────────────────────────────────────────────────────
# Warp physics — control-interface divergence sanity check
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def env_pos_pd():
    from jax_rl.envs.manipulation.factory.factory_peg_insert import (
        FactoryPegInsert,
        default_config,
    )
    cfg = default_config()
    # Default already position_pd, but be explicit.
    cfg.actuator_mode = "position_pd"
    return FactoryPegInsert(cfg)


@pytest.fixture(scope="module")
def env_motor():
    from jax_rl.envs.manipulation.factory.factory_peg_insert import (
        FactoryPegInsert,
        default_config,
    )
    cfg = default_config()
    cfg.actuator_mode = "motor"
    return FactoryPegInsert(cfg)


def _rollout_raw(env, ctrl_arm, n_steps=200):
    """Raw mjx.step rollout from env's reset state with explicit ctrl.

    Bypasses env.step (which has a Phase 1 placeholder that overrides ctrl
    with DEFAULT_ARM_QPOS regardless of the action argument). This isolates
    the actuator schema effect: same ctrl, different actuator schema → very
    different trajectories.
    """
    from mujoco import mjx
    nu = env._mj_model.nu
    ctrl = jp.zeros(nu)
    ctrl = ctrl.at[:7].set(jp.asarray(ctrl_arm))
    state = env.reset(jax.random.PRNGKey(0))
    data = state.data
    qpos_hist = [np.asarray(data.qpos)]
    for _ in range(n_steps):
        data = mjx.step(env.mjx_model, data.replace(ctrl=ctrl))
        qpos_hist.append(np.asarray(data.qpos))
    return np.stack(qpos_hist)


@pytest.mark.gpu
@pytest.mark.warp
def test_pos_pd_vs_motor_ctrl_zero_diverges(env_pos_pd, env_motor):
    """With identical zero action, pos-PD vs motor trajectories must diverge.

    Position-PD with ctrl=0 derives torque = -kp*q - kd*qdot, dragging arm
    toward q_target=0 (straight up). Motor with ctrl=0 = zero torque, arm
    falls under gravity from DEFAULT_ARM_QPOS. Should diverge >> noise.
    """
    qpos_pd = _rollout_raw(env_pos_pd, ctrl_arm=jp.zeros(7), n_steps=200)
    qpos_mt = _rollout_raw(env_motor,  ctrl_arm=jp.zeros(7), n_steps=200)

    arm_adr = env_pos_pd._arm_qposadr
    # Final-state divergence on the 7 arm joints.
    final_pd = qpos_pd[-1, arm_adr]
    final_mt = qpos_mt[-1, arm_adr]
    delta = np.abs(final_mt - final_pd)
    assert delta.max() > 0.3, (
        f"trajectories must diverge >0.3rad on at least one joint; got max={delta.max():.4f}, "
        f"per-joint={delta}"
    )


@pytest.mark.gpu
@pytest.mark.warp
def test_pos_pd_ctrl_zero_pulls_toward_zero_pose(env_pos_pd):
    """Position-PD with ctrl=0: joint4 should move AWAY from -1.97 toward 0."""
    qpos = _rollout_raw(env_pos_pd, ctrl_arm=jp.zeros(7), n_steps=200)
    arm_adr = env_pos_pd._arm_qposadr
    j4_traj = qpos[:, arm_adr[3]]
    init_j4 = env_pos_pd._init_arm_qpos[3]
    # joint4 starts at the IK-resolved init. ctrl=0 → q_target=0 → arm yanks
    # toward q=0. j4 should march toward 0 (gain in the +q4 direction).
    assert j4_traj[0] == pytest.approx(init_j4, abs=1e-4)
    assert j4_traj[-1] > j4_traj[0] + 0.1, (
        f"joint4 should drift toward 0 from {init_j4}; got {j4_traj[-1]}"
    )


@pytest.mark.gpu
@pytest.mark.warp
def test_motor_ctrl_zero_does_not_pull_toward_zero(env_motor):
    """Motor with ctrl=0: arm drifts under gravity, NOT toward q_target=0.

    With position-PD off, ctrl=0 means literally no torque. Joints sag under
    gravity. joint4 (elbow bent at -1.97) should NOT march to 0 like in pos-PD.
    """
    qpos = _rollout_raw(env_motor, ctrl_arm=jp.zeros(7), n_steps=200)
    arm_adr = env_motor._arm_qposadr
    j4_traj = qpos[:, arm_adr[3]]
    init_j4 = env_motor._init_arm_qpos[3]
    # With zero torque, j4 should NOT approach 0 (no position-PD bias).
    # Concretely: must not have crossed the half-way point from init toward 0.
    halfway = init_j4 / 2.0
    assert j4_traj[-1] < halfway, (
        f"motor + ctrl=0 should NOT drive joint4 toward 0; got {j4_traj[-1]:.4f}, halfway={halfway:.4f}"
    )
