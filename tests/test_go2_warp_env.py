"""Tests for Go2 Warp environment."""
import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]

from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick, default_config


@pytest.fixture
def env():
    return WarpJoystick(task="flat_terrain")

@pytest.fixture
def state(env):
    return env.reset(jax.random.PRNGKey(0))


class TestWarpGo2Loads:
    def test_action_size(self, env):
        assert env.action_size == 12

    def test_root_body_is_base_link(self, env):
        body_id = env.mj_model.body("base_link").id
        assert body_id == env._torso_body_id

    def test_obs_dict_keys(self, state):
        assert isinstance(state.obs, dict)
        assert "state" in state.obs
        assert "privileged_state" in state.obs

    def test_obs_dims(self, state):
        assert state.obs["state"].shape == (48,)
        assert state.obs["privileged_state"].shape == (122,)

    def test_reset_shapes(self, state):
        assert state.reward.shape == ()
        assert state.done.shape == ()

    def test_config_impl_is_warp(self):
        cfg = default_config()
        assert cfg.impl == "warp"
        assert cfg.contact_mode == "training"

    def test_unitree_native_values(self, env):
        assert env.mj_model.dof_damping[6] == pytest.approx(0.1, abs=0.01)
        assert env.mj_model.dof_frictionloss[6] == pytest.approx(0.2, abs=0.01)


class TestWarpGo2Steps:
    def test_step_zero_action(self, env, state):
        action = jnp.zeros(12)
        next_state = env.step(state, action)
        assert isinstance(next_state.obs, dict)
        assert next_state.obs["state"].shape == (48,)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))
        assert not jnp.any(jnp.isnan(next_state.reward))

    def test_step_random_action(self, env, state):
        key = jax.random.PRNGKey(42)
        action = jax.random.uniform(key, (12,), minval=-1.0, maxval=1.0)
        next_state = env.step(state, action)
        assert next_state.obs["state"].shape == (48,)

    def test_reward_nonzero_after_steps(self, env, state):
        action = jnp.zeros(12)
        for _ in range(5):
            state = env.step(state, action)
        assert jnp.isfinite(state.reward)


class TestWarpDomainRand:
    def test_dr_specs_declared(self, env):
        specs = env.get_domain_randomization_spec()
        model_specs = [s for s in specs if s.type == "model"]
        runtime_specs = [s for s in specs if s.type == "runtime"]
        assert len(model_specs) == 8
        assert len(runtime_specs) == 0
        names = {s.name for s in model_specs}
        assert "torso_com_jitter" in names
        assert "body_inertia" in names


class TestWarpContactModes:
    def test_training_mode_overrides_contacts(self):
        env = WarpJoystick(task="flat_terrain")
        gid = env.mj_model.geom("FL").id
        assert env.mj_model.geom_condim[gid] == 3

    def test_deploy_mode_keeps_native(self):
        cfg = default_config()
        cfg.contact_mode = "deploy"
        env = WarpJoystick(task="flat_terrain", config=cfg)
        gid = env.mj_model.geom("FL").id
        assert env.mj_model.geom_condim[gid] == 6


class TestWarpBatched:
    def test_make_envs_integration(self):
        """Test Warp Go2 through the full make_envs pipeline."""
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(
            env_name="Go2WarpJoystickFlat",
            num_envs=4,
            total_timesteps=1000,
        )
        env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(
            cfg, seed=0
        )
        assert obs_dim == 48
        assert action_dim == 12
        assert isinstance(env_state.obs, dict)
        assert env_state.obs["state"].shape == (4, 48)

        # Test batched step
        action = jnp.zeros((4, 12))
        next_state = env_step(env_state, action)
        assert isinstance(next_state.obs, dict)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))

    def test_existing_envs_still_work(self):
        """Regression: Go2WarpJoystickFlat still loads after registry change."""
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(env_name="Go2WarpJoystickFlat", num_envs=2, total_timesteps=1000)
        _, _, env_state, _, obs_dim, _, _ = make_envs(cfg, seed=0)
        assert obs_dim == 48


class TestTorqueSpeedModel:
    """Optional linear torque-speed actuator limit.

    tau_limit = stall_torque * max(1 - |dq| / velocity_limit, 0)

    Disabled by default: the MJCF actuator_ctrlrange is the only clip.
    Enabled via config.torque_speed_model: adds velocity-dependent dropoff.
    """

    def test_default_flag_is_off(self, env):
        assert env._torque_speed_model is False

    def test_stall_torque_from_mjcf(self, env):
        # Joint order (qpos[7:]) is body-tree: (FL, FR, RL, RR) × (hip, thigh, calf).
        # Unitree go2.xml: abduction+hip classes = 23.7 Nm, knee = 45.43 Nm.
        assert env._stall_torque.shape == (12,)
        expected = jnp.array([23.7, 23.7, 45.43] * 4)
        assert jnp.allclose(env._stall_torque, expected, atol=1e-4)

    def test_velocity_limit_from_urdf(self, env):
        # URDF: hip/thigh 30.1 rad/s, calf 20.07 rad/s.
        assert env._velocity_limit.shape == (12,)
        expected = jnp.array([30.1, 30.1, 20.07] * 4)
        assert jnp.allclose(env._velocity_limit, expected, atol=1e-4)

    def test_helper_is_noop_when_disabled(self, env):
        tau = jnp.ones(12) * 1000.0  # far above any stall torque
        dq = jnp.ones(12) * 50.0     # far above any velocity limit
        out = env._apply_torque_speed_limit(tau, dq)
        assert jnp.array_equal(out, tau)

    def test_enabled_via_override(self):
        env = WarpJoystick(task="flat_terrain",
                           config_overrides={"torque_speed_model": True})
        assert env._torque_speed_model is True

    def test_clip_at_zero_velocity(self):
        """At dq=0, limit equals stall_torque (MJCF ctrlrange equivalent)."""
        env = WarpJoystick(task="flat_terrain",
                           config_overrides={"torque_speed_model": True})
        dq = jnp.zeros(12)
        huge_tau = jnp.ones(12) * 1000.0
        out = env._apply_torque_speed_limit(huge_tau, dq)
        # Clipped to per-joint stall torque.
        assert jnp.allclose(out, env._stall_torque, atol=1e-4)

    def test_clip_at_velocity_limit(self):
        """At |dq|=velocity_limit, allowance reaches zero."""
        env = WarpJoystick(task="flat_terrain",
                           config_overrides={"torque_speed_model": True})
        dq = env._velocity_limit  # exactly at limit
        tau = jnp.ones(12) * 100.0
        out = env._apply_torque_speed_limit(tau, dq)
        assert jnp.allclose(out, 0.0, atol=1e-5)

    def test_clip_beyond_velocity_limit_stays_zero(self):
        """Beyond the limit, scale is clamped to 0 (no negative allowance)."""
        env = WarpJoystick(task="flat_terrain",
                           config_overrides={"torque_speed_model": True})
        dq = env._velocity_limit * 2.0
        tau = jnp.ones(12) * 100.0
        out = env._apply_torque_speed_limit(tau, dq)
        assert jnp.allclose(out, 0.0, atol=1e-5)

    def test_clip_at_half_velocity(self):
        """At |dq|=0.5 * velocity_limit, tau_limit = 0.5 * stall."""
        env = WarpJoystick(task="flat_terrain",
                           config_overrides={"torque_speed_model": True})
        dq = env._velocity_limit * 0.5
        huge_tau = jnp.ones(12) * 1000.0
        out = env._apply_torque_speed_limit(huge_tau, dq)
        expected = env._stall_torque * 0.5
        assert jnp.allclose(out, expected, atol=1e-4)

    def test_small_torques_unchanged(self):
        """Sub-limit torques pass through unmodified."""
        env = WarpJoystick(task="flat_terrain",
                           config_overrides={"torque_speed_model": True})
        dq = jnp.ones(12) * 5.0  # within all vel limits
        small_tau = jnp.ones(12) * 0.5  # well below any stall torque
        out = env._apply_torque_speed_limit(small_tau, dq)
        assert jnp.allclose(out, small_tau, atol=1e-5)

    def test_symmetric_clip_negative_tau(self):
        """Negative torques clipped symmetrically."""
        env = WarpJoystick(task="flat_terrain",
                           config_overrides={"torque_speed_model": True})
        dq = jnp.zeros(12)
        huge_neg_tau = jnp.ones(12) * -1000.0
        out = env._apply_torque_speed_limit(huge_neg_tau, dq)
        assert jnp.allclose(out, -env._stall_torque, atol=1e-4)

    def test_step_with_flag_on_no_nan(self):
        """End-to-end: step with flag on produces finite obs/reward."""
        env = WarpJoystick(task="flat_terrain",
                           config_overrides={"torque_speed_model": True})
        state = env.reset(jax.random.PRNGKey(0))
        action = jnp.zeros(12)
        next_state = env.step(state, action)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))
        assert not jnp.any(jnp.isnan(next_state.reward))
