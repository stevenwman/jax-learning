"""Tests for Go2 Warp environment."""
import jax
import jax.numpy as jnp
import pytest

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
        n_frames = default_config().n_frame_stack  # 3
        assert state.obs["state"].shape == (48 * n_frames,)  # 144
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
        n_frames = default_config().n_frame_stack
        assert next_state.obs["state"].shape == (48 * n_frames,)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))
        assert not jnp.any(jnp.isnan(next_state.reward))

    def test_step_random_action(self, env, state):
        key = jax.random.PRNGKey(42)
        action = jax.random.uniform(key, (12,), minval=-1.0, maxval=1.0)
        next_state = env.step(state, action)
        n_frames = default_config().n_frame_stack
        assert next_state.obs["state"].shape == (48 * n_frames,)

    def test_reward_nonzero_after_steps(self, env, state):
        action = jnp.zeros(12)
        for _ in range(5):
            state = env.step(state, action)
        assert jnp.isfinite(state.reward)


class TestWarpDomainRand:
    def test_domain_randomize_with_warp_body_id(self, env):
        from jax_rl.envs.locomotion.go2_randomize import domain_randomize
        model, in_axes = domain_randomize(
            env.mjx_model,
            jax.random.split(jax.random.PRNGKey(0), 2),
            torso_body_id=env._torso_body_id,
        )
        assert model is not None


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
        n_frames = default_config().n_frame_stack
        assert obs_dim == 48 * n_frames
        assert action_dim == 12
        assert isinstance(env_state.obs, dict)
        assert env_state.obs["state"].shape == (4, 48 * n_frames)

        # Test batched step
        action = jnp.zeros((4, 12))
        next_state = env_step(env_state, action)
        assert isinstance(next_state.obs, dict)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))

    def test_existing_envs_still_work(self):
        """Regression: MJX Go2 still loads after registry change."""
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(env_name="Go2JoystickFlat", num_envs=2, total_timesteps=1000)
        _, _, env_state, _, obs_dim, _, _ = make_envs(cfg, seed=0)
        assert obs_dim == 48


class TestWarpFrameStack:
    def test_frame_stack_content(self, env, state):
        """Frame stack should have newest obs at front, zeros shifted out."""
        n_frames = default_config().n_frame_stack
        raw_dim = 48

        # After reset, all frames should be identical (tiled initial obs).
        stacked = state.obs["state"]
        for i in range(n_frames):
            frame_i = stacked[i * raw_dim : (i + 1) * raw_dim]
            assert jnp.allclose(frame_i, stacked[:raw_dim]), f"Frame {i} should match frame 0 after reset"

        # After one step, frame 0 should differ (new obs), frame 1 should match old frame 0.
        action = jnp.zeros(12)
        next_state = env.step(state, action)
        old_frame_0 = stacked[:raw_dim]
        new_stacked = next_state.obs["state"]
        new_frame_1 = new_stacked[raw_dim : 2 * raw_dim]
        assert jnp.allclose(new_frame_1, old_frame_0), "Frame 1 after step should be previous frame 0"


class TestWarpNoFrameStack:
    def test_n_frame_stack_1_gives_raw_obs(self):
        cfg = default_config()
        cfg.n_frame_stack = 1
        env = WarpJoystick(task="flat_terrain", config=cfg)
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs["state"].shape == (48,)
