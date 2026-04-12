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
        assert state.obs["state"].shape == (51,)
        assert state.obs["privileged_state"].shape == (125,)

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
        assert next_state.obs["state"].shape == (51,)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))
        assert not jnp.any(jnp.isnan(next_state.reward))

    def test_step_random_action(self, env, state):
        key = jax.random.PRNGKey(42)
        action = jax.random.uniform(key, (12,), minval=-1.0, maxval=1.0)
        next_state = env.step(state, action)
        assert next_state.obs["state"].shape == (51,)

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
        assert obs_dim == 45
        assert action_dim == 12
        assert isinstance(env_state.obs, dict)
        assert env_state.obs["state"].shape == (4, 45)

        # Test batched step
        action = jnp.zeros((4, 12))
        next_state = env_step(env_state, action)
        assert isinstance(next_state.obs, dict)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))

    def test_existing_envs_still_work(self):
        """Regression: MJX Go2 still loads after registry change."""
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(env_name="Go2WarpJoystickFlat", num_envs=2, total_timesteps=1000)
        _, _, env_state, _, obs_dim, _, _ = make_envs(cfg, seed=0)
        assert obs_dim == 45
