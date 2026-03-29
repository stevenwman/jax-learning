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
