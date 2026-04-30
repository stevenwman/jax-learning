"""Tests for Go2 Bongo Board Handstand environment."""
import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]

from jax_rl.envs.locomotion.go2_bongo_handstand import (
    BongoHandstand,
    default_config,
)


@pytest.fixture
def env():
    return BongoHandstand(task="bongo_handstand")


@pytest.fixture
def state(env):
    return env.reset(jax.random.PRNGKey(0))


class TestBongoLoads:
    def test_action_size(self, env):
        assert env.action_size == 12

    def test_board_body_exists(self, env):
        board_id = env.mj_model.body("board").id
        assert board_id > 0

    def test_roller_joints_exist(self, env):
        slide_id = env.mj_model.joint("roller_slide").id
        spin_id = env.mj_model.joint("roller_spin").id
        assert slide_id > 0
        assert spin_id > 0

    def test_equality_constraint_exists(self, env):
        assert env.mj_model.neq > 0

    def test_config_impl_is_warp(self):
        cfg = default_config()
        assert cfg.impl == "warp"

    def test_obs_dict_keys(self, state):
        assert isinstance(state.obs, dict)
        assert "state" in state.obs
        assert "privileged_state" in state.obs

    def test_obs_dims_with_board_state(self, state):
        assert state.obs["state"].shape == (46,)

    def test_obs_dims_without_board_state(self):
        cfg = default_config()
        cfg.observe_board_state = False
        env = BongoHandstand(task="bongo_handstand", config=cfg)
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs["state"].shape == (42,)

    def test_reset_shapes(self, state):
        assert state.reward.shape == ()
        assert state.done.shape == ()

    def test_privileged_state_shape(self, state):
        priv = state.obs["privileged_state"]
        # 46 (state w/ board) + 52 (privileged extras) = 98
        assert priv.shape == (98,)

    def test_privileged_state_without_board(self):
        cfg = default_config()
        cfg.observe_board_state = False
        env = BongoHandstand(task="bongo_handstand", config=cfg)
        state = env.reset(jax.random.PRNGKey(0))
        # 42 (state w/o board) + 52 (privileged extras) = 94
        assert state.obs["privileged_state"].shape == (94,)


class TestBongoSteps:
    def test_step_zero_action(self, env, state):
        action = jnp.zeros(12)
        next_state = env.step(state, action)
        assert isinstance(next_state.obs, dict)
        assert next_state.obs["state"].shape == (46,)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))
        assert not jnp.any(jnp.isnan(next_state.reward))

    def test_step_random_action(self, env, state):
        key = jax.random.PRNGKey(42)
        action = jax.random.uniform(key, (12,), minval=-1.0, maxval=1.0)
        next_state = env.step(state, action)
        assert next_state.obs["state"].shape == (46,)

    def test_reward_finite_after_steps(self, env, state):
        action = jnp.zeros(12)
        for _ in range(5):
            state = env.step(state, action)
        assert jnp.isfinite(state.reward)


class TestBongoBatched:
    def test_make_envs_integration(self):
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(
            env_name="Go2BongoHandstand",
            num_envs=4,
            total_timesteps=1000,
        )
        env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(
            cfg, seed=0
        )
        assert obs_dim == 46
        assert action_dim == 12
        assert isinstance(env_state.obs, dict)
        assert env_state.obs["state"].shape == (4, 46)

        action = jnp.zeros((4, 12))
        next_state = env_step(env_state, action)
        assert isinstance(next_state.obs, dict)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))
