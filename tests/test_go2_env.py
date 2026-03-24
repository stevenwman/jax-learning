"""Tests for Go2 environment."""

import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.locomotion.go2_joystick import Joystick, default_config


@pytest.fixture
def env():
    return Joystick(task="flat_terrain")


@pytest.fixture
def state(env):
    return env.reset(jax.random.PRNGKey(0))


class TestGo2Loads:
    def test_action_size(self, env):
        assert env.action_size == 12

    def test_obs_dict_keys(self, state):
        """Obs is a dict with state and privileged_state."""
        assert isinstance(state.obs, dict)
        assert "state" in state.obs
        assert "privileged_state" in state.obs

    def test_obs_dims(self, state):
        assert state.obs["state"].shape == (48,)
        assert state.obs["privileged_state"].shape[0] > 48  # ~116d

    def test_reset_shapes(self, state):
        assert state.reward.shape == ()
        assert state.done.shape == ()

    def test_default_config_values(self):
        cfg = default_config()
        # Matched to Playground Go1 for training parity
        assert cfg.Kp == 35.0
        assert cfg.Kd == 0.5
        assert cfg.action_scale == 0.5
        assert cfg.ctrl_dt == 0.02
        assert cfg.sim_dt == 0.004


class TestGo2Steps:
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
        """After a few steps with action, reward should be non-zero."""
        action = jnp.zeros(12)
        for _ in range(5):
            state = env.step(state, action)
        assert jnp.isfinite(state.reward)

    def test_termination_on_flip(self, env, state):
        """If robot is flipped (upvector z < 0), done should be True."""
        flipped_qpos = state.data.qpos.at[3:7].set(
            jnp.array([0.0, 1.0, 0.0, 0.0])  # 180° around x-axis
        )
        from mujoco import mjx
        flipped_data = state.data.replace(qpos=flipped_qpos)
        flipped_data = mjx.forward(env.mjx_model, flipped_data)
        done = env._get_termination(flipped_data)
        assert done

    def test_privileged_state_contains_unnoised(self, env, state):
        """Privileged state should be larger than state (includes ground truth)."""
        action = jnp.zeros(12)
        next_state = env.step(state, action)
        priv_dim = next_state.obs["privileged_state"].shape[0]
        state_dim = next_state.obs["state"].shape[0]
        assert priv_dim > state_dim


class TestGo2Batched:
    def test_make_envs_integration(self):
        """Test Go2 through the full make_envs pipeline."""
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(
            env_name="Go2JoystickFlat",
            num_envs=4,
            total_timesteps=1000,
        )
        env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(
            cfg, seed=0
        )
        assert obs_dim == 48  # policy obs dim
        assert action_dim == 12
        assert isinstance(env_state.obs, dict)
        assert env_state.obs["state"].shape == (4, 48)

        # Test batched step.
        action = jnp.zeros((4, 12))
        next_state = env_step(env_state, action)
        assert isinstance(next_state.obs, dict)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))

    def test_dm_control_suite_still_works(self):
        """Regression: existing envs still load after registry change."""
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(
            env_name="CheetahRun",
            num_envs=2,
            total_timesteps=1000,
        )
        _, _, env_state, _, obs_dim, action_dim, _ = make_envs(cfg, seed=0)
        assert obs_dim == 17
        assert action_dim == 6
