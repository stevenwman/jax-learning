"""Tests for FrameStackWrapper."""
import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.wrappers.frame_stack import FrameStackWrapper


@pytest.fixture
def base_env():
    """Load Go2 Warp env (raw, no wrapper)."""
    from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick
    return WarpJoystick(task="flat_terrain")


@pytest.fixture
def wrapped_env(base_env):
    return FrameStackWrapper(base_env, n_frames=3)


@pytest.fixture
def state(wrapped_env):
    return wrapped_env.reset(jax.random.PRNGKey(0))


class TestFrameStackWrapped:
    def test_stacked_obs_shape(self, state):
        assert state.obs["state"].shape == (144,)  # 3 * 48

    def test_privileged_state_unchanged(self, state):
        assert state.obs["privileged_state"].shape == (122,)

    def test_reset_tiles_initial_obs(self, state):
        stacked = state.obs["state"]
        for i in range(3):
            frame = stacked[i * 48 : (i + 1) * 48]
            assert jnp.allclose(frame, stacked[:48])

    def test_step_shifts_frames(self, wrapped_env, state):
        old_frame_0 = state.obs["state"][:48]
        next_state = wrapped_env.step(state, jnp.zeros(12))
        new_frame_1 = next_state.obs["state"][48:96]
        assert jnp.allclose(new_frame_1, old_frame_0)

    def test_action_size_passthrough(self, wrapped_env):
        assert wrapped_env.action_size == 12


class TestFrameStackSingle:
    def test_n_frames_1_is_identity(self, base_env):
        wrapped = FrameStackWrapper(base_env, n_frames=1)
        state = wrapped.reset(jax.random.PRNGKey(0))
        assert state.obs["state"].shape == (48,)


class TestFrameStackIntegration:
    def test_with_make_envs(self):
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(
            env_name="Go2WarpJoystickFlat",
            num_envs=4,
            total_timesteps=1000,
            n_frame_stack=3,
        )
        env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed=0)
        assert obs_dim == 144  # 3 * 48
        assert env_state.obs["state"].shape == (4, 144)

        # Step should work
        next_state = env_step(env_state, jnp.zeros((4, 12)))
        assert next_state.obs["state"].shape == (4, 144)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))

    def test_no_frame_stack_by_default(self):
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(
            env_name="Go2WarpJoystickFlat",
            num_envs=2,
            total_timesteps=1000,
        )
        _, _, env_state, _, obs_dim, _, _ = make_envs(cfg, seed=0)
        assert obs_dim == 48  # No stacking by default
