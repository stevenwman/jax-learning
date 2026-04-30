"""Tests for vendored training wrappers."""
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.gpu

from jax_rl.envs.wrappers.training import (
    VmapWrapper, EpisodeWrapper, AutoResetWrapper, wrap_for_training,
)


@pytest.fixture
def raw_env():
    """Load a raw (unwrapped) env for testing."""
    from mujoco_playground import dm_control_suite
    return dm_control_suite.load("CartpoleBalance")


@pytest.fixture
def raw_go2():
    """Load raw Go2 Warp env."""
    from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick
    return WarpJoystick(task="flat_terrain")


class TestVmapWrapper:
    def test_batched_reset(self, raw_env):
        env = VmapWrapper(raw_env)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        state = env.reset(keys)
        assert state.obs.shape[0] == 4

    def test_batched_step(self, raw_env):
        env = VmapWrapper(raw_env)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        state = env.reset(keys)
        action = jnp.zeros((4, raw_env.action_size))
        next_state = env.step(state, action)
        assert next_state.obs.shape[0] == 4
        assert next_state.reward.shape == (4,)

    def test_action_size_passthrough(self, raw_env):
        env = VmapWrapper(raw_env)
        assert env.action_size == raw_env.action_size


class TestEpisodeWrapper:
    def test_step_counter(self, raw_env):
        env = VmapWrapper(raw_env)
        env = EpisodeWrapper(env, episode_length=10)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        assert jnp.all(state.info['steps'] == 0)

        action = jnp.zeros((2, raw_env.action_size))
        state = env.step(state, action)
        assert jnp.all(state.info['steps'] == 1)

    def test_truncation_at_episode_length(self, raw_env):
        env = VmapWrapper(raw_env)
        env = EpisodeWrapper(env, episode_length=5)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        action = jnp.zeros((2, raw_env.action_size))

        for i in range(4):
            state = env.step(state, action)
            assert jnp.all(state.done == 0), f"Should not be done at step {i+1}"

        state = env.step(state, action)
        assert jnp.all(state.done == 1), "Should be done at step 5"
        assert jnp.all(state.info['truncation'] == 1), "Should be truncation, not termination"

    def test_truncation_flag_zero_before_limit(self, raw_env):
        env = VmapWrapper(raw_env)
        env = EpisodeWrapper(env, episode_length=1000)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        action = jnp.zeros((2, raw_env.action_size))
        state = env.step(state, action)
        assert jnp.all(state.info['truncation'] == 0)


class TestAutoResetWrapper:
    def test_done_envs_get_reset_obs(self, raw_env):
        env = VmapWrapper(raw_env)
        env = EpisodeWrapper(env, episode_length=3)
        env = AutoResetWrapper(env)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        initial_obs = state.obs.copy()
        action = jnp.zeros((2, raw_env.action_size))

        for _ in range(3):
            state = env.step(state, action)

        assert jnp.allclose(state.obs, initial_obs, atol=1e-5), \
            "Auto-reset should restore cached initial obs"

    def test_step_counter_resets_on_done(self, raw_env):
        env = VmapWrapper(raw_env)
        env = EpisodeWrapper(env, episode_length=3)
        env = AutoResetWrapper(env)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        action = jnp.zeros((2, raw_env.action_size))

        for _ in range(3):
            state = env.step(state, action)

        state = env.step(state, action)
        assert jnp.all(state.info['steps'] == 1)

    def test_done_count_increments(self, raw_env):
        env = VmapWrapper(raw_env)
        env = EpisodeWrapper(env, episode_length=2)
        env = AutoResetWrapper(env)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        action = jnp.zeros((2, raw_env.action_size))

        assert jnp.all(state.info['AutoResetWrapper_done_count'] == 0)
        state = env.step(state, action)
        state = env.step(state, action)
        assert jnp.all(state.info['AutoResetWrapper_done_count'] == 1)


class TestWrapForTraining:
    def test_full_pipeline_dmc(self, raw_env):
        env = wrap_for_training(raw_env, episode_length=5)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        state = env.reset(keys)
        assert state.obs.shape[0] == 4

        action = jnp.zeros((4, raw_env.action_size))
        for _ in range(10):
            state = env.step(state, action)
        assert jnp.all(jnp.isfinite(state.obs))
        assert jnp.all(jnp.isfinite(state.reward))

    def test_full_pipeline_go2_warp(self, raw_go2):
        env = wrap_for_training(raw_go2, episode_length=5)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        assert isinstance(state.obs, dict)
        assert state.obs["state"].shape == (2, 48)

        action = jnp.zeros((2, 12))
        for _ in range(10):
            state = env.step(state, action)
        assert not jnp.any(jnp.isnan(state.obs["state"]))

    def test_matches_playground_wrapper(self, raw_env):
        from mujoco_playground._src.wrapper import wrap_for_brax_training
        from mujoco_playground import dm_control_suite

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        ours = wrap_for_training(raw_env, episode_length=10)
        our_state = ours.reset(keys)

        raw_env2 = dm_control_suite.load("CartpoleBalance")
        theirs = wrap_for_brax_training(raw_env2, episode_length=10)
        their_state = theirs.reset(keys)

        assert jnp.allclose(our_state.obs, their_state.obs, atol=1e-6), \
            "Vendored wrapper should produce identical obs to Playground"

        action = jnp.zeros((4, raw_env.action_size))
        our_state = ours.step(our_state, action)
        their_state = theirs.step(their_state, action)
        assert jnp.allclose(our_state.obs, their_state.obs, atol=1e-6)
        assert jnp.allclose(our_state.reward, their_state.reward, atol=1e-6)
