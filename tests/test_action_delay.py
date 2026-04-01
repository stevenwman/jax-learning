"""Tests for ActionDelayWrapper."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco_playground._src.mjx_env import State

from jax_rl.envs.wrappers.action_delay import ActionDelayWrapper


class FakeEnv:
    """Minimal env stub for testing the wrapper in isolation."""

    def __init__(self, action_dim=12):
        self._action_dim = action_dim
        self._dt = 0.02  # 50 Hz

    @property
    def action_size(self):
        return self._action_dim

    @property
    def dt(self):
        return self._dt

    def reset(self, rng):
        obs = jnp.zeros(48)
        data = None
        return State(
            data=data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            metrics={},
            info={"rng": rng},
        )

    def step(self, state, action):
        obs = jnp.zeros(48)  # fresh obs each step (important for composition tests)
        info = {**state.info, "received_action": action}
        return state.replace(obs=obs, info=info)


def test_fixed_delay_buffers_actions():
    """With delay_ms=40 at 50Hz (2 steps), action should appear 2 steps later."""
    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_ms=40)

    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)

    a1 = jnp.ones(4) * 1.0
    state = wrapped.step(state, a1)
    np.testing.assert_allclose(state.info["received_action"], jnp.zeros(4), atol=1e-6)

    a2 = jnp.ones(4) * 2.0
    state = wrapped.step(state, a2)
    np.testing.assert_allclose(state.info["received_action"], jnp.zeros(4), atol=1e-6)

    a3 = jnp.ones(4) * 3.0
    state = wrapped.step(state, a3)
    np.testing.assert_allclose(state.info["received_action"], a1, atol=1e-6)

    a4 = jnp.ones(4) * 4.0
    state = wrapped.step(state, a4)
    np.testing.assert_allclose(state.info["received_action"], a2, atol=1e-6)


def test_delay_zero_raises():
    env = FakeEnv(action_dim=4)
    with pytest.raises(AssertionError):
        ActionDelayWrapper(env, delay_ms=0)


def test_shape_preservation():
    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_ms=40)
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    action = jnp.ones(4)
    state = wrapped.step(state, action)
    assert state.obs.shape == (48,)
    assert state.reward.shape == ()
    assert state.done.shape == ()


def test_reset_clears_buffer():
    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_ms=40)
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)

    state = wrapped.step(state, jnp.ones(4) * 99.0)
    state = wrapped.step(state, jnp.ones(4) * 99.0)

    state = state.replace(done=jnp.float32(1.0))
    state = wrapped.step(state, jnp.ones(4) * 5.0)

    state = state.replace(done=jnp.float32(0.0))
    state = wrapped.step(state, jnp.ones(4) * 6.0)
    np.testing.assert_allclose(state.info["received_action"], jnp.zeros(4), atol=1e-6)


def test_randomized_delay_range():
    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_range_ms=(20, 120))

    delays = []
    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        state = wrapped.reset(rng)
        delays.append(int(state.info["action_delay_steps"]))

    assert len(set(delays)) > 1, f"All delays identical: {delays}"
    assert all(1 <= d <= 6 for d in delays), f"Delay out of range: {delays}"


def test_buffer_info_keys_exist():
    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_ms=60)
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    assert "action_delay_buffer" in state.info
    assert "action_delay_steps" in state.info
    assert "action_delay_rng" in state.info
    assert state.info["action_delay_buffer"].shape == (3, 4)
    assert state.info["action_delay_steps"] == 3


def test_composed_with_frame_stack():
    from jax_rl.envs.wrappers.frame_stack import FrameStackWrapper

    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_ms=40)
    wrapped = FrameStackWrapper(wrapped, n_frames=3)

    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)

    assert "action_delay_buffer" in state.info
    assert "frame_stack" in state.info

    for i in range(5):
        state = wrapped.step(state, jnp.ones(4) * float(i))

    state = state.replace(done=jnp.float32(1.0))
    state = wrapped.step(state, jnp.zeros(4))

    np.testing.assert_allclose(
        state.info["action_delay_buffer"], jnp.zeros((2, 4)), atol=1e-6
    )
