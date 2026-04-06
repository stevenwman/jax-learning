"""Action delay wrapper — FIFO buffer to simulate real-robot latency."""
from __future__ import annotations

import jax
import jax.numpy as jp
from jax_rl.envs.wrappers.training import Wrapper


class ActionDelayWrapper(Wrapper):
    """Delays actions by K control steps to simulate real-robot latency.

    Maintains a FIFO buffer in state.info. Supports fixed delay or
    per-episode randomized delay (uniform over a range).
    """

    def __init__(
        self,
        env: Wrapper,
        delay_ms: int = 0,
        delay_range_ms: tuple[int, int] | None = None,
    ):
        """Initialize ActionDelayWrapper.

        Args:
            env: Base environment to wrap.
            delay_ms: Fixed delay in milliseconds. Converted to control steps
                via env.dt. Ignored if delay_range_ms is provided.
            delay_range_ms: Tuple (min_ms, max_ms) for per-episode random delay.
                Overrides delay_ms.
        """
        super().__init__(env)
        ctrl_dt_ms = env.dt * 1000

        if delay_range_ms is not None:
            self._min_delay = int(round(delay_range_ms[0] / ctrl_dt_ms))
            self._max_delay = int(round(delay_range_ms[1] / ctrl_dt_ms))
        else:
            steps = int(round(delay_ms / ctrl_dt_ms))
            self._min_delay = steps
            self._max_delay = steps

        assert self._max_delay >= 1, "ActionDelayWrapper requires delay >= 1 step"
        assert self._min_delay >= 1, "Minimum delay must be >= 1 step"

    def reset(self, rng: jax.Array):
        rng, wrapper_rng, sample_rng = jax.random.split(rng, 3)
        state = self.env.reset(rng)

        delay = jax.random.randint(
            sample_rng, (), self._min_delay, self._max_delay + 1
        )
        buffer = jp.zeros((self._max_delay, self.env.action_size))

        state.info["action_delay_buffer"] = buffer
        state.info["action_delay_steps"] = delay
        state.info["action_delay_rng"] = wrapper_rng
        return state

    def step(self, state, action):
        buffer = state.info["action_delay_buffer"]
        delay = state.info["action_delay_steps"]
        rng = state.info["action_delay_rng"]

        # Pop: read the delayed action
        read_idx = self._max_delay - delay
        delayed_action = buffer[read_idx]

        # Push: shift buffer left, write new action to end
        buffer = jp.roll(buffer, -1, axis=0)
        buffer = buffer.at[-1].set(action)

        # Reset handling: on done, zero buffer + re-sample delay
        rng, sample_key = jax.random.split(rng)
        new_delay = jax.random.randint(
            sample_key, (), self._min_delay, self._max_delay + 1
        )
        zero_buffer = jp.zeros_like(buffer)

        buffer = jp.where(state.done, zero_buffer, buffer)
        delay = jp.where(state.done, new_delay, delay)

        # Step inner env with delayed action
        state = self.env.step(state, delayed_action)

        # Update state.info
        state.info["action_delay_buffer"] = buffer
        state.info["action_delay_steps"] = delay
        state.info["action_delay_rng"] = rng

        return state
