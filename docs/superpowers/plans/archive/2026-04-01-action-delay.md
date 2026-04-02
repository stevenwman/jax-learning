# Action Delay (Latency FIFO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `ActionDelayWrapper` that simulates real-robot latency by buffering actions in a FIFO, with fixed or per-episode randomized delay specified in milliseconds.

**Architecture:** Wrapper pattern (same as `FrameStackWrapper`) — sits between training loop and base env, stores FIFO buffer in `state.info`, converts ms delay to control steps via `env.dt`. Handles auto-reset via `jp.where(state.done, ...)`.

**Tech Stack:** JAX, Brax/MJX env interface, jax.numpy, pytest

**Spec:** `docs/superpowers/specs/2026-04-01-action-delay-design.md`

---

## File Structure

| File | Role |
|------|------|
| `jax_rl/envs/wrappers/action_delay.py` | NEW — ActionDelayWrapper class |
| `jax_rl/envs/wrappers/__init__.py` | Export ActionDelayWrapper |
| `jax_rl/configs/train_config.py` | Add `action_delay_ms` and `action_delay_range_ms` fields |
| `jax_rl/training/env_setup.py` | Wire wrapper for train + eval envs |
| `train_ppo_fast.py` | CLI flags `--action-delay-ms`, `--action-delay-range-ms` |
| `train_offpolicy.py` | Same CLI flags |
| `record_video.py` | Apply wrapper from checkpoint meta |
| `deploy/go2_constants.py` | `ACTION_DELAY_MS = 120` constant |
| `tests/test_action_delay.py` | NEW — wrapper unit tests |

---

### Task 1: ActionDelayWrapper — Core Implementation

**Files:**
- Create: `jax_rl/envs/wrappers/action_delay.py`
- Create: `tests/test_action_delay.py`

- [ ] **Step 1: Write failing tests for fixed delay**

```python
# tests/test_action_delay.py
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
        data = None  # not needed for wrapper tests
        return State(
            data=data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            metrics={},
            info={"rng": rng},
        )

    def step(self, state, action):
        # Store the action that was actually passed in, so tests can verify delay.
        info = {**state.info, "received_action": action}
        return state.replace(info=info)


def test_fixed_delay_buffers_actions():
    """With delay_ms=40 at 50Hz (2 steps), action should appear 2 steps later."""
    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_ms=40)  # 40ms / 20ms = 2 steps

    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)

    # First 2 steps: should receive zeros (buffer not yet flushed)
    a1 = jnp.ones(4) * 1.0
    state = wrapped.step(state, a1)
    np.testing.assert_allclose(state.info["received_action"], jnp.zeros(4), atol=1e-6)

    a2 = jnp.ones(4) * 2.0
    state = wrapped.step(state, a2)
    np.testing.assert_allclose(state.info["received_action"], jnp.zeros(4), atol=1e-6)

    # Step 3: a1 should come out
    a3 = jnp.ones(4) * 3.0
    state = wrapped.step(state, a3)
    np.testing.assert_allclose(state.info["received_action"], a1, atol=1e-6)

    # Step 4: a2 should come out
    a4 = jnp.ones(4) * 4.0
    state = wrapped.step(state, a4)
    np.testing.assert_allclose(state.info["received_action"], a2, atol=1e-6)


def test_delay_zero_raises():
    """delay_ms=0 should not be wrapped — constructor rejects it."""
    env = FakeEnv(action_dim=4)
    with pytest.raises(AssertionError):
        ActionDelayWrapper(env, delay_ms=0)


def test_shape_preservation():
    """Wrapper should not change obs, reward, or done."""
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
    """After done=1, buffer should be zeros — stale actions must not leak."""
    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_ms=40)  # 2 steps
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)

    # Push some actions
    state = wrapped.step(state, jnp.ones(4) * 99.0)
    state = wrapped.step(state, jnp.ones(4) * 99.0)

    # Simulate done
    state = state.replace(done=jnp.float32(1.0))
    state = wrapped.step(state, jnp.ones(4) * 5.0)

    # Buffer should have been cleared — next outputs should be zeros, not 99
    state = state.replace(done=jnp.float32(0.0))
    state = wrapped.step(state, jnp.ones(4) * 6.0)
    np.testing.assert_allclose(state.info["received_action"], jnp.zeros(4), atol=1e-6)


def test_randomized_delay_range():
    """With delay_range_ms, different resets should produce different delays."""
    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_range_ms=(20, 120))  # 1-6 steps

    delays = []
    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        state = wrapped.reset(rng)
        delays.append(int(state.info["action_delay_steps"]))

    # Should have some variation (not all the same)
    assert len(set(delays)) > 1, f"All delays identical: {delays}"
    # All within range [1, 6]
    assert all(1 <= d <= 6 for d in delays), f"Delay out of range: {delays}"


def test_buffer_info_keys_exist():
    """Wrapper should add expected keys to state.info."""
    env = FakeEnv(action_dim=4)
    wrapped = ActionDelayWrapper(env, delay_ms=60)  # 3 steps
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    assert "action_delay_buffer" in state.info
    assert "action_delay_steps" in state.info
    assert "action_delay_rng" in state.info
    assert state.info["action_delay_buffer"].shape == (3, 4)
    assert state.info["action_delay_steps"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_action_delay.py -v`
Expected: FAIL — `ImportError: cannot import name 'ActionDelayWrapper'`

- [ ] **Step 3: Implement ActionDelayWrapper**

```python
# jax_rl/envs/wrappers/action_delay.py
"""Action delay wrapper — FIFO buffer to simulate real-robot latency."""
from __future__ import annotations

import jax
import jax.numpy as jp
from jax_rl.envs.wrappers.training import Wrapper


class ActionDelayWrapper(Wrapper):
    """Delays actions by K control steps to simulate real-robot latency.

    Maintains a FIFO buffer in state.info. Supports fixed delay or
    per-episode randomized delay (uniform over a range).

    Args:
        env: Base environment to wrap.
        delay_ms: Fixed delay in milliseconds. Converted to control steps
            via env.dt. Ignored if delay_range_ms is provided.
        delay_range_ms: Tuple (min_ms, max_ms) for per-episode random delay.
            Overrides delay_ms.
    """

    def __init__(
        self,
        env,
        delay_ms: int = 0,
        delay_range_ms: tuple[int, int] | None = None,
    ):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_action_delay.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/wrappers/action_delay.py tests/test_action_delay.py
git commit -m "feat: ActionDelayWrapper — FIFO latency buffer for sim2real"
```

---

### Task 2: Export & Wrapper __init__

**Files:**
- Modify: `jax_rl/envs/wrappers/__init__.py`

- [ ] **Step 1: Add export**

Add `ActionDelayWrapper` to imports and `__all__` in `jax_rl/envs/wrappers/__init__.py`:

```python
from jax_rl.envs.wrappers.action_delay import ActionDelayWrapper
```

Add `"ActionDelayWrapper"` to the `__all__` list.

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from jax_rl.envs.wrappers import ActionDelayWrapper; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add jax_rl/envs/wrappers/__init__.py
git commit -m "feat: export ActionDelayWrapper from wrappers package"
```

---

### Task 3: Wire into env_setup.py (train + eval)

**Files:**
- Modify: `jax_rl/configs/train_config.py`
- Modify: `jax_rl/training/env_setup.py:113-131`

- [ ] **Step 1: Add fields to TrainConfig**

In `jax_rl/configs/train_config.py`, add to the `TrainConfig` dataclass:

```python
action_delay_ms: int = 0
action_delay_range_ms: tuple[int, int] | None = None
```

Without these fields, `cfg_overrides` via `dataclasses.replace()` will crash with `TypeError`.

- [ ] **Step 2: Add action delay wrapper to training env**

In `env_setup.py`, insert action delay wrapping **before** FrameStackWrapper (before line 113). The wrapper reads `cfg.action_delay_ms` and `cfg.action_delay_range_ms`:

```python
# Action delay (optional, simulates real-robot latency).
action_delay_ms = getattr(cfg, "action_delay_ms", 0)
action_delay_range_ms = getattr(cfg, "action_delay_range_ms", None)
if action_delay_range_ms is not None or action_delay_ms > 0:
    from jax_rl.envs.wrappers import ActionDelayWrapper
    env = ActionDelayWrapper(env, delay_ms=action_delay_ms, delay_range_ms=action_delay_range_ms)

# Frame stacking (optional, universal wrapper).
if cfg.n_frame_stack > 1:
    ...
```

- [ ] **Step 3: Add action delay wrapper to eval env**

In the eval env construction (around line 127-131), apply with fixed delay (max of range, or the fixed value):

```python
eval_delay_ms = action_delay_range_ms[1] if action_delay_range_ms else action_delay_ms
if eval_delay_ms > 0:
    from jax_rl.envs.wrappers import ActionDelayWrapper
    eval_env = ActionDelayWrapper(eval_env, delay_ms=eval_delay_ms)
```

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All existing tests pass

- [ ] **Step 5: Commit**

```bash
git add jax_rl/configs/train_config.py jax_rl/training/env_setup.py
git commit -m "feat: wire ActionDelayWrapper into env_setup (train + eval)"
```

---

### Task 4: CLI Flags in Training Scripts

**Files:**
- Modify: `train_ppo_fast.py` (argparse section ~line 428-466, config merge ~line 468-487)
- Modify: `train_offpolicy.py` (argparse section ~line 323-361, config merge ~line 363-382)

- [ ] **Step 1: Add CLI flags to train_ppo_fast.py**

In the argparse section, add:

```python
parser.add_argument("--action-delay-ms", type=int, default=None,
                    help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Randomized action delay range in ms (e.g., 40 120)")
```

In the config merge section, add:

```python
if args.action_delay_ms is not None:
    cfg_overrides["action_delay_ms"] = args.action_delay_ms
if args.action_delay_range_ms is not None:
    cfg_overrides["action_delay_range_ms"] = tuple(args.action_delay_range_ms)
```

- [ ] **Step 2: Add same CLI flags to train_offpolicy.py**

Same two `add_argument` calls and config merge lines as above.

- [ ] **Step 3: Verify flags parse correctly**

Run: `uv run python train_ppo_fast.py --help | grep delay`
Expected: Shows `--action-delay-ms` and `--action-delay-range-ms`

Run: `uv run python train_offpolicy.py --help | grep delay`
Expected: Same

- [ ] **Step 4: Commit**

```bash
git add train_ppo_fast.py train_offpolicy.py
git commit -m "feat: --action-delay-ms and --action-delay-range-ms CLI flags"
```

---

### Task 5: record_video.py Integration

**Files:**
- Modify: `record_video.py:178-183`

- [ ] **Step 1: Apply delay wrapper from checkpoint metadata**

After the frame stacking block (line 183), add analogous action delay block. The delay config is saved in `meta["train_config"]` by the training scripts:

```python
# Apply action delay if checkpoint was trained with it.
action_delay_ms = meta.get("train_config", {}).get("action_delay_ms", 0)
action_delay_range_ms = meta.get("train_config", {}).get("action_delay_range_ms", None)
# For recording, use fixed delay (max of range if randomized).
if action_delay_range_ms:
    record_delay_ms = action_delay_range_ms[1]
elif action_delay_ms:
    record_delay_ms = action_delay_ms
else:
    record_delay_ms = 0
if record_delay_ms > 0:
    from jax_rl.envs.wrappers import ActionDelayWrapper
    env = ActionDelayWrapper(env, delay_ms=record_delay_ms)
    print(f"  Action delay: {record_delay_ms}ms")
```

- [ ] **Step 2: Verify record_video.py still loads**

Run: `uv run python record_video.py --help`
Expected: No import errors

- [ ] **Step 3: Commit**

```bash
git add record_video.py
git commit -m "feat: record_video applies action delay from checkpoint meta"
```

---

### Task 6: Deploy Constants

**Files:**
- Modify: `deploy/go2_constants.py`

- [ ] **Step 1: Add action delay constant**

After the PD gain constants (around line 35), add:

```python
# Action delay — expected real-robot latency (communication + motor response).
# Training uses randomized range; deploy uses this fixed value.
ACTION_DELAY_MS = 120  # WTW default for Unitree robots
```

- [ ] **Step 2: Commit**

```bash
git add deploy/go2_constants.py
git commit -m "feat: ACTION_DELAY_MS constant for deploy parity"
```

---

### Task 7: Composed Wrapper Test + Integration Smoke Test

**Files:**
- Modify: `tests/test_action_delay.py`

- [ ] **Step 1: Add composed wrapper test**

Append to `tests/test_action_delay.py`:

```python
def test_composed_with_frame_stack():
    """ActionDelay + FrameStack composed — both reset cleanly on done."""
    from jax_rl.envs.wrappers.frame_stack import FrameStackWrapper

    env = FakeEnv(action_dim=4)
    # ActionDelay first (modifies actions), then FrameStack (modifies obs)
    wrapped = ActionDelayWrapper(env, delay_ms=40)  # 2 steps
    wrapped = FrameStackWrapper(wrapped, n_frames=3)

    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)

    # Both buffers should exist
    assert "action_delay_buffer" in state.info
    assert "frame_stack" in state.info

    # Step a few times
    for i in range(5):
        state = wrapped.step(state, jnp.ones(4) * float(i))

    # Simulate done — both buffers should reset
    state = state.replace(done=jnp.float32(1.0))
    state = wrapped.step(state, jnp.zeros(4))

    # Action buffer should be zeros
    np.testing.assert_allclose(
        state.info["action_delay_buffer"], jnp.zeros((2, 4)), atol=1e-6
    )
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/test_action_delay.py -v`
Expected: All PASS

- [ ] **Step 3: Run full repo tests for regression**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All existing tests still pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_action_delay.py
git commit -m "test: composed wrapper test + verify no regressions"
```

---

### Task 8: Smoke Test — Training with Action Delay

- [ ] **Step 1: Run short training with fixed delay**

Run: `uv run python train_offpolicy.py --algo fast_sac --env Go2WarpJoystickFlat --total-steps 50000 --num-envs 64 --action-delay-ms 120`
Expected: Runs without crash, prints step metrics. Eval reward doesn't matter (50k steps is just a smoke test).

- [ ] **Step 2: Run short training with randomized delay**

Run: `uv run python train_offpolicy.py --algo fast_sac --env Go2WarpJoystickFlat --total-steps 50000 --num-envs 64 --action-delay-range-ms 40 120`
Expected: Runs without crash.

- [ ] **Step 3: Update docs**

Update `.context/TODO.md` — mark "action delay (120ms FIFO from WTW)" as done under "Wider DR ranges".

Update `.context/journals/2026-04-01.md` — add session notes about action delay implementation.
