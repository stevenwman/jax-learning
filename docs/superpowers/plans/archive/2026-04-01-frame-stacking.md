# Go2 Frame Stacking Implementation Plan (Revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a general-purpose `FrameStackWrapper` that can wrap any Playground env, so any env gets temporal observation history. Default 3 frames (3x48=144d for Go2).

**Architecture:** `FrameStackWrapper` subclasses Playground's `Wrapper` base class. Applied in `env_setup.py` when `cfg.n_frame_stack > 1`. Frame stack lives in `state.info["frame_stack"]`. Env code stays clean — no frame stacking logic inside envs. `TrainConfig.n_frame_stack` controls it, `--frame-stack` CLI flag overrides.

**Tech Stack:** JAX (jnp), MuJoCo Playground (`Wrapper` base class, `mjx_env.State`)

**Revision note:** v1 had frame stacking inline in go2_warp_joystick.py. Revised to wrapper pattern per user feedback — frame stacking should be universal, not per-env.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `jax_rl/envs/wrappers/__init__.py` | Create | Package init |
| `jax_rl/envs/wrappers/frame_stack.py` | Create | FrameStackWrapper class |
| `jax_rl/envs/locomotion/go2_warp_joystick.py` | Modify | Remove inline frame stack (revert to raw 48d) |
| `jax_rl/configs/train_config.py` | Modify | Add `n_frame_stack: int = 1` |
| `jax_rl/training/env_setup.py` | Modify | Apply FrameStackWrapper when n_frame_stack > 1 |
| `train_offpolicy.py` | Modify | Add `--frame-stack` CLI arg |
| `train_ppo_fast.py` | Modify | Add `--frame-stack` CLI arg |
| `train_ppo.py` | Modify | Add `--frame-stack` CLI arg |
| `tests/test_frame_stack_wrapper.py` | Create | Wrapper-specific tests |
| `tests/test_go2_warp_env.py` | Modify | Revert obs_dim assertions back to 48 (raw env) |
| `deploy/obs_builder.py` | Modify | Add n_frame_stack param, numpy frame stack buffer |

---

### Task 1: Revert inline frame stack from go2_warp_joystick.py

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_warp_joystick.py`
- Modify: `tests/test_go2_warp_env.py`

- [ ] **Step 1: Remove `n_frame_stack` from `default_config()`**

Remove the line `n_frame_stack=3,` from `default_config()`.

- [ ] **Step 2: Remove frame stack caching from `_post_init()`**

Remove these lines from end of `_post_init()`:
```python
self._n_frames = self._config.n_frame_stack
self._raw_state_dim = 48
```

- [ ] **Step 3: Remove frame stack from `reset()`**

Remove these lines after `obs = self._get_obs(data, info)`:
```python
# Frame stack: fill all frames with initial obs.
raw_state = obs["state"]
info["frame_stack"] = jp.tile(raw_state, self._n_frames)
obs = {**obs, "state": info["frame_stack"]}
```

- [ ] **Step 4: Remove frame stack from `step()`**

Remove these lines after `obs = self._get_obs(data, state.info)`:
```python
# Frame stack: push new state to front, shift old frames right.
raw_state = obs["state"]
old_stack = state.info["frame_stack"]
state.info["frame_stack"] = jp.concatenate([raw_state, old_stack[:-self._raw_state_dim]])
obs = {**obs, "state": state.info["frame_stack"]}
```

- [ ] **Step 5: Revert test assertions back to 48d**

In `tests/test_go2_warp_env.py`, revert all `48 * n_frames` assertions back to `48`. Remove the `TestWarpFrameStack` and `TestWarpNoFrameStack` test classes (they'll be replaced by wrapper tests).

- [ ] **Step 6: Run Warp env tests**

Run: `uv run python -m pytest tests/test_go2_warp_env.py -v`
Expected: All original tests pass with 48d obs.

- [ ] **Step 7: Commit**

```bash
git add jax_rl/envs/locomotion/go2_warp_joystick.py tests/test_go2_warp_env.py
git commit -m "revert: remove inline frame stack from Go2 Warp env (moving to wrapper)"
```

---

### Task 2: Create FrameStackWrapper + wire into env_setup.py

**Files:**
- Create: `jax_rl/envs/wrappers/__init__.py`
- Create: `jax_rl/envs/wrappers/frame_stack.py`
- Modify: `jax_rl/configs/train_config.py`
- Modify: `jax_rl/training/env_setup.py`

- [ ] **Step 1: Create wrapper package**

Create `jax_rl/envs/wrappers/__init__.py`:
```python
from jax_rl.envs.wrappers.frame_stack import FrameStackWrapper

__all__ = ["FrameStackWrapper"]
```

- [ ] **Step 2: Create FrameStackWrapper**

Create `jax_rl/envs/wrappers/frame_stack.py`:

```python
"""Frame stacking wrapper — wraps any Playground env to provide temporal obs history.

Maintains a FIFO buffer of the last N observations in state.info["frame_stack"].
Newest frame at index [0:obs_dim], oldest at the end.

Works with both flat obs (array) and dict obs (stacks the "state" key only).
"""

import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper


class FrameStackWrapper(Wrapper):
    """Stack N consecutive observations for temporal context.

    Applied between env creation and wrap_for_brax_training.
    Operates per-env (pre-vmap).
    """

    def __init__(self, env: mjx_env.MjxEnv, n_frames: int = 3):
        super().__init__(env)
        self._n_frames = n_frames

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        obs = state.obs
        if isinstance(obs, dict):
            raw = obs["state"]
            stack = jp.tile(raw, self._n_frames)
            obs = {**obs, "state": stack}
        else:
            stack = jp.tile(obs, self._n_frames)
            obs = stack
        state.info["frame_stack"] = stack
        return state.replace(obs=obs)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        obs = state.obs
        if isinstance(obs, dict):
            raw = obs["state"]
        else:
            raw = obs
        raw_dim = raw.shape[-1]
        old_stack = state.info["frame_stack"]
        new_stack = jp.concatenate([raw, old_stack[:-raw_dim]])
        state.info["frame_stack"] = new_stack
        if isinstance(obs, dict):
            obs = {**obs, "state": new_stack}
        else:
            obs = new_stack
        return state.replace(obs=obs)
```

- [ ] **Step 3: Add `n_frame_stack` to TrainConfig**

In `jax_rl/configs/train_config.py`, add after `domain_rand`:

```python
# Observation preprocessing
n_frame_stack: int = 1  # 1 = no stacking, 3 = standard for locomotion
```

- [ ] **Step 4: Apply wrapper in env_setup.py**

In `jax_rl/training/env_setup.py`, after domain randomization setup but before `wrap_for_brax_training`, add:

```python
# Frame stacking (optional, universal wrapper).
if cfg.n_frame_stack > 1:
    from jax_rl.envs.wrappers import FrameStackWrapper
    env = FrameStackWrapper(env, n_frames=cfg.n_frame_stack)
```

Also apply to eval_env:
```python
if cfg.n_frame_stack > 1:
    from jax_rl.envs.wrappers import FrameStackWrapper
    eval_env = FrameStackWrapper(eval_env, n_frames=cfg.n_frame_stack)
```

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/wrappers/ jax_rl/configs/train_config.py jax_rl/training/env_setup.py
git commit -m "feat: add FrameStackWrapper — universal frame stacking for any env"
```

---

### Task 3: Add --frame-stack CLI arg to training scripts

**Files:**
- Modify: `train_offpolicy.py`
- Modify: `train_ppo_fast.py`
- Modify: `train_ppo.py`

- [ ] **Step 1: Add CLI arg to each script**

In each script's argparse block, add:
```python
parser.add_argument("--frame-stack", type=int, default=None,
                    help="Number of stacked observation frames (default: 1, use 3 for locomotion)")
```

- [ ] **Step 2: Wire CLI arg to TrainConfig**

In each script where CLI args override config, add:
```python
if args.frame_stack is not None:
    cfg = dataclasses.replace(cfg, n_frame_stack=args.frame_stack)
```

- [ ] **Step 3: Commit**

```bash
git add train_offpolicy.py train_ppo_fast.py train_ppo.py
git commit -m "feat: add --frame-stack CLI flag to all training scripts"
```

---

### Task 4: Tests for FrameStackWrapper

**Files:**
- Create: `tests/test_frame_stack_wrapper.py`

- [ ] **Step 1: Write wrapper tests**

```python
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
```

- [ ] **Step 2: Run wrapper tests**

Run: `uv run python -m pytest tests/test_frame_stack_wrapper.py -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_frame_stack_wrapper.py
git commit -m "test: add FrameStackWrapper tests (unit + integration)"
```

---

### Task 5: Add frame stacking to deploy ObsBuilder

**Files:**
- Modify: `deploy/obs_builder.py`
- Modify: `deploy/sim2sim_direct.py`

Same as original plan Task 3 — ObsBuilder gets `n_frame_stack` param, maintains numpy buffer. sim2sim_direct.py infers n_frame_stack from `runner.obs_dim // 48`.

---

### Task 6: Run full test suite + update docs

Same as original plan Task 4.
