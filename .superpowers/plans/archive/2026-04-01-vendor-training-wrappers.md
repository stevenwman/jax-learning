# Vendor Training Wrappers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Own the 3 training wrappers (VmapWrapper, EpisodeWrapper, AutoResetWrapper) + the DR vmap wrapper + the Wrapper base class, removing the Brax training wrapper import and Playground wrapper import from core infra.

**Architecture:** Copy ~200 lines of pure JAX wrapper code into `jax_rl/envs/wrappers/training.py`. Update `env_setup.py` to import from our code instead of Brax/Playground. The Wrapper base class still delegates to `mjx_env.MjxEnv` (we still use Playground's env base class — that's env code, not training infra). Update `FrameStackWrapper` to import from our Wrapper.

**Tech Stack:** JAX, MuJoCo MJX (for type hints only in DR wrapper)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `jax_rl/envs/wrappers/training.py` | Create | Wrapper base + VmapWrapper + EpisodeWrapper + AutoResetWrapper + DomainRandomizationVmapWrapper |
| `jax_rl/envs/wrappers/frame_stack.py` | Modify | Import Wrapper from our training.py instead of Playground |
| `jax_rl/envs/wrappers/__init__.py` | Modify | Export new wrappers + wrap_for_training function |
| `jax_rl/training/env_setup.py` | Modify | Import from our wrappers instead of Playground/Brax |
| `tests/test_determinism.py` | Modify | Import from our wrappers instead of Playground |
| `tests/test_training_wrappers.py` | Create | Unit tests for all vendored wrappers |

**Files NOT touched:**
- Go2 env files — they inherit from `mjx_env.MjxEnv` (Playground), that's fine, they're env code
- Training scripts — they call `make_envs()`, no direct wrapper imports
- `record_video.py` — uses `pg_registry.load()` (Playground registry), that's env loading not wrapping

---

### Task 1: Create vendored wrappers

**Files:**
- Create: `jax_rl/envs/wrappers/training.py`

- [ ] **Step 1: Create `training.py` with Wrapper base class**

The Wrapper base class delegates everything to the wrapped env. This is from Playground's `wrapper.py` lines 29-83, adapted to remove the Playground import of `Wrapper` (we define our own).

```python
"""Training wrappers — vectorization, episode management, auto-reset.

Vendored from Brax (VmapWrapper, EpisodeWrapper) and MuJoCo Playground
(AutoResetWrapper, DomainRandomizationVmapWrapper) to own core training
infrastructure. Pure JAX — no Brax dependency.
"""

import contextlib
from typing import Any, Callable, List, Optional, Sequence, Tuple

import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np


class Wrapper(mjx_env.MjxEnv):
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Any):
        self.env = env

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return self.env.reset(rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.env.mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self.env.mjx_model

    @property
    def xml_path(self) -> str:
        return self.env.xml_path

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[
            Sequence[Callable[[mujoco.MjvScene], None]]
        ] = None,
    ) -> Sequence[np.ndarray]:
        return self.env.render(
            trajectory, height, width, camera, scene_option, modify_scene_fns
        )
```

- [ ] **Step 2: Add VmapWrapper**

Append to `training.py`. From Brax `training.py` lines 59-72:

```python
class VmapWrapper(Wrapper):
    """Vectorizes env reset/step via jax.vmap."""

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return jax.vmap(self.env.step)(state, action)
```

- [ ] **Step 3: Add EpisodeWrapper**

Append to `training.py`. From Brax `training.py` lines 75-126:

```python
class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode_length (truncation)."""

    def __init__(self, env: Any, episode_length: int, action_repeat: int = 1):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        state.info['steps'] = jp.zeros(rng.shape[:-1])
        state.info['truncation'] = jp.zeros(rng.shape[:-1])
        state.info['episode_done'] = jp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps

        prev_done = state.info['episode_done']
        state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
        state.info['episode_metrics']['sum_reward'] += jp.sum(rewards, axis=0)
        state.info['episode_metrics']['length'] *= (1 - prev_done)
        state.info['episode_metrics']['length'] += self.action_repeat
        for metric_name in state.metrics.keys():
            if metric_name != 'reward':
                state.info['episode_metrics'][metric_name] *= (1 - prev_done)
                state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_done'] = done
        return state.replace(done=done)
```

- [ ] **Step 4: Add AutoResetWrapper**

Append to `training.py`. From Playground `wrapper.py` lines 121-206:

```python
class AutoResetWrapper(Wrapper):
    """Automatically resets done envs.

    Default (fast): replays cached initial data+obs. state.info is NOT reset.
    full_reset=True: calls env.reset() per done env. Slower but resets info.
    """

    def __init__(self, env: Any, full_reset: bool = False):
        super().__init__(env)
        self._full_reset = full_reset
        self._info_key = 'AutoResetWrapper'

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(rng)
        rng, key = rng_key[..., 0], rng_key[..., 1]
        state = self.env.reset(key)
        state.info[f'{self._info_key}_first_data'] = state.data
        state.info[f'{self._info_key}_first_obs'] = state.obs
        state.info[f'{self._info_key}_rng'] = rng
        state.info[f'{self._info_key}_done_count'] = jp.zeros(
            key.shape[:-1], dtype=int
        )
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        reset_state = None
        rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
        reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
        if self._full_reset:
            reset_state = self.reset(reset_key)
            reset_data = reset_state.data
            reset_obs = reset_state.obs
        else:
            reset_data = state.info[f'{self._info_key}_first_data']
            reset_obs = state.info[f'{self._info_key}_first_obs']

        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, reset_data, state.data)
        obs = jax.tree.map(where_done, reset_obs, state.obs)

        next_info = state.info
        done_count_key = f'{self._info_key}_done_count'
        if self._full_reset and reset_state:
            next_info = jax.tree.map(where_done, reset_state.info, state.info)
            next_info[done_count_key] = state.info[done_count_key]
            if 'steps' in next_info:
                next_info['steps'] = state.info['steps']
            preserve_info_key = f'{self._info_key}_preserve_info'
            if preserve_info_key in next_info:
                next_info[preserve_info_key] = state.info[preserve_info_key]

        next_info[done_count_key] += state.done.astype(int)
        next_info[f'{self._info_key}_rng'] = reset_rng

        return state.replace(data=data, obs=obs, info=next_info)
```

- [ ] **Step 5: Add DomainRandomizationVmapWrapper**

Append to `training.py`. From Playground `wrapper.py` lines 209-246:

```python
class DomainRandomizationVmapWrapper(Wrapper):
    """Vectorized env with per-env domain randomization."""

    def __init__(
        self,
        env: Any,
        randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    ):
        super().__init__(env)
        self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

    @contextlib.contextmanager
    def _v_env_fn(self, mjx_model: mjx.Model):
        env = self.env.unwrapped
        old_mjx_model = env._mjx_model
        try:
            env.unwrapped._mjx_model = mjx_model
            yield env
        finally:
            env.unwrapped._mjx_model = old_mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        def reset(mjx_model, rng):
            with self._v_env_fn(mjx_model) as v_env:
                return v_env.reset(rng)
        return jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        def step(mjx_model, s, a):
            with self._v_env_fn(mjx_model) as v_env:
                return v_env.step(s, a)
        return jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
            self._mjx_model_v, state, action
        )
```

- [ ] **Step 6: Add `wrap_for_training` function**

Append to `training.py`:

```python
def wrap_for_training(
    env: Any,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
    """Wrap a raw env for training: vmap + episode management + auto-reset."""
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env)
    return env
```

- [ ] **Step 7: Commit**

```bash
git add jax_rl/envs/wrappers/training.py
git commit -m "feat: vendor training wrappers (Vmap, Episode, AutoReset, DR) from Brax/Playground"
```

---

### Task 2: Wire vendored wrappers into env_setup.py and update imports

**Files:**
- Modify: `jax_rl/envs/wrappers/__init__.py`
- Modify: `jax_rl/envs/wrappers/frame_stack.py`
- Modify: `jax_rl/training/env_setup.py`
- Modify: `tests/test_determinism.py`

- [ ] **Step 1: Update wrappers `__init__.py`**

```python
from jax_rl.envs.wrappers.frame_stack import FrameStackWrapper
from jax_rl.envs.wrappers.training import (
    Wrapper,
    VmapWrapper,
    EpisodeWrapper,
    AutoResetWrapper,
    DomainRandomizationVmapWrapper,
    wrap_for_training,
)

__all__ = [
    "FrameStackWrapper",
    "Wrapper",
    "VmapWrapper",
    "EpisodeWrapper",
    "AutoResetWrapper",
    "DomainRandomizationVmapWrapper",
    "wrap_for_training",
]
```

- [ ] **Step 2: Update FrameStackWrapper imports**

In `jax_rl/envs/wrappers/frame_stack.py`, change:
```python
from mujoco_playground._src.wrapper import Wrapper
```
To:
```python
from jax_rl.envs.wrappers.training import Wrapper
```

Also remove the `from mujoco_playground._src import mjx_env` import — instead import `mjx_env` through our training module or keep it since it's used for type hints. Actually, `mjx_env.State` is used in type hints, so keep `from mujoco_playground._src import mjx_env` since frame_stack IS env code.

- [ ] **Step 3: Update env_setup.py**

Change:
```python
from mujoco_playground._src.wrapper import wrap_for_brax_training
```
To:
```python
from jax_rl.envs.wrappers import wrap_for_training
```

Then replace both calls to `wrap_for_brax_training(` with `wrap_for_training(`.

- [ ] **Step 4: Update test_determinism.py**

Change:
```python
from mujoco_playground._src.wrapper import wrap_for_brax_training
```
To:
```python
from jax_rl.envs.wrappers import wrap_for_training
```

Replace `wrap_for_brax_training(` with `wrap_for_training(` (2 occurrences).

- [ ] **Step 5: Run existing tests to verify no breakage**

Run: `uv run python -m pytest tests/test_go2_warp_env.py tests/test_frame_stack_wrapper.py tests/test_determinism.py -v`
Expected: All pass — behavior is identical.

- [ ] **Step 6: Commit**

```bash
git add jax_rl/envs/wrappers/__init__.py jax_rl/envs/wrappers/frame_stack.py jax_rl/training/env_setup.py tests/test_determinism.py
git commit -m "refactor: use vendored wrappers instead of Brax/Playground imports"
```

---

### Task 3: Write tests for vendored wrappers

**Files:**
- Create: `tests/test_training_wrappers.py`

- [ ] **Step 1: Write VmapWrapper tests**

```python
"""Tests for vendored training wrappers."""
import jax
import jax.numpy as jnp
import pytest

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
        # obs should have batch dim
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
```

- [ ] **Step 2: Write EpisodeWrapper tests**

```python
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

    def test_truncation_flag_zero_when_env_terminates(self, raw_env):
        """If env sets done=1 before episode_length, truncation should be 0."""
        env = VmapWrapper(raw_env)
        env = EpisodeWrapper(env, episode_length=1000)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        action = jnp.zeros((2, raw_env.action_size))
        state = env.step(state, action)
        # CartpoleBalance won't terminate in 1 step, but truncation should be 0
        assert jnp.all(state.info['truncation'] == 0)
```

- [ ] **Step 3: Write AutoResetWrapper tests**

```python
class TestAutoResetWrapper:
    def test_done_envs_get_reset_obs(self, raw_env):
        """After done=1, obs should be replaced with cached initial obs."""
        env = VmapWrapper(raw_env)
        env = EpisodeWrapper(env, episode_length=3)
        env = AutoResetWrapper(env)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        initial_obs = state.obs.copy()
        action = jnp.zeros((2, raw_env.action_size))

        # Step to episode end
        for _ in range(3):
            state = env.step(state, action)

        # After auto-reset, obs should match initial (cached)
        assert jnp.allclose(state.obs, initial_obs, atol=1e-5), \
            "Auto-reset should restore cached initial obs"

    def test_step_counter_resets_on_done(self, raw_env):
        env = VmapWrapper(raw_env)
        env = EpisodeWrapper(env, episode_length=3)
        env = AutoResetWrapper(env)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        state = env.reset(keys)
        action = jnp.zeros((2, raw_env.action_size))

        # Step to done
        for _ in range(3):
            state = env.step(state, action)

        # Next step should have steps=1 (reset to 0 then stepped)
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
        # 2 steps → done
        state = env.step(state, action)
        state = env.step(state, action)
        assert jnp.all(state.info['AutoResetWrapper_done_count'] == 1)
```

- [ ] **Step 4: Write wrap_for_training integration test**

```python
class TestWrapForTraining:
    def test_full_pipeline_dmc(self, raw_env):
        """End-to-end: wrap → reset → step → auto-reset cycle."""
        env = wrap_for_training(raw_env, episode_length=5)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        state = env.reset(keys)
        assert state.obs.shape[0] == 4

        action = jnp.zeros((4, raw_env.action_size))
        for _ in range(10):  # 2 full episodes
            state = env.step(state, action)
        assert jnp.all(jnp.isfinite(state.obs))
        assert jnp.all(jnp.isfinite(state.reward))

    def test_full_pipeline_go2_warp(self, raw_go2):
        """End-to-end with dict obs (Go2 Warp)."""
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
        """Our wrap_for_training should produce identical results to Playground's."""
        from mujoco_playground._src.wrapper import wrap_for_brax_training

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        # Our wrapper
        ours = wrap_for_training(raw_env, episode_length=10)
        our_state = ours.reset(keys)

        # Playground's wrapper
        raw_env2 = type(raw_env)()  # fresh instance
        theirs = wrap_for_brax_training(raw_env2, episode_length=10)
        their_state = theirs.reset(keys)

        # Obs should match exactly
        assert jnp.allclose(our_state.obs, their_state.obs, atol=1e-6), \
            "Vendored wrapper should produce identical obs to Playground"

        # Step and compare
        action = jnp.zeros((4, raw_env.action_size))
        our_state = ours.step(our_state, action)
        their_state = theirs.step(their_state, action)
        assert jnp.allclose(our_state.obs, their_state.obs, atol=1e-6)
        assert jnp.allclose(our_state.reward, their_state.reward, atol=1e-6)
```

- [ ] **Step 5: Run all tests**

Run: `uv run python -m pytest tests/test_training_wrappers.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_training_wrappers.py
git commit -m "test: add unit tests for vendored training wrappers"
```

---

### Task 4: Run full test suite + DMC sanity bench

**Files:**
- Run: `tests/` (full suite)

- [ ] **Step 1: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: All 125+ tests pass.

- [ ] **Step 2: DMC quick sanity bench**

Run a short SAC training on CartpoleBalance to verify the full pipeline works end-to-end with vendored wrappers:

```bash
uv run python train_offpolicy.py --algo sac --env CartpoleBalance --total-timesteps 200000 --num-envs 64 --eval-every 50000
```

Expected: Training runs, eval scores improve (CartpoleBalance should reach ~800+ in 200k steps with SAC).

- [ ] **Step 3: Commit docs update**

Update `.context/TODO.md` and `.context/journals/2026-04-01.md` to record the vendoring.
