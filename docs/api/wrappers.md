# Wrappers

Environment wrappers for observation transforms, action delays, vectorization, and episode management.

## Observation & Action Wrappers

Applied per-environment *before* vectorization. Order matters — see [Pipeline](#pipeline).

---

### FrameStackWrapper

```python
from jax_rl.envs.wrappers.frame_stack import FrameStackWrapper
```

Stacks N consecutive observations for temporal context. Maintains a FIFO buffer in `state.info["frame_stack"]`.

```python
FrameStackWrapper(env, n_frames=3)
```

`reset(rng) → State`
: Tiles initial observation N times.

`step(state, action) → State`
: Pushes new obs to front, shifts old frames right. Re-tiles on episode done for clean slate.

---

### ActionDelayWrapper

```python
from jax_rl.envs.wrappers.action_delay import ActionDelayWrapper
```

Delays actions by K control steps to simulate real-robot latency. Stores a FIFO action buffer in `state.info`.

```python
ActionDelayWrapper(env, delay_ms=0, delay_range_ms=None)
```

- `delay_ms` — Fixed delay in milliseconds (converted to control steps via `env.dt`)
- `delay_range_ms` — `(min_ms, max_ms)` for per-episode random delay (overrides `delay_ms`)

`reset(rng) → State`
: Initialize action buffer and sample per-episode delay.

`step(state, action) → State`
: Read delayed action from buffer, shift left, write new action. Zeros buffer and resamples delay on done.

---

## Pipeline

```python
from jax_rl.envs.wrappers.pipeline import build_wrapper_pipeline, apply_wrapper_pipeline
```

Declarative wrapper composition from `TrainConfig` fields.

**build_wrapper_pipeline**(cfg) → list[tuple[str, class, kwargs]]
: Reads config and returns ordered `(name, wrapper_cls, kwargs)` tuples.
  Application order: (1) ActionDelayWrapper if configured, (2) FrameStackWrapper if `n_frame_stack > 1`.

**apply_wrapper_pipeline**(env, cfg) → wrapped_env
: Convenience — calls `build_wrapper_pipeline` then applies each wrapper sequentially.

---

## Training Wrappers

Applied *after* per-env wrappers. Compose the full training stack via [`wrap_for_training`](#wrap_for_training).

---

### VmapWrapper

```python
from jax_rl.envs.wrappers.training import VmapWrapper
```

Vectorizes `reset` and `step` with `jax.vmap` to run parallel environments.

```python
VmapWrapper(env)
```

---

### EpisodeWrapper

```python
from jax_rl.envs.wrappers.training import EpisodeWrapper
```

Tracks episode step count, implements action repeat, and sets truncation at `episode_length`.

```python
EpisodeWrapper(env, episode_length=1000, action_repeat=1)
```

Stores `steps`, `truncation`, `episode_done`, and `episode_metrics` in `state.info`.

---

### AutoResetWrapper

```python
from jax_rl.envs.wrappers.training import AutoResetWrapper
```

Automatically resets done environments with no data loss between episodes.

```python
AutoResetWrapper(env, full_reset=False)
```

- `full_reset=False` (default) — replays cached initial data (fast)
- `full_reset=True` — calls `env.reset()` per done env (slower, resets all info)

---

### DomainRandomizationVmapWrapper

```python
from jax_rl.envs.wrappers.training import DomainRandomizationVmapWrapper
```

Vectorized env where each instance gets a different randomized physics model (mass, friction, etc.).

```python
DomainRandomizationVmapWrapper(env, randomization_fn)
```

- `randomization_fn(mjx_model) → (randomized_model_vmap, in_axes)` — returns vmappable randomized models

---

### wrap_for_training

```python
from jax_rl.envs.wrappers.training import wrap_for_training
```

Composes the full training wrapper stack in one call:

```python
wrap_for_training(env, episode_length=1000, action_repeat=1, randomization_fn=None)
```

**Stack order:**

1. **VmapWrapper** (or **DomainRandomizationVmapWrapper** if `randomization_fn` provided)
2. **EpisodeWrapper**
3. **AutoResetWrapper**
