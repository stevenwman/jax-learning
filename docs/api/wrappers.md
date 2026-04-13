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

The full wrapper stack for training, in order, is:

```
Raw env (MuJoCo Playground)
  → ActionDelayWrapper      (if action_delay_ms > 0 or action_delay_range_ms set)
  → FrameStackWrapper       (if n_frame_stack > 1)
  → [Training wrappers — choice depends on reset_mode:]

  reset_mode="legacy" (default):
    → VmapWrapper             (vectorize across num_envs)
    → EpisodeWrapper          (episode length, truncation flag)
    → AutoResetWrapper        (auto-reset with cached initial state)

  reset_mode="per_step":
    → DomainRandWrapper       (vectorization + episode tracking + fresh ICs + per-episode DR)
```

`DomainRandWrapper` replaces the entire `VmapWrapper + EpisodeWrapper + AutoResetWrapper` stack. It is the correct choice for Go2 locomotion and any policy intended for sim-to-real transfer — it applies fresh initial conditions and per-episode domain randomization (friction, mass, PD gain scales, etc.) declared by the env via `get_domain_randomization_spec()`. The legacy stack remains available for lightweight DM Control benchmarks where fresh resets are not needed and throughput matters.

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

!!! warning
    With `full_reset=False` (the default), every episode starts from the same cached initial state. This works fine **with domain randomization enabled** (e.g., `--reset-mode per_step`), where physical parameters vary per episode. However, without domain randomization, the policy exploits the fixed initial conditions and fails catastrophically on real hardware or varied simulation settings. Always pair cached initial state with domain randomization for policies intended for deployment.

---

### DomainRandWrapper

```python
from jax_rl.envs.wrappers.domain_rand import DomainRandWrapper
```

Unified wrapper that replaces the `AutoResetWrapper + EpisodeWrapper` stack for envs that need fresh resets and per-episode domain randomization. Handles vectorization, episode length tracking, auto-reset, and DR in a single wrapper. DR parameters are declared by the env via `get_domain_randomization_spec()`.

```python
DomainRandWrapper(env, episode_length=1000, mode="per_step")
```

- `mode="per_step"` — per-env reset every step when done; fresh IC and per-episode DR.

`reset(rng) → State`
: Initial reset; initializes wrapper keys in `state.info`.

`step(state, action) → State`
: Step all envs with auto-reset and DR.

---

### wrap_for_training

```python
from jax_rl.envs.wrappers.training import wrap_for_training
```

Composes the full training wrapper stack in one call:

```python
wrap_for_training(env, episode_length=1000, action_repeat=1)
```

**Stack order:**

1. **VmapWrapper**
2. **EpisodeWrapper**
3. **AutoResetWrapper**
