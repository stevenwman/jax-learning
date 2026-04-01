# Wrapper Pipeline + Action Delay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-driven wrapper pipeline so that `env_setup.py` and `record_video.py` never need per-wrapper `if` blocks again, then use it to add `ActionDelayWrapper` as the first consumer.

**Architecture:** A `WrapperPipeline` reads `TrainConfig` and returns the list of wrappers to apply. `env_setup.py` calls `build_wrapper_pipeline(cfg) → list[WrapperFactory]`, applies them in order, then `wrap_for_training()`. `record_video.py` rebuilds the same pipeline from `meta.json["train_config"]`. Adding a new wrapper means: (1) write the wrapper class, (2) add a config field to `TrainConfig`, (3) add one entry to the pipeline builder. Zero changes to consumers.

**Tech Stack:** JAX, dataclasses, existing wrapper base class

**Scope:** This plan covers **Option B** (wrapper pipeline + action delay) — execute this now. **Option C** (RewardSpec + ObsSpec composability) is documented at the end as a future plan for reference but is NOT executed.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `jax_rl/envs/wrappers/pipeline.py` | Create | `build_wrapper_pipeline()` — reads TrainConfig, returns ordered wrapper list |
| `jax_rl/envs/wrappers/action_delay.py` | Create | ActionDelayWrapper (FIFO latency buffer) |
| `jax_rl/envs/wrappers/__init__.py` | Modify | Export new classes |
| `jax_rl/configs/train_config.py` | Modify | Add `action_delay_ms`, `action_delay_range_ms` fields |
| `jax_rl/training/env_setup.py` | Modify | Replace per-wrapper `if` blocks with pipeline |
| `record_video.py` | Modify | Replace per-wrapper `if` blocks with pipeline |
| `train_offpolicy.py` | Modify | Add `--action-delay-ms`, `--action-delay-range-ms` CLI flags |
| `train_ppo_fast.py` | Modify | Same CLI flags |
| `train_ppo.py` | Modify | Same CLI flags |
| `deploy/go2_constants.py` | Modify | Add `ACTION_DELAY_MS = 120` |
| `tests/test_action_delay.py` | Create | ActionDelayWrapper unit tests |
| `tests/test_wrapper_pipeline.py` | Create | Pipeline builder tests |

---

### Task 1: Create wrapper pipeline builder

**Files:**
- Create: `jax_rl/envs/wrappers/pipeline.py`
- Create: `tests/test_wrapper_pipeline.py`

- [ ] **Step 1: Write pipeline builder**

```python
# jax_rl/envs/wrappers/pipeline.py
"""Config-driven wrapper pipeline builder.

Reads TrainConfig and returns an ordered list of wrappers to apply.
Adding a new wrapper = (1) write the class, (2) add config field, (3) add entry here.
No changes to env_setup.py or record_video.py needed.
"""
from __future__ import annotations
from typing import Any


def build_wrapper_pipeline(cfg) -> list[tuple[str, Any, dict]]:
    """Return ordered list of (name, wrapper_cls, kwargs) from config.

    Wrappers are applied in the returned order, before wrap_for_training().
    Order matters: action-modifying wrappers first, obs-modifying wrappers second.

    Args:
        cfg: TrainConfig or dict (from meta.json["train_config"]).
              Supports both dataclass attribute access and dict key access.
    """
    # Support both dataclass (training) and dict (inference from meta.json)
    def _get(key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    pipeline = []

    # 1. Action delay (modifies actions going in)
    action_delay_ms = _get("action_delay_ms", 0)
    action_delay_range_ms = _get("action_delay_range_ms", None)
    if action_delay_range_ms is not None or action_delay_ms > 0:
        from jax_rl.envs.wrappers.action_delay import ActionDelayWrapper
        pipeline.append((
            "action_delay",
            ActionDelayWrapper,
            {"delay_ms": action_delay_ms, "delay_range_ms": action_delay_range_ms},
        ))

    # 2. Frame stacking (modifies obs coming out)
    n_frame_stack = _get("n_frame_stack", 1)
    if n_frame_stack > 1:
        from jax_rl.envs.wrappers.frame_stack import FrameStackWrapper
        pipeline.append((
            "frame_stack",
            FrameStackWrapper,
            {"n_frames": n_frame_stack},
        ))

    return pipeline


def apply_wrapper_pipeline(env, cfg) -> Any:
    """Apply all wrappers from config to env. Returns wrapped env."""
    for name, wrapper_cls, kwargs in build_wrapper_pipeline(cfg):
        env = wrapper_cls(env, **kwargs)
    return env
```

- [ ] **Step 2: Write pipeline tests**

```python
# tests/test_wrapper_pipeline.py
"""Tests for config-driven wrapper pipeline."""
import pytest
from dataclasses import dataclass

from jax_rl.envs.wrappers.pipeline import build_wrapper_pipeline, apply_wrapper_pipeline


@dataclass
class FakeConfig:
    n_frame_stack: int = 1
    action_delay_ms: int = 0
    action_delay_range_ms: tuple[int, int] | None = None


class TestBuildPipeline:
    def test_empty_config_returns_empty(self):
        cfg = FakeConfig()
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 0

    def test_frame_stack_only(self):
        cfg = FakeConfig(n_frame_stack=3)
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 1
        assert pipeline[0][0] == "frame_stack"

    def test_action_delay_only(self):
        cfg = FakeConfig(action_delay_ms=120)
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 1
        assert pipeline[0][0] == "action_delay"

    def test_both_wrappers_ordered(self):
        cfg = FakeConfig(n_frame_stack=3, action_delay_ms=60)
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 2
        assert pipeline[0][0] == "action_delay"  # action wrappers first
        assert pipeline[1][0] == "frame_stack"    # obs wrappers second

    def test_dict_config_works(self):
        """Pipeline should work with dict config (from meta.json)."""
        cfg = {"n_frame_stack": 3, "action_delay_ms": 60}
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 2

    def test_delay_range_overrides_fixed(self):
        cfg = FakeConfig(action_delay_ms=0, action_delay_range_ms=(40, 120))
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 1
        assert pipeline[0][0] == "action_delay"
        assert pipeline[0][2]["delay_range_ms"] == (40, 120)
```

- [ ] **Step 3: Run tests**

Run: `JAX_PLATFORMS=cpu uv run python -m pytest tests/test_wrapper_pipeline.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/envs/wrappers/pipeline.py tests/test_wrapper_pipeline.py
git commit -m "feat: config-driven wrapper pipeline builder"
```

---

### Task 2: Create ActionDelayWrapper

**Files:**
- Create: `jax_rl/envs/wrappers/action_delay.py`
- Create: `tests/test_action_delay.py`

Use the implementation and tests from `docs/superpowers/plans/2026-04-01-action-delay.md` Tasks 1 and 7. The wrapper code and test code are fully specified there — copy them exactly.

Key points:
- Wrapper stores FIFO in `state.info["action_delay_buffer"]`
- Handles auto-reset with `jp.where(state.done, zero_buffer, buffer)`
- Fixed delay: `delay_ms` converted to steps via `env.dt`
- Randomized delay: `delay_range_ms=(min, max)` per-episode sampling
- Imports `Wrapper` from `jax_rl.envs.wrappers.training`

- [ ] **Step 1: Write tests** (from action delay plan Task 1 Step 1 — all 7 tests)
- [ ] **Step 2: Write ActionDelayWrapper** (from action delay plan Task 1 Step 3)
- [ ] **Step 3: Add composed wrapper test** (from action delay plan Task 7 Step 1)
- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu uv run python -m pytest tests/test_action_delay.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/wrappers/action_delay.py tests/test_action_delay.py
git commit -m "feat: ActionDelayWrapper — FIFO latency buffer for sim2real"
```

---

### Task 3: Wire pipeline into env_setup.py and record_video.py

**Files:**
- Modify: `jax_rl/envs/wrappers/__init__.py`
- Modify: `jax_rl/configs/train_config.py`
- Modify: `jax_rl/training/env_setup.py`
- Modify: `record_video.py`

- [ ] **Step 1: Update wrappers `__init__.py`**

Add exports for `ActionDelayWrapper`, `build_wrapper_pipeline`, `apply_wrapper_pipeline`:

```python
from jax_rl.envs.wrappers.action_delay import ActionDelayWrapper
from jax_rl.envs.wrappers.pipeline import build_wrapper_pipeline, apply_wrapper_pipeline
```

Add all three to `__all__`.

- [ ] **Step 2: Add config fields to TrainConfig**

In `jax_rl/configs/train_config.py`, add after `n_frame_stack`:

```python
# Action delay (sim2real latency simulation)
action_delay_ms: int = 0          # 0 = no delay, 120 = WTW default for Go2
action_delay_range_ms: tuple[int, int] | None = None  # randomized per-episode
```

- [ ] **Step 3: Replace per-wrapper `if` blocks in env_setup.py**

Replace the frame stack block (lines 113-116):

```python
# Frame stacking (optional, universal wrapper).
if cfg.n_frame_stack > 1:
    from jax_rl.envs.wrappers import FrameStackWrapper
    env = FrameStackWrapper(env, n_frames=cfg.n_frame_stack)
```

With:

```python
# Apply wrapper pipeline (action delay, frame stacking, etc.)
from jax_rl.envs.wrappers import apply_wrapper_pipeline
env = apply_wrapper_pipeline(env, cfg)
```

Do the same for the eval env block (lines 128-130). Replace:

```python
if cfg.n_frame_stack > 1:
    from jax_rl.envs.wrappers import FrameStackWrapper
    eval_env = FrameStackWrapper(eval_env, n_frames=cfg.n_frame_stack)
```

With:

```python
# Eval env: same pipeline, but action delay uses fixed max (not randomized).
eval_cfg = cfg
if cfg.action_delay_range_ms is not None:
    import dataclasses
    eval_cfg = dataclasses.replace(cfg, action_delay_ms=cfg.action_delay_range_ms[1], action_delay_range_ms=None)
eval_env = apply_wrapper_pipeline(eval_env, eval_cfg)
```

- [ ] **Step 4: Replace per-wrapper `if` blocks in record_video.py**

Replace the frame stacking block (lines 178-183):

```python
# Apply frame stacking if checkpoint was trained with it.
n_frame_stack = meta.get("train_config", {}).get("n_frame_stack", 1)
if n_frame_stack > 1:
    from jax_rl.envs.wrappers import FrameStackWrapper
    env = FrameStackWrapper(env, n_frames=n_frame_stack)
    print(f"  Frame stacking: {n_frame_stack} frames")
```

With:

```python
# Apply wrapper pipeline from checkpoint config (action delay, frame stacking, etc.)
train_cfg = meta.get("train_config", {})
from jax_rl.envs.wrappers import build_wrapper_pipeline, apply_wrapper_pipeline
pipeline = build_wrapper_pipeline(train_cfg)
if pipeline:
    # For recording, use fixed delay (max of range if randomized)
    if train_cfg.get("action_delay_range_ms"):
        train_cfg = {**train_cfg, "action_delay_ms": train_cfg["action_delay_range_ms"][1], "action_delay_range_ms": None}
    env = apply_wrapper_pipeline(env, train_cfg)
    print(f"  Wrappers: {[name for name, _, _ in build_wrapper_pipeline(train_cfg)]}")
```

- [ ] **Step 5: Run existing tests**

Run: `uv run python -m pytest tests/test_go2_warp_env.py tests/test_frame_stack_wrapper.py tests/test_training_wrappers.py tests/test_determinism.py -v`
Expected: All pass — pipeline produces identical behavior to the old `if` blocks.

- [ ] **Step 6: Commit**

```bash
git add jax_rl/envs/wrappers/__init__.py jax_rl/configs/train_config.py jax_rl/training/env_setup.py record_video.py
git commit -m "refactor: replace per-wrapper if blocks with config-driven pipeline"
```

---

### Task 4: CLI flags for action delay

**Files:**
- Modify: `train_offpolicy.py`
- Modify: `train_ppo_fast.py`
- Modify: `train_ppo.py`

- [ ] **Step 1: Add CLI args to all 3 scripts**

In each script's argparse block, add:

```python
parser.add_argument("--action-delay-ms", type=int, default=None,
                    help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Randomized action delay range in ms (e.g., 40 120)")
```

In each script's config override section, add:

```python
if args.action_delay_ms is not None:
    cfg_overrides["action_delay_ms"] = args.action_delay_ms
if args.action_delay_range_ms is not None:
    cfg_overrides["action_delay_range_ms"] = tuple(args.action_delay_range_ms)
```

- [ ] **Step 2: Add deploy constant**

In `deploy/go2_constants.py`, add:

```python
# Action delay — expected real-robot latency (communication + motor response).
ACTION_DELAY_MS = 120  # WTW default for Unitree robots
```

- [ ] **Step 3: Commit**

```bash
git add train_offpolicy.py train_ppo_fast.py train_ppo.py deploy/go2_constants.py
git commit -m "feat: --action-delay-ms and --action-delay-range-ms CLI flags"
```

---

### Task 5: Full test suite + smoke test

- [ ] **Step 1: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests pass (145+ existing + new pipeline + action delay tests).

- [ ] **Step 2: Smoke test training with action delay**

Run (short, no GPU concern):
```bash
uv run python train_offpolicy.py --algo sac --env CartpoleBalance --total-timesteps 50000 --num-envs 32 --action-delay-ms 40
```
Expected: Runs without crash, prints step metrics.

- [ ] **Step 3: Smoke test with randomized delay**

```bash
uv run python train_offpolicy.py --algo sac --env CartpoleBalance --total-timesteps 50000 --num-envs 32 --action-delay-range-ms 20 80
```
Expected: Runs without crash.

- [ ] **Step 4: Update docs**

Update `.context/TODO.md` — mark action delay as done.
Update `.context/journals/2026-04-01.md` — add action delay + pipeline notes.

- [ ] **Step 5: Commit**

```bash
git add .context/
git commit -m "docs: update TODO and journal for wrapper pipeline + action delay"
```

---

## Option C: Future Plan (DO NOT EXECUTE — reference only)

This section documents the RewardSpec and ObsSpec refactors for when we need them (DIAYN / vision RL). It's here so Option B's pipeline design accounts for these future needs.

### C1: RewardSpec — Composable reward terms (before DIAYN)

**When to do this:** When we start DIAYN implementation. DIAYN replaces the env's reward with `reward = discriminator(obs, skill_z)`. With inline `_get_reward()`, this requires forking the env. With RewardSpec, it's a config swap.

**Design:**

```python
# jax_rl/envs/reward_spec.py
@dataclass
class RewardTerm:
    name: str
    weight: float
    fn: Callable  # (data, action, info) -> scalar

@dataclass
class RewardSpec:
    terms: list[RewardTerm]
    clip_min: float = 0.0
    clip_max: float = 10000.0
    scale_by_dt: bool = True
```

**Refactor `_get_reward()`:**

Current (inline):
```python
def _get_reward(self, data, action, info, ...):
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(...),
        "tracking_ang_vel": self._reward_tracking_ang_vel(...),
        ...  # 16 methods
    }
```

After (spec-driven):
```python
def _get_reward(self, data, action, info, ...):
    return {term.name: term.fn(self, data, action, info) for term in self._reward_spec.terms}
```

The 16 reward methods stay on the class (they compute physics quantities). The spec just lists which ones to use and their weights. DIAYN creates a spec with one term: `RewardTerm("discriminator", 1.0, lambda self, data, action, info: discriminator(obs, z))`.

**Estimated effort:** ~2 hours, ~80 lines. No behavior change for existing envs.

### C2: ObsSpec — Config-driven observation groups (before vision RL)

**When to do this:** When we add CNN obs for vision RL, or when DIAYN needs skill vector injection.

**Design:**

```python
# jax_rl/envs/obs_spec.py
@dataclass
class ObsTerm:
    name: str
    fn: Callable       # (data, info) -> array
    noise_scale: float = 0.0

@dataclass
class ObsGroup:
    name: str          # "state", "privileged_state", "pixels"
    terms: list[ObsTerm]
```

**Refactor `_get_obs()`:**

Current: 80 lines of inline hstack with per-sensor noise.

After: ObsGroup iterates terms, applies noise, hstacks. Adding a skill vector z = `ObsTerm("skill_z", lambda data, info: info["skill_z"])` appended to the "state" group. Adding pixels = new `ObsGroup("pixels", [ObsTerm("camera_0", render_fn)])`.

**Estimated effort:** ~2 hours, ~100 lines. No behavior change for existing envs.

### C3: Why these are separate from the wrapper pipeline

The wrapper pipeline (Option B) handles transformations *between* the env and the training loop — action delay, frame stacking, future DrQ augmentation. These are composable, order-dependent, and env-agnostic.

RewardSpec and ObsSpec live *inside* the env class — they're about how a specific env computes its outputs. Different concern, different abstraction layer.

The pipeline doesn't need to know about RewardSpec/ObsSpec. They're orthogonal refactors that happen to serve the same north star (DIAYN composability).
