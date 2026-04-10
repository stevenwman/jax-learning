# Off-Policy Loop Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the shared off-policy training loop into a single function-level utility (`jax_rl/training/offpolicy_loop.py`) so each per-algo script (`train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py`) becomes a thin ~60-line file showing only algo-specific decisions (optimizer, explore closure, log fields). FlashSAC stays standalone.

**Architecture:** One new helper `run_offpolicy_loop()` takes the already-constructed algo + an `EnvBundle` + an `explore_fn` + log fields, and runs the entire loop (env setup prints, W&B, pipe/buffer/norm_state, resume, tracker/ctx/ckpt_mgr, main loop, eval/checkpoint, wandb finish). New `make_env_bundle()` wraps `make_envs` with dict-obs detection. No Trainer base class, no algo registry, no inheritance — just a function.

**Tech Stack:** JAX, Flax, Optax, MuJoCo Playground, pytest.

**Spec:** `.superpowers/specs/2026-04-10-offpolicy-loop-extraction.md`
**Review source:** `.superpowers/specs/2026-04-10-offpolicy-training-scripts-review.md`

**Working environment:** This plan is being executed in a worktree at `../jax-learning-offpolicy-loop/` on branch `refactor/offpolicy-loop`. All file paths below are relative to that worktree root.

---

## Task ordering rationale

1. **Task 1** (EnvBundle) creates a prerequisite that Tasks 4-8 depend on.
2. **Task 2** (metrics fmt bug) and **Task 3** (non-breaking ObsPipeline extension) are independent cheap wins — done early while context is fresh.
3. **Task 4** (run_offpolicy_loop helper) is the main event. Uses the new `make_buffer(..., critic_obs_dim=...)` signature added in Task 3.
4. **Tasks 5-8** (thin each script) each depend on 1 + 4 but are independent of each other. They switch from the old `make_buffer_with_critic` to the new unified `make_buffer` as a side effect.
5. **Task 9** (delete `make_buffer_with_critic`) is the cleanup half of the ObsPipeline merge — safe now that Tasks 4-8 have migrated all callers.
6. **Tasks 10-11** are documentation + verification.
7. **Smoke tests** are deferred to a "Tomorrow morning TODO" section because the GPU is in use today (10.3 / 16.3 GiB, 86% util, two python processes).

**No broken window:** Task 3 is deliberately non-breaking — it adds the new `critic_obs_dim` parameter without removing `make_buffer_with_critic`. The old method is deleted in Task 9 only after every caller has been migrated. Scripts remain runnable at every intermediate commit.

---

## Task 1: Add EnvBundle dataclass + `make_env_bundle` wrapper

**Files:**
- Modify: `jax_rl/training/env_setup.py` (add dataclass + wrapper below `make_envs`)
- Modify: `jax_rl/training/__init__.py` (export new symbols)
- Test: `tests/test_env_bundle.py` (new file)

**Background:** `make_envs` currently returns a 7-tuple. The 4 training scripts each do a 14-line dict-obs detection block after calling it. Add a wrapper `make_env_bundle(cfg, seed) -> EnvBundle` that calls `make_envs` internally and adds the detection, returning a dataclass. Leave `make_envs` untouched — 11 existing callers use it and we don't need to churn them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_env_bundle.py`:

```python
"""Tests for EnvBundle + make_env_bundle wrapper."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.configs.env_presets import get_sac_preset
from jax_rl.training import make_env_bundle, EnvBundle


def test_env_bundle_flat_obs_cheetahrun():
    """CheetahRun has flat array obs — bundle should have dict_obs=False."""
    cfg, _ = get_sac_preset("CheetahRun")
    cfg.num_envs = 4  # keep test fast
    bundle = make_env_bundle(cfg, seed=0)

    assert isinstance(bundle, EnvBundle)
    assert bundle.dict_obs is False
    assert bundle.has_privileged is False
    assert bundle.critic_obs_dim is None
    assert bundle.obs_dim > 0  # don't hardcode — Playground spec could change
    assert bundle.action_dim > 0
    assert bundle.key is not None


@pytest.mark.slow
def test_env_bundle_dict_obs_go2warp():
    """Go2WarpJoystickFlat has dict obs with privileged_state — bundle should reflect that."""
    cfg, _ = get_sac_preset("Go2WarpJoystickFlat")
    cfg.num_envs = 4
    bundle = make_env_bundle(cfg, seed=0)

    assert bundle.dict_obs is True
    assert bundle.has_privileged is True
    assert bundle.critic_obs_dim is not None and bundle.critic_obs_dim > bundle.obs_dim
    assert bundle.obs_dim == 48  # Go2Warp actor obs
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m pytest tests/test_env_bundle.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_env_bundle'` or similar.

- [ ] **Step 3: Implement `EnvBundle` + `make_env_bundle` in `env_setup.py`**

Add to `jax_rl/training/env_setup.py` (after `make_envs`, before `make_identity_norm_state`):

```python
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class EnvBundle:
    """Env setup bundle for off-policy training scripts.

    Wraps make_envs output with dict-obs detection so training scripts don't
    need to re-detect asymmetric critic structure.
    """
    env: Any
    env_step: Callable
    env_state: Any
    eval_env: Any
    obs_dim: int
    action_dim: int
    critic_obs_dim: int | None  # None if symmetric
    has_privileged: bool
    dict_obs: bool
    key: Any  # jax.Array


def make_env_bundle(cfg: TrainConfig, seed: int) -> EnvBundle:
    """Wrap make_envs + dict obs detection. For off-policy training scripts.

    Returns EnvBundle with dict_obs / has_privileged / critic_obs_dim populated.
    When dict obs with privileged_state is present, obs_dim is the actor
    ("state") dim and critic_obs_dim is the privileged dim.

    Prints a one-line summary when dict obs is detected (matches existing
    per-script print behavior).
    """
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)

    dict_obs = isinstance(env_state.obs, dict)
    has_privileged = False
    critic_obs_dim = None

    if dict_obs:
        obs_dim = env_state.obs["state"].shape[-1]
        has_privileged = "privileged_state" in env_state.obs
        if has_privileged:
            critic_obs_dim = env_state.obs["privileged_state"].shape[-1]
            print(f"  Dict obs detected: actor={obs_dim}d, critic={critic_obs_dim}d (asymmetric)")
        else:
            print(f"  Dict obs detected: using 'state' key ({obs_dim}d) for off-policy")

    return EnvBundle(
        env=env, env_step=env_step, env_state=env_state, eval_env=eval_env,
        obs_dim=obs_dim, action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        has_privileged=has_privileged,
        dict_obs=dict_obs,
        key=key,
    )
```

- [ ] **Step 4: Export new symbols from `jax_rl/training/__init__.py`**

Change line 10 from:
```python
from jax_rl.training.env_setup import make_envs, make_identity_norm_state
```
to:
```python
from jax_rl.training.env_setup import make_envs, make_env_bundle, EnvBundle, make_identity_norm_state
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run python -m pytest tests/test_env_bundle.py::test_env_bundle_flat_obs_cheetahrun -v`
Expected: PASS.

(The `@pytest.mark.slow` test for Go2Warp is skipped unless `-m slow` is passed — don't run it now, it's validated during smoke tests tomorrow.)

- [ ] **Step 6: Commit**

```bash
git add jax_rl/training/env_setup.py jax_rl/training/__init__.py tests/test_env_bundle.py
git commit -m "feat: EnvBundle dataclass + make_env_bundle wrapper

Wraps make_envs with dict-obs detection. Replaces the 14-line dict-obs
detection block duplicated across the 4 per-algo off-policy scripts.
make_envs itself is unchanged — PPO/FlashSAC/tests keep their existing
callers."
```

---

## Task 2: Fix `metrics_logger.py` fmt-string bug (review item #3)

**Files:**
- Modify: `jax_rl/training/metrics_logger.py:99`
- Test: `tests/test_wandb_metrics.py` (append)

**Background:** `log_training_step` line 99 has `parts.append(f"{label} {val:.3e}")` — it ignores the `fmt` parameter from the caller's `extra_fields` tuple. Training scripts pass `.3f`, `.4f` that get silently dropped. This is a one-line bug.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wandb_metrics.py`:

```python
def test_log_training_step_respects_extra_field_fmt(capsys):
    """log_training_step must use the fmt from extra_fields, not hardcoded .3e."""
    from jax_rl.training import log_training_step, EpisodeTracker

    tracker = EpisodeTracker(num_envs=1)
    # Seed a fake episode so tracker.recent_stats() has values.
    import numpy as np
    tracker.step(np.array([1.0]), np.array([1.0]))

    last_metrics = {"entropy": 1.2345, "alpha": 0.98765}
    log_training_step(
        total_steps=100,
        tracker=tracker,
        last_metrics=last_metrics,
        sps=1000,
        is_training=True,
        buffer_size=100,
        min_buffer=50,
        extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        elapsed=1.0,
    )
    captured = capsys.readouterr()
    # ".3f" format on 1.2345 → "1.234" or "1.235", NOT scientific notation
    assert "Ent 1.234" in captured.out or "Ent 1.235" in captured.out, \
        f"Expected Ent in .3f format, got: {captured.out}"
    assert "Alpha 0.9877" in captured.out or "Alpha 0.9876" in captured.out, \
        f"Expected Alpha in .4f format, got: {captured.out}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m pytest tests/test_wandb_metrics.py::test_log_training_step_respects_extra_field_fmt -v`
Expected: FAIL — current output will show `Ent 1.235e+00` (scientific notation) instead of `Ent 1.235`.

- [ ] **Step 3: Fix `metrics_logger.py` line 99**

Change line 99 from:
```python
            parts.append(f"{label} {val:.3e}")
```
to:
```python
            parts.append(f"{label} {val:{fmt}}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python -m pytest tests/test_wandb_metrics.py::test_log_training_step_respects_extra_field_fmt -v`
Expected: PASS.

- [ ] **Step 5: Run all of `test_wandb_metrics.py` to confirm nothing else broke**

Run: `uv run python -m pytest tests/test_wandb_metrics.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add jax_rl/training/metrics_logger.py tests/test_wandb_metrics.py
git commit -m "fix: log_training_step respects fmt from extra_fields

Line 99 was hardcoded to .3e, silently dropping the fmt string passed
by callers. Training scripts pass .3f/.4f that never took effect."
```

---

## Task 3: Add `critic_obs_dim` parameter to `ObsPipeline.make_buffer` (non-breaking)

**Files:**
- Modify: `jax_rl/training/obs_pipeline.py` (extend `make_buffer` signature — DO NOT delete `make_buffer_with_critic` yet)
- Test: `tests/test_obs_pipeline.py` (append)

**Background:** The two methods share ~80% of their code. Review item #5 calls for merging them. To avoid the "broken window" where the 4 training scripts temporarily reference a deleted method, this task is split:
- **Task 3 (this one)**: non-breaking — add `critic_obs_dim` parameter to `make_buffer` alongside the existing `make_buffer_with_critic`. Both coexist briefly.
- **Task 9 (after Tasks 5-8 migrate all callers)**: delete `make_buffer_with_critic`.

This way, the scripts keep working throughout the refactor — no broken intermediate commits.

**Note on `JaxReplayBuffer` internals:** the "extra obs dims" attribute is **private**: `self._extra_obs_dims`. There is no public accessor. Tests that inspect this attribute should use the underscore-private name, matching the pattern in existing tests (e.g. `test_replay_buffer.py` uses `buf._fsc` for frame stack config).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_obs_pipeline.py`:

```python
def test_make_buffer_without_critic_obs_dim():
    """make_buffer with critic_obs_dim=None → no extra critic buffer."""
    from jax_rl.training import ObsPipeline
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=False)
    buffer = pipe.make_buffer(obs_dim=17, action_dim=6, buffer_size=1000)
    assert buffer is not None
    assert buffer._extra_obs_dims is None or buffer._extra_obs_dims == {}


def test_make_buffer_with_critic_obs_dim():
    """make_buffer with critic_obs_dim=64 → allocates critic_obs buffer."""
    from jax_rl.training import ObsPipeline
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False)
    buffer = pipe.make_buffer(obs_dim=17, action_dim=6, buffer_size=1000,
                              critic_obs_dim=64)
    assert buffer is not None
    assert "critic_obs" in buffer._extra_obs_dims
    assert buffer._extra_obs_dims["critic_obs"] == 64


def test_make_buffer_frame_stack_with_critic_obs_dim():
    """Frame stacking × privileged critic — this is the Go2Warp production path."""
    from jax_rl.training import ObsPipeline
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False,
                       n_frame_stack=3)
    # obs_dim = raw_dim * n_frame_stack = 16 * 3 = 48 (mimics Go2Warp)
    buffer = pipe.make_buffer(obs_dim=48, action_dim=12, buffer_size=1000,
                              critic_obs_dim=120, num_envs=4)
    assert buffer is not None
    assert "critic_obs" in buffer._extra_obs_dims
    assert buffer._extra_obs_dims["critic_obs"] == 120
    # Frame stack config should be populated.
    assert buffer._fsc is not None
    assert buffer._fsc.n_frames == 3
    assert buffer._fsc.raw_dim == 16


def test_make_buffer_privileged_without_critic_obs_dim_raises():
    """has_privileged=True with critic_obs_dim=None should raise ValueError."""
    import pytest
    from jax_rl.training import ObsPipeline
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False)
    with pytest.raises(ValueError, match="critic_obs_dim required"):
        pipe.make_buffer(obs_dim=17, action_dim=6, buffer_size=1000)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m pytest tests/test_obs_pipeline.py::test_make_buffer_with_critic_obs_dim tests/test_obs_pipeline.py::test_make_buffer_frame_stack_with_critic_obs_dim tests/test_obs_pipeline.py::test_make_buffer_privileged_without_critic_obs_dim_raises -v`
Expected: FAIL — `make_buffer` doesn't accept `critic_obs_dim` keyword arg yet, and doesn't raise on missing critic_obs_dim.

- [ ] **Step 3: Extend `make_buffer` in `obs_pipeline.py` — non-breaking**

Replace the existing `make_buffer` method (lines 111-148) with this extended version. **Do NOT delete `make_buffer_with_critic` (lines 150-186) — it stays until Task 9.**

```python
    def make_buffer(self, obs_dim, action_dim, buffer_size,
                    critic_obs_dim=None, num_envs=None):
        """Create JaxReplayBuffer with correct frame_stack + extra_obs_dims.

        Args:
            obs_dim: Actor observation dim (stacked if frame stacking,
                i.e. raw_dim * n_frames).
            action_dim: Action dimensionality.
            buffer_size: Maximum number of transitions.
            critic_obs_dim: Privileged critic obs dim. Required when
                has_privileged is True. Ignored otherwise.
            num_envs: Number of parallel envs (required when n_frame_stack > 1).

        Returns:
            JaxReplayBuffer configured for this pipeline.
        """
        extra_obs_dims = None
        if self.has_privileged:
            if critic_obs_dim is None:
                raise ValueError("critic_obs_dim required when has_privileged=True")
            extra_obs_dims = {"critic_obs": critic_obs_dim}

        frame_stack_config = None
        if self.n_frame_stack > 1:
            if num_envs is None:
                raise ValueError("num_envs required when n_frame_stack > 1")
            raw_dim = obs_dim // self.n_frame_stack
            frame_stack_config = FrameStackConfig(
                n_frames=self.n_frame_stack, raw_dim=raw_dim, num_envs=num_envs
            )
            return JaxReplayBuffer(
                raw_dim, action_dim, max_size=buffer_size,
                frame_stack_config=frame_stack_config,
                extra_obs_dims=extra_obs_dims,
            )

        return JaxReplayBuffer(
            obs_dim, action_dim, max_size=buffer_size,
            extra_obs_dims=extra_obs_dims,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m pytest tests/test_obs_pipeline.py -v`
Expected: all PASS (new tests AND all existing ObsPipeline tests).

- [ ] **Step 5: Commit**

```bash
git add jax_rl/training/obs_pipeline.py tests/test_obs_pipeline.py
git commit -m "feat: add critic_obs_dim param to ObsPipeline.make_buffer

Non-breaking addition. Existing callers of make_buffer(obs_dim, action_dim,
buffer_size) keep working. make_buffer_with_critic is untouched and will
be removed in a follow-up after all callers migrate to the unified
signature."
```

---

## Task 4: Create `run_offpolicy_loop` helper

**Files:**
- Create: `jax_rl/training/offpolicy_loop.py`
- Modify: `jax_rl/training/__init__.py` (export `run_offpolicy_loop`)
- Test: `tests/test_offpolicy_loop.py` (new)

**Background:** This is the main event. Extract the full training loop into a single function. Copy the loop body verbatim from `train_sac.py` (which is the reference), parameterize the 4 variation points (algo, explore_fn, log_extra_fields, log_extra_keys), and take an `EnvBundle` instead of re-detecting obs structure.

The helper is a function, not a class. No inheritance. No hook callbacks beyond `explore_fn`.

- [ ] **Step 1: Write the skeleton test**

Create `tests/test_offpolicy_loop.py`:

```python
"""Smoke test for run_offpolicy_loop — runs SAC on CheetahRun for a handful of steps."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.slow
def test_run_offpolicy_loop_sac_cheetah(tmp_path, monkeypatch):
    """Run SAC + CheetahRun for 2000 env steps — helper should complete without error.

    This is a smoke test: it verifies the glue works end-to-end with a real
    env + real algo. It does NOT verify training quality (too few steps).

    Uses monkeypatch.chdir (pytest fixture) so cwd is restored even on failure —
    `run_offpolicy_loop` creates `checkpoints/<timestamp>_<algo>_<env>_<seed>/`
    relative to cwd and we don't want it in the repo.
    """
    import dataclasses
    import optax

    from jax_rl.algos.sac import SAC
    from jax_rl.configs.env_presets import get_sac_preset
    from jax_rl.training import make_env_bundle, run_offpolicy_loop

    cfg, algo_cfg = get_sac_preset("CheetahRun")
    cfg = dataclasses.replace(cfg, num_envs=4, total_timesteps=2000)
    algo_cfg = dataclasses.replace(algo_cfg, min_buffer_size=500, batch_size=64)

    monkeypatch.chdir(tmp_path)

    bundle = make_env_bundle(cfg, seed=0)

    optimizer = optax.adam(cfg.lr)
    alpha_opt = optax.adam(algo_cfg.alpha_lr)
    algo = SAC(
        config=algo_cfg, obs_dim=bundle.obs_dim, action_dim=bundle.action_dim,
        optimizer=optimizer, alpha_optimizer=alpha_opt,
        gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
        critic_obs_dim=bundle.critic_obs_dim,
    )

    def explore(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key)

    # Should complete without exception.
    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="sac",
        env_bundle=bundle, explore_fn=explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
        seed=0, resume=None, use_wandb=False,
    )
```

- [ ] **Step 2: Run the test — expect failure (helper doesn't exist)**

Run: `uv run python -m pytest tests/test_offpolicy_loop.py -v -m slow`
Expected: FAIL with `ImportError` or `cannot import name 'run_offpolicy_loop'`.

- [ ] **Step 3: Implement `run_offpolicy_loop` in `jax_rl/training/offpolicy_loop.py`**

Create the new file. Structure:

```python
"""Single off-policy training loop — shared across SAC, TD3, FastSAC, FastTD3.

Each per-algo script builds its optimizer + algo + explore closure, then calls
this function. FlashSAC does NOT use this helper (BN state, Zeta noise, and
reward normalization don't fit the shared shape).

Variation points (the 4 things each script passes in):
    - algo: already-constructed SAC/TD3/FastSAC/FastTD3 with optimizer bound
    - explore_fn: (actor_params, obs, key) -> action — handles any noise injection
    - log_extra_fields: algo-specific metrics for stdout logging
    - log_extra_keys: algo-specific metrics for W&B CSV rows
"""
import dataclasses
import os
import time
from datetime import datetime
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from jax_rl.configs.train_config import TrainConfig
from jax_rl.training.checkpointing import CheckpointManager, load_checkpoint
from jax_rl.training.env_setup import EnvBundle
from jax_rl.training.episode_tracker import EpisodeTracker
from jax_rl.training.eval_runner import maybe_eval_and_checkpoint, final_eval_and_checkpoint
from jax_rl.training.metrics_logger import (
    log_training_step, make_metrics_row,
    wandb_init, wandb_setup_metrics, wandb_log, wandb_finish,
)
from jax_rl.training.obs_pipeline import ObsPipeline
from jax_rl.training.train_context import TrainContext


def run_offpolicy_loop(
    cfg: TrainConfig,
    algo_cfg,
    algo,
    algo_name: str,
    env_bundle: EnvBundle,
    explore_fn: Callable,
    log_extra_fields: list,
    log_extra_keys: list,
    seed: int = 0,
    resume: str | None = None,
    use_wandb: bool = False,
    wandb_project: str = "jax-rl",
) -> None:
    """Run the off-policy training loop.

    See train_sac.py, train_td3.py, train_fast_sac.py, train_fast_td3.py for
    concrete usage. FlashSAC does NOT use this helper — it has algo-specific
    loop state (BN stats, Zeta noise, adaptive reward scaling) that don't fit.
    """
    # ── Unpack bundle ──────────────────────────────────────────────────────
    env = env_bundle.env
    env_step = env_bundle.env_step
    env_state = env_bundle.env_state
    eval_env = env_bundle.eval_env
    obs_dim = env_bundle.obs_dim
    action_dim = env_bundle.action_dim
    critic_obs_dim = env_bundle.critic_obs_dim
    has_privileged = env_bundle.has_privileged
    dict_obs = env_bundle.dict_obs
    key = env_bundle.key

    total_env_steps = cfg.total_timesteps

    # ── Banner ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"{algo_name.upper()} — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, episode_length={cfg.episode_length}")
    print(f"  total_timesteps={total_env_steps:,}")
    print(f"  buffer_size={algo_cfg.buffer_size:,}, min_buffer={algo_cfg.min_buffer_size:,}")
    print(f"  batch_size={algo_cfg.batch_size}, grad_updates_per_step={algo_cfg.grad_updates_per_step}")
    print(f"  tau={algo_cfg.tau}, lr={cfg.lr}, gamma={cfg.gamma}")
    print(f"  reward_scaling={cfg.reward_scaling}")

    # ── Timestamp (shared by checkpoint dir + W&B run name) ────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")

    # ── W&B (optional) ─────────────────────────────────────────────────────
    if use_wandb:
        wandb_init(
            project=wandb_project,
            name=f"{timestamp}_{algo_name}_{env_short}_seed{seed}",
            config={
                "algo": algo_name,
                "env": cfg.env_name,
                "seed": seed,
                "timestamp": timestamp,
                **{k: v for k, v in dataclasses.asdict(cfg).items() if k != "ppo"},
                **{f"algo_{k}": v for k, v in dataclasses.asdict(algo_cfg).items()},
            },
        )
        wandb_setup_metrics()

    # ── Algo init ──────────────────────────────────────────────────────────
    key, init_key = jax.random.split(key)
    training_state = algo.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    # ── ObsPipeline + buffer + norm_state ──────────────────────────────────
    use_obs_norm = algo_cfg.obs_normalization
    obs_norm_eps = getattr(algo_cfg, "obs_norm_eps", 1e-8)
    n_frame_stack = cfg.n_frame_stack

    pipe = ObsPipeline(dict_obs, has_privileged, use_obs_norm, n_frame_stack, obs_norm_eps)
    buffer = pipe.make_buffer(
        obs_dim, action_dim, algo_cfg.buffer_size,
        critic_obs_dim=critic_obs_dim, num_envs=cfg.num_envs,
    )
    norm_state = pipe.init_norm_state(obs_dim)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if resume is not None:
        print(f"\n  Resuming from {resume}")
        training_state, norm_state, start_step = load_checkpoint(resume, training_state, norm_state)
        print(f"  Resuming from step {start_step:,}")

    # ── Tracker + ctx + checkpoint manager ─────────────────────────────────
    tracker = EpisodeTracker(cfg.num_envs)
    metrics_log: list[dict] = []
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_{algo_name}_{env_short}_seed{seed}")
    ckpt_mgr = CheckpointManager(ckpt_dir)
    ctx = TrainContext(
        cfg=cfg, algo_cfg=algo_cfg, algo_name=algo_name,
        ckpt_dir=ckpt_dir, obs_dim=obs_dim, action_dim=action_dim,
        metrics_log=metrics_log, ckpt_mgr=ckpt_mgr, resume=resume,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\nCollecting {algo_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // cfg.num_envs)
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    for outer_step in range(start_step // cfg.num_envs, total_env_steps // cfg.num_envs):
        raw_steps = (outer_step + 1) * cfg.num_envs
        total_steps = raw_steps
        raw_obs = pipe.get_obs(env_state.obs)
        critic_raw_obs = pipe.get_critic_obs(env_state.obs) if has_privileged else None

        # Obs normalization
        norm_state = pipe.update_stats(raw_obs, norm_state)
        obs_for_action = pipe.normalize_for_action(raw_obs, norm_state)

        # Action selection
        if len(buffer) < algo_cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action = explore_fn(training_state.actor_params, obs_for_action, ak)

        # Env step
        env_state = env_step(env_state, action)
        truncation = (env_state.info.get("truncation", jnp.zeros_like(env_state.done))
                      if cfg.handle_truncation else jnp.zeros_like(env_state.done))

        # Buffer
        next_raw_obs = pipe.get_obs(env_state.obs)
        extra_kwargs = {}
        if has_privileged:
            extra_kwargs["critic_obs"] = critic_raw_obs
            extra_kwargs["critic_next_obs"] = pipe.get_critic_obs(env_state.obs)
        buffer.add_batch(
            obs=raw_obs, action=action,
            reward=env_state.reward * cfg.reward_scaling,
            next_obs=next_raw_obs, done=env_state.done,
            truncation=truncation, **extra_kwargs,
        )
        tracker.step(np.asarray(env_state.reward), np.asarray(env_state.done))

        # Gradient updates
        if len(buffer) >= algo_cfg.min_buffer_size:
            for _ in range(algo_cfg.grad_updates_per_step):
                key, sample_key = jax.random.split(key)
                jax_batch = buffer.sample(algo_cfg.batch_size, key=sample_key)
                jax_batch = pipe.normalize_batch(jax_batch, norm_state)
                training_state, step_metrics = algo.update(training_state, jax_batch)
                total_gradient_steps += 1
                last_metrics = step_metrics

        # Logging
        if outer_step % log_every == 0 or total_steps >= total_env_steps:
            elapsed = time.time() - t0
            sps = int(total_steps / elapsed) if elapsed > 0 else 0
            is_training = last_metrics and len(buffer) >= algo_cfg.min_buffer_size

            log_training_step(
                total_steps, tracker, last_metrics, sps,
                is_training=is_training,
                buffer_size=len(buffer), min_buffer=algo_cfg.min_buffer_size,
                extra_fields=log_extra_fields,
                elapsed=elapsed,
            )

            if is_training:
                row = make_metrics_row(
                    total_steps, tracker, last_metrics, total_gradient_steps, sps, elapsed,
                    extra_keys=log_extra_keys,
                )
                metrics_log.append(row)
                wandb_log(row, step=raw_steps)

        # Eval + checkpoint
        obs_norm_fn = pipe.make_obs_norm_fn(norm_state)
        # Rebind training_state into _ts to avoid Python loop-variable capture
        # in the q_fn lambda below. The lambda is called later (from inside
        # eval_runner) and must see the CURRENT state, not a later iteration's.
        _ts = training_state
        last_eval_eps, key = maybe_eval_and_checkpoint(
            algo.select_action, training_state.actor_params, eval_env, tracker,
            ctx, training_state, norm_state, last_eval_eps, key,
            obs_normalize_fn=obs_norm_fn,
            q_fn=lambda obs, action: algo.get_q_value(
                _ts, pipe.get_obs(obs), action,
                critic_obs=obs["privileged_state"]
                           if isinstance(obs, dict) and "privileged_state" in obs else None),
        )

    # ── Final eval ─────────────────────────────────────────────────────────
    obs_norm_fn = pipe.make_obs_norm_fn(norm_state)
    final_eval_and_checkpoint(
        algo.select_action, training_state.actor_params, eval_env, tracker,
        ctx, training_state, norm_state, key, total_gradient_steps,
        obs_normalize_fn=obs_norm_fn,
        q_fn=lambda obs, action: algo.get_q_value(
            training_state, pipe.get_obs(obs), action,
            critic_obs=obs["privileged_state"]
                       if isinstance(obs, dict) and "privileged_state" in obs else None),
    )

    wandb_finish()
```

**Note on the `pipe.make_buffer` call:** this depends on Task 3 having merged `make_buffer`/`make_buffer_with_critic`. If Task 3 is skipped or reordered, use the two-method fork instead:

```python
    if has_privileged:
        buffer = pipe.make_buffer_with_critic(
            obs_dim, action_dim, algo_cfg.buffer_size,
            critic_obs_dim, num_envs=cfg.num_envs,
        )
    else:
        buffer = pipe.make_buffer(obs_dim, action_dim, algo_cfg.buffer_size, num_envs=cfg.num_envs)
```

- [ ] **Step 4: Export `run_offpolicy_loop` from `jax_rl/training/__init__.py`**

Add at the end of the import block:
```python
from jax_rl.training.offpolicy_loop import run_offpolicy_loop
```

- [ ] **Step 5: Run the test to verify the helper works (DEFERRED if GPU busy)**

Check GPU: `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv`

- **If GPU is free or below 50% util**: run `uv run python -m pytest tests/test_offpolicy_loop.py::test_run_offpolicy_loop_sac_cheetah -v -m slow`. Expected: PASS (takes ~1-2 min for JIT + loop).
- **If GPU is busy**: skip this step. Add a TODO to the "Tomorrow morning" section to run the test.

- [ ] **Step 6: Commit**

```bash
git add jax_rl/training/offpolicy_loop.py jax_rl/training/__init__.py tests/test_offpolicy_loop.py
git commit -m "feat: run_offpolicy_loop — shared training loop for 4 non-flash off-policy scripts

Extracts the full training loop (env setup prints, W&B, buffer, resume,
tracker, main loop, eval, checkpoint) into a single function. Takes an
already-constructed algo + EnvBundle + explore_fn. FlashSAC stays
standalone (Zeta noise + BN state + reward norm don't fit)."
```

---

## Task 5: Thin `train_sac.py`

**Files:**
- Modify: `train_sac.py` (rewrite `train()` function body)

**Background:** Replace the ~250-line train() function with a ~60-line version that builds the SAC-specific pieces (optimizer, algo, explore pass-through, log fields) and delegates to `run_offpolicy_loop`. CLI argparse block stays unchanged.

- [ ] **Step 1: Rewrite the imports and `train()` function**

Replace lines 14-249 of `train_sac.py` (everything from `import argparse` through the end of the `train()` function) with:

```python
import argparse

import optax

from jax_rl.algos.sac import SAC
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_sac_preset
from jax_rl.training import (
    make_env_bundle, apply_cli_overrides, run_offpolicy_loop,
)


def train(cfg: TrainConfig, algo_cfg, seed: int = 0, resume: str | None = None,
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    env_bundle = make_env_bundle(cfg, seed)

    # SAC-specific: optimizer with optional grad clipping
    if algo_cfg.grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(algo_cfg.grad_clip_norm),
            optax.adam(cfg.lr),
        )
    else:
        optimizer = optax.adam(cfg.lr)
    alpha_optimizer = optax.adam(algo_cfg.alpha_lr)

    # SAC-specific: algo
    algo = SAC(
        config=algo_cfg,
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        optimizer=optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma,
        handle_truncation=cfg.handle_truncation,
        critic_obs_dim=env_bundle.critic_obs_dim,
    )

    # SAC-specific: explore is pass-through (stochastic policy)
    def explore(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key)

    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="sac",
        env_bundle=env_bundle, explore_fn=explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
        seed=seed, resume=resume,
        use_wandb=use_wandb, wandb_project=wandb_project,
    )
```

The module-level `os`/`sys`/XLA env var setup block at the top (lines 9-12) stays. The CLI `if __name__ == "__main__":` block at the bottom (lines 254-300) stays.

- [ ] **Step 2: Verify the thinned script imports cleanly**

Run: `uv run python -c "import train_sac"`
Expected: no error.

- [ ] **Step 3: Verify CLI parses correctly**

Run: `uv run python train_sac.py --help`
Expected: prints help text, no error.

- [ ] **Step 4: Run the existing off-policy algo tests to make sure SAC is still healthy**

Run: `uv run python -m pytest tests/test_offpolicy_algos.py -v -k sac`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add train_sac.py
git commit -m "refactor: thin train_sac.py — delegate to run_offpolicy_loop

train() drops from ~210 lines to ~40. Only SAC-specific bits remain:
optimizer (adam + optional grad clip), algo construction, pass-through
explore closure. CLI args unchanged."
```

---

## Task 6: Thin `train_td3.py`

**Files:**
- Modify: `train_td3.py`

- [ ] **Step 1: Rewrite the imports and `train()` function**

Replace lines 13-253 of `train_td3.py` with:

```python
import argparse

import jax
import optax

from jax_rl.algos.td3 import TD3
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_td3_preset
from jax_rl.training import (
    make_env_bundle, apply_cli_overrides, run_offpolicy_loop,
)


def train(cfg: TrainConfig, algo_cfg, seed: int = 0, resume: str | None = None,
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    env_bundle = make_env_bundle(cfg, seed)

    # TD3-specific: plain adam for both actor and critic
    algo = TD3(
        config=algo_cfg,
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        actor_optimizer=optax.adam(cfg.lr),
        critic_optimizer=optax.adam(cfg.lr),
        gamma=cfg.gamma,
        handle_truncation=cfg.handle_truncation,
        critic_obs_dim=env_bundle.critic_obs_dim,
    )

    # TD3-specific: explore injects Gaussian noise. noise_min/max optional
    # (randomized noise range from the Fast papers).
    exploration_noise_std = getattr(algo_cfg, "exploration_noise_std", 0.1)
    noise_min = getattr(algo_cfg, "noise_min", None)
    noise_max = getattr(algo_cfg, "noise_max", None)

    def explore(actor_params, obs, key):
        key, noise_key = jax.random.split(key)
        if noise_min is not None:
            noise_std = jax.random.uniform(noise_key, (), minval=noise_min, maxval=noise_max)
        else:
            noise_std = exploration_noise_std
        return algo.select_action(
            actor_params, obs, key, deterministic=False, exploration_noise=noise_std,
        )

    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="td3",
        env_bundle=env_bundle, explore_fn=explore,
        log_extra_fields=[],
        log_extra_keys=[],
        seed=seed, resume=resume,
        use_wandb=use_wandb, wandb_project=wandb_project,
    )
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "import train_td3"`
Expected: no error.

- [ ] **Step 3: Verify CLI**

Run: `uv run python train_td3.py --help`
Expected: prints help.

- [ ] **Step 4: Run existing TD3 tests**

Run: `uv run python -m pytest tests/test_offpolicy_algos.py -v -k td3`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_td3.py
git commit -m "refactor: thin train_td3.py — delegate to run_offpolicy_loop

train() drops from ~215 lines to ~50. TD3-specific bits: plain adam
optimizers, explore closure with Gaussian noise (supports noise_min/max
range from Fast papers)."
```

---

## Task 7: Thin `train_fast_sac.py`

**Files:**
- Modify: `train_fast_sac.py`

**Background:** Same as train_sac.py, but FastSAC uses a cosine LR schedule (warmup-aware) and adamw with weight decay.

- [ ] **Step 1: Rewrite the `train()` function**

Replace lines 14-250 of `train_fast_sac.py` with:

```python
import argparse

import optax

from jax_rl.algos.fast_sac import FastSAC
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_fast_sac_preset
from jax_rl.training import (
    make_env_bundle, apply_cli_overrides, run_offpolicy_loop,
)


def train(cfg: TrainConfig, algo_cfg, seed: int = 0, resume: str | None = None,
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    env_bundle = make_env_bundle(cfg, seed)

    # FastSAC-specific: cosine LR schedule (warmup-aware) + adamw
    warmup_steps = algo_cfg.min_buffer_size // cfg.num_envs
    train_iters = (cfg.total_timesteps // cfg.num_envs) - warmup_steps
    total_grad_est = train_iters * algo_cfg.grad_updates_per_step
    lr_schedule = (
        optax.cosine_decay_schedule(cfg.lr, total_grad_est,
                                    alpha=algo_cfg.lr_end / cfg.lr)
        if algo_cfg.lr_end < cfg.lr else cfg.lr
    )
    optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
    alpha_optimizer = optax.adam(algo_cfg.alpha_lr)

    algo = FastSAC(
        config=algo_cfg,
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        optimizer=optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma,
        handle_truncation=cfg.handle_truncation,
        critic_obs_dim=env_bundle.critic_obs_dim,
    )

    # FastSAC-specific: explore is pass-through (SAC-family stochastic policy)
    def explore(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key)

    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="fast_sac",
        env_bundle=env_bundle, explore_fn=explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
        seed=seed, resume=resume,
        use_wandb=use_wandb, wandb_project=wandb_project,
    )
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "import train_fast_sac"`

- [ ] **Step 3: Verify CLI**

Run: `uv run python train_fast_sac.py --help`

- [ ] **Step 4: Run FastSAC tests**

Run: `uv run python -m pytest tests/test_offpolicy_algos.py -v -k fast_sac`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_fast_sac.py
git commit -m "refactor: thin train_fast_sac.py — delegate to run_offpolicy_loop

train() drops from ~220 lines to ~55. FastSAC-specific bits: cosine LR
schedule + adamw optimizer (b2=0.95, wd=1e-3)."
```

---

## Task 8: Thin `train_fast_td3.py`

**Files:**
- Modify: `train_fast_td3.py`

- [ ] **Step 1: Rewrite the `train()` function**

Replace lines 13-258 of `train_fast_td3.py` with:

```python
import argparse

import jax
import optax

from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_fast_td3_preset
from jax_rl.training import (
    make_env_bundle, apply_cli_overrides, run_offpolicy_loop,
)


def train(cfg: TrainConfig, algo_cfg, seed: int = 0, resume: str | None = None,
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    env_bundle = make_env_bundle(cfg, seed)

    # FastTD3-specific: cosine LR schedule (warmup-aware) + adamw for actor AND critic
    warmup_steps = algo_cfg.min_buffer_size // cfg.num_envs
    train_iters = (cfg.total_timesteps // cfg.num_envs) - warmup_steps
    total_grad_est = train_iters * algo_cfg.grad_updates_per_step
    lr_schedule = (
        optax.cosine_decay_schedule(cfg.lr, total_grad_est,
                                    alpha=algo_cfg.lr_end / cfg.lr)
        if algo_cfg.lr_end < cfg.lr else cfg.lr
    )
    actor_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
    critic_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)

    algo = FastTD3(
        config=algo_cfg,
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        gamma=cfg.gamma,
        handle_truncation=cfg.handle_truncation,
        critic_obs_dim=env_bundle.critic_obs_dim,
    )

    # FastTD3-specific: explore with Gaussian noise (like TD3)
    exploration_noise_std = getattr(algo_cfg, "exploration_noise_std", 0.1)
    noise_min = getattr(algo_cfg, "noise_min", None)
    noise_max = getattr(algo_cfg, "noise_max", None)

    def explore(actor_params, obs, key):
        key, noise_key = jax.random.split(key)
        if noise_min is not None:
            noise_std = jax.random.uniform(noise_key, (), minval=noise_min, maxval=noise_max)
        else:
            noise_std = exploration_noise_std
        return algo.select_action(
            actor_params, obs, key, deterministic=False, exploration_noise=noise_std,
        )

    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="fast_td3",
        env_bundle=env_bundle, explore_fn=explore,
        log_extra_fields=[],
        log_extra_keys=[],
        seed=seed, resume=resume,
        use_wandb=use_wandb, wandb_project=wandb_project,
    )
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "import train_fast_td3"`

- [ ] **Step 3: Verify CLI**

Run: `uv run python train_fast_td3.py --help`

- [ ] **Step 4: Run FastTD3 tests**

Run: `uv run python -m pytest tests/test_offpolicy_algos.py -v -k fast_td3`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_fast_td3.py
git commit -m "refactor: thin train_fast_td3.py — delegate to run_offpolicy_loop

train() drops from ~225 lines to ~65. FastTD3-specific bits: cosine LR
schedule + adamw (separate actor/critic optimizers), explore closure
with Gaussian noise."
```

---

## Task 9: Delete `ObsPipeline.make_buffer_with_critic` (cleanup half of review item #5)

**Files:**
- Modify: `jax_rl/training/obs_pipeline.py` (delete `make_buffer_with_critic` method)

**Background:** Task 3 added `critic_obs_dim` to `make_buffer` as a non-breaking extension, leaving `make_buffer_with_critic` in place. Tasks 4 and 5-8 migrated all internal callers to the unified signature. This task deletes the old method.

- [ ] **Step 1: Grep to confirm no remaining callers**

Grep the whole worktree for `make_buffer_with_critic`:

Run the Grep tool (or `uv run python -c "import subprocess; subprocess.run(['grep', '-rn', 'make_buffer_with_critic', '--include=*.py', '.'])"` as a fallback).

Expected: **zero hits** in source code and tests. If any hit remains, STOP and update that caller to use `make_buffer(..., critic_obs_dim=...)` before deleting.

- [ ] **Step 2: Delete the `make_buffer_with_critic` method**

Remove the `make_buffer_with_critic` method from `jax_rl/training/obs_pipeline.py` entirely (was lines 150-186 before Task 3; may have shifted slightly). Keep `make_buffer` (the unified version from Task 3).

- [ ] **Step 3: Run ObsPipeline tests + the 4 algo tests to confirm nothing broke**

Run: `uv run python -m pytest tests/test_obs_pipeline.py tests/test_offpolicy_algos.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/training/obs_pipeline.py
git commit -m "refactor: delete ObsPipeline.make_buffer_with_critic

All callers now use the unified make_buffer(..., critic_obs_dim=...)
added in the earlier non-breaking step. Completes review item #5 from
2026-04-10-offpolicy-training-scripts-review.md."
```

---

## Task 10: Update `AGENT_HANDOFF.md` (review item #9)

**Files:**
- Modify: `.context/AGENT_HANDOFF.md`

- [ ] **Step 1: Find stale references to `train_offpolicy.py`**

Run Grep on the pattern `train_offpolicy\.py` scoped to `.context/AGENT_HANDOFF.md`. Each hit is a candidate for update — use judgment:
- References in Parts 4 "Key entry points" / 6 "Commands": almost certainly stale (these docs should show the per-algo scripts users should actually run).
- References noting `train_offpolicy.py` exists as a legacy unified dispatcher: can stay if accurate (the script still exists; its deletion is a separate follow-up, review item #2).

Also Grep for `train_sac\.py|train_td3\.py|train_fast_sac\.py|train_fast_td3\.py` to verify the per-algo scripts ARE listed somewhere in the handoff. If any are missing from the "Key entry points" table in Part 4, add them.

- [ ] **Step 2: Update entry points and commands**

In Part 4 "Key entry points" — ensure `train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py`, `train_flashsac.py` are all listed. Add a sentence below the table:

> The 4 non-FlashSAC off-policy scripts delegate to `jax_rl/training/offpolicy_loop.py::run_offpolicy_loop` for the shared training loop and only contain algo-specific optimizer/explore decisions (~60 lines each). FlashSAC stays standalone (it has algo-specific BatchNorm state, Zeta noise, and adaptive reward scaling that don't fit the shared shape).

In Part 6 "Commands" — replace any `train_offpolicy.py --algo <x>` example with `train_<x>.py`. For example:
```bash
# Old (stale):
uv run python train_offpolicy.py --algo fast_sac --env Go2WarpJoystickFlat ...
# New:
uv run python train_fast_sac.py --env Go2WarpJoystickFlat ...
```

- [ ] **Step 3: Commit**

```bash
git add .context/AGENT_HANDOFF.md
git commit -m "docs: update AGENT_HANDOFF for thinned off-policy scripts + run_offpolicy_loop"
```

---

## Task 11: Run full test suite + verify line count reduction

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run python -m pytest tests/ -v --ignore=tests/test_offpolicy_loop.py -x`
Expected: all PASS. (Skip the slow loop test unless GPU is free.)

If anything fails that isn't pre-existing, diagnose before continuing. Common culprits:
- A test calling `make_buffer_with_critic` (removed in Task 3)
- A test calling `make_envs` unpacking that changed (shouldn't happen — `make_envs` unchanged)

- [ ] **Step 2: Run the slow loop test (if GPU available)**

Check GPU again. If free:
Run: `uv run python -m pytest tests/test_offpolicy_loop.py -v -m slow`
Expected: PASS in ~1-2 min.

If GPU busy, skip and add to Tomorrow TODO.

- [ ] **Step 3: Verify git status is clean**

Run: `git status`
Expected: working tree clean (all task commits landed).

- [ ] **Step 4: Verify line count reduction — total file size**

Run: `wc -l train_sac.py train_td3.py train_fast_sac.py train_fast_td3.py`
Expected: each file now ~100-130 lines total (down from ~300).

- [ ] **Step 5: Verify `train()` body is ≤ 60 non-blank lines**

For each of the 4 thinned scripts, open the file and count the non-blank, non-comment lines between `def train(...)` and the CLI block. Expected: ≤ 60 per script. If any script exceeds 60, investigate whether there's boilerplate that could have been pushed into the helper. (Spec success criterion #1.)

Quick check: open each file and eyeball. The thinned scripts should have: make_env_bundle → optimizer → algo construction → explore closure → run_offpolicy_loop call. That's it.

---

## Tomorrow morning TODO (smoke tests — GPU currently busy)

Check GPU first: `nvidia-smi`. When it's free, run these in order:

### Smoke test 1: SAC on CheetahRun (quick, ~5 min)
```bash
uv run python train_sac.py --env CheetahRun --num-envs 128 --total-timesteps 500000
```
**Expected:** finishes without crash, final eval reward > 200 (CheetahRun is easy for SAC).

### Smoke test 2: TD3 on CheetahRun (quick, ~5 min)
```bash
uv run python train_td3.py --env CheetahRun --num-envs 128 --total-timesteps 500000
```
**Expected:** finishes without crash, final eval reward > 200.

### Smoke test 3: FastSAC on CheetahRun (~10 min at 1024 envs)
```bash
uv run python train_fast_sac.py --env CheetahRun --num-envs 1024 --total-timesteps 2000000
```
**Expected:** finishes without crash, eval curve trending up.

### Smoke test 4: FastTD3 on CheetahRun (~10 min at 1024 envs)
```bash
uv run python train_fast_td3.py --env CheetahRun --num-envs 1024 --total-timesteps 2000000
```
**Expected:** finishes without crash, eval curve trending up.

### Smoke test 5: FastSAC on Go2Warp (asymmetric critic, DR off, ~15 min)
```bash
uv run python train_fast_sac.py --env Go2WarpJoystickFlat --num-envs 1024 --total-timesteps 3000000
```
**Expected:** finishes without crash, dict-obs detection print appears, asymmetric critic_obs_dim > obs_dim in the log, eval curve trending up. This one exercises the `has_privileged=True` code path.

### Slow unit test (if skipped in Task 4 / Task 11)
```bash
uv run python -m pytest tests/test_offpolicy_loop.py -v -m slow
uv run python -m pytest tests/test_env_bundle.py -v -m slow  # Go2Warp EnvBundle test
```

### After all smoke tests pass
- If everything is green, this refactor is ready for review + merge.
- Consider opening a PR: `gh pr create --title "refactor: extract run_offpolicy_loop helper" --body "..."`
- Remove the worktree when merged: `git worktree remove ../jax-learning-offpolicy-loop`

### Review items NOT covered by this plan (for future work)
From `.superpowers/specs/2026-04-10-offpolicy-training-scripts-review.md`:
- **#2** delete old `train_offpolicy.py` (after confirming nothing depends on it)
- **#4** move FlashSAC to ObsPipeline + `apply_cli_overrides`
- **#7** fix FlashSAC's `algo._default_actor_bs` mutation
- **#8** fix FlashSAC unsafe truncation access
- Algo-file clean-code nits: `sac.py:98-103`, `td3.py:168-170`, `flash_sac.py:492`
