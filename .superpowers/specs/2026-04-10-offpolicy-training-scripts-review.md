# Off-Policy Training Scripts — Code Review

**Date:** 2026-04-10
**Reviewer:** code-reviewer subagent (strict senior-SWE perspective)
**Scope:** `train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py`, `train_flashsac.py` + `jax_rl/training/` helpers + `jax_rl/algos/{sac,td3,fast_sac,fast_td3,flash_sac}.py`
**Reference (not reviewed):** `train_offpolicy.py` (older unified dispatcher, refactored out)
**Goal:** Brutally honest feedback on clarity/readability/duplication. North star = CLEAN CODE, not over-abstraction.

---

## Executive Summary

The 5 per-algo training scripts share **85-90% identical code**. The helper layer (`jax_rl/training/`) is well-factored — `ObsPipeline`, `TrainContext`, `eval_runner`, `cli_utils` all pull their weight. The algo files are clean and consistent. **The biggest problem:** the decomposition from `train_offpolicy.py` into 4 separate scripts (SAC, TD3, FastSAC, FastTD3) created massive duplication without gaining meaningful per-algo clarity — the "algo-specific" code in each is only 5-15 lines out of ~300. FlashSAC is the one script that genuinely earns its standalone existence. **Second biggest problem:** `train_offpolicy.py` still exists as a working script, doesn't use `ObsPipeline` or `apply_cli_overrides`, and is referenced in `AGENT_HANDOFF.md` as the primary entry point — so there are **6 scripts** that can train the same 4 algorithms.

---

## Is the 5-script decomposition the right call?

**Premature for 4 of the 5. Correct for FlashSAC.**

The differences between `train_sac.py`, `train_td3.py`, `train_fast_sac.py`, and `train_fast_td3.py` reduce to: (1) which algo class to import, (2) which preset to call, (3) the `explore()` closure (SAC=pass-through, TD3=noise injection), (4) optimizer construction, and (5) `log_extra_fields`/`log_extra_keys`. That's 10-20 lines of variation in 250+ lines of identical boilerplate. The old `train_offpolicy.py` handled these via `if family == "sac"` branches and `_make_algo()`.

FlashSAC is different — Zeta noise state management, adaptive reward scaling outside JIT, custom BatchNorm coordination. It earns its own script.

**Verdict:** Re-unify train_sac/td3/fast_sac/fast_td3 into a single modernized `train_offpolicy.py` (using ObsPipeline + apply_cli_overrides). Keep `train_flashsac.py` standalone. Delete the old `train_offpolicy.py`.

---

## Duplication map

| Logic block | Files (lines approx) | Differences | Should be |
|---|---|---|---|
| Env setup + dict obs detection | sac:44-58, td3:44-58, fast_sac:44-58, fast_td3:44-58, flashsac:64-77 (14 lines x5) | FlashSAC prints shorter messages | Unified |
| Print banner | sac:62-71, td3:62-71, fast_sac:62-71, fast_td3:62-71 (10 lines x4) | Only `algo_name` string differs | Unified |
| W&B init block | All 5 scripts (~13 lines x5) | Identical | Unified |
| ObsPipeline + buffer setup | 4 non-FlashSAC scripts (~10 lines x4) | Identical | Unified |
| Resume block | 4 scripts (~5 lines x4) | Identical | Unified |
| Tracker + ctx + ckpt_mgr | 4 scripts (~9 lines x4) | Identical | Unified |
| Training loop (obs/action/step/buffer/grad/log) | 4 scripts (~67 lines x4) | Identical | Unified |
| Eval + checkpoint block | 4 scripts (~22 lines x4) | Identical (q_fn lambda is copy-pasted) | Unified |
| CLI argparse | 4 scripts (~45-52 lines x4) | ~30 args identical, ~3-5 algo-specific | Unified with branches |

**Total duplicated:** ~200 of 250 lines per `train()` function (80%+), ~35 of 45 CLI args (78%).

---

## Per-file findings

### train_sac.py
- **Line 60:** `total_env_steps = cfg.total_timesteps` — zero-value alias. Same in all 5 scripts.
- **Lines 124-126:** `explore()` closure is a pass-through to `algo.select_action`. Exists only for interface parity with TD3's noise injection. Fine if unified, pointless if standalone.
- **Lines 228, 233-235:** `_ts = training_state` closure workaround — cryptic name, no comment explaining the Python loop-variable capture issue. Same in all 5 scripts.

### train_td3.py
- **Lines 93-96:** Inconsistent optimizer style — `optimizer` named variable for actor, inline `optax.adam(cfg.lr)` for critic.
- **Lines 119-134:** The one genuinely algo-specific block (noise injection). Uses `getattr` defensively on TD3Config — suggests config design issue if fields might be missing.
- **Line 130:** `log_extra_fields = []`, `log_extra_keys = []` — empty lists just to satisfy the logging API. Confirms this is boilerplate.

### train_fast_sac.py
- **Lines 94-98:** LR schedule (warmup, cosine, adamw) — **identical copy-paste** from `train_fast_td3.py:93-97`. 5 lines duplicated between the two "Fast" scripts and also in `train_offpolicy.py`.
- Everything else is identical to `train_sac.py` except import and preset.

### train_fast_td3.py
- **Lines 93-98:** Same LR schedule as fast_sac (see above).
- Everything else is identical to `train_td3.py` except import, preset, and LR schedule.

### train_flashsac.py
- **Lines 198-202:** `_get_obs()` / `_get_critic_obs()` replicate what `ObsPipeline` does. FlashSAC doesn't use ObsPipeline at all — divergence from the other 4 scripts.
- **Line 207:** `dummy_norm_state = make_identity_norm_state(obs_dim)` — forced by `load_checkpoint`'s signature. Smell in the checkpoint API, not FlashSAC's fault.
- **Lines 360, 378:** `algo._default_actor_bs = training_state.actor_batch_stats` — **mutating algo internals from the training script.** Most fragile coupling in the codebase. Violates "algo is a black box."
- **Line 263:** `env_state.info["truncation"]` — direct dict access, no `.get()` with default. Other 4 scripts use safe `.get()`. Will crash on envs without truncation key.
- **Lines 442-460:** Manual CLI overrides instead of `apply_cli_overrides`. Two override systems in one codebase.

---

## Helpers audit

### jax_rl/training/obs_pipeline.py
- **Lines 111-186:** `make_buffer()` and `make_buffer_with_critic()` share ~80% of code. The `make_buffer` method even has a dead code block acknowledging it can't handle privileged obs. Should be a single method with `critic_obs_dim=None` default.
- **Line 235:** Lazy import of `make_identity_norm_state` — presumably for circular import avoidance, but no comment saying so.

### jax_rl/training/cli_utils.py
- Clean, declarative, does one thing well. **Problem:** FlashSAC doesn't use it. Two override systems coexist.

### jax_rl/training/eval_runner.py
- **Line 37:** `total_steps = metrics_log[-1]["total_steps"] if metrics_log else 0` — reaches into `metrics_log` (list of dicts) to get step count. Should be on `TrainContext` or passed explicitly.
- **Lines 53-63 vs 125-137:** Eval print formatting duplicated between `maybe_eval_and_checkpoint` and `final_eval_and_checkpoint` (~30% shared code).

### jax_rl/training/metrics_logger.py
- **Line 99: BUG** — `parts.append(f"{label} {val:.3e}")` **ignores the `fmt` parameter.** Training scripts pass format strings like `".3f"`, `".4f"` that are silently dropped. Should be `f"{label} {val:{fmt}}"`.

### jax_rl/training/env_setup.py
- **Line 107:** `getattr(cfg, 'reset_mode', 'legacy')` — dead defensive code. `reset_mode` already has a default on `TrainConfig`.
- **Line 130:** Dict obs detection logic duplicated — `make_envs` already computes `obs_dim` correctly, but every training script re-does the dict obs detection at lines 48-58.

### jax_rl/training/checkpointing.py
- **Lines 89-115:** `save_checkpoint` does heavy side-effect work (subprocess for git hash, reloads env for DR specs via `pg_registry.load`). Expensive, non-obvious, undocumented in docstring.

---

## Algo files — clean code notes

### jax_rl/algos/sac.py
- **Lines 98-103:** `_q_values` and `_target_q_values` have identical bodies. The alias adds nothing since callers already name their variables `target_q1_params` etc.

### jax_rl/algos/td3.py
- **Lines 168-170:** `_do_actor_update` unpacks a 9-element tuple positionally. Getting unwieldy — easy to swap two same-typed values.

### jax_rl/algos/fast_sac.py + fast_td3.py
- C51-specific code (projection, cross-entropy) correctly factored into `jax_rl/utils/distributional.py`. No further unification needed.

### jax_rl/algos/flash_sac.py
- **Line 492:** `self._default_actor_bs` — mutable state on the algo object, set by `init()` and mutated by the training script. Only algo with mutable state outside `TrainingState`. Wart forced by eval_runner not passing batch_stats.

---

## Ranked refactor action list

**#1 — Re-unify the 4 non-FlashSAC scripts into one `train_offpolicy.py`**
- **What:** Merge train_sac/td3/fast_sac/fast_td3 back into a single script using modernized helpers (ObsPipeline, TrainContext, apply_cli_overrides).
- **Why:** Eliminates ~800 lines of duplication. Any training loop fix currently requires 4 identical edits.
- **Files:** `train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py`, `train_offpolicy.py`
- **Risk:** Low — the old train_offpolicy.py already proved this works
- **Effort:** Medium
- **Depends on:** None

**#2 — Delete old train_offpolicy.py**
- **What:** The old script is stale (no ObsPipeline, no apply_cli_overrides) and creates confusion (6 scripts for 4 algos).
- **Why:** Maintenance nightmare. AGENT_HANDOFF.md still points to it.
- **Files:** `train_offpolicy.py`, `.context/AGENT_HANDOFF.md`
- **Risk:** Low
- **Effort:** Small
- **Depends on:** #1 (old script becomes the unification target or gets deleted)

**#3 — Fix ignored format string bug in metrics_logger.py**
- **What:** Line 99 uses `.3e` instead of the caller's `fmt` parameter.
- **Why:** This is a bug — format strings passed by training scripts are silently ignored.
- **Files:** `jax_rl/training/metrics_logger.py:99`
- **Risk:** Low
- **Effort:** Small (one-line fix)
- **Depends on:** None

**#4 — Move FlashSAC to ObsPipeline + apply_cli_overrides**
- **What:** Replace `_get_obs()` / `_get_critic_obs()` with `ObsPipeline(use_obs_norm=False)`, and use `apply_cli_overrides` (extending it for FlashSAC-specific fields).
- **Why:** Two obs-extraction systems and two CLI-override systems in one codebase.
- **Files:** `train_flashsac.py:198-202, 442-460`, `jax_rl/training/cli_utils.py`
- **Risk:** Low-Med (needs testing with FlashSAC's no-norm path)
- **Effort:** Medium
- **Depends on:** None

**#5 — Merge ObsPipeline.make_buffer / make_buffer_with_critic**
- **What:** Single `make_buffer` method with `critic_obs_dim=None` default.
- **Why:** 80% code duplication + dead code block acknowledging the split is awkward.
- **Files:** `jax_rl/training/obs_pipeline.py:111-186`
- **Risk:** Low
- **Effort:** Small
- **Depends on:** None

**#6 — Eliminate redundant dict obs detection in training scripts**
- **What:** Move `critic_obs_dim` extraction into `make_envs` or a small `detect_obs_structure()` helper.
- **Why:** 14 identical lines x 5 scripts = 70 lines of boilerplate.
- **Files:** `jax_rl/training/env_setup.py`, all 5 training scripts
- **Risk:** Low
- **Effort:** Small
- **Depends on:** #1 (collapses naturally if re-unified)

**#7 — Fix FlashSAC's `algo._default_actor_bs` mutation**
- **What:** Pass `actor_batch_stats` through the eval path explicitly instead of mutating algo internals.
- **Why:** Violates "algo is a black box." Forgetting this line = silent stale BN stats in eval.
- **Files:** `train_flashsac.py:360,378`, `jax_rl/algos/flash_sac.py:492`, `jax_rl/training/eval_runner.py`
- **Risk:** Medium (changes eval_runner API)
- **Effort:** Medium
- **Depends on:** None

**#8 — Fix FlashSAC unsafe truncation access**
- **What:** `env_state.info["truncation"]` → `.get("truncation", jnp.zeros_like(...))`.
- **Why:** Will crash on envs without truncation key.
- **Files:** `train_flashsac.py:263`
- **Risk:** Low
- **Effort:** Small (one-line fix)
- **Depends on:** None

**#9 — Update AGENT_HANDOFF.md to reflect current entry points**
- **What:** Remove stale `train_offpolicy.py --algo sac` references, document actual script structure.
- **Files:** `.context/AGENT_HANDOFF.md` (Parts 4 and 6)
- **Risk:** Low
- **Effort:** Small
- **Depends on:** #1 or #2 (update after deciding final structure)

---

## Things NOT recommended (and why)

1. **Trainer base class / AbstractTrainingLoop** — Project philosophy explicitly forbids it. The duplication is better solved by re-unifying into one script with `if family == "sac"` branches. The variation points are small enough that branching is clearer than inheritance.

2. **Algo registry / plugin system** — The old `train_offpolicy.py` has `_make_algo()` which is fine. Wouldn't expand it into a plugin system. 4 algos are known at write time; a registry adds indirection without real extensibility.

3. **Shared TrainingState base across algo files** — `@flax.struct.dataclass` doesn't support inheritance, and fields genuinely differ (TD3 has `target_actor_params`; FlashSAC has `*_batch_stats` + `noise_state`). Forcing a shared structure would be premature generalization.

4. **Rich env_context return struct from `make_envs`** — Would move env-awareness into the helper layer, violating "env is a black box, training script mediates." Compromise: a small `detect_obs_structure(env_state)` function (item #6) that doesn't live inside `make_envs`.
