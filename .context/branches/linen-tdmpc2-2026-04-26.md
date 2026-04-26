# Linen TDMPC2 changes — merge helper for env-backend-refactor

> Heads-up doc for the agent merging `env-backend-refactor` (commit `aca64a6`)
> into `new_slate_linen` (current HEAD `e003d32`). Authored 2026-04-26 to
> mirror the pattern in `.context/branches/env-backend-refactor.md`.
>
> Both branches forked from `7a9daf6`. Linen has 8 commits since; refactor
> branch has 8 commits since (per its own doc). Conflicts cataloged below.

---

## 1. What linen shipped (8 commits)

```
e003d32 feat(tdmpc2): HopperHop preset (4-dim, validates JHop benchmark)
c6c4ad7 plan(tdmpc2): refactor implementation plan — 10 tasks
cf1382c spec(tdmpc2): refactor design — split monofile into sub-package
b407edf docs(lessons): tdmpc2 — 5 port bugs + reusable patterns
b6c89a0 docs(tdmpc2): J4 HumanoidRun paper-band match journal + handoff/TODO
fd3e814 feat(tdmpc2): record_video_tdmpc2.py — MPPI + prior mode rollouts
43b71ad fix(tdmpc2): action_repeat=2 + episode_length=500 (source DMC parity)
c080d32 fix(tdmpc2): 2 more correctness bugs from J4 HumanoidRun debug
```

Story arc: J4 HumanoidRun debug → 2 algo bugs (truncated/done conflation,
Q dropout in policy/qscale) → 1 env-parity fix (action_repeat=2) → JHop
benchmark validation → refactor spec + plan written but **not executed**.
The refactor execution is a separate post-merge task.

## 2. File-by-file changes (linen-side)

| File | Status | Linen change | Conflict risk |
|---|---|---|---|
| `jax_rl/algos/tdmpc2.py` | M | Bug A (TD target terminated/truncated semantics) at lines 325-330; Bug B (Q dropout in policy_loss + qscale recompute) at lines 542-567 + 1077-1086 | LOW. Refactor branch doesn't edit this file (verified). Will move into `tdmpc2/{losses,agent}.py` per refactor plan but that's a SEPARATE post-merge task. |
| `jax_rl/algos/tdmpc2_runtime.py` | M | Added 1 line: `action_repeat=tdmpc2_cfg.action_repeat` in `build_train_config_from_tdmpc2` | LOW. Refactor branch doesn't edit this file. |
| `jax_rl/configs/env_presets.py` | M | Changed all 4 DMC TDMPC2 presets episode_length=1000→500. Added HopperHop preset. | LOW. Refactor branch doesn't edit `TDMPC2_PRESETS`. |
| `jax_rl/configs/tdmpc2_config.py` | M | Added `action_repeat: int = 2` field to `TDMPC2Config` | LOW. Refactor branch doesn't edit this file. |
| `jax_rl/configs/train_config.py` | M | Added `action_repeat: int = 1` field to `TrainConfig` | **MEDIUM** — refactor also added a field (`env_kwargs: dict`). See §3. |
| `jax_rl/training/env_setup.py` | M | Added `action_repeat=cfg.action_repeat` arg to TWO `wrap_for_training()` calls (lines ~176 + ~190) | **HIGH** — refactor moved `make_envs` body to `env_backends/mjx_backend.py`. See §3. |
| `scripts/record_video_tdmpc2.py` | A | New file, 191 LOC | NONE. Refactor branch doesn't add this file. |
| `.context/AGENT_HANDOFF.md` | M | One line: benchmark summary adds J4 HumanoidRun + JHop results | LOW. Different lines from refactor's edits (env_setup paragraph, codebase tree). Auto-merge clean. |
| `.context/LESSONS.md` | M | Added 10-line `## [TD-MPC2]` section | LOW. Different position from refactor's adds. Auto-merge or trivial manual. |
| `.context/TODO.md` | M | Restructured: "🔥 high pri TDMPC2 collapse" entry demoted; added "Completed (2026-04-26) TDMPC2 J4" section | **MEDIUM** — refactor also prepended "Completed (2026-04-26) Env-backend refactor" section. See §3. |
| `.context/journals/2026-04-26.md` | A | New file, 164 LOC, all TDMPC2 J4 content | **HARD CONFLICT** — refactor also created this file with completely different content (~150 LOC env-backend). Per existing branches/env-backend-refactor.md §4: combine into one file with two H2 sections under shared H1 date header. |
| `.context/lessons/tdmpc2.md` | A | New file, 210 LOC | NONE. Refactor branch doesn't add this. |
| `.superpowers/specs/2026-04-26-tdmpc2-refactor-design.md` | A | New file, 145 LOC | NONE. |
| `.superpowers/plans/2026-04-26-tdmpc2-refactor.md` | A | New file, 826 LOC | NONE. |

## 3. Hard / medium conflicts (resolution recipes)

### 3.1 `jax_rl/training/env_setup.py` (HIGH risk)

Linen added `action_repeat=cfg.action_repeat` to two `wrap_for_training()`
calls in `make_envs`. Refactor moved `make_envs` body into
`jax_rl/training/env_backends/mjx_backend.py`.

**Resolution**: in the merged tree, `env_setup.py` is a thin shim and the
real `make_envs` lives in `mjx_backend.py`. Port linen's two arg additions
into `mjx_backend.make_envs`:

```python
# in mjx_backend.make_envs, change both wrap_for_training() calls:
env = wrap_for_training(env, episode_length=cfg.episode_length, action_repeat=cfg.action_repeat)
# (and same for eval_env)
```

This is a mechanical port. No design decision.

### 3.2 `jax_rl/configs/train_config.py` (MEDIUM)

Linen added `action_repeat: int = 1`. Refactor added `env_kwargs: dict =
field(default_factory=dict)`. Both are additive; keep both. Place adjacent
in the env section. No semantic conflict.

```python
# Environment
env_name: str = "CartpoleBalance"
episode_length: int = 1000
action_repeat: int = 1                                  # from linen
env_kwargs: dict = field(default_factory=dict)          # from refactor
```

### 3.3 `.context/journals/2026-04-26.md` (HARD)

Both branches CREATED this file. Combine into one file with two H2 sections
under a shared H1 date header. Linen's TDMPC2 J4 content goes first
(chronologically earlier in the day), env-backend refactor content second.
Pattern matches `branches/env-backend-refactor.md` §4 guidance.

```
# 2026-04-26

## TDMPC2 J4 HumanoidRun debug → paper-band match
[linen content here, currently 164 lines]

## Env-backend refactor — Phase 0-5 + bonus
[refactor content here, currently ~150 lines]
```

### 3.4 `.context/TODO.md` (MEDIUM)

Linen and refactor both prepended new sections at the top. Place refactor's
"Completed (2026-04-26) — Env-backend refactor" first (matches the
existing convention of completed sections at top, and the env-backend
refactor doc is the canonical one for that work). Linen's "Completed
(2026-04-26) — TDMPC2 J4" follows immediately after.

Linen also rewrote the "🔥 high pri TDMPC2 end-of-run collapse"
entry to "Medium priority — TDMPC2 J3 Cheetah re-run" — keep linen's
version (the collapse turned out to be eval-key-bleed and is fixed in v3).

## 4. action_repeat rationalization (deferred design call)

Linen's `cfg.action_repeat` (universal field, default 1) and refactor
branch's PushT factory `cfg.env_kwargs["action_repeat"]` (default 2 for
PushT) become two sources of truth post-merge. The refactor branch's
heads-up doc §5 already flagged this and recommended a resolution:

> "Recommended resolution at merge time (option C):" — see env-backend-refactor.md §5 for the full set of options.

**My recommendation post-merge**: option C from that doc, OR migrate PushT's
factory to read `cfg.action_repeat` (single source of truth). Either way,
the merger should write a follow-up TODO to converge the two paths.

## 5. Refactor execution comes AFTER merge

The refactor spec + plan (commits cf1382c, c6c4ad7) describe splitting
`jax_rl/algos/tdmpc2.py` (1138 LOC) into `jax_rl/algos/tdmpc2/` sub-package.
**Do not execute this during merge resolution.** The plan is committed; the
execution is queued for a separate session (see plan execution choice
documented in commit message of c6c4ad7).

When the refactor IS executed post-merge, Tasks 1-6 are pure file-move
operations under deterministic XLA flag, with byte-identical loss values
expected. Won't conflict with anything env-backend touched.

## 6. Validation post-merge

After resolving conflicts, validate the merged tree with:

```bash
# 1. TDMPC2 algo + env_setup integration
PYTHONPATH=$PWD XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/train_tdmpc2.py --env CheetahRun --total-timesteps 3000 \
  --seed 0 --num-envs 8 --eval-every 100000 --ckpt-dir .temp/merge_smoke
# Expected: completes; final EVAL line shows mppi > 50.

# 2. Eval-only path
PYTHONPATH=$PWD XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/eval_tdmpc2.py --env CheetahRun \
  --load-ckpt .temp/merge_smoke --num-evals 1 --seed 100
# Expected: completes; mppi within seed band.

# 3. Record video (uses MPPI plan_fn + env step)
PYTHONPATH=$PWD XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/record_video_tdmpc2.py --env CheetahRun \
  --load-ckpt .temp/merge_smoke --mode mppi --num-steps 100 --seed 100
# Expected: mppi.mp4 saved.

# 4. SAC HalfCheetah (refactor's gym backend smoke from env-backend doc)
PYTHONPATH=$PWD uv run python scripts/train_sac.py --env HalfCheetah-v4 \
  --total-timesteps 50000  # short version
# Expected: runs end-to-end through the new bundle dispatch.

# 5. Existing test suite
uv run pytest tests/ -k "tdmpc or env" -v 2>&1 | tail -10
```

If any validation fails, the merge introduced a regression.

## 7. Verified facts (sanity-check before merging)

- Linen branch HEAD is `e003d32` (run `git log -1 --oneline new_slate_linen`)
- Fork point with refactor branch is `7a9daf6` (run `git merge-base new_slate_linen env-backend-refactor`)
- Linen has 8 commits since fork (run `git log --oneline 7a9daf6..new_slate_linen | wc -l`)
- Linen does NOT touch `train_pusht.py`, `eval_runner.py`, `offpolicy_loop.py`, `train_context.py`, `utils/eval.py`, `train_ppo.py`, `train_ppo_fast.py`, `record_video.py` (the heavy refactor edits) — verified `git diff 7a9daf6..new_slate_linen --stat` lists none of these.

## 8. Known-good benchmark numbers post-linen

For sanity-checking that the merge didn't regress numerically:

| Task | Mppi @ best ckpt | Eval std (40 eps where available) |
|---|---|---|
| CheetahRun 1M (J3 v1, pre Bug A+B + ar=2 fixes) | 837.51 | ± 1.5 |
| HumanoidRun 1M (J4 v3, all fixes) | 556.99 | ± 4.06 |
| HopperHop 500k (JHop, all fixes) | 580.66 | ± 9.07 (only 16 eps) |

J3 was run pre-action_repeat=2 and pre-Bug-A/B. A re-run is queued
(TODO.md medium-pri) but not blocking.
