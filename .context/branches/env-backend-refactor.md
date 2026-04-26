# Branch: `env-backend-refactor`

> Heads-up doc for any agent working on `new_slate_linen`. This branch
> is parked, waiting to merge. Read here before you touch any of the
> files listed in §3 — your changes there will conflict.
>
> **Worktree:** `../jax-learning-envrefactor/` (sibling dir to `jax-learning/`)
> **Branched from:** `7a9daf6` (linen head as of 2026-04-25 evening)
> **Status (2026-04-26):** 8 commits, end-to-end validated, docs swept, **NOT merged yet** — waiting for in-flight TDMPC2 / FastSAC linen work to wrap before resolving conflicts.
>
> To peek without checking out: `git show env-backend-refactor:<path>` or `git diff new_slate_linen..env-backend-refactor -- <path>`.

---

## 1. Why the branch exists

The training pipeline was MJX-only. `make_envs(cfg, seed)` hardcoded `pg_registry.load()`. Adding a non-MJX env (PushT, gymnasium, eventually IsaacLab) meant forking the SAC stack — `scripts/train_pusht.py` was a 476-line copy of the off-policy stack with its own buffer, wrappers, eval, and loop.

Goal: make env construction backend-agnostic so SAC/PPO/record_video work across MJX, gym, and IsaacLab through one entrypoint and one set of harnesses.

---

## 2. What the branch does

Five-phase refactor (Phase 3 IsaacLab deferred — no install/test target):

| Phase | Commit | What |
|---|---|---|
| 0 | `9273c2b` | Relocate `EnvBundle` → `jax_rl/training/env_bundle.py`. Add `backend_kind`, `num_envs`, `render_fn` fields. |
| 1 | `2c3f68a` | New `jax_rl/training/env_backends/` registry + `mjx_backend.py` (MJX code relocated from `env_setup.py`). `env_setup.py` becomes a thin shim. |
| 2 | `f4ac6ad` | New `gym_backend.py` with `GymState` + PushT factory + `evaluate_gym()`. `train_sac --env PushT` works end-to-end. |
| 3 | — | IsaacLab — DEFERRED (plan only). |
| 4 | `8d23ca3` | `train_ppo.py` + `train_ppo_fast.py` route through bundle dispatch; fast path raises if `backend_kind != "mjx"`. |
| 5 | `ccc851f` | `record_video.py` early-dispatches by backend; `_record_gym()` for gym envs. |
| Bonus | `07e7707` | 7 gymnasium[mujoco] benchmark factories (HalfCheetah/Hopper/Walker2d/Humanoid/Ant/Pendulum/LunarLanderContinuous). |
| Docs | `aca64a6` | Journal, lessons, TODO, AGENT_HANDOFF, NEW_AGENT_PROMPT, README + 3 lessons-doc fixes. |

Validated end-to-end: SAC HalfCheetah 200k @ num_envs=8 → eval **5697 ± 43** (above published SAC baselines for that step count). Zero code outside `gym_backend.py:_make_gymnasium_mujoco_factory` was needed.

---

## 3. Files touched (read this before editing on linen)

**Heavy edits (the files you'll most likely conflict on):**

| File | Refactor change | Risk for linen edits |
|---|---|---|
| `jax_rl/training/env_setup.py` | Reduced from 253 → 35 lines. Now a thin shim — `make_env_bundle()` dispatches to registry; `make_envs` and `_make_nan_safe_step` re-exported for legacy callers. | **HIGH.** If you edit `make_envs` body on linen, port the change into `jax_rl/training/env_backends/mjx_backend.py:make_envs` instead — that's where the real code lives now. |
| `jax_rl/training/__init__.py` | Adds `EnvBundle, BackendKind` to public API. | Low. |
| `jax_rl/training/env_bundle.py` | NEW. EnvBundle dataclass + Protocol. | None on linen. |
| `jax_rl/training/env_backends/` | NEW DIR. `__init__.py`, `mjx_backend.py`, `gym_backend.py`. | None on linen. |
| `jax_rl/training/eval_runner.py` | Dispatches `evaluate` vs `evaluate_gym` via `TrainContext.backend_kind`. | Low — only the dispatch site at the top changed. |
| `jax_rl/training/train_context.py` | Added `backend_kind: str = "mjx"` field. | Low — additive. |
| `jax_rl/training/offpolicy_loop.py` | Threads `bundle.backend_kind` into ctx. One line added. | Low. |
| `jax_rl/utils/eval.py` | Added `evaluate_gym()` (Python-loop eval for gym backend). | Low — appended only. |
| `jax_rl/configs/train_config.py` | Added `env_kwargs: dict = field(default_factory=dict)` field. | **MEDIUM.** Linen also added a field here (`action_repeat`); both should coexist. |
| `scripts/train_ppo.py` | Switched `make_envs` → `make_env_bundle` (one block). | Medium. |
| `scripts/train_ppo_fast.py` | Same + `backend_kind != "mjx"` guard with clear error. | Medium. |
| `scripts/record_video.py` | Added backend dispatch + `_record_gym()` helper. | Low. |
| `train_pusht.py` | **Marked DEPRECATED** in tree comments. Code unchanged this branch — Phase 6 deletes it after a 2M reproduction validates parity. | Low. |

**Doc updates:** `.context/AGENT_HANDOFF.md`, `.context/LESSONS.md`, `.context/TODO.md`, `.context/NEW_AGENT_PROMPT.md`, `.context/journals/2026-04-26.md`, `.context/lessons/{frame_stack,infrastructure,mjx}.md`, `.context/plans/skill-discovery-plan.md`, `README.md`. **Linen also wrote `.context/journals/2026-04-26.md` independently** — see §4.

---

## 4. Known conflicts to anticipate when merging

Linen has 5 commits since the fork point (`7a9daf6`):

```
b407edf docs(lessons): tdmpc2 — 5 port bugs + reusable patterns
b6c89a0 docs(tdmpc2): J4 HumanoidRun paper-band match journal + handoff/TODO
fd3e814 feat(tdmpc2): record_video_tdmpc2.py — MPPI + prior mode rollouts
43b71ad fix(tdmpc2): action_repeat=2 + episode_length=500 (source DMC parity)
c080d32 fix(tdmpc2): 2 more correctness bugs from J4 HumanoidRun debug
```

**Hard conflicts:**

1. **`.context/journals/2026-04-26.md`** — both branches CREATED this file with totally different content (linen: 164 lines TDMPC2 J4 debug; this branch: ~150 lines env-backend refactor). Resolution: combine into one file with two H2 sections under a single H1 date header. Linen's TDMPC2 first (chronologically earlier), env-backend second.

2. **`jax_rl/training/env_setup.py` + `jax_rl/training/env_backends/mjx_backend.py`** — linen added `action_repeat=cfg.action_repeat` to two `wrap_for_training()` calls in `make_envs`. This branch moved `make_envs` to `mjx_backend.py`. Resolution: port linen's two-arg change into `mjx_backend.make_envs`. **Mechanical, no design decision.**

3. **`jax_rl/configs/train_config.py`** — both branches added a field. Linen: `action_repeat: int = 1`. This branch: `env_kwargs: dict = field(default_factory=dict)`. Resolution: keep both, place adjacent in the env section.

**Doc conflicts (minor, mostly auto-merge):**

4. **`.context/AGENT_HANDOFF.md`** — linen edits one line (current-best summary, line 234, adds J4 HumanoidRun benchmark). This branch edits different sections (env_setup paragraph rewrite, gym envs table, codebase tree). Different lines → auto-merge clean.

5. **`.context/LESSONS.md`** — linen inserts a `## [TD-MPC2]` section (~10 lines). This branch adds entries in Infrastructure section + 2 entries at the end. Adjacent but non-overlapping.

6. **`.context/TODO.md`** — both edited the top. Linen rewrote "🔥 TD-MPC2 end-of-run collapse" → "Medium priority — TD-MPC2 J3 Cheetah re-run". This branch prepended "Completed (2026-04-26) — Env-backend refactor". Resolution: this branch's "Completed" section stays at the top (matches existing convention); linen's revised "Medium priority" follows.

---

## 5. Open design question — `action_repeat` rationalization (deferred until merge)

Linen's `cfg.action_repeat` (universal field, default 1) and this branch's PushT factory `cfg.env_kwargs["action_repeat"]` (default 2) become **two sources of truth** post-merge. A user setting `cfg.action_repeat=3` and `--env PushT` would silently get repeat=2.

**Recommended resolution at merge time (option C):**
- `cfg.action_repeat` is the canonical field (linen's design wins).
- `mjx_backend.make_envs` honors it via `wrap_for_training` (port linen's change).
- gymnasium[mujoco] factories in `gym_backend.py` apply `_ActionRepeatWrapper` IFF `cfg.action_repeat > 1`.
- PushT factory: `k = cfg.action_repeat if cfg.action_repeat > 1 else 2` — preserves PushT's historic default of 2 when caller hasn't explicitly overridden.
- Drop `kwargs.pop("action_repeat", ...)` from `_make_pusht_factory` (no longer ambiguous).

Document in `AGENT_HANDOFF` gym envs table: "PushT defaults action_repeat=2; others default to 1".

This isn't urgent and isn't part of the merge mechanics — it's a clean-up after the conflicts resolve.

---

## 6. Coordination

**If you (other agent on linen) want to:**

- **Edit `jax_rl/training/env_setup.py`** → don't. The real code lives in `env_backends/mjx_backend.py` post-merge. Make your change there directly on linen (it'll merge cleanly as a re-relocation), or coordinate with whoever merges this branch.
- **Add a field to `TrainConfig`** → fine. `train_config.py` is a 3-way merge; both branches added different fields without overlap.
- **Edit `make_envs`** → see above; route to `mjx_backend.make_envs` once merged.
- **Touch any other file in §3** → check the diff first. Most are low-risk.
- **Write a journal entry for 2026-04-26** → don't; this branch already has one. Add to it (combined post-merge) instead of creating a parallel file.

**Whoever merges this branch:**

1. Wait until in-flight TDMPC2 / FastSAC work on linen settles (check `git log new_slate_linen ^7a9daf6 --oneline` for any new commits).
2. From the worktree, `git merge new_slate_linen` to bring linen's tip into env-backend-refactor.
3. Resolve conflicts per §4 + §5 (option C for action_repeat).
4. Smoke-test: `train_sac --env Go2WarpJoystickFlat --total-timesteps 5000` (MJX path) and `train_sac --env HalfCheetah --total-timesteps 5000 --num-envs 8` (gym path).
5. Run the full test suite (expect `test_generated_scene_file_written` to fail — pre-existing on linen).
6. Push the resolved branch; merge into linen via PR or fast-forward.

---

## 7. Pointers

- Plan: `.superpowers/plans/2026-04-25-env-backend-refactor.md`
- Today's journal (this branch): `.context/journals/2026-04-26.md`
- Today's journal (linen, conflicts with above): `git show new_slate_linen:.context/journals/2026-04-26.md`
- HalfCheetah validation ckpt: `checkpoints/20260426_095946_sac_halfcheetah_seed0/`
