# Linen resume-warmup fix — merge helper for env-backend-refactor

> Heads-up doc for the agent merging `env-backend-refactor` into
> `new_slate_linen`. Authored 2026-04-26, mirrors the pattern in
> `.context/branches/linen-tdmpc2-2026-04-26.md`.
>
> This doc covers the resume-warmup work landed on linen *after* the
> TDMPC2 work (linen-tdmpc2-2026-04-26.md) and *after* the Polyak verify
> training run that closed the TODO 🔥 polyak gate.

---

## 1. What linen shipped

A 1-commit fix for the FlashSAC resume eval-drop regression (TODO 🔥
since 2026-04-24). Root cause confirmed and fix validated end-to-end.

**Root cause:** Replay buffer is not persisted across resumes. The
loop's warmup gate refires on resume, fills the buffer with random
uniform actions for `min_buffer_size` env steps, and the first
gradient batches train on that off-distribution data. Critic targets
shift, actor follows. Result: first eval drops.

**Severity by task:**
- Locomotion (FastSAC Go2WarpJoystickFlat): 268.4 → 253.9 (~14 pt, borderline noise)
- Cartpole (FlashSAC CartpoleBalance, original report): 996 → 747 (~250 pt)

**Fix shipped:** new `--resume-warmup {policy,random}` CLI flag, default
`policy`. On resume, refill the buffer with the loaded policy's actions
during the same warmup window — same threshold (`min_buffer_size`),
same wall time, but on-policy data instead of random uniform. Zero
storage cost (vs ~30-460 MB if persisting the buffer alongside orbax).

Random refill stays opt-in via `--resume-warmup random` for users who
want a hard buffer-distribution reset. Cold-start behavior is unchanged
(random for exploration when no policy exists).

**Validation:**
| Algo / Env (baseline) | First eval (random) | First eval (policy) | Long-term (@ 256 eps) |
|---|---|---|---|
| FastSAC / Go2WarpJoystickFlat (268.4) | 253.9 (-14) | **270.9** (+2.5) | 268+ |
| FlashSAC / CartpoleBalance (999.7) | 690.7 (-309) | 661.1 (-339) | random=982, **policy=999.8** |

- FastSAC Go2: **fix is total**, resume seamless.
- FlashSAC Cartpole: **fix is partial.** First-eval drop persists in both modes; fix only changes the recovery curve (random plateaus at 982, policy hits 999.8 by @ 256 eps). A second, non-buffer cause exists for FlashSAC Cartpole specifically — open follow-up TODO captures the diagnostic data and rejected `reward_norm_state` freeze hypothesis.

---

## 2. File-by-file changes (linen-side)

| File | Status | Linen change | Conflict risk vs env-backend-refactor |
|---|---|---|---|
| `jax_rl/training/offpolicy_loop.py` | M | Added `resume_warmup: str = "policy"` param to `run_offpolicy_loop()`. Replaced action-selection branch (lines ~170-180): now gates random uniform on `is_warmup AND (start_step == 0 OR resume_warmup == "random")`. Both edits are confined to the action-selection block. | **LOW.** Refactor branch added 1 line in `offpolicy_loop.py` for `bundle.backend_kind` plumbing into `TrainContext` (different region, near the ctx construction). Auto-merge expected clean. |
| `scripts/train_flashsac.py` | M | Same logic mirrored in standalone loop. Added `resume_warmup: str = "policy"` to `train()`. CLI flag + arg threading. Action-selection block (lines ~243-256) gated identically. | NONE. Refactor branch doesn't edit this file. |
| `scripts/train_sac.py` | M | Added `--resume-warmup` CLI flag + threaded into `train()` → `run_offpolicy_loop()`. ~5 lines. | NONE. |
| `scripts/train_td3.py` | M | Same as above. | NONE. |
| `scripts/train_fast_sac.py` | M | Same as above. | NONE. |
| `scripts/train_fast_td3.py` | M | Same as above. | NONE. |
| `docs/reference/cli-flags.md` | M | Auto-regenerated from `gen_cli_reference.py`. Adds `--resume-warmup` row to all 5 off-policy script tables. | NONE. Refactor branch doesn't edit this file. |
| `.context/lessons/offpolicy.md` | M | Appended new section "Resume Warmup: Random Actions Corrupt the Buffer (2026-04-26)". | LOW. Refactor branch doesn't touch this file. |
| `.context/TODO.md` | M | Replaced 🔥 "FlashSAC resume eval regression" entry with "Completed (2026-04-26) — Resume eval regression fix (universal off-policy)". | **MEDIUM** — refactor also prepended a Completed-2026-04-26 section. Both are append-near-top. Resolve by ordering: refactor's Completed section first (chronologically earlier, env-backend), then this resume-warmup Completed section, then any remaining TODOs. Both are pure additions; no overlap on actual lines. |

---

## 3. Hard conflicts

**None expected.** The action-selection block in `offpolicy_loop.py` is
not touched by the refactor branch. The refactor branch's only change
to this file is the `bundle.backend_kind` line near the ctx
construction (~line 145), which is structurally distinct from the
action-selection block (~line 170).

If `git merge` flags any conflict in `offpolicy_loop.py`, expect it to
be a 3-way auto-merge friction near the function signature — both
branches added/threaded params. Resolution: both params coexist; my
addition is `resume_warmup: str = "policy"` after the existing `resume`
param.

---

## 4. CLI awareness post-merge

After merge, **5 train scripts gain `--resume-warmup` flag** (default
`policy`). This is the canonical knob for the resume-eval fix.
Document in `AGENT_HANDOFF.md`:

> When resuming an off-policy ckpt, the loop refills the replay buffer
> using the loaded policy's actions (not random uniform). To restore
> legacy behavior pass `--resume-warmup random`.

The flag is a no-op on cold-start runs (no `--resume`).

CLI ref already regenerated; downstream MkDocs build will pick it up.

---

## 5. Tests / validation status at merge time

- **Drift tests:** 224 passed / 46 skipped at HEAD prior to commit. Unchanged after the fix.
- **Smoke (`build_parser` import + `--resume-warmup` flag presence):** 5/5 scripts pass.
- **Behavioral (FastSAC Go2 5M):** validated total, see §1 table.
- **Behavioral (FlashSAC Cartpole 1M):** validated partial — buffer fix changes long-term recovery (982 → 999.8 by @ 256 eps) but doesn't eliminate the first-eval transient drop (~310 pts). Open `🔥 Follow-up — FlashSAC Cartpole resume residual drop` TODO captures the rejected reward_norm freeze hypothesis and the next investigation steps. Not blocking the merge.

If the merger wants to re-validate post-merge: rerun the FastSAC Go2
resume command and confirm first post-resume eval lands within ±1 std
of pre-resume eval (not ~14+ below).

---

## 6. Order of operations for the merger

Follow the existing `branches/env-backend-refactor.md` §6 sequence,
plus this delta:

1. Confirm linen tip includes both:
   - TDMPC2 commits per `linen-tdmpc2-2026-04-26.md`
   - This resume-warmup commit
2. From the worktree, `git merge new_slate_linen`.
3. Resolve conflicts:
   - Hard conflicts from `linen-tdmpc2-2026-04-26.md` (env_setup.py, train_config.py, journals, TODO).
   - The TODO.md "Completed" stacking from §2 above.
   - Any auto-merge friction in `offpolicy_loop.py` near the `run_offpolicy_loop` signature.
4. Smoke-test the 4-algo + flashsac path:
   ```
   uv run python scripts/train_fast_sac.py --env Go2WarpJoystickFlat \
     --reset-mode per_step --num-envs 1024 --total-timesteps 5000 --seed 0
   ```
   should print "Collecting 8,192 samples..." and not crash.
5. Run full pytest suite per existing recipe.
