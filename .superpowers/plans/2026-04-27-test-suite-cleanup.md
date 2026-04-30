# Test-Suite Cleanup — Execution Plan

> Driven by `.context/audits/2026-04-27_test_suite_audit.md` + the
> companion playbook `.context/audits/2026-04-27_test_suite_execution_playbook.md`.
>
> **For the executing agent:** treat each unchecked box below as a
> standalone session. Read the linked playbook §, execute, verify,
> commit, tick the box, append a log entry, stop. Do not bundle
> multiple boxes into one commit unless the box says so explicitly.

**Goal:** harden the pytest surface — partition by markers, prune dead
tests, close coverage gaps, add CI.

**Hard rules** (carried from `.context/lessons/algo_port_protocol.md`):
- `uv run python <cmd>` always
- Don't push to origin without owner approval
- Don't touch `.context/journals/` or `branches/`
- Don't gate on cross-run bit-identity on GPU
- Drift canaries (`tests/test_docs_drift.py` +
  `tests/test_docs_code_blocks.py`) must keep passing — pass count
  cannot drop

---

## Phase ordering

```
Phase 1 (mechanical) → Phase 2 (CI) → Phase 3 (coverage)
   1.1, 1.2, 1.3, 1.4    2.1            3.1, 3.2, 3.3, 3.4
   (parallel-safe)       (after 1.1)    (parallel-safe)
```

Phase 3 boxes can land in any order. They are independent of each
other; only Phase 1.1 (marker taxonomy) is a hard dependency for
verifying them on CI.

---

## Phase 1 — Mechanical cleanup

- [x] **1.1 Marker taxonomy + retarget default lane** (1 hr)
  - Spec: playbook §"Task 1" — Diff 1 (pyproject), Diff 2 (9 files
    table), Diff 3 (deploy_e2e per-test), Diff 4 (drift split).
  - Verify: `pytest --markers` shows 6 marks; default lane ~600
    green in 1-2 min; docs canaries unchanged.
  - Commit: playbook §"Task 1 Commit".
  - **Blocks:** 2.1, and is the verification baseline for 3.x.

- [x] **1.1.5 Stamp deploy marker on deploy-intent tests** (5 min)
  - Follow-on to 1.1, closes audit/playbook drift. Audit §2 said the
    2 deploy_e2e warp tests get `[gpu, warp, go2, deploy]`; playbook
    Diff 3 had only `[gpu, warp, go2]`. `deploy` was a registered
    marker selecting zero tests post-1.1 — dead taxonomy entry.
  - Spec: updated playbook §"Task 1 Diff 3" + companion stamp on
    `deploy/test_policy_runner.py`.
  - Edits: add `@pytest.mark.deploy` to the 2 named tests in
    `tests/test_deploy_e2e.py`; add module-level
    `pytestmark = pytest.mark.deploy` to `deploy/test_policy_runner.py`
    (after `import pytest`).
  - Verify: `pytest -m deploy --collect-only` selects ≥9 tests
    (was 0); default lane count unchanged.

- [x] **1.2 Delete dead TDMPC2 train smoke tests** (5 min)
  - Spec: playbook §"Task 3".
  - `git rm tests/test_train_tdmpc2_h{1,2,3,4,5}.py`
  - Verify: `pytest -m slow --collect-only` collects 10 not 15;
    default lane unchanged.
  - Commit: playbook §"Task 3 Commit".

- [ ] **1.3 Delete or rewrite test_tdmpc2_i3_eval_isolation.py** (5 min delete / 30 min rewrite)
  - Spec: playbook §"Task 4". **Default to Option A (delete).**
  - If owner has not explicitly requested Option B (rewrite to use
    in-process import from `scripts.train_tdmpc2`), do Option A and
    move on.
  - Verify (Option A): `pytest -m slow --collect-only` collects 9
    not 10.
  - Commit: short caveman message — `test: drop dead tdmpc2 eval-isolation subprocess test (broken worktree path)`

- [ ] **1.4 Refresh docs/contributing.md** (15 min)
  - Spec: playbook §"Task 10".
  - Replace stale "~299 tests" claim. Add the 5 named pytest
    invocations from playbook §"Task 10 What to add".
  - Verify: `uv run mkdocs build --strict` if mkdocs installed;
    otherwise visual review.
  - Commit: `docs(contributing): refresh pytest invocations post-marker taxonomy`
  - **Depends on:** 1.1 (so the documented invocations are real).

---

## Phase 2 — CI gate

- [ ] **2.1 Add CPU-only GitHub Actions pytest workflow** (30 min)
  - Spec: playbook §"Task 9".
  - New file: `.github/workflows/tests.yml`.
  - Use the YAML in playbook §"Task 9 Diff" verbatim. Watch the
    `--no-group docs` gotcha.
  - Verify: push to a PR branch, CI runs green in <5 min.
  - **DO NOT push the workflow file to a public branch yourself.**
    Commit locally; owner pushes after review.
  - Commit: playbook §"Task 9 Commit".
  - **Depends on:** 1.1 merged (markers must exist for the addopts
    exclusion to be meaningful).

---

## Phase 3 — Coverage gaps

These four are independent. Each is its own session and own PR. Pick
in priority order or whatever the owner asks for next.

- [ ] **3.1 Synthetic policy_runner test + TDMPC2 rejection** (1 hr) ★ recommended first
  - Spec: playbook §"Task 5".
  - Replaces the artifact-fixture-driven test with a hermetic synthetic
    ckpt; adds the consumer-side rejection path coverage codex flagged.
  - Gotcha: `actor_params` pytree shape is algo-specific. Read
    `deploy/policy_runner.py` carefully before writing the synthetic.
  - Verify: `JAX_PLATFORMS=cpu pytest -q deploy/test_policy_runner.py`
    → 10 passed (was 9; +1 new rejection test).
  - Commit: playbook §"Task 5 Commit".

- [ ] **3.2 Fix test_offpolicy_loop stub-env + parameterize across off-policy algos** (1 hr)
  - Spec: playbook §"Task 6".
  - Bug fix: `EnvBundle(num_envs=NUM_ENVS, ...)` in the existing
    test (currently defaults `num_envs=1` → broadcast error).
  - Add 3 sibling tests for TD3 / FastSAC / FastTD3. **Don't
    parametrize** — TD3/FastTD3 ctors take separate
    `actor_optimizer + critic_optimizer`, not `optimizer + alpha_optimizer`.
  - Extract `_stub_env_bundle` + `_common_cfg` + `_patch_eval` helpers
    (likely to `tests/_loop_helpers.py` so 3.4 can reuse them).
  - Remove this file's `pytestmark = pytest.mark.gpu` after the 4 CPU
    tests pass — they don't need GPU anymore.
  - Verify: `JAX_PLATFORMS=cpu pytest -q tests/test_offpolicy_loop.py`
    → 4 passed, 1 deselected.
  - Commit: `test(offpolicy): fix EnvBundle.num_envs bug; add stub-env loop smoke for sac/td3/fast_sac/fast_td3`

- [ ] **3.3 DomainRand per-episode persistence regression test** (1 hr)
  - Spec: playbook §"Task 7".
  - New file: `tests/test_domain_rand_persistence.py`.
  - Reuses the mock-wrapper pattern from `tests/test_domain_rand_compose.py`
    but walks reset → step → step.
  - **This is the most likely task to overrun.** If past 90 min,
    stop and ask owner.
  - Gotchas in playbook are load-bearing — read them.
  - Verify: `JAX_PLATFORMS=cpu pytest -q tests/test_domain_rand_persistence.py`
    → 2 passed.
  - Commit: `test(domain_rand): add per-episode DR persistence regression`

- [ ] **3.4 Resume warmup behavioral test** (1 hr)
  - Spec: playbook §"Task 8".
  - New file: `tests/test_resume_warmup_behavior.py`.
  - Hardest task in the plan. Use the spy-on-`jax.random.uniform`
    approach from playbook §"Task 8 Approach" — filter by shape
    `(num_envs, action_dim)` to isolate the warmup-action call.
  - Reuses helpers from 3.2.
  - Fallback: positive assertion (count `explore_fn` calls during
    warmup) is easier than negative assertion on `random.uniform`.
  - Verify: `JAX_PLATFORMS=cpu pytest -q tests/test_resume_warmup_behavior.py`
    → 1 passed.
  - Commit: `test(offpolicy): add resume-warmup behavioral guard`

---

## Out-of-scope (do NOT touch in this plan)

- **Task 2** (move 9 module-level `KEY = jax.random.PRNGKey` to fixtures):
  demoted P3. Cosmetic violation. Skip unless a future session hits
  collection-time OOM on a tight GPU.
- **Splitting tests/test_tdmpc2.py** (1334 LOC, 48 tests): file is
  well-organized with section banners; runtime is 41s (mostly JIT, won't
  shrink). Defer indefinitely.
- **Codex P0 #2 (curriculum scene file test stale)** + **codex P0 #3
  (`EnvBundle.num_envs` `__post_init__` validation)**: both flagged in
  the audit but they're runtime/test-correctness fixes, not test-suite
  partitioning. Owner triages separately.

---

## Verification matrix (run between every task)

```bash
# Default lane still green + pass count holds
JAX_PLATFORMS=cpu uv run python -m pytest -q

# Drift canaries
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_docs_drift.py tests/test_docs_code_blocks.py

# Marker registry (after 1.1)
uv run python -m pytest --markers | grep -E "gpu|warp|go2|deploy|network|slow"
```

Pass count drop without a corresponding deletion → STOP, investigate,
report.

---

## Execution log

> Append one entry per checked box. Format:
> `### YYYY-MM-DD — N.N <name>`
> Then: commit hash, default-lane test count before/after, verify
> deltas, deviations, gotchas hit.

<!-- ### 2026-04-30 — 1.1 Marker taxonomy ... -->

### 2026-04-30 — 1.1 Marker taxonomy + retarget default lane

- Task commit: `8e857ea6bbe54cd08577c80eb906ee2fbe9bb6f1`
- Default-lane test count before/after: collect-only `808/823 collected (15 deselected)` → `712/823 collected (111 deselected)`; full after-run `666 passed, 46 skipped, 111 deselected`.
- Verify deltas: marker registry lists `gpu`, `warp`, `go2`, `deploy`, `network`, `slow`; `-m gpu` collects `98/823`; `-m "warp or go2"` collects `82/823`; docs canaries held at `224 passed, 46 skipped, 7 deselected`.
- Deviations/gotchas hit: current-tree collection counts differ from the audit estimate (`~600` default, `~200` gpu), but the target file hunks matched the playbook and the CPU default lane passed. Audit table mentioned a `deploy` mark on the two deploy E2E tests; playbook Diff 3 literally adds only `gpu`/`warp`/`go2`, so this execution followed the playbook.

### 2026-04-30 — 1.1.5 Stamp deploy marker on deploy-intent tests

- Task commit: `11d2d6213d7c2b3dbd9ad353affa1b4a32313884`
- Default-lane test count: `712/823 collected (111 deselected)` unchanged; full run `666 passed, 46 skipped, 111 deselected, 122s` — no regression.
- `-m deploy --collect-only`: `0/823` → `11/823` (9 in `deploy/test_policy_runner.py` + 2 stamped in `tests/test_deploy_e2e.py`).
- Closes audit/playbook drift surfaced by 1.1's deviation note. Playbook Diff 3 updated in same commit so future agents stamp 4 marks not 3. Also added the audit + playbook files to the repo (were untracked despite being the spec the plan points to).
- No deviations.

### 2026-04-30 — 1.2 Delete dead TDMPC2 train smoke tests

- Task commit: `7fdb406a03d535ece5e50b8aef8625e851f03329`
- Default-lane test count before/after: `666 passed, 46 skipped, 111 deselected` → `666 passed, 46 skipped, 106 deselected`.
- Verify deltas: `uv run python -m pytest --collect-only -q -m slow` went from `14/823 collected (809 deselected)` to `9/818 collected (809 deselected)`; docs canaries held at `224 passed, 46 skipped, 7 deselected`; marker registry still lists `gpu`, `warp`, `go2`, `deploy`, `network`, `slow`.
- Deviations/gotchas hit: playbook's old absolute expectation says slow collect should be `10 not 15`, but after Task 1 moved the arXiv check from `slow` to `network`, the current baseline was `14`; deleting exactly five dead files produced the expected relative delta to `9`.

---

## Cross-references

- Audit: `.context/audits/2026-04-27_test_suite_audit.md`
- Playbook (literal diffs + gotchas): `.context/audits/2026-04-27_test_suite_execution_playbook.md`
- Codex audit (outsider view, untracked): `codex_audit.md`
- Algo-port protocol: `.context/lessons/algo_port_protocol.md`
- Resume warmup lesson: `.context/lessons/offpolicy.md` §"Resume Warmup"
- Env backend contract: `.context/lessons/env_backends.md`
- Structural-hardening parent plan: `.superpowers/plans/2026-04-25-structural-hardening.md` Phase 0 (this plan implements that phase)
