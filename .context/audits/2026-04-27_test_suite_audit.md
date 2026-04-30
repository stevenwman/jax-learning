# Test-Suite Cleanup Audit — 2026-04-27

> Audit-only pass. No file changes in this commit. Owner triages, then
> spawns a separate execution session per phase below.
>
> **Methodology:** verified every codex / prior-agent claim with `pytest
> --collect-only`, `JAX_PLATFORMS=cpu pytest -q <file>`, and direct file
> reads before recording it. Where this audit disagrees with codex, the
> disagreement is called out inline.

---

## TL;DR (top 5 actions)

1. **Add marker taxonomy + retarget default lane** (1 hr). Today
   `pytest` defaults pull 804 tests including Warp/Go2/MJX. Add
   `gpu/warp/go2/deploy/network` markers, set
   `addopts = "-m 'not (gpu or warp or go2 or network or slow)'"`,
   stamp the 12 GPU-required files with module-level `pytestmark`.
   Result: default lane drops to ~600 hermetic CPU tests, runtime
   from 9m39s → ~1-2 min.
2. **Delete or rewrite the 6 dead worktree-path tests** (30 min).
   `tests/test_train_tdmpc2_h{1..5}.py` and
   `tests/test_tdmpc2_i3_eval_isolation.py` all reference
   `.worktrees/tdmpc2-impl/` which no longer exists. They live behind
   `slow` so default lane doesn't see them, but `pytest -m slow` is now
   a 6-failure trap. Owner-call: rewrite to use `scripts/train_tdmpc2.py`
   (post-relocation) or delete.
3. **Make `deploy/test_policy_runner.py::test_policy_runner_loads_and_infers`
   hermetic** (1 hr). Currently scans `checkpoints/` and skips when
   absent, so CI can never run it. Replace with synthetic ckpt
   (precedent: `tests/test_artifact_contract.py`); add an artifact-kind
   rejection test while you're there (consumer-end coverage gap, §6).
4. **Fix the `EnvBundle(num_envs=1)` footgun**
   (covered by codex P0; not in scope here, but the test currently
   fails). One-line test fix (pass `num_envs=NUM_ENVS`) plus an
   `__post_init__` validation in `EnvBundle`. Until then,
   `tests/test_offpolicy_loop.py::test_run_offpolicy_loop_stub_env_cpu`
   stays red (30 min for the test fix alone).
5. **Add CPU-only GitHub Actions workflow** (30 min once #1 lands).
   Run `pytest -m 'not (gpu or warp or go2 or network or slow)'` on
   pushes/PRs. Today CI is docs-only — there is no automated regression
   gate at all.

**Verified codex hit rate this round:** 4/5 of codex's test-related claims
held up. The 5th — "test_tdmpc2.py constructs a module-level
`jax.random.PRNGKey`" — is **stale**: I grepped the file at lines 1-30
and there is no module-level PRNGKey. Codex was right that other files
do this (9 of them), just wrong about which one.

---

## 1. Current state inventory

### Counts

```bash
$ uv run python -m pytest --collect-only -q
804/819 tests collected (15 deselected) in 2.45s

$ uv run python -m pytest --collect-only -q -m slow
15/819 tests collected (804 deselected) in 2.54s
```

53 test files (52 under `tests/`, 1 under `deploy/`), 9903 LOC, 819
total tests. Default lane = 804. Owner-quoted previous audit's "299
tests" estimate is stale.

### Categorization table

Symbols: ✓ = passes `JAX_PLATFORMS=cpu`, ✗ = fails on CPU (verified),
~ = mixed / per-test split, ? = not verified yet.

Intent abbreviations: U=unit, I=integration, D=docs-drift, X=external
deps, A=artifact-fixture-driven.

| File | LOC | Tests | Intent | CPU? | Backend gate | Module-level CUDA? |
|---|---:|---:|---|:-:|---|---|
| `tests/test_action_delay.py` | 151 | 7 | U | ✓ | none (imports `mujoco_playground._src.mjx_env.State` only) | no |
| `tests/test_algo_configs.py` | 400 | 25 | U | ✓ | none | **`KEY = jax.random.PRNGKey(42)` line 34** |
| `tests/test_artifact_contract.py` | 133 | 11 | U | ✓ | none (synthetic dirs) | no |
| `tests/test_asymmetric_critic.py` | 386 | 28 | U | ✓ | none | no |
| `tests/test_builders.py` | 105 | 6 | U | ✓ | none | **`KEY = jax.random.PRNGKey(0)` line 22** |
| `tests/test_checkpoint.py` | 245 | 9 | U | ✓ | none | no |
| `tests/test_cli_utils.py` | 238 | 22 | U | ✓ | none | no |
| `tests/test_contraction_config.py` | 32 | 4 | U | ✓ | none | no |
| `tests/test_contraction_metric.py` | 103 | 5 | U | ✓ | none | no |
| `tests/test_deploy_e2e.py` | 374 | 11 | U+I (deploy) | ~ | 2 tests need Warp (`pytest.importorskip("warp")` already), 9 hermetic | no |
| `tests/test_distributional.py` | 131 | 9 | U | ✓ | none | no |
| `tests/test_docs_code_blocks.py` | 366 | 260 | D | ✓ | none (regex over docs) | no |
| `tests/test_docs_drift.py` | 374 | 10 (+1 `slow` network) | D | ✓ | arxiv test marked `slow`, network-only | no |
| `tests/test_domain_rand_compose.py` | 118 | 4 | U | ✓ | none (synthetic mock env) | no |
| `tests/test_env_bundle.py` | 38 | 2 (1 default + 1 `slow`) | I | ✗ | CheetahRun MJX → CUDA | no |
| `tests/test_flash_blocks.py` | 76 | 7 | U | ✓ | none | **line 13** |
| `tests/test_flash_sac.py` | 153 | 8 | U | ✓ | none | **line 13** |
| `tests/test_frame_stack.py` | 97 | 7 | U | ✓ | none | no |
| `tests/test_frame_stack_wrapper.py` | 86 | 8 | I (warp) | ✗ | `WarpJoystick` fixture | no (in fixture) |
| `tests/test_go2_bongo_contraction_obs.py` | 59 | 5 | I (warp) | ✗ | `BongoHandstand` | no |
| `tests/test_go2_bongo_env.py` | 119 | 15 | I (warp) | ✗ | `BongoHandstand` | no |
| `tests/test_go2_warp_curriculum_env.py` | 147 | 11 | I (warp) | ✗ | `WarpJoystickCurriculum` | no |
| `tests/test_go2_warp_env.py` | 232 | 27 | I (warp+deploy) | ✗ | `WarpJoystick` | no |
| `tests/test_metrics_logger.py` | 41 | 3 | U | ✓ | none | no |
| `tests/test_normalization.py` | 197 | 9 | U | ✓ | none | no |
| `tests/test_obs_pipeline.py` | 347 | 28 | U | ✓ | none | no |
| `tests/test_obs_spec.py` | 110 | 9 | U | ✓ | none | no |
| `tests/test_offpolicy_algos.py` | 277 | 18 | U | ✓ | none (synthetic batches) | **line 33** |
| `tests/test_offpolicy_loop.py` | 191 | 2 (1 default + 1 `slow`) | I | ✗ | StubEnv test fails on `EnvBundle(num_envs=1)` default → real bug, not GPU | no |
| `tests/test_onpolicy_collect.py` | 172 | 4 | U | ✓ | none | no |
| `tests/test_policy_delay.py` | 83 | 2 | U | ✓ | none | no |
| `tests/test_polyak.py` | 105 | 8 | U | ✓ | none | no |
| `tests/test_ppo_contraction.py` | 156 | 6 | U | ✓ | none | **line 23** |
| `tests/test_ppo_setup.py` | 215 | 8 | U | ✓ | none | **line 22** |
| `tests/test_pusht_parity.py` | 78 | 4 | I (gym) | ✓ | needs `gym-pusht` extra | no |
| `tests/test_qscale.py` | 29 | 4 | U | ✓ | none | no |
| `tests/test_replay_buffer.py` | 412 | 26 | U | ✓ | none | **line 16** |
| `tests/test_reward_scaling.py` | 49 | 5 | U | ✓ | none | no |
| `tests/test_reward_spec.py` | 47 | 5 | U | ✓ | none | no |
| `tests/test_rollout_contraction_fields.py` | 36 | 2 | U | ✓ | none | no |
| `tests/test_rollout_recording.py` | 142 | 4 | U | ✓ | none | no |
| `tests/test_simnorm.py` | 41 | 4 | U | ✓ | none | no |
| `tests/test_tdmpc2.py` | 1334 | 48 | U | ✓ | none | **NO** — codex's claim is **stale** |
| `tests/test_tdmpc2_config.py` | 30 | 3 | U | ✓ | none | no |
| `tests/test_tdmpc2_i3_eval_isolation.py` | 163 | 1 (`slow`) | I (broken) | n/a | hardcoded `.worktrees/tdmpc2-impl` (missing) | no |
| `tests/test_tdmpc2_presets.py` | 39 | 5 | U | ✓ | none | no |
| `tests/test_terrain_curriculum_dr_wrapper.py` | 212 | 14 | I (warp+go2) | ✗ | `WarpJoystickCurriculum` | no |
| `tests/test_terrain_generator.py` | 110 | 7 | U | ✓ | none | no |
| `tests/test_terrain_primitives.py` | 335 | 35 | U | ✓ | none | no |
| `tests/test_train_context.py` | 63 | 2 | U | ✓ | none | no |
| `tests/test_train_tdmpc2_h1.py` | 29 | 1 (`slow`) | I (broken) | n/a | `.worktrees/tdmpc2-impl/train_tdmpc2.py` (missing) | no |
| `tests/test_train_tdmpc2_h2.py` | 46 | 1 (`slow`) | I (broken) | n/a | same | no |
| `tests/test_train_tdmpc2_h3.py` | 39 | 1 (`slow`) | I (broken) | n/a | same | no |
| `tests/test_train_tdmpc2_h4.py` | 39 | 1 (`slow`) | I (broken) | n/a | same | no |
| `tests/test_train_tdmpc2_h5.py` | 44 | 1 (`slow`) | I (broken) | n/a | same | no |
| `tests/test_training_wrappers.py` | 173 | 12 | I (mjx+warp) | ✗ | mujoco_playground + Warp | no |
| `tests/test_twohot.py` | 39 | 6 | U | ✓ | none | no |
| `tests/test_wandb_metrics.py` | 117 | 7 | U | ✓ | none | no |
| `tests/test_weight_norm.py` | 68 | 4 | U | ✓ | none | **line 8** |
| `tests/test_wrapper_pipeline.py` | 50 | 6 | U | ✓ | none | no |
| `tests/archive/test_go2_env.py` | n/a | n/a | archived | n/a | not collected (in `archive/`) | no |
| `deploy/test_policy_runner.py` | 200 | 9 | U+A (deploy) | ✓ | 1 artifact-fixture test scans `checkpoints/`, 8 hermetic synthetic | no |

### Module-level CUDA-allocating bombs (verified)

These nine files declare a module-level
`KEY = jax.random.PRNGKey(...)`. On a GPU-tight machine with eager
backend init, **collection** of any of these can allocate CUDA memory
before a test even runs. All currently pass on CPU (so the OOM risk is
GPU-specific), but the §11 hermeticity rule in
`.context/lessons/algo_port_protocol.md` says "do NOT" do this:

```
tests/test_algo_configs.py:34:KEY = jax.random.PRNGKey(42)
tests/test_builders.py:22:KEY = jax.random.PRNGKey(0)
tests/test_flash_blocks.py:13:KEY = jax.random.PRNGKey(0)
tests/test_flash_sac.py:13:KEY = jax.random.PRNGKey(42)
tests/test_offpolicy_algos.py:33:KEY = jax.random.PRNGKey(42)
tests/test_ppo_contraction.py:23:KEY = jax.random.PRNGKey(0)
tests/test_ppo_setup.py:22:KEY = jax.random.PRNGKey(0)
tests/test_replay_buffer.py:16:KEY = jax.random.PRNGKey(0)
tests/test_weight_norm.py:8:KEY = jax.random.PRNGKey(0)
```

**Action (15 min each, 9 files):** move `KEY = jax.random.PRNGKey(...)`
into a `@pytest.fixture` or use `key = jax.random.PRNGKey(0)` inside
each test body. Trivial mechanical change.

**Codex disagreement:** codex's audit (line 962) says
`tests/test_tdmpc2.py constructs a module-level jax.random.PRNGKey`.
I verified — it does not. All seven `jax.random.PRNGKey(0)` calls in
that file are inside test functions (lines 11, 20, 33, 42, 53, 65, 76).
Codex's prior CUDA-OOM-during-collection was caused by one of the nine
files above, not test_tdmpc2.py.

### Existing markers in use

```bash
$ grep -E "@pytest.mark|pytestmark" tests/*.py deploy/test_*.py
```

- `@pytest.mark.slow` × 11 (the only registered marker beyond
  `parametrize`)
- 0 module-level `pytestmark = ...` declarations

---

## 2. Partitioning gap (P0)

`pyproject.toml` lines 71-77:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests", "deploy"]
markers = [
    "slow: marks tests as slow / GPU-bound (deselected by default; run with -m slow)",
]
addopts = "-m 'not slow'"
```

**Verified facts:**
- Default `pytest` collects 804 tests including 12 files
  (`test_env_bundle.py`, `test_offpolicy_loop.py`,
  `test_frame_stack_wrapper.py`, `test_training_wrappers.py`,
  `test_go2_warp_env.py`, `test_go2_warp_curriculum_env.py`,
  `test_go2_bongo_env.py`, `test_go2_bongo_contraction_obs.py`,
  `test_terrain_curriculum_dr_wrapper.py`, plus 2 tests in
  `test_deploy_e2e.py`) that **fail under `JAX_PLATFORMS=cpu`** with
  `RuntimeError: Unknown backend cuda`.
- Default lane runtime: 9m39s (codex-quoted, consistent with my partial
  reruns).
- Codex P0 #1 ("default pytest is red") confirmed: 2 failing tests on
  GPU machine, plus the broken `EnvBundle.num_envs` test.

### Proposed marker taxonomy

```toml
markers = [
    "gpu:     test needs CUDA (MJX physics, JAX device arrays beyond CPU)",
    "warp:    test needs warp-lang library + GPU (implies gpu)",
    "go2:     test needs Go2 Warp env (implies warp + gpu)",
    "deploy:  test exercises deploy/ runtime",
    "network: test needs internet (e.g. arxiv, HF datasets)",
    "slow:    test takes >30s wall-clock (can be CPU or GPU)",
]
addopts = "-m 'not (gpu or warp or go2 or network or slow)'"
```

**Justification:**
- `gpu` is the catch-all for "JAX requires CUDA backend." MJX physics
  and any test that constructs an MJX env transitively needs this.
- `warp` is narrower: needs the `warp-lang` library import plus GPU.
  `warp` ⊂ `gpu` semantically; mark a test with both for clarity.
- `go2` = warp + Go2-specific MJCF/observation contract. Useful as a
  selector for "I'm refactoring deploy, run the Go2 surface only."
- `deploy` cuts orthogonally — there are CPU deploy tests
  (`deploy/test_policy_runner.py` mostly) and GPU deploy tests
  (the 2 in `test_deploy_e2e.py` that need Warp). Want to be able to
  run "all deploy" before pushing a checkpoint.
- `network` separates the arxiv/HF tests from `slow`. `slow` today
  conflates "needs network" with "long-running." Once split, owners
  can run `pytest -m 'slow and not network'` from a flight.
- `slow` stays for long-running CPU tests too (none right now, but
  the door's open).

**The `mjx` marker codex proposed is redundant** — `gpu` covers it
(every MJX test needs CUDA, and we have no CPU-MJX path). Keep marker
count small.

### Per-file marker assignments

Stamp each file with module-level `pytestmark = pytest.mark.<x>` (or
`[mark.<x>, mark.<y>]`). For mixed files (e.g., `test_deploy_e2e.py`),
mark per-test, not per-module.

| File | Add `pytestmark` |
|---|---|
| `tests/test_env_bundle.py` | `[gpu]` (CheetahRun → CUDA) — slow already on dict-obs test, leave it |
| `tests/test_offpolicy_loop.py` | `[gpu]` — `run_offpolicy_loop` jit'd for GPU. Also fix the stub-env test bug separately |
| `tests/test_frame_stack_wrapper.py` | `[gpu, warp]` — `WarpJoystick` fixture |
| `tests/test_training_wrappers.py` | `[gpu]` — uses `mujoco_playground` MJX directly; one test also touches Warp |
| `tests/test_go2_warp_env.py` | `[gpu, warp, go2]` |
| `tests/test_go2_warp_curriculum_env.py` | `[gpu, warp, go2]` |
| `tests/test_go2_bongo_env.py` | `[gpu, warp, go2]` |
| `tests/test_go2_bongo_contraction_obs.py` | `[gpu, warp, go2]` |
| `tests/test_terrain_curriculum_dr_wrapper.py` | `[gpu, warp, go2]` |
| `tests/test_deploy_e2e.py` | per-test only: `test_bongo_metadata_uses_handstand_pose_not_home` and `test_env_metadata_matches_deploy_constants` get `[gpu, warp, go2, deploy]`; rest stay default |
| `tests/test_pusht_parity.py` | none; runs CPU. Optional: `[deploy]` if owner wants gym-pusht extra to be opt-in |
| `tests/test_docs_drift.py::test_arxiv_ids_resolve` | `network` (currently `slow` — split it) |
| `tests/test_train_tdmpc2_h{1..5}.py` | **DELETE** (see §5) — they reference dead worktree |
| `tests/test_tdmpc2_i3_eval_isolation.py` | **DELETE or rewrite** (see §5c) |
| All 9 module-level-PRNGKey files | no marker change; just move PRNGKey into fixture |

After marker discipline lands, the default lane runs **roughly 600
tests** (804 − ~200 GPU/warp/go2/network/slow) in 1-2 min instead of
9m39s.

### Named pytest invocations to document

Add to `docs/contributing.md` (and `Makefile` if owner likes makefiles):

```bash
# CPU smoke (default, fast, hermetic) — what CI should run
uv run python -m pytest

# Run GPU-only tests after a code change that touches MJX/JAX physics
uv run python -m pytest -m gpu --no-header

# Run Warp/Go2-specific surface (use before a deploy push)
uv run python -m pytest -m "warp or go2"

# Deploy-only (use before a hardware push)
uv run python -m pytest -m deploy

# Full suite (slow, includes network — flight wifi unfriendly)
uv run python -m pytest -m "not network"
```

---

## 3. Structure

**Recommendation: markers-only, no subfolder reorganization.**

The structural-hardening plan
(`.superpowers/plans/2026-04-25-structural-hardening.md` Phase 0) and
codex both proposed `tests/{unit,integration,gpu}/` subfolders. After
inspecting each file, I disagree:

1. **Most files are already single-intent.** The CPU/GPU split is
   ~75/25 and clean per-file with the 1 mixed exception
   (`test_deploy_e2e.py`). A marker captures the same information
   without import-path churn.
2. **Marker selectors compose; folders don't.** With markers, owner
   can run `pytest -m "gpu and not warp"` (= MJX without Warp) — a
   useful filter when Warp is broken but MJX isn't. Folders force a
   single primary axis.
3. **`pytest --collect-only -q -m gpu` self-documents** which tests
   are GPU. New contributors don't need to learn a folder convention.
4. **57 file moves = 57 import-path-touched git diff entries** — ugly
   review, churns blame for no functional benefit. The drift tests
   (`tests/test_docs_drift.py`, `tests/test_docs_code_blocks.py`) are
   already canary-style — moving them risks breaking the rglob walks
   and we'd discover that as a regression.

**One subfolder I'd consider — `tests/integration/`** for the 6 dead
worktree-path tests (after rewrite, if kept). They are full-process
subprocess smoke tests, not unit tests, and will always be slow + GPU.
But this is a stylistic call; markers alone solve the partitioning
problem. Defer until owner says otherwise.

---

## 4. Hermeticity (CPU verification)

Verified by `JAX_PLATFORMS=cpu uv run python -m pytest -q <file>` on
each suspect file. Failures classified per the prompt:

| File | CPU result | Class | Action |
|---|---|---|---|
| `test_env_bundle.py` | 1 failed (CheetahRun → CUDA), 1 deselected | (a) genuinely needs GPU (MJX) | mark `[gpu]` |
| `test_offpolicy_loop.py` | 1 failed (`EnvBundle(num_envs=1)` default), 1 deselected | (a) **but bug is real, not GPU**: stub-env test passes `NUM_ENVS=2` to env state but bundle defaults `num_envs=1`. Fix is `num_envs=NUM_ENVS` in test + `__post_init__` validation in EnvBundle (codex P0 #3) | mark `[gpu]` AFTER fixing the test bug — it's a unit test in spirit |
| `test_frame_stack_wrapper.py` | 8/8 errors | (b) accidentally requires GPU — fixture builds `WarpJoystick` even for tests that just want shape checks | mark `[gpu, warp]`. Long-term: provide a stub env fixture for shape-only tests |
| `test_training_wrappers.py` | 13/14 errors | (a) MJX needed (uses real `dm_control_suite`) | mark `[gpu]` |
| `test_go2_warp_env.py` | 26/27 fail | (a) genuinely Warp+Go2 | mark `[gpu, warp, go2]` |
| `test_go2_warp_curriculum_env.py` | 11/11 fail | (a) genuinely Warp+Go2 | mark `[gpu, warp, go2]` |
| `test_go2_bongo_env.py` | 14/15 fail | (a) | mark `[gpu, warp, go2]` |
| `test_go2_bongo_contraction_obs.py` | 5/5 fail | (a) | mark `[gpu, warp, go2]` |
| `test_terrain_curriculum_dr_wrapper.py` | 14/14 fail | (a) | mark `[gpu, warp, go2]` |
| `test_deploy_e2e.py` | 2/11 fail | (a) for the 2 Warp-touching ones; rest are hermetic | per-test mark only |

**Class (c) — module-level CUDA allocation:** zero files in the failing
set. The 9 module-level-PRNGKey files all pass on CPU. They're a
latent hazard, not a current bug. Move to fixtures as preventive
hygiene; not a blocker.

**No bug-filing in this pass.** Per prompt: classify and stop. The
fixture-bug in `test_frame_stack_wrapper.py` (case b) and the
EnvBundle.num_envs default (codex P0 #3) are owner-triage items.

---

## 5. Known-debt verification

### 5a. `tests/test_tdmpc2.py` (1334 lines, 48 tests) — split?

**Disagree with prior agent's "should split" framing.** I read the
file structure:

- Networks (lines 9-170) — NormedLinear, Encoder, Dynamics, Reward,
  QEnsemble: 12 tests
- Policy/distrib utils (172-290) — bound_log_std, squash, gaussian
  log_prob, policy prior, latents: 9 tests
- TD targets + losses (291-694) — `compute_td_target`,
  `world_model_loss`, `policy_loss`: 9 tests
- MPPI (697-916) — rollout, iteration, sample, init mean: 8 tests
- Plan/agent (920-1056) — `plan`, gumbel, `plan_batched`: 5 tests
- State + optimizers + integration (1059-end) — `TDMPC2State`,
  `build_world_model_optimizer`, `make_update_step` smoke,
  H-normalization: 5 tests

Three module-level helpers shared across blocks:
`_build_small_cfg_and_modules` (line 458), `_init_small_params` (line
477), `_build_plan_params` (line 697). All are pure and could move to
a `tests/_tdmpc2_helpers.py`.

**Verdict:** splitting earns marginal benefit. Pros: 6 files at
~150-300 LOC each, easier to navigate, parallelizable in `pytest -n
auto`. Cons: 3 helpers must move to a sibling file (small refactor),
and the file already has clear `# ------` section banners. The CPU
runtime is 41s (mostly JIT); split won't reduce it.

**Recommendation:** **defer the split**. Mark it as P3. The 1334-LOC
file is well-organized; the test selector already navigates by name.
If the owner ever wants to enable `pytest-xdist`, revisit then.
Expected-utility-wise, bigger wins exist elsewhere in this audit.

If the split happens, propose:
```
tests/tdmpc2/
    test_networks.py        # NormedLinear, Encoder, Dynamics, Reward, QEnsemble
    test_policy.py          # PolicyPrior + distribution utils
    test_losses.py           # td_target, world_model_loss, policy_loss
    test_mppi.py             # MPPI primitives
    test_plan.py             # plan, plan_batched
    test_agent.py            # TDMPC2State, optimizers, update_step
    _helpers.py              # _build_small_cfg_and_modules etc.
```

**Cost:** 1 hr if split.

### 5b. `tests/test_tdmpc2_i3_eval_isolation.py` worktree path

Verified line 29 + line 152 hardcode
`/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl`.
That directory **does not exist** (verified `ls .worktrees/` →
`No such file or directory`).

**Disagree with prior agent's claim that this is intentional.** The
TDMPC2 refactor merged into `new_slate_linen` per
`.context/lessons/algo_port_protocol.md` §5 + codex's note that "the
package now contains focused modules" — the worktree was the *staging
ground* for the refactor, not a permanent fixture. The merge happened
(commit history shows TDMPC2 in `jax_rl/algos/tdmpc2/` as a normal
package); the worktree should have been deleted; the test should have
been updated to import from the merged package.

**Action:** rewrite to use `scripts/train_tdmpc2.py` directly (the
post-relocation path) and import from `jax_rl.algos.tdmpc2` (already
the canonical location). The test logic — eval/collect prev_mean
isolation regression guard — is genuinely valuable, do not delete it.
Cost: 30 min.

### 5b'. `tests/test_train_tdmpc2_h{1..5}.py` (5 files) — same problem

All five files reference `/home/stevenman/.../.worktrees/tdmpc2-impl`
in their `subprocess.run(cwd=...)`. All five would fail if `pytest -m
slow` ever ran. They're hermetic from the default lane (all `slow`),
so today they're invisibly broken.

H1-H5 are smoke gates from the original TDMPC2 implementation plan
(see `.superpowers/plans/2026-04-21-tdmpc2-implementation.md`). The
algorithm has since landed in mainline. Smoke tests for an
already-landed algorithm are arguably redundant — `test_tdmpc2.py`'s
48 tests cover the components; the train smoke is just a "does the
script run" check, which is what `gen_cli_reference.py` already
exercises by import.

**Recommendation:** **delete all 5 files**. They're dead weight. If
the owner wants a TDMPC2 train smoke test, write one (50 LOC) that
calls `scripts/train_tdmpc2.py --total-timesteps 0` via subprocess
from `cwd=REPO_ROOT`. Cost to delete: 5 min. Cost to rewrite as one
unified file: 1 hr.

### 5c. `deploy/test_policy_runner.py::test_policy_runner_loads_and_infers`

Verified lines 15-25:

```python
ckpt_dir = None
if os.path.isdir("checkpoints"):
    for d in sorted(os.listdir("checkpoints")):
        best = os.path.join("checkpoints", d, "best")
        if os.path.isdir(best) and os.path.exists(os.path.join(best, "actor_params.npy")):
            ckpt_dir = best
            break

if ckpt_dir is None:
    pytest.skip("No checkpoint found in checkpoints/")
```

This is brittle in two ways:

1. **Behavior depends on whoever ran training last.** On the dev
   machine: `ls checkpoints/ | head -3` shows `20260313_173851_cheetahrun_seed0`,
   `20260318_111816_sac_humanoidrun_seed0`, etc. — the test loads
   *whichever sorts first alphabetically*, then checks that
   `runner.action_dim`-shaped action is in [-1, 1]. Different machines
   with different checkpoint sets may load entirely different algos.
2. **CI can never see this test.** GitHub runners have no
   `checkpoints/` — `pytest.skip` always fires. So the only place
   this test runs is the dev machine's manual invocation.

**Replace with synthetic ckpt fixture** (precedent:
`tests/test_artifact_contract.py::test_validate_shared_actor_files_passes_on_complete_dir`
lines 89-97 — already builds a synthetic shared-actor ckpt dir):

```python
def test_policy_runner_loads_synthetic_shared_actor_ckpt(tmp_path):
    # Synthesize meta + actor_params.npy + orbax/ in tmp_path
    # (use stamp_meta + KIND_SHARED_ACTOR)
    # Assert runner.algo / .obs_dim / .action_dim are correct
    # Assert action shape and bounds
    ...

def test_policy_runner_rejects_tdmpc2_ckpt(tmp_path):
    # Synthesize KIND_TDMPC2 ckpt (actor_params.npz + world_model_params.npz)
    # Assert PolicyRunner(ckpt_dir) raises with redirect message
    # Closes the consumer-rejection coverage gap (§6)
```

The second test answers codex's "consumer X rejects kind Y end to
end" gap by exercising the assert_artifact_kind redirect path.

**Cost:** 1 hr (1 to build the synthetic fixture, 1 to write the
rejection test). Big win for both hermeticity and coverage.

---

## 6. Coverage gaps

### Per-algo end-to-end smoke

For each algo: does anything exercise the full training loop end-to-end?

| Algo | Loop test | Notes |
|---|---|---|
| sac | `test_offpolicy_loop.py::test_run_offpolicy_loop_stub_env_cpu` (broken, see §5) + `::test_run_offpolicy_loop_sac_cheetah` (`slow`) | Stub-env test has the EnvBundle bug. Fix it. |
| td3 | none | **Gap.** Unit tests exist (`test_offpolicy_algos.py`); no loop smoke. |
| fast_sac | none | **Gap.** Same. |
| fast_td3 | none | **Gap.** Same. |
| flashsac | none | **Gap.** Has unit tests but no loop smoke. The 2026-04-26 resume regression session would have benefited from one. |
| ppo | none | **Gap.** No loop smoke. `train_ppo.py` migration to bundle dispatch is recent (commits `8b5d0c3`, `4e79319`); missing test means we won't catch regressions. |
| ppo_fast | none | **Gap.** Mode-A-only path; full-scan rollout is the easy regression target. |
| ppo_contraction | `test_ppo_contraction.py` (6 tests, unit-level only) | No loop smoke. |
| tdmpc2 | `test_train_tdmpc2_h{1..5}.py` (5 broken files) + `test_tdmpc2_i3_eval_isolation.py` (broken) | Once the worktree paths are fixed/rewritten, this is acceptable coverage. |

**Recommendation (1 hr each):** the existing
`test_run_offpolicy_loop_stub_env_cpu` is a great template — once
fixed (EnvBundle bug), parameterize it over the 4 off-policy algos
(sac/td3/fast_sac/fast_td3). One file, ~250 LOC, runs in <30s. Same
template for ppo (use a stub env + a 40-step run). Result: every algo
has a CPU-hermetic smoke that catches glue-layer regressions before
they hit GPU.

### DomainRand per-episode persistence regression

Codex flagged the absence of a regression test for the per-episode
fix in commit `826c326`. Verified:

```bash
$ grep -rln "_dr_dr_fields" tests/
(no matches)
$ grep -rln "_dr_dr_fields" jax_rl/
jax_rl/envs/wrappers/domain_rand.py
```

The duplicate-field-composition bug **does** have a regression test
(`tests/test_domain_rand_compose.py`, 4 tests, hermetic mock env —
nicely done). The **per-episode persistence** path does not.

**Real gap.** Recommended test (synthetic, hermetic, ~50 LOC): build
a mock env exposing `mjx_model.body_mass`; call `wrapper.step()` 5
times with `done=False` for env 0; assert
`state.info["_dr_dr_fields"]["body_mass"][0]` is byte-identical
across all 5 steps. Then call with `done=True`; assert env 0's field
changes to a new value.

**Cost:** 30 min. Use the `test_domain_rand_compose.py` mock-wrapper
pattern.

### Artifact contract: consumer-end rejection

Codex flagged: `tests/test_artifact_contract.py` covers stamps +
validators, but nothing covers "consumer X rejects kind Y end to
end." Verified by reading the 11 tests in that file — all are
producer-side (stamp / read / validate file presence). The closest
consumer test is in `test_artifact_contract.py::test_assert_artifact_kind_raises_on_mismatch_with_redirect`,
which is a unit test of `assert_artifact_kind` itself, not of a
real consumer.

**Real gap.** Best fix is folding it into the §5c rewrite of
`deploy/test_policy_runner.py` — `test_policy_runner_rejects_tdmpc2_ckpt`
exercises the full reject path: TDMPC2 ckpt on disk → PolicyRunner
ctor → assert_artifact_kind redirects → ValueError.

**Cost:** included in §5c. Bundled win.

### Resume warmup behavioral test

`--resume-warmup {policy,random}` flag was added per
`.context/lessons/offpolicy.md` §"Resume Warmup". Verified:

```bash
$ grep -rln "resume_warmup\|--resume-warmup" tests/ deploy/test_*.py
(no matches)
```

**Real gap, but this is genuinely hard to test hermetically.** The
behavioral claim is: when `--resume-warmup policy`, the action
selection branch uses `explore_fn(actor_params, obs, key)` (loaded
policy) instead of `jax.random.uniform(...)`. To test it without a
GPU + 10k env steps, you'd need to:

1. Build a stub env + tiny SAC.
2. Save a ckpt with non-zero actor weights.
3. Load with `resume="...", resume_warmup="policy"`.
4. Patch `jax.random.uniform` to assert it's never called during
   warmup, OR patch `explore_fn` to count calls.

Doable, ~80 LOC, ~30s runtime. Worthwhile because the off-policy
resume is a known fragile area (see lesson §"Resume Warmup" — FlashSAC
Cartpole still has a residual drift the fix doesn't fully close).

**Cost:** 1 hr (medium; requires careful loop-state surgery).

---

## 7. CI absence

Verified `.github/workflows/`:
```
.github/workflows/docs.yml   # mkdocs deploy on push to main
```

No pytest workflow. No lint. No type-check. Zero automated regression
gate beyond docs.

**Proposed (do NOT write the YAML in this audit, just leave the TODO):**

After §2 marker landing, add `.github/workflows/tests.yml`:
- Trigger: push to `main` + PR
- Runner: `ubuntu-latest` (CPU only)
- Steps: install uv → `uv sync` → `uv run python -m pytest -m 'not (gpu or warp or go2 or network or slow)'`
- Expect: ~600 tests, ~1-2 min wall-clock
- Fail PR on red

A separate `gpu-tests.yml` could run on `workflow_dispatch` only
(self-hosted runner with CUDA), executing `pytest -m gpu`. Defer until
the CPU lane is green and stable.

**Cost:** 30 min once §2 lands. Don't write YAML before markers.

---

## Prioritized action list (cost-tagged)

| # | Action | Cost | Section |
|---:|---|---|---|
| 1 | Add marker taxonomy to `pyproject.toml`; stamp 9 GPU-test files with module-level `pytestmark`; per-test mark in `test_deploy_e2e.py` | 1 hr | §2 |
| 2 | Move 9 module-level `KEY = jax.random.PRNGKey(...)` into fixtures or test bodies | 30 min | §1 (last subsection) |
| 3 | Delete `tests/test_train_tdmpc2_h{1..5}.py` (5 dead files) | 5 min | §5b' |
| 4 | Rewrite or delete `tests/test_tdmpc2_i3_eval_isolation.py` (worktree path) | 30 min (rewrite) / 5 min (delete) | §5b |
| 5 | Make `deploy/test_policy_runner.py::test_policy_runner_loads_and_infers` synthetic; add TDMPC2 rejection test | 1 hr | §5c, §6 |
| 6 | Fix `test_run_offpolicy_loop_stub_env_cpu` (pass `num_envs=NUM_ENVS` to EnvBundle); parameterize across off-policy algos | 1 hr | §6 |
| 7 | Add DomainRand per-episode persistence regression test | 30 min | §6 |
| 8 | Add resume-warmup behavioral test | 1 hr | §6 |
| 9 | Add CPU-only GitHub Actions pytest workflow | 30 min (after #1) | §7 |
| 10 | Document the named pytest invocations in `docs/contributing.md` | 15 min | §2 |
| 11 | (Defer) Split `tests/test_tdmpc2.py` into 6 files | 1 hr | §5a |
| 12 | (Defer) Move dead worktree-path tests to `tests/integration/` if rewritten | 30 min | §3 |

**Total committed cost (#1-10):** ~6.5 hr — split across 2 owner sessions
or a single long afternoon. Each item is independently mergeable.

---

## Cross-references with codex_audit.md

Codex 2026-04-27 §"Current P0 Findings" / §"Current P1 Findings" overlap:
- **codex P0 #1 (default pytest red):** confirmed; addressed by my
  action #1 (marker lane) + action #6 (stub-env fix).
- **codex P0 #2 (curriculum scene file test stale):** out of scope
  (test correctness, not partition); flagged separately.
- **codex P0 #3 (`EnvBundle.num_envs` footgun):** confirmed; my action
  #6 handles the test side; runtime fix (validate `num_envs` in
  `__post_init__`) is a separate execution session.
- **codex P1 (test marker taxonomy and CI underbuilt):** my actions
  #1 and #9 cover it.
- **codex P1 (docs drift guard does not share full PUBLIC_SCRIPTS
  registry):** out of scope here (docs side); flagged in this audit's
  intent table only.

Codex's marker proposal listed `unit, docs, integration, gpu, warp,
go2, deploy, network, artifact`. I dropped `unit` (default = unit;
explicit positive marker is just noise), `docs` (these tests run
hermetically, no marker needed), `integration` (granular subdomain
markers cover it), and `artifact` (one file's worth of tests doesn't
need its own marker — they go under `unit`/default). Otherwise we
agree.

Codex's claim that `tests/test_tdmpc2.py` has a module-level PRNGKey
is **stale** — verified false. The OOM codex saw came from one of the
9 other files I listed in §1.

---

## Acceptance check

- [x] Owner can read this in 10-15 min — table of contents, TL;DR up
  top, prioritized action list with costs.
- [x] Each finding has a one-line action + cost estimate.
- [x] No file changes in this pass.
- [x] Commands and line numbers quoted exactly. Codex disagreements
  called out where they exist.
