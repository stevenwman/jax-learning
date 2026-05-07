# Structural Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the repo's structure by making test tiers, artifact contracts, deploy metadata, backend boundaries, and maturity labels explicit.

**Architecture:** This is an umbrella implementation plan, not a single atomic patch. Each phase is intentionally scoped as an independently testable sub-project because the audit spans test infrastructure, checkpoint artifacts, Go2 deploy, DomainRand semantics, resume behavior, docs, and script structure. Execute phases in order unless the owner explicitly reprioritizes.

**Tech Stack:** Python 3.13, JAX, Flax, Orbax, MuJoCo Playground/Warp, pytest, pure-numpy deploy runtime, Superpowers plan conventions.

**Date:** 2026-04-25
**Author:** Codex audit, agent-agnostic execution plan

**Non-goals:**
- Do not rewrite algorithms for style.
- Do not force FlashSAC, TD-MPC2, PushT, and shared off-policy loops into one trainer abstraction.
- Do not change Go2 training/deploy behavior without a regression test or explicit operator-facing migration.
- Do not require IsaacLab/gym backend integration in this plan. This plan prepares boundaries; backend expansion can follow.

**Primary principle:** Define contracts before moving code. The repo needs sharper boundaries more than it needs more abstraction.

---

## Scope Check

This audit covers multiple independent subsystems. Do not implement the whole file as one patch. Treat each phase below as either:

- a standalone execution task, or
- a prompt for generating a more detailed phase-specific plan with `superpowers:writing-plans`.

Recommended first executable subplan: **Phase 0 - Establish A Safe Test Surface**. It improves validation reliability before touching runtime behavior.

## Execution Checklist

- [ ] **Task 0: Establish a safe default test surface** - add pytest markers and exclude GPU/Warp/Go2 integration tests from default runs.
- [ ] **Task 1: Define the shared artifact contract** - add versioned metadata fields and validation for shared checkpoints.
- [ ] **Task 2: Add Go2 control metadata and deploy fallbacks** - persist deploy-critical control settings and make sim2sim read them.
- [ ] **Task 3: Fix DomainRand model DR semantics** - make model randomization per-episode and compose duplicate-field specs.
- [ ] **Task 4: Clarify resume semantics** - make warm-start explicit and add restored-policy warmup behavior.
- [ ] **Task 5: Add backend metadata and logging hooks** - remove Go2-specific imports from generic training loops.
- [ ] **Task 6: Create a public entrypoint registry** - keep CLI docs and parser tests aligned.
- [ ] **Task 7: Thin large scripts after contracts are stable** - start with TD-MPC2 script decomposition.
- [ ] **Task 8: Add maturity labels in docs and tooling** - mark stable, deploy-critical, experimental, diagnostics, and archived surfaces.
- [ ] **Task 9: Optionally implement full resume** - only after owner approval.

## Current State Summary

The repo has three maturity levels mixed into one surface:

- Stable framework: algorithms, networks, buffers, normalization, shared off-policy loop, configs.
- Deploy-critical Go2 path: Warp env, obs schema, deploy `PolicyRunner`, sim2sim, hardware validation docs.
- Research workbench: TD-MPC2, PushT, Bongo, terrain curriculum, diagnostics, one-off experiments.

The structural risk is that docs, CLIs, checkpoint formats, and tests do not consistently signal which layer a feature belongs to.

Key validated issues from `codex_audit.md`:

- `DomainRandWrapper` model randomization changes within a non-done episode.
- `_build_dr_model()` overwrites duplicate-field DR specs.
- Go2 sim deploy uses archived MJX PD gains while Warp training uses `Kp=20.0`, `Kd=0.5`.
- "Resume" is warm-start with optimizer state, not exact continuation.
- Default pytest can run unmarked Go2/Warp tests and OOM the GPU.
- Checkpoint artifacts differ across shared algos, TD-MPC2, PushT, deploy, export, and video recording.

---

## North-Star Boundaries

### Pure JAX Implementation Layer

Owns:
- `jax_rl/algos/`
- `jax_rl/networks/`
- `jax_rl/buffers/`
- math utilities and normalization helpers

Rules:
- No env imports.
- No deploy imports.
- No checkpoint directory assumptions.
- Tests should run CPU-only unless explicitly GPU-specific.

### Backend-Neutral Training Layer

Owns:
- loops
- obs extraction
- eval scheduling
- metrics plumbing
- checkpoint calls
- backend hooks

Rules:
- Training loops know that envs step and reset.
- Training loops do not know about Go2 terrain images, MuJoCo Playground registries, or hardware constants.
- Backend-specific metadata/logging arrives through `EnvBundle` hooks.

### Environment Layer

Owns:
- MuJoCo Playground/Warp envs
- wrapper composition
- obs schema generation
- DR specs
- curriculum-specific metrics/images
- physics/control metadata

Rules:
- Env packages may know about terrain, PD gains, DR, contact modes.
- Env packages provide metadata to training, not the other way around.

### Artifact Layer

Owns:
- checkpoint schema
- metadata validators
- deploy compatibility checks
- consumer capability checks

Rules:
- A checkpoint declares its contract.
- Consumers fail early when a contract is unsupported.
- Deploy-critical constants come from metadata, not duplicated source constants.

### Script Layer

Owns:
- CLI parsing
- config overrides
- calling reusable functions

Rules:
- Scripts should not own artifact formats.
- Scripts should not hold large reusable training/eval logic.
- Public scripts expose `build_parser()`.

---

## Global Execution Rules For Agents

Use these rules for every phase:

1. Read the relevant files before editing.
2. Add or update tests before changing behavior when a bug is being fixed.
3. Keep changes narrowly scoped to the phase.
4. Do not delete archived or generated local artifacts unless explicitly asked.
5. Do not rename public CLI flags without docs and migration notes.
6. After each phase, update `codex_audit.md` with status and any changed assumptions.
7. Prefer synthetic fixtures over local checkpoint artifacts.
8. Use CPU-only tests for fast validation where possible.
9. Run explicit GPU/Warp tests only under their marker or with operator approval.

Recommended baseline commands:

```bash
JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/test_polyak.py tests/test_replay_buffer.py tests/test_twohot.py tests/test_reward_spec.py tests/test_docs_code_blocks.py -q
JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/test_docs_drift.py tests/test_cli_utils.py tests/test_checkpoint.py deploy/test_policy_runner.py -q
```

---

## Phase 0 - Establish A Safe Test Surface

**Purpose:** Make default test results trustworthy before deeper refactors.

**Files likely touched:**
- `pyproject.toml`
- Go2/Warp test modules under `tests/`
- docs or README test instructions

**Steps:**
1. Add pytest markers:
   - `gpu`
   - `warp`
   - `go2`
   - keep `slow`
2. Change default `addopts` to exclude GPU/Warp/Go2 tests.
3. Add module-level `pytestmark` to tests that instantiate:
   - `WarpJoystick`
   - `WarpJoystickCurriculum`
   - `Go2BongoHandstand`
   - MuJoCo Playground full training wrapper paths that allocate Warp graphs
4. Document explicit integration commands:
   - `uv run python -m pytest -m warp`
   - `uv run python -m pytest -m go2`
   - `uv run python -m pytest -m gpu`
5. Keep slow network/doc tests separately gated.

**Acceptance criteria:**
- Default `uv run python -m pytest -q` does not instantiate Warp Go2 envs.
- CPU-safe unit/docs subsets pass.
- `pytest --markers` lists the new markers.
- Docs clearly explain how to run GPU integration tests.

**Risks:**
- Over-marking could hide useful tests from default runs.
- Under-marking keeps the current OOM problem.

**Agent prompt:**

```text
Implement Phase 0 of .superpowers/plans/2026-04-25-structural-hardening.md.
Only change pytest marker config, test module markers, and test documentation.
Do not change runtime code. Validate that default pytest selection excludes Go2/Warp tests.
```

---

## Phase 1 - Define The Shared Artifact Contract

**Purpose:** Create one explicit checkpoint contract for shared PPO/off-policy/FlashSAC artifacts.

**Files likely touched:**
- new `jax_rl/artifacts/` package or `jax_rl/training/artifacts.py`
- `jax_rl/training/checkpointing.py`
- `tests/test_checkpoint.py`
- deploy and video loaders later, but not necessarily in this phase

**Contract fields to introduce:**
- `artifact_version`
- `artifact_kind`
- `producer`
- `supported_consumers`
- `algo`
- `env_name`
- `obs_dim`
- `action_dim`
- `train_config`
- algorithm config block
- optional `control`
- optional `obs_schema`

**Recommended values:**
- `artifact_version`: integer, start at `1`
- `artifact_kind`: `"shared_actor_checkpoint"` for current shared checkpoint format
- `producer`: script/module name if available
- `supported_consumers`: list such as `["record_video", "policy_runner"]`

**Steps:**
1. Add a small dataclass or plain helper for shared artifact metadata.
2. Add `validate_shared_checkpoint_meta(meta)` with clear error messages.
3. Update `save_checkpoint()` to write the new fields.
4. Keep backward compatibility for old checkpoints.
5. Update tests to assert required fields exist.
6. Do not migrate TD-MPC2/PushT yet; label them later.

**Acceptance criteria:**
- New shared checkpoints have explicit artifact contract fields.
- Existing checkpoint tests pass.
- Old metadata can still be loaded by inference helpers where practical.
- Error messages tell users which file/field is missing.

**Risks:**
- Too much schema strictness can break old checkpoints.
- Too little strictness will not solve consumer ambiguity.

**Agent prompt:**

```text
Implement Phase 1 of .superpowers/plans/2026-04-25-structural-hardening.md.
Add a shared checkpoint artifact contract with validation and tests.
Maintain backward compatibility for old checkpoints. Do not change deploy behavior yet.
```

---

## Phase 2 - Add Go2 Control Metadata And Deploy Fallbacks

**Purpose:** Make Go2 deploy/sim2sim read control constants from checkpoint metadata.

**Files likely touched:**
- `jax_rl/training/checkpointing.py`
- Go2 env modules that can provide control metadata
- `deploy/go2_constants.py`
- `deploy/sim2sim_direct.py`
- `deploy/robot_interface.py`
- `deploy/policy_runner.py` or a new deploy metadata helper
- `deploy/test_policy_runner.py` or new deploy tests

**Metadata block:**

```json
"control": {
  "backend": "warp",
  "impl": "warp",
  "Kp": 20.0,
  "Kd": 0.5,
  "action_scale": 0.5,
  "policy_dt": 0.02,
  "physics_dt": 0.004,
  "contact_mode": "training",
  "torque_speed_model": false,
  "joint_order": "policy_FL_FR_RL_RR",
  "actuator_order": "unitree_FR_FL_RR_RL"
}
```

**Steps:**
1. Add an env-side helper for Go2 Warp metadata, for example `get_control_metadata()`.
2. Have checkpointing include that metadata when available.
3. Add a deploy helper such as `load_control_config(ckpt_dir)`.
4. `sim2sim_direct.py` should use metadata by default.
5. `Go2Interface(sim=True)` should not silently select archived MJX gains for Warp checkpoints.
6. Keep constants as fallback for old checkpoints, but print a loud warning.
7. Add tests with synthetic metadata:
   - Warp checkpoint picks `20.0/0.5`.
   - Missing metadata warns and falls back.
   - Explicit CLI override wins and is logged.

**Acceptance criteria:**
- New Warp checkpoints carry control metadata.
- Direct sim2sim prints the metadata source.
- Old checkpoints still run but warn.
- No path silently uses `35.0/0.1` for a Warp checkpoint.

**Risks:**
- Existing old checkpoints may not have enough metadata to infer backend.
- CLI override behavior needs to be explicit to avoid hiding mistakes.

**Agent prompt:**

```text
Implement Phase 2 of .superpowers/plans/2026-04-25-structural-hardening.md.
Add Go2 control metadata to shared checkpoints and make sim2sim/deploy read it with tested fallback behavior.
Do not change policy network inference.
```

---

## Phase 3 - Fix DomainRand Model DR Semantics

**Purpose:** Make model DR truly per-episode and compose duplicate-field specs.

**Files likely touched:**
- `jax_rl/envs/wrappers/domain_rand.py`
- new or existing DomainRand tests
- possibly Go2/Bongo DR tests

**Validated bugs:**
- Model DR changes every step for non-done envs.
- Duplicate model-field specs overwrite earlier specs.

**Design:**
1. Split model DR sampling from model construction.
2. Store sampled model replacements in wrapper state.
3. Step with the stored per-env replacements.
4. Precompute reset replacements for all envs.
5. After done is known, merge reset replacements into stored replacements only where done.
6. Compose duplicate-field specs on a per-field accumulator.

**State storage options:**
- Store replacement arrays under `state.info["_dr_model_replacements"]`.
- Avoid storing full `mjx.Model` in `state.info` if that makes tree shape large or awkward.
- Keep arrays static in structure: every randomized field gets an array with leading env dimension.

**Regression tests:**
- A synthetic one-field model randomizer stays fixed from reset through step when `done=0`.
- It changes only after truncation or env done.
- Duplicate-field specs on disjoint slices compose.
- Wrapper composition with action delay/frame stack still preserves info keys.

**Acceptance criteria:**
- Synthetic validation that previously failed now passes.
- Bongo duplicate `body_mass` specs compose correctly.
- Existing DomainRand/Warp tests pass under explicit marker.

**Risks:**
- Storing replacements in `state.info` increases state size.
- `jax.tree.map` requires reset and step info trees to match.
- Need to preserve `in_axes` construction for vmapped model stepping.

**Agent prompt:**

```text
Implement Phase 3 of .superpowers/plans/2026-04-25-structural-hardening.md.
Fix DomainRand model DR so model randomization is per-episode, not per-step, and duplicate-field specs compose.
Start with synthetic tests. Preserve current public API.
```

---

## Phase 4 - Clarify Resume Semantics

**Purpose:** Stop implying exact continuation when the code performs warm-start.

**Files likely touched:**
- training scripts CLI help
- README/docs
- `jax_rl/training/checkpointing.py`
- `jax_rl/training/offpolicy_loop.py`
- `scripts/train_flashsac.py`
- tests around resume behavior

**Decision required from owner:**
- Does `--resume` mean exact continuation?
- Or should it be renamed/documented as warm-start?

**Recommended short-term path:**
1. Keep `--resume` for compatibility.
2. Add `--resume-mode` with values:
   - `warm_start`
   - `full` reserved or not yet implemented
3. Print a warning when current behavior is warm-start.
4. On warm-start, use restored-policy exploration for replay warmup instead of random uniform actions.

**Recommended long-term full resume:**
- Persist:
  - replay buffer arrays
  - buffer pointer and size
  - episode IDs
  - env state
  - PRNG key
  - episode tracker state
  - `last_eval_eps`
  - best eval score
  - loop counters
  - W&B run id if needed

**Acceptance criteria for short-term:**
- Help text and docs no longer claim exact continuation.
- Warm-start behavior is explicit in logs.
- FlashSAC/off-policy resumed warmup does not use random actions by default.
- Tests verify resume-mode messaging and warmup action path.

**Acceptance criteria for full resume:**
- A small deterministic training run resumed from checkpoint matches uninterrupted training within expected deterministic tolerance.

**Risks:**
- Full replay buffer snapshots can be large.
- Exact env-state serialization may be backend-specific.
- W&B continuity has separate operational constraints.

**Agent prompt:**

```text
Implement the short-term path of Phase 4 in .superpowers/plans/2026-04-25-structural-hardening.md.
Do not implement full replay-buffer persistence yet. Make warm-start semantics explicit and use restored-policy warmup on resume.
```

---

## Phase 5 - Add Backend Metadata And Logging Hooks

**Purpose:** Remove Go2/locomotion imports from generic training loops.

**Files likely touched:**
- `jax_rl/training/env_setup.py`
- `jax_rl/training/offpolicy_loop.py`
- `jax_rl/training/metrics_logger.py`
- `jax_rl/training/checkpointing.py`
- Go2 env modules
- curriculum logging module

**Hook shape:**

```python
@dataclass
class EnvBundle:
    ...
    metadata_provider: Callable[[], dict] | None = None
    train_metrics_provider: Callable[[Any], dict] | None = None
    visualization_provider: Callable[[Any], dict] | None = None
    debug_dump_provider: Callable[[Any, int], None] | None = None
```

**Steps:**
1. Extend `EnvBundle` with optional hooks.
2. Wire Go2 terrain metrics/images through the hooks.
3. Remove direct locomotion imports from `offpolicy_loop.py`.
4. Move checkpoint obs schema/DR metadata discovery behind env metadata provider where feasible.
5. Keep backward compatibility while hooks are introduced.

**Acceptance criteria:**
- `offpolicy_loop.py` no longer imports `jax_rl.envs.locomotion.*`.
- Go2 terrain metrics still log.
- Checkpoint metadata still includes obs schema where available.
- Non-Go2 envs do not need dummy locomotion imports.

**Risks:**
- Hook return values must remain JAX-safe where called from hot paths.
- Avoid making `EnvBundle` too broad; hooks should be optional and narrow.

**Agent prompt:**

```text
Implement Phase 5 of .superpowers/plans/2026-04-25-structural-hardening.md.
Move terrain/curriculum logging out of generic offpolicy_loop via EnvBundle hooks.
Do not change training algorithm behavior.
```

---

## Phase 6 - Public Entrypoint Registry

**Purpose:** Keep CLI docs, parser tests, and public scripts in sync.

**Files likely touched:**
- new `jax_rl/training/public_scripts.py` or `docs/scripts/public_scripts.py`
- `docs/scripts/gen_cli_reference.py`
- `tests/test_docs_drift.py`
- scripts missing `build_parser()`

**Steps:**
1. Define one registry of public scripts.
2. Use it in CLI reference generation.
3. Use it in tests.
4. Add `build_parser()` to intended public scripts.
5. Decide whether diagnostics get their own section.
6. Either document TD-MPC2 as experimental or include it fully.

**Acceptance criteria:**
- A test fails when a public script lacks `build_parser()`.
- Generated docs include exactly the registry entries.
- TD-MPC2 support claim matches actual artifact support.

**Risks:**
- Some scripts may be intentionally internal.
- Diagnostics may need a separate registry to avoid bloating user docs.

**Agent prompt:**

```text
Implement Phase 6 of .superpowers/plans/2026-04-25-structural-hardening.md.
Create a single public script registry used by docs generation and tests.
Do not change training behavior.
```

---

## Phase 7 - Thin Large Scripts

**Purpose:** Reduce script complexity after contracts and tests are stable.

**First target:** `scripts/train_tdmpc2.py`

**Reason:**
- It is public-ish, large, experimental, and owns artifact helpers that differ from shared checkpointing.

**Suggested split:**
- `jax_rl/algos/tdmpc2_runtime.py`: already exists; extend if needed.
- `jax_rl/training/tdmpc2_train.py`: reusable train/eval orchestration.
- `scripts/train_tdmpc2.py`: parser and call only.
- `scripts/eval_tdmpc2.py`: parser and call only.

**Do not do this before:**
- artifact contract is clear
- TD-MPC2 maturity label is decided
- current TD-MPC2 tests are preserved

**Acceptance criteria:**
- CLI behavior unchanged.
- Tests still pass.
- Script line count drops substantially.
- Reusable functions can be imported by tests without running CLI code.

**Risks:**
- TD-MPC2 is active research; over-structuring can slow iteration.
- Refactor can hide semantic changes if tests are weak.

**Agent prompt:**

```text
Implement Phase 7 for train_tdmpc2.py only.
Preserve behavior. Move reusable helpers into importable modules, keep the script as CLI glue.
Run existing TD-MPC2 tests and targeted smoke tests.
```

---

## Phase 8 - Maturity Labels In Docs And Tooling

**Purpose:** Make stable, deploy-critical, experimental, and archived surfaces obvious.

**Files likely touched:**
- README
- docs index/reference pages
- CLI reference
- checkpoint artifact metadata

**Labels:**
- Stable framework
- Deploy-critical
- Experimental
- Diagnostics
- Archived

**Examples:**
- SAC/TD3/FastSAC/FastTD3 shared loop: stable framework.
- Go2 Warp deploy path: deploy-critical.
- TD-MPC2: experimental until artifact support and benchmarks stabilize.
- PushT: research workbench unless promoted.
- Old MJX Go2: archived.

**Acceptance criteria:**
- README has a maturity table.
- CLI docs do not imply unsupported consumer paths.
- Artifact metadata can express experimental vs shared checkpoint contracts.

**Risks:**
- Labels can become stale unless tied to tests/docs generation.

**Agent prompt:**

```text
Implement Phase 8 of .superpowers/plans/2026-04-25-structural-hardening.md.
Add concise maturity labels to README/docs and align script/artifact wording.
Do not change runtime behavior.
```

---

## Phase 9 - Optional Full Resume

**Purpose:** Implement exact continuation only if the owner confirms it is worth the cost.

**Prerequisite decision:** Owner says exact continuation matters.

**Files likely touched:**
- `jax_rl/buffers/jax_replay_buffer.py`
- checkpointing
- off-policy loop
- FlashSAC loop
- tests

**Implementation outline:**
1. Add replay buffer serialization.
2. Add loop-state dataclass.
3. Save full loop state periodically.
4. Restore full loop state when `resume_mode=full`.
5. Keep `warm_start` as cheaper default if desired.

**Acceptance criteria:**
- Deterministic uninterrupted run equals resumed run for a tiny synthetic env.
- Replay buffer content, pointer, size, and episode IDs roundtrip.
- PRNG and env state roundtrip.

**Risks:**
- Large checkpoint size.
- Backend-specific env state serialization.
- More complicated migration story for old checkpoints.

**Agent prompt:**

```text
Implement Phase 9 only after owner approval.
Add full resume as a separate resume mode, preserving warm_start behavior.
Prove exact continuation with a deterministic small test.
```

---

## Recommended Sequencing

Best order:

1. Phase 0 - safe test surface.
2. Phase 3 - DomainRand fixes, because this affects real training semantics.
3. Phase 1 - shared artifact contract.
4. Phase 2 - Go2 control metadata and deploy fallbacks.
5. Phase 4 short-term - resume semantics clarity and restored-policy warmup.
6. Phase 5 - backend hooks.
7. Phase 6 - public script registry.
8. Phase 8 - maturity labels.
9. Phase 7 - thin large scripts.
10. Phase 9 - full resume only if approved.

Reason for this order:

- Phase 0 makes validation reliable.
- Phase 3 fixes confirmed training semantics.
- Phases 1-2 make deploy/artifact boundaries explicit.
- Phase 4 reduces misleading behavior without a huge persistence project.
- Phases 5-8 clean structure once contracts exist.
- Phase 7 is safer after the semantics are pinned.

---

## Handoff Template For Agents

Every agent implementing a phase should end with:

```text
Phase completed:
- Phase:
- Files changed:
- Behavior changes:
- Tests run:
- Tests not run and why:
- Compatibility notes:
- Follow-up risks:
```

Every agent reviewing a phase should answer:

```text
Review result:
- Contract preserved?
- Go2 deploy path affected?
- Public CLI affected?
- Old checkpoints affected?
- Default tests reliable?
- Any untested GPU/Warp behavior?
```
