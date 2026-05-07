# Codex Audit Notes

## Fresh Audit Rerun - 2026-04-27

Repo: `/home/stevenman/Desktop/Work/Research/jax-learning`
Branch observed: `new_slate_linen`
Audit mode: comprehensive rerun after the Claude-agent update pass, with prior
findings rechecked against current code and a fresh unbiased test/docs/deploy
scan.

This section is the current source of truth. The older 2026-04-27 refresh and
2026-04-25 notes below are retained as historical context only.

### Current Working Tree

Observed before this edit:

- Untracked: `.superpowers/plans/2026-04-25-structural-hardening.md`
- Untracked: `codex_audit.md`

The previously observed local TD-MPC2 modifications are no longer present as
working-tree changes. Recent history contains targeted fixes for the prior
audit: DomainRand persistence, sim2sim control metadata, artifact contracts,
backend `num_envs`, PPO eval dispatch, TD-MPC2 package split, CLI docs, and
maturity docs.

### Executive Status

The Claude-agent update pass addressed several high-value findings, but the
audit should remain open. The repo is materially healthier than the previous
snapshot, especially around artifact contracts, TD-MPC2 packaging, and direct
sim2sim metadata. The remaining risks are more specific:

- default pytest currently fails,
- real/DDS deploy still ignores checkpoint control metadata,
- DomainRand still has a duplicate-field composition bug,
- test marker taxonomy and CI are still too thin,
- docs drift checks do not yet share the full public-script registry.

### Prior Finding Disposition

| Prior issue | Current status | Notes |
|---|---|---|
| DomainRand model DR was per-step | Mostly addressed | `DomainRandWrapper` now persists sampled model DR fields in `state.info` and uses them for active steps. |
| DomainRand duplicate field overwrite | Still open | `_sample_dr_fields()` still starts each spec from the original model field and overwrites `replacements[spec.field]`. |
| Direct sim2sim used archived MJX PD gains | Addressed for `sim2sim_direct.py` | It reads `meta["control"]` and falls back loudly for legacy checkpoints. |
| Real/DDS deploy should be metadata-driven | Still open | `deploy_go2.py` constructs `Go2Interface` without checkpoint control metadata; `robot_interface.py` still uses hard-coded gains/action scale. |
| Artifact kind/version missing | Addressed | `artifact_contract.py` adds kind/version stamping, readers, allowlist checks, and file validators. |
| Consumers failed opaquely on TD-MPC2 artifacts | Addressed for key consumers | `load_actor_for_inference()` and `deploy.PolicyRunner` validate artifact kind and redirect away from TD-MPC2. |
| Backend `cfg.num_envs`/actual `num_envs` mismatch | Mostly addressed | Hot off-policy and PPO paths now use `bundle.num_envs`; one manual `EnvBundle` test failure exposes that the dataclass default remains a footgun. |
| PPO Python-loop eval used MJX evaluator for gym | Addressed | `train_ppo.py` dispatches to `evaluate_gym` when `bundle.backend_kind == "gym"`. |
| TD-MPC2 monolith/refactor incomplete | Addressed | `jax_rl/algos/tdmpc2/` is now a focused package; `tdmpc2_old.py` is gone. |
| Public CLI/docs omitted TD-MPC2 tools | Partly addressed | CLI generator includes TD-MPC2 tools, but docs drift tests still check only six train scripts. |
| Test partitioning / CI gate unreliable | Still open | Only `slow` is configured, default pytest still runs a broad heavy lane, and GitHub Actions appears docs-only. |
| Deploy safety boundary thin | Still open | Manual Enter gate exists; runtime finite checks, fall stop, command clamps, and robust stand-down/finally are still missing. |

### Resolved Or Improved Areas

#### DomainRand Per-Episode Persistence

`jax_rl/envs/wrappers/domain_rand.py` now persists sampled model DR fields under
`state.info["_dr_dr_fields"]` after reset and reuses those fields for active
steps. Fresh reset fields are merged only for envs whose episodes ended. That
addresses the prior high-severity concern that non-done envs saw fresh physics
every step.

Remaining caveat: there is no focused regression test proving a non-done env's
model field remains stable across multiple steps.

#### Artifact Contract

`jax_rl/training/artifact_contract.py` now defines:

- `KIND_SHARED_ACTOR`
- `KIND_TDMPC2`
- `KIND_PUSHT_LEGACY`
- `KIND_LEGACY_SHARED_ACTOR`
- `stamp_meta()`
- `assert_artifact_kind()`
- `validate_shared_actor_files()`
- `validate_tdmpc2_files()`

Shared checkpoint writers stamp kind/version metadata, and TD-MPC2 writers stamp
`tdmpc2_v1`. Shared consumers validate artifact kind before loading files, so a
TD-MPC2 checkpoint now fails with a clear redirect instead of a confusing
missing-file error.

#### Direct Sim2sim Control Metadata

`deploy/sim2sim_direct.py` now reads deploy-critical control values from
checkpoint `meta["control"]`:

- `Kp`
- `Kd`
- `action_scale`
- `physics_dt`
- `policy_dt`
- `contact_mode`

For legacy checkpoints it falls back to `deploy/go2_constants.py` with an
explicit warning that the fallback values are archived MJX values.

#### Backend Hot-Path `num_envs`

`run_offpolicy_loop()` and `scripts/train_ppo.py` now use the backend bundle's
effective `num_envs` for buffers, trackers, action shapes, rollout sizes, and
PPO config. This directly addresses the earlier gym backend cap mismatch in the
normal constructed-backend path.

#### TD-MPC2 Package Split

TD-MPC2 is no longer a single `tdmpc2_old.py` file. The package now contains
focused modules:

- `agent.py`
- `losses.py`
- `mppi.py`
- `networks.py`
- `runtime.py`
- `__init__.py`

The public package re-exports preserve test/import compatibility.

### Current P0 Findings

#### P0: Default Pytest Is Red

Command run:

```bash
uv run python -m pytest -q --maxfail=10
```

Result:

```text
2 failed, 745 passed, 46 skipped, 15 deselected, 2 warnings in 579.33s
```

Failures:

- `tests/test_go2_warp_curriculum_env.py::test_generated_scene_file_written`
- `tests/test_offpolicy_loop.py::test_run_offpolicy_loop_stub_env_cpu`

The default lane now produces a useful result, but it is not green and takes
about 9m39s on this machine.

Proposed solution:

- Fix these two failures before treating any later audit as closed.
- Add a shorter explicit smoke command for local/CI use, separate from the full
  default suite.
- Keep a full suite command documented, but do not let it be the only health
  signal if it takes nearly ten minutes and includes heavy integration coverage.

#### P0: Curriculum Generated Scene Test Is Stale

`WarpJoystickCurriculum` now writes generated scene XMLs to a system temp file
using `tempfile.mkstemp(prefix=f"go2_curriculum_scene_{os.getpid()}_", ...)`
and tracks them for cleanup at process exit. This is documented in the module
docstring as intentional behavior to avoid source-tree pollution and
multiprocess races.

The failing test still expects a source-tree file named
`jax_rl/envs/locomotion/xmls/_generated_curriculum_scene_<pid>.xml`.

Affected files:

- `jax_rl/envs/locomotion/go2_warp_curriculum.py`
- `tests/test_go2_warp_curriculum_env.py`

Proposed solution:

- Expose the generated `scene_path` on the env as an internal diagnostic field,
  for example `self._generated_scene_path`.
- Update the test to assert that path exists, lives under the system temp dir,
  and is registered for cleanup.
- Do not revert to writing generated XML into the source tree.

#### P0: Manual `EnvBundle` Construction Can Still Lie About `num_envs`

The failing off-policy stub test manually constructs `EnvBundle(...)` without
`num_envs`, so the dataclass default remains `num_envs=1`. The stub env emits
two rewards/dones, and `run_offpolicy_loop()` correctly trusts
`env_bundle.num_envs`, creating a one-slot `EpisodeTracker`. The first step then
fails with:

```text
ValueError: non-broadcastable output operand with shape (1,) doesn't match the broadcast shape (2,)
```

This does not invalidate the normal backend-builder fix, but it exposes a
remaining contract weakness: `EnvBundle` can be manually instantiated with
metadata inconsistent with `env_state`.

Affected files:

- `jax_rl/training/env_bundle.py`
- `tests/test_offpolicy_loop.py`

Proposed solution:

- Fix the test by passing `num_envs=NUM_ENVS`.
- Add `EnvBundle.__post_init__()` validation where feasible: if `env_state.obs`,
  `reward`, or `done` has an obvious leading batch dimension, require it to
  match `num_envs`.
- Consider removing the `num_envs=1` default and making `num_envs` explicit.

#### P0: Real/DDS Deploy Still Does Not Consume Checkpoint Control Metadata

The direct sim2sim path is fixed, but the real/DDS deploy path is not.

`deploy/deploy_go2.py` loads `PolicyRunner` and `ObsBuilder`, then constructs:

```python
iface = Go2Interface(sim=sim, interface=interface)
```

`Go2Interface.__init__()` still chooses gains from constants:

```python
self.kp = KP_SIM if sim else KP_REAL
self.kd = KD_SIM if sim else KD_REAL
```

`send_action()` still uses the module-level `ACTION_SCALE`.

Why this matters:

- A checkpoint can now be self-describing, but the highest-risk deploy path
  ignores that metadata.
- DDS sim mode still uses archived MJX `KP_SIM/KD_SIM`.
- Real mode currently happens to match Warp gains, but that is by constant
  coincidence rather than artifact contract.

Proposed solution:

- Add a small control-metadata resolver shared by `sim2sim_direct.py` and
  `deploy_go2.py`.
- Pass `kp`, `kd`, and `action_scale` into `Go2Interface`.
- Print the resolved values in both sim and real modes.
- For real mode, require explicit operator confirmation if checkpoint metadata
  is missing or disagrees with the deploy defaults.

### Current P1 Findings

#### P1: DomainRand Duplicate-Field Composition Is Still Wrong

`_sample_dr_fields()` still computes `field_data = getattr(model, spec.field)`
for each spec, applies that spec to the original field, then stores
`replacements[spec.field] = new_field`.

If two model specs target different slices of the same field, the later spec
overwrites the earlier replacement. `Go2BongoHandstand` currently has two specs
targeting `body_mass` (`board_mass` slice and full `body_mass`), so this is not
only theoretical.

Affected file:

- `jax_rl/envs/wrappers/domain_rand.py`

Proposed solution:

- Change the source field to `replacements.get(spec.field, getattr(model, spec.field))`.
- Add a synthetic test with two specs targeting disjoint slices of the same
  field and assert both replacements survive.
- Add one current-env regression covering the `Go2BongoHandstand` `body_mass`
  specs if practical.

#### P1: Test Marker Taxonomy And CI Are Still Underbuilt

`pyproject.toml` still defines only:

```toml
markers = [
    "slow: marks tests as slow / GPU-bound (deselected by default; run with -m slow)",
]
addopts = "-m 'not slow'"
```

GitHub Actions still appears docs-only:

- `.github/workflows/docs.yml` installs docs deps and deploys MkDocs.
- No CPU pytest workflow is present.
- No lint/type/pre-commit workflow is present.

Proposed solution:

- Add markers such as `unit`, `docs`, `integration`, `gpu`, `warp`, `go2`,
  `deploy`, `network`, and `artifact`.
- Make the default local/CI lane CPU-safe and fast.
- Add a GitHub Actions CPU workflow running the smoke lane.
- Keep Warp/GPU/Go2 tests in an explicit manual or self-hosted job.

#### P1: Docs Drift Guard Does Not Share The Full Public Script Registry

`docs/scripts/gen_cli_reference.py` now includes TD-MPC2 tools, which is an
improvement. But `tests/test_docs_drift.py::TRAIN_SCRIPTS` still checks only:

- `scripts/train_sac.py`
- `scripts/train_td3.py`
- `scripts/train_fast_sac.py`
- `scripts/train_fast_td3.py`
- `scripts/train_flashsac.py`
- `scripts/train_ppo_fast.py`

That means CLI docs can drift for `train_tdmpc2.py`, `eval_tdmpc2.py`,
`record_video_tdmpc2.py`, `check_tdmpc2_determinism.py`, `train_ppo.py`, and
other intended public scripts without the drift test noticing.

Proposed solution:

- Create one importable public-script registry used by both the generator and
  the drift test.
- Require every public script in that registry to expose `build_parser()`.
- Decide whether diagnostics live in the main CLI reference or a separate
  diagnostics reference, then encode that decision in the registry.

#### P1: Artifact Contract Is Implemented But Not Yet Complete As A Product Boundary

The core artifact contract exists and consumers validate major kinds. Remaining
gaps are mostly productization:

- docs should list required files for each artifact kind,
- old checkpoints should have an explicit support policy,
- TD-MPC2 tooling should validate `tdmpc2_v1` files before loading,
- deploy metadata should be treated as part of the contract for deployable
  policies, not a best-effort optional block.

Proposed solution:

- Add `docs/reference/artifacts.md` or a checkpoint-artifacts subsection.
- Add a metadata validator for deployable shared-actor checkpoints that checks
  `obs_schema` and `control`.
- Add TD-MPC2 runtime validation through `validate_tdmpc2_files()`.

#### P1: Deploy Safety Envelope Is Still Thin

The real-robot path has a manual Enter gate and a stand-up FSM, but the policy
loop still lacks an explicit runtime safety envelope.

Observed gaps:

- no finite obs/action guard at the deploy boundary,
- no command clamp beyond CLI input,
- no joint/IMU validity stop,
- no fall/tilt stop,
- no default max-duration guard for real hardware,
- no broad `try/finally` around stand-down/zero-torque cleanup.

Proposed solution:

- Add a `SafetyConfig` for deploy.
- Clamp velocity/yaw commands before constructing observations.
- Reject non-finite observations and actions.
- Stop on excessive tilt, stale state, out-of-range joints, or large action
  jumps.
- Wrap policy execution in `try/finally` and stand down or zero torque on every
  exit path.

### Recommended Execution Order

1. Fix the two default pytest failures.
   These are immediate trust blockers and should be small, bounded changes.

2. Push deploy control metadata through `deploy_go2.py` and `Go2Interface`.
   `sim2sim_direct.py` is now contract-driven; real/DDS deploy should match.

3. Fix DomainRand duplicate-field composition and add regression tests.
   The per-episode bug is mostly closed, but multi-spec same-field behavior is
   still wrong.

4. Add the CPU smoke CI lane and marker taxonomy.
   Keep the first gate small and reliable before broadening.

5. Unify public script registry for CLI docs and drift tests.
   This prevents the docs generator and drift checker from diverging again.

6. Productize artifact/deploy contracts in docs.
   Define artifact kinds, required files, legacy behavior, and deploy metadata
   requirements.

7. Add the real deploy safety envelope.
   Treat this as deploy-critical, not general framework cleanup.

### Verification Performed In This Rerun

Commands run:

```bash
git status --short
git log --oneline -30
git diff --stat
uv run python -m pytest tests/test_artifact_contract.py tests/test_checkpoint.py tests/test_env_bundle.py tests/test_docs_drift.py tests/test_offpolicy_loop.py deploy/test_policy_runner.py -q
uv run python -m pytest -q --maxfail=10
```

Results:

- Focused audit suite: `40 passed, 9 deselected, 1 failed`.
- Focused failure: `tests/test_offpolicy_loop.py::test_run_offpolicy_loop_stub_env_cpu`.
- Default suite: `745 passed, 46 skipped, 15 deselected, 2 failed`.
- Default failures:
  - `tests/test_go2_warp_curriculum_env.py::test_generated_scene_file_written`
  - `tests/test_offpolicy_loop.py::test_run_offpolicy_loop_stub_env_cpu`

### Bottom Line

The prior audit was acted on in good faith and many high-impact items are now
closed or mostly closed. The repo's current problem is no longer "large areas
are unaudited"; it is that a handful of contracts are halfway implemented and
need to be tightened:

- `EnvBundle.num_envs` must be impossible to lie about accidentally.
- Generated terrain scene files now live in temp space, and tests should encode
  that current behavior.
- Deployable checkpoints have useful control metadata, but real/DDS deploy does
  not use it yet.
- DomainRand now persists per episode, but duplicate field composition still
  drops earlier replacements.
- Artifact and CLI contracts exist, but docs/tests need one shared source of
  truth.

## Superseded Fresh Audit Refresh - 2026-04-27

Repo: `/home/stevenman/Desktop/Work/Research/jax-learning`
Branch observed: `new_slate_linen`
Audit mode: comprehensive refresh from current repo state, with four read-only subsystem reviews.

This section supersedes the older 2026-04-25 audit notes below. The older notes
are retained as historical context, but the findings and priorities in this
refresh reflect the current tree.

### Scope

The repo changed substantially since the prior audit. The env-backend refactor
landed, PushT moved toward the generic gym backend, TD-MPC2 reached paper-band
benchmarks and started a package split, and off-policy resume behavior gained a
policy-warmup mitigation.

The goal of this refresh is not broad refactoring. The goal is to identify what
is now healthy, what is still risky, and which contracts need to become explicit
before future agents or contributors can safely build on this system.

### Current State Summary

The repo is now best described as a JAX-native robot-learning RL framework with
three active surfaces:

- Stable-ish framework core: algorithms, networks, buffers, configs, shared
  off-policy loop, normalization, checkpointing, eval helpers.
- Deploy-critical Go2 stack: Warp Unitree MJCF envs, observation schema,
  sim2sim/direct deploy, policy runner, real-hardware interfaces.
- Active research workbench: TD-MPC2, terrain curriculum, PushT shape
  generalization, PPOContraction, FlashSAC variants.

The high-level architecture is still sound: environment construction is
separated from training glue, and algorithms mostly remain pure JAX math with no
environment imports. The largest structural improvement is that backend
abstraction is no longer just a plan: `EnvBundle`, backend detection,
`env_backends`, and `gym_backend` are present.

The main risk is that the public surface now implies more stability than some
subsystems actually have. Backend-agnostic training is partly real, but several
loops still assume MJX/Warp shapes and CUDA availability. Checkpointing is
centralized for shared algos, but artifact contracts are not versioned and
TD-MPC2 remains separate. Deploy observation schema improved substantially, but
deploy control metadata is still hardcoded.

### Working Tree Notes

The workspace was already dirty before this audit. I did not revert or clean
those changes.

Observed pre-existing local state:

- Modified: `.superpowers/plans/2026-04-26-tdmpc2-refactor.md`
- Modified: `jax_rl/algos/tdmpc2/__init__.py`
- Modified: `jax_rl/algos/tdmpc2_old.py`
- Untracked: `.superpowers/plans/2026-04-25-structural-hardening.md`
- Untracked: `jax_rl/algos/tdmpc2/networks.py`
- `codex_audit.md` was already untracked and is now updated by this audit.

Recent history shows active structural work:

- `feat(training): gym backend + PushT factory + gym eval path`
- `refactor(scripts): route train_ppo + train_ppo_fast through bundle dispatch`
- `feat(record_video): backend dispatch + gym recorder for PushT et al.`
- `feat(tdmpc2): HopperHop preset`
- `refactor(tdmpc2): create sub-package skeleton with re-exports`
- `plan(tdmpc2-refactor): relax validation gate, drop byte-ID expectation`

### Major Improvements Since Prior Audit

#### Backend Abstraction Is Real Now

`jax_rl/training/env_bundle.py` defines an explicit backend bundle, and
`jax_rl/training/env_backends/` dispatches MJX/Warp and gym environments through
a registry. `gym_backend.py` wires PushT plus gymnasium MuJoCo/classic-control
envs into the same high-level training path.

This is the largest structural improvement. The old audit recommendation to
build backend adapters is no longer hypothetical.

Remaining caveat: the implementation is not consistently backend-neutral yet.
Several loops still use `cfg.num_envs` instead of `env_bundle.num_envs`, banners
still say MuJoCo Playground for all backends, PPO eval still calls the MJX-style
`evaluate()` path, and generic checkpointing still imports MuJoCo Playground to
load env metadata.

#### PushT Is Moving Into The Common Training Surface

README and `.context/TODO.md` now point to `train_sac.py --env PushT` as the
future path, with `scripts/train_pusht.py` marked deprecated. The gym backend
encapsulates PushT defaults, action repeat, observation normalization, and
action scaling.

This is a good direction: PushT should be a backend/env configuration, not a
parallel training universe. The remaining step is to validate parity with the
old script before deleting or archiving `train_pusht.py`.

#### Resume Behavior Improved, But Is Still Warm-Start

The prior audit identified resume as a major misnomer. That is partly addressed:
shared off-policy scripts and FlashSAC now expose
`--resume-warmup {policy,random}` and default to `policy`, avoiding the old
behavior where a restored policy immediately refilled its buffer with random
actions.

This is a useful mitigation, and the documented FastSAC Go2 result suggests it
solves the most visible resume drop there.

It is still not full resume. Replay buffer, env state, PRNG stream, episode
tracker, best-eval state, W&B/logging continuity, and some loop state are
recreated.

#### Deploy Observation Schema Is Much Stronger

Shared checkpoints now try to save `obs_schema` when envs expose `_obs_groups`,
and `deploy/obs_builder.py` reads the schema to build deploy observations by
term name. Tests cover default layout, schema reordering, dropped accelerometer,
unknown terms, checkpoint schema, and fallback behavior.

This materially reduces train/deploy observation drift risk. It is one of the
clearest resolved areas from the old audit.

#### TD-MPC2 Has Advanced From Prototype To Active Research Surface

TD-MPC2 now has benchmark evidence in `.context/TODO.md` and journals:
CheetahRun paper-match, HumanoidRun paper-band, HopperHop strong early result,
eval-key isolation, dedicated eval/video helpers, and several correctness
fixes.

It is also mid-refactor from `tdmpc2_old.py` into `jax_rl/algos/tdmpc2/`, with
`networks.py` already present locally and `__init__.py` preserving imports.

This is progress, but the refactor is currently incomplete and local changes are
uncommitted.

### Highest-Priority Findings

#### P0: DomainRand Model Randomization Still Appears Per-Step

`DomainRandWrapper.step()` builds a fresh randomized `dr_model` from `reset_rng`
on every environment step, then uses that fresh model for both reset candidates
and the active `env.step` path for all envs.

That means active non-done episodes can see model physics change every step,
even though the wrapper and docs describe per-episode model randomization.

Affected file:

- `jax_rl/envs/wrappers/domain_rand.py`

Why this matters:

- It changes the physics process from per-episode domain randomization into
  per-timestep stochastic dynamics.
- It can distort Go2 training conclusions under `--reset-mode per_step`.
- It is exactly the kind of bug that can produce apparently robust policies
  that are robust to the wrong perturbation process.

Recommended fix:

- Persist sampled model DR parameters per env episode.
- Use each env's persisted randomized model for active steps.
- Resample only for envs whose episode resets.
- Add a synthetic regression test proving a non-done env sees stable model
  parameters across reset -> step -> step.

#### P0: Test Partitioning Is Still Not A Reliable Signal

`pyproject.toml` registers only a `slow` marker and defaults to
`-m 'not slow'`. That is too coarse for the current repo.

A read-only test audit found about 797 collected tests with only about 15
deselected by `slow`. Go2/Warp/MJX/GPU tests remain in the default collection
surface.

Verification during this audit:

- `uv run python -m pytest tests/test_docs_drift.py tests/test_env_bundle.py tests/test_checkpoint.py tests/test_tdmpc2.py -q`
  failed during collection because `tests/test_tdmpc2.py` constructs a
  module-level `jax.random.PRNGKey`, triggering CUDA allocation and an OOM.
- `JAX_PLATFORMS=cpu uv run python -m pytest tests/test_docs_drift.py tests/test_env_bundle.py tests/test_checkpoint.py tests/test_tdmpc2.py -q`
  produced `67 passed, 1 failed, 8 deselected`.
- The CPU-forced failure was `tests/test_env_bundle.py::test_env_bundle_flat_obs_cheetahrun`;
  CheetahRun construction still asked MuJoCo/JAX for a CUDA backend even though
  only CPU was available.

Why this matters:

- The default test suite is not a hermetic local or CI gate.
- Test collection itself can depend on CUDA memory availability.
- CI currently appears docs-only in `.github/workflows/docs.yml`; there is no
  pytest/lint/type/coverage gate.

Recommended fix:

- Add markers: `unit`, `docs`, `integration`, `gpu`, `warp`, `go2`, `deploy`,
  `network`, and maybe `artifact`.
- Make default pytest CPU/hermetic only.
- Move GPU/Warp/Go2 integration tests behind explicit markers.
- Remove module-level JAX device allocation in test files.
- Add at least one GitHub Actions CPU test workflow and keep docs deployment
  separate.

#### P0: Sim2sim Still Uses Archived MJX PD Gains

Warp training and real deploy use `Kp=20.0`, `Kd=0.5`, but direct sim2sim still
imports `KP_SIM=35.0`, `KD_SIM=0.1`, which are explicitly marked archived MJX
values in `deploy/go2_constants.py`.

Affected files:

- `deploy/go2_constants.py`
- `deploy/sim2sim_direct.py`
- `jax_rl/envs/locomotion/go2_warp_joystick.py`

Why this matters:

- Sim2sim is supposed to validate a Warp-trained policy against the Unitree
  MJCF.
- Running it with archived MJX gains can produce misleading failures or
  misleading successes.

Recommended fix:

- Save deploy-critical control metadata in checkpoint `meta.json`.
- Read that metadata in `sim2sim_direct.py` and deploy code.
- Keep constants only as fallback for old checkpoints, with a loud warning.

Suggested metadata block:

```json
{
  "artifact_kind": "shared_actor_checkpoint",
  "artifact_version": 1,
  "control": {
    "kp": 20.0,
    "kd": 0.5,
    "action_scale": 0.5,
    "policy_dt": 0.02,
    "physics_dt": 0.004,
    "impl": "warp",
    "contact_mode": "training",
    "torque_speed_model": false,
    "joint_order": "policy_FL_FR_RL_RR",
    "action_order": "policy_FL_FR_RL_RR"
  }
}
```

#### P1: Backend-Agnostic Training Has Shape/Dispatch Holes

The env-backend refactor is directionally right, but some downstream code still
uses old assumptions.

Examples:

- `gym_backend.py` can cap `cfg.num_envs` to `os.cpu_count()`, but
  `run_offpolicy_loop()` uses `cfg.num_envs` for buffer construction, action
  shape, trackers, loop counters, and logging instead of `env_bundle.num_envs`.
- `train_ppo.py` now constructs an `EnvBundle`, but eval still calls
  `jax_rl.utils.eval.evaluate()`, which expects a JAX/MJX-style env, not a gym
  vector env.
- `train_ppo_fast.py` correctly guards `backend_kind != "mjx"`, but it still
  reports old MuJoCo Playground language.
- `offpolicy_loop.py` directly imports locomotion curriculum logging.
- `checkpointing.py` imports MuJoCo Playground registry to discover DR specs and
  obs schema during generic checkpoint save.

Recommended fix:

- Treat `EnvBundle` as the source of runtime truth, including actual `num_envs`.
- Add optional provider hooks to `EnvBundle` for checkpoint metadata, train
  metrics, render/eval behavior, and visualization.
- Route PPO Python-loop eval through `evaluate_gym` when `backend_kind == "gym"`.
- Move locomotion curriculum logging behind an env-provided metrics hook.

#### P1: Artifact Contracts Are Still Implicit And Split

Shared algos write:

- `meta.json`
- `metrics.csv`
- `actor_params.npy`
- `orbax/`

TD-MPC2 writes a separate artifact shape:

- `meta.json`
- `actor_params.npz`
- `world_model_params.npz`
- optional `best/`

The docs still mostly describe the shared artifact as if it is universal and as
if `orbax/` means full resume. TD-MPC2 is a valid exception, but it needs to be
labeled as an exception through a formal contract.

Recommended fix:

- Add `artifact_kind` and `artifact_version` to every checkpoint.
- Document required files per kind.
- Add artifact validators for shared actor checkpoints and TD-MPC2 checkpoints.
- Make consumers fail clearly when passed an unsupported artifact.

#### P1: Public CLI And Docs Surface Drifted Again

Current docs drift is narrower than before, but still important.

Examples:

- `docs/scripts/gen_cli_reference.py` excludes `train_tdmpc2.py`,
  `eval_tdmpc2.py`, `check_tdmpc2_determinism.py`, `record_video_tdmpc2.py`,
  and `train_ppo.py`.
- `train_ppo.py` is documented as live but does not expose `build_parser()`, so
  generated CLI docs cannot reflect it.
- `scripts/train_tdmpc2.py` docstring still says `uv run python train_tdmpc2.py`
  instead of `uv run python scripts/train_tdmpc2.py`.
- `docs/reference/architecture.md` still documents `train_pusht.py` as a
  standalone manipulation script rather than a deprecated reference path.
- `docs/api/index.md` says "Six" algorithms and omits newer surfaces.
- `load_actor_for_inference()` returns 4 values but its type hint/docstring say
  3.

Recommended fix:

- Add a single `PUBLIC_SCRIPTS` registry shared by CLI docs generation and docs
  drift tests.
- Give every intended public script an importable `build_parser()`.
- Decide whether diagnostics belong in the main CLI reference or a diagnostics
  reference.
- Decide whether TD-MPC2 is public, experimental-public, or internal, then align
  README, API docs, CLI docs, artifact docs, and video tooling.

#### P1: Deploy Safety Boundary Is Thin

The real-robot deploy path has a manual Enter gate, but the policy loop does
not yet look like a hardened runtime boundary.

Observed gaps:

- No explicit finite obs/action guard at the deploy boundary.
- No command clamp policy beyond what upstream code happens to pass.
- No joint/IMU validity stop.
- No fall/tilt stop.
- No default max-duration guard for real hardware.
- Cleanup behavior should be guaranteed under more exception paths.

Recommended fix:

- Add a deploy safety envelope before further real-robot use: finite checks,
  action clamps, command clamps, joint/IMU range checks, fall stop, watchdog or
  max-duration, and a robust `finally` path that stands down or zeroes torque.

### Subsystem Notes

#### Training And Checkpointing

Strengths:

- SAC, TD3, FastSAC, and FastTD3 share `run_offpolicy_loop()`.
- `ObsPipeline` centralizes actor/critic obs extraction, normalization, frame
  stacking, and buffer construction.
- `CheckpointManager` centralizes latest/best checkpoint behavior.
- Critic normalization state is now persisted for shared paths.
- Resume warmup mitigation is a real improvement.

Risks:

- Resume remains warm-start, not full continuation.
- `start_step` is recovered from `metrics.csv`, not from a persisted loop state.
- Replay buffer, env state, PRNG, tracker, best eval, and W&B state are not
  persisted.
- TD-MPC2 has no resume path and uses a separate checkpoint contract.
- `CheckpointManager.best_eval` is initialized fresh after resume.

Recommendation:

- Rename docs/help text from "resume" to "warm-start" unless full state
  persistence is implemented.
- Add explicit `resume_mode` semantics: `weights_only`, `optimizer`, `full`.
- Do not implement large replay snapshots until the artifact contract is
  versioned and validated.

#### Environments And Domain Randomization

Strengths:

- Go2 Warp is the clear primary sim-to-real path.
- Actor obs and privileged critic obs are documented and tested.
- Torque-speed variants are registered.
- Observation schema is now serialized for deploy.
- Current Go2 DR specs avoid duplicate field targets, lowering exposure to the
  generic replacement-overwrite bug.

Risks:

- `DomainRandWrapper` still appears to resample model DR every step.
- `_build_dr_model()` still overwrites earlier replacements when multiple specs
  target the same MJX model field.
- Terrain curriculum generated XML moved to system temp, but at least one test
  still appears to expect an old source-tree generated filename.
- Go2/Warp tests are not consistently marked as GPU/Warp/Go2.

Recommendation:

- Fix DomainRand before interpreting new DR-heavy Go2 training as final.
- Add regression coverage for per-episode model persistence and duplicate-field
  replacement composition.
- Update curriculum tests to assert the current temp-file behavior.

#### Deploy And Sim2sim

Strengths:

- `PolicyRunner` is pure NumPy and avoids JAX in deploy.
- `ObsBuilder` is schema-driven and tested against reordered/dropped terms.
- Real deploy gains now match Warp training gains (`20.0/0.5`).

Risks:

- Sim2sim still uses archived MJX gains by default.
- `PolicyRunner` supports only a subset of trained algorithms.
- Generic `record_video.py` still claims TD-MPC2 support even though TD-MPC2 has
  separate `.npz` artifacts and a dedicated recorder.
- ONNX export is narrow and partly stale: FastSAC-only, no obs norm/BN support,
  and docs encode a fixed 48d assumption instead of using `obs_schema`.

Recommendation:

- Make deploy consumers contract-driven: read `artifact_kind`, `obs_schema`, and
  `control` metadata before constructing runtime behavior.
- Remove unsupported algo claims from generic tools or dispatch to dedicated
  TD-MPC2 tooling.

#### Tests And CI

Strengths:

- Docs drift tests are valuable and catch real classes of stale documentation.
- TD-MPC2 has a much broader focused test set than before.
- Deploy obs schema tests are mostly synthetic and hermetic.

Risks:

- Default pytest is not hermetic.
- CI appears docs-only.
- `docs/contributing.md` says "Full test suite (~299 tests)" and recommends
  `uv run python -m pytest tests/ -v`, but the current collection is closer to
  797 tests plus `deploy`.
- Some deploy tests depend on local checkpoint artifacts and skip otherwise.
- No configured lint/type/coverage/pre-commit gate is visible.

Recommendation:

- Establish a CPU-safe default gate first.
- Add a manual GPU/Warp job later.
- Add `ruff check` before investing in broad type coverage.
- Replace local-artifact deploy tests with synthetic checkpoint fixtures or mark
  them as artifact-dependent.

#### TD-MPC2

Strengths:

- Recent fixes and benchmark notes show serious validation work.
- Dedicated runtime/eval/video helpers exist.
- Refactor plan and package skeleton are in progress.
- `tdmpc2/__init__.py` preserves import compatibility while symbols move.

Risks:

- The refactor is mid-flight in the working tree.
- `tdmpc2_old.py` still holds much of the implementation.
- Some tests reportedly hardcode a `.worktrees/tdmpc2-impl` path, which risks
  testing stale code instead of this checkout.
- TD-MPC2 artifact shape is separate but not versioned or documented as a first
  class artifact kind.

Recommendation:

- Finish the planned package split in small commits.
- Remove any hardcoded worktree paths from tests.
- Treat TD-MPC2 as experimental-public: documented and testable, but with
  explicit caveats around artifacts, resume, and video/deploy support.

### Recommended Execution Order

1. Fix the test gate first.
   Add marker taxonomy, update `docs/contributing.md`, and make the default test
   command CPU/hermetic. This gives future agents a reliable signal.

2. Fix DomainRand model persistence.
   This is the highest-confidence correctness issue that can affect current Go2
   conclusions.

3. Add artifact/control metadata.
   Version shared and TD-MPC2 artifacts, then make deploy/sim2sim/video consume
   contracts instead of implicit filenames/constants.

4. Close backend dispatch gaps.
   Use `env_bundle.num_envs`, route PPO gym eval correctly, and move locomotion
   logging/checkpoint metadata behind env-provider hooks.

5. Finish TD-MPC2 package split.
   Keep this as a pure refactor with import compatibility and focused tests.

6. Add maturity labels.
   README/docs should distinguish stable core, deploy-critical Go2, experimental
   research, deprecated scripts, and archived references.

### Suggested Maturity Labels

| Surface | Suggested label | Notes |
|---|---|---|
| SAC/TD3/FastSAC/FastTD3 shared loop | Stable core | Main reusable training path |
| EnvBundle MJX/gym registry | Beta | Real but still has downstream holes |
| Go2 Warp env + obs schema | Deploy-critical | Must stay tightly tested |
| `deploy/` real robot runtime | Deploy-critical / needs safety envelope | Treat changes conservatively |
| TD-MPC2 | Experimental public | Benchmarked, but artifact/refactor still moving |
| PushT via `train_sac.py --env PushT` | Experimental public | New canonical path pending reproduction |
| `scripts/train_pusht.py` | Deprecated reference | Delete/archive after parity run |
| `scripts/archive/*` and `tools/archive/*` | Archived | Keep out of public docs except historical notes |

### Verification Performed In This Audit

Commands run:

```bash
git status --short
git log --oneline -25
git diff --stat
uv run python -m pytest tests/test_docs_drift.py tests/test_env_bundle.py tests/test_checkpoint.py tests/test_tdmpc2.py -q
JAX_PLATFORMS=cpu uv run python -m pytest tests/test_docs_drift.py tests/test_env_bundle.py tests/test_checkpoint.py tests/test_tdmpc2.py -q
```

Results:

- First pytest command failed during collection with CUDA OOM from a module-level
  `jax.random.PRNGKey` in `tests/test_tdmpc2.py`.
- CPU-forced rerun: `67 passed, 1 failed, 8 deselected`.
- The failure was `test_env_bundle_flat_obs_cheetahrun`, which still requested a
  CUDA backend through MuJoCo/JAX even under `JAX_PLATFORMS=cpu`.
- Full pytest was not run because the current default test surface is known to
  include GPU/Warp tests and is not a reliable audit signal yet.

### Bottom Line

The repo is healthier than the older audit implied: backend abstraction,
PushT integration, docs drift checks, resume warmup, deploy obs schema, and
TD-MPC2 validation all moved forward.

The priority now is making the repo's contracts explicit:

- what counts as a default test,
- what kind of artifact a checkpoint is,
- what deploy metadata a policy requires,
- what "resume" means,
- what each backend is allowed to assume,
- and which surfaces are stable versus experimental.

Once those contracts are explicit, the existing architecture is strong enough
to support incremental cleanup without destabilizing the working Go2 and
TD-MPC2 paths.

Date: 2026-04-25
Repo: `/home/stevenman/Desktop/Work/Research/jax-learning`
Branch: `new_slate_linen`, ahead of origin by 26 commits

## Mission

Audit the full repo structure and implementation with emphasis on:

- Project goals, context, and current technical state
- Usability for the owner and future contributors
- Readability, structure, and maintainability
- Runtime performance and training efficiency
- Correctness risks, especially train/deploy parity and resume behavior
- Prioritized improvement opportunities with alignment questions

This is an audit-first pass. Do not refactor blindly. Findings should be grounded in code, docs, tests, or repo state.

Follow-up execution plan:

- `.superpowers/plans/2026-04-25-structural-hardening.md` translates the structural audit into agent-agnostic implementation phases, acceptance criteria, test commands, and handoff prompts.

## Current Repo Model

The repo is a JAX-native reinforcement learning framework for robot learning research, centered on:

- Unitree Go2 locomotion using MuJoCo Playground, especially Warp backend for sim-to-real
- Off-policy algorithms: SAC, TD3, FastSAC, FastTD3, FlashSAC
- On-policy algorithms: PPO, PPOContraction
- TD-MPC2 implementation in progress
- PushT manipulation experiments
- Deployment and sim2sim utilities for Go2

The strongest design idea is a three-layer split:

- Env layer: MuJoCo Playground envs, Go2 Warp envs, PushT, wrappers
- Training layer: env setup, obs pipeline, checkpointing, eval, logging, shared loops
- Algo layer: pure math/state updates, with no environment imports

The main Go2 path is currently:

- `Go2WarpJoystickFlat` or curriculum variants
- Unitree MJCF through Warp
- external PD at physics rate
- actor obs: 48d deployable state
- critic obs: 122d privileged state
- action: 12d position target offsets in policy joint order
- deployment: `deploy/PolicyRunner` plus schema-driven `ObsBuilder`

## High-Level Structural Assessment

The repo has a strong technical core, but its structure now mixes three maturity levels in one surface:

- Stable framework pieces: algorithms, networks, replay buffers, shared off-policy training loop, config dataclasses.
- Production-critical Go2 deployment pieces: Warp env, obs schema, deploy runner, sim2sim, hardware validation docs.
- Research workbench pieces: TD-MPC2, PushT, Bongo, curriculum experiments, diagnostics, one-off scripts.

That is the main structural tension. The code is not failing because it lacks abstractions; it is failing because the repo does not consistently label which abstractions are stable contracts and which are experiment-local conveniences.

Recommended structural direction:

- Define a small stable core: algos, networks, buffers, normalization, checkpoint artifact schema.
- Define env backends as adapters with metadata/logging hooks rather than letting `training/` import Go2/MuJoCo details.
- Keep scripts thin: parser + config overrides + call a training/eval entrypoint. Large scripts should move reusable pieces into modules.
- Treat Go2 deploy as a first-class product boundary. Anything deploy consumes should come from checkpoint metadata or a versioned artifact contract, not copied constants.
- Move experimental systems behind explicit labels: `research`, `experimental`, or per-domain modules. TD-MPC2 can stay fast-moving, but tooling should not imply it supports every generic workflow until it does.
- Split tests by intent: hermetic unit/docs tests by default, GPU/Warp integration by explicit marker, hardware/deploy checks by explicit operator workflow.

A possible target shape:

- `jax_rl/algos/`, `jax_rl/networks/`, `jax_rl/buffers/`: pure JAX implementation layer.
- `jax_rl/training/`: backend-neutral loops, checkpoint contracts, logging interfaces, eval interfaces.
- `jax_rl/envs/`: backend adapters and domain envs, including Go2/Warp-specific hooks.
- `jax_rl/artifacts/`: checkpoint schema, metadata validators, deploy compatibility checks.
- `scripts/`: thin CLI entrypoints only.
- `deploy/`: pure deploy runtime that consumes versioned artifacts.
- `tests/unit`, `tests/integration`, `tests/gpu`, or equivalent marker discipline.

Highest-leverage structural cleanup:

- Introduce an explicit checkpoint/artifact contract first.
  This means every checkpoint declares what kind of artifact it is, what files are required, what metadata fields are guaranteed, and which consumers support it. Today shared algos, TD-MPC2, PushT, deploy, export, and video recording all imply different contracts. First patch: add `artifact_version`, `artifact_kind`, and a small metadata validator for shared checkpoints; make deploy/video fail clearly when the contract is unsupported.

- Add deploy-critical metadata to checkpoints.
  Go2 deployment should not depend on copied constants. The checkpoint should say which action scale, PD gains, backend, control timestep, contact mode, torque-speed setting, and obs schema the policy was trained with. First patch: write a `meta["control"]` block for Go2 Warp checkpoints and teach `deploy/sim2sim_direct.py` to read it with a loud fallback warning for old checkpoints.

- Introduce backend metadata/logging hooks second.
  Generic training code currently imports locomotion curriculum logging and MuJoCo Playground registry details. That makes future gym/IsaacLab backends harder than necessary. First patch: let `EnvBundle` carry optional `metadata_provider`, `train_metrics_provider`, and `visualization_provider` callables, then move terrain logging behind those hooks.

- Make `training/` backend-neutral by policy, not by hope.
  The line should be: training loops know about batched env stepping, obs extraction, eval, checkpointing, and metrics; env packages know about terrain images, obs schemas, DR specs, and physics metadata. First patch: remove direct `jax_rl.envs.locomotion.*` imports from `offpolicy_loop.py` and replace them with optional bundle hooks.

- Split tests by intent before adding more coverage.
  The current default test command can hit Warp CUDA OOM, then poison unrelated JAX tests. That makes test results hard to trust. First patch: add `gpu`, `warp`, and `go2` pytest markers, mark the Go2/Warp modules, and make default `pytest` run hermetic CPU-safe unit/docs tests only.

- Keep scripts thin.
  A script should parse CLI args, apply overrides, and call a reusable training/eval function. Large scripts are harder to test and easy to drift from each other. First patch: start with `train_tdmpc2.py`: move parser/config construction, checkpoint artifact helpers, and eval helpers into importable module functions before changing algorithm behavior.

- Label maturity explicitly.
  Stable framework APIs, deploy-critical Go2 code, and experimental research code should not present the same stability signal. First patch: add a small “Maturity” table in README or docs: stable, experimental, deploy-critical, archived. Then align CLI docs and artifact contracts with those labels.

- Consolidate repeated training semantics, not necessarily every training loop.
  FlashSAC and TD-MPC2 may need separate loops for good reasons. The structural problem is duplicated artifact, resume, eval, and logging semantics. First patch: centralize artifact writing/validation and resume-mode naming before trying to force all algorithms through one trainer abstraction.

- Only then split large implementation files.
  File size is a symptom, not the root cause. Splitting before contracts are clear risks moving ambiguity into more places. First patch after contracts: split TD-MPC2 tests by concern, then split implementation along real boundaries such as model components, planner/runtime, update step, and checkpoint/eval runtime.

## Git And Artifact State

Observed git state:

- Branch `new_slate_linen...origin/new_slate_linen [ahead 26]`
- Untracked: `.superpowers/plans/2026-04-25-env-backend-refactor.md`
- No tracked generated artifacts found under `site`, `checkpoints`, `wandb`, `.temp`, `dist`, `.venv`, `deploy/.venv`

Ignored/generated directory sizes:

- `checkpoints`: 3.1G
- `wandb`: 638M
- `.temp`: 516M
- `site`: 6.3M
- `dist`: 6.5M
- `.venv`: 6.9G
- `deploy/.venv`: 143M

Finding: generated/runtime state is ignored correctly, but living directly in the repo creates search noise, backup/sync pressure, and newcomer friction. Consider a documented artifact root outside the source checkout, or a `JAX_RL_RUNS_DIR`/`JAX_RL_ARTIFACTS_DIR` convention.

Additional local noise:

- Source-tree `__pycache__` directories exist under `jax_rl/`, `tests/`, `docs/scripts/`, `tools/`, and `deploy/`. They are ignored, but still pollute local search and directory scans.
- This is not a correctness problem, but the repo would benefit from a maintenance/cleanup command instead of relying on manual deletion.

## Active Project Context

Current high-priority threads from `.context/TODO.md`:

- TD-MPC2 end-of-run collapse: J3 CheetahRun reached paper-match best return around 500k steps, then final checkpoint collapsed sharply by 1M. Best checkpointing prevents deployment of the collapsed policy, but the cause is unresolved.
- Polyak refactor verification: unit equivalence is proven, but a real Go2 FastSAC behavioral verification run is still pending.
- FlashSAC resume eval regression: reward normalizer persistence is fixed, but post-resume eval still drops. The leading hypothesis is empty replay buffer on resume.

Current active research/infra themes:

- Terrain curriculum: overall strong, but `pyramid_up` stalls because level 1 has an 8cm step face after level 0 is flat. Candidate fix is quadratic difficulty scaling.
- PushT letter-shape generalization: V2 dense keypoints improved DR generalization, but held-out letters and variable-N set encoders remain open.
- Env-backend refactor: active Superpowers plan proposes a unified MJX/gym/IsaacLab `EnvBundle` protocol and backend registry.

Audit implication: the biggest repo-level improvements should not be abstract cleanup. They should either reduce experiment ambiguity, make checkpoint/resume semantics explicit, or improve cross-backend portability without destabilizing the working Warp Go2 path.

## Public Entrypoint And Docs Findings

Good:

- Most training scripts expose `build_parser()`.
- `docs/scripts/gen_cli_reference.py` reflects argparse definitions instead of hand-mirroring flags.
- Drift tests exist for common stale-doc classes.

Issues:

- `scripts/train_tdmpc2.py` is a public training script but is not included in `docs/scripts/gen_cli_reference.py`.
- `scripts/eval_tdmpc2.py` and `scripts/check_tdmpc2_determinism.py` are public diagnostic/eval scripts but are not documented in the generated CLI reference.
- `scripts/train_ppo.py` is mentioned in README as the readable Python-loop PPO variant, but it does not expose `build_parser()`, so it cannot be reflected by CLI docs.
- README usage examples in script docstrings still say `uv run python train_tdmpc2.py` and `uv run python train_fast_sac.py` in some places, while the current layout uses `scripts/...`.
- `scripts/record_video.py` top docstring says "JIT-scan the rollout on GPU", but implementation intentionally uses a Python loop over a jitted step to reduce peak HBM.
- `scripts/record_video.py` top docstring claims support for "ALL algos ... TDMPC2", but `record()` calls `load_actor_for_inference()`, which requires `actor_params.npy`. TD-MPC2 checkpoints write `actor_params.npz` and `world_model_params.npz`, and `_build_select_action()` has no TD-MPC2 branch.
- `jax_rl/training/checkpointing.py::load_actor_for_inference()` returns 4 values including `actor_batch_stats`, but its type hint and docstring still describe 3 return values.

Improvement:

- Make every intended public script expose `build_parser()`.
- Add a single `PUBLIC_SCRIPTS` registry used by docs generation and a test.
- Decide whether diagnostics belong in the public CLI reference or a separate diagnostics reference.
- Either make `record_video.py` actually support TD-MPC2 artifacts or remove TD-MPC2 from the support claim and point users at `scripts/eval_tdmpc2.py`.

## Training Architecture Findings

Good:

- `run_offpolicy_loop` centralizes SAC/TD3/FastSAC/FastTD3.
- `EnvBundle` removes repeated dict-obs/asymmetric-critic detection from scripts.
- `ObsPipeline` localizes actor/critic obs extraction, sample-time normalization, and frame-stack-aware buffer construction.
- Config dataclasses are explicit and mostly separated by concern.
- `CheckpointManager` centralizes latest/best checkpoint behavior.

Major risk:

- Off-policy "resume" is not a full training resume. Checkpointing saves model state and normalization state, but not replay buffer, env state, PRNG key, episode tracker, optimizer schedule position outside model state assumptions, or W&B/logging continuity state. After resume, training restarts with an empty replay buffer and fresh envs but a trained model. This is closer to warm-start fine-tuning than exact continuation.
- Validation: `save_checkpoint()` stores `training_state`, `norm_state`, and optional `critic_norm_state` in Orbax. `run_offpolicy_loop()` and `train_flashsac.py` recreate `JaxReplayBuffer`, `env_state`, `EpisodeTracker`, checkpoint manager, PRNG flow, and `last_eval_eps` before loading. `start_step` is recovered from `metrics.csv`, not from a persisted training-loop state.

Why it matters:

- SAC/TD3/FastSAC/FastTD3 behavior after resume differs materially from uninterrupted training.
- FlashSAC has extra state and already has a suspected resume/eval regression thread.
- TD-MPC2 is even more sensitive because sequence replay is part of the algorithm.

Improvement options:

- Rename CLI/help/docs from "resume" to "warm-start" unless full state persistence is implemented.
- Or implement real resume: replay buffer, env state, PRNG, tracker, metrics, best eval, and any algorithm-specific state.
- Add a `resume_mode` concept: `weights_only`, `optimizer`, `full`.
- Cheap mitigation if keeping warm-start semantics: when `resume is not None` and the replay buffer is empty, populate the warmup buffer with restored-policy exploration, not random uniform actions. This directly targets the FlashSAC resume regression hypothesis without pretending to be exact continuation.
- Full-resume design: add `JaxReplayBuffer.state_dict()/from_state_dict()`, persist buffer arrays/position/size/episode IDs, persist `env_state`, `key`, tracker counters, `last_eval_eps`, `ckpt_mgr.best_eval`, and loop counters. Make this opt-in because buffer snapshots can be large.

Medium risk:

- `TrainConfig` still contains PPO-specific `ppo` config while off-policy paths also use it for metadata. This is pragmatic but weakens type clarity.
- TD-MPC2 uses its own config shape where loop fields and algo fields live together, unlike other algorithms. That may be appropriate for research velocity, but it should be documented as an exception.
- `jax_rl/training/__init__.py` imports `env_setup` and `offpolicy_loop`, which pull MuJoCo Playground and locomotion-specific code into the package-level training import. This is convenient but works against a lightweight backend-neutral public API.

Performance observations:

- `JaxReplayBuffer` stores on GPU and JITs add/sample paths.
- Extra obs fields are scattered outside the main JIT path. The comment says overhead is tiny; worth measuring for asymmetric Go2 at scale.
- `DomainRandWrapper` in `per_step` computes reset states for all envs every step and selects reset values for done envs. This is deliberate and documented, but it is a permanent performance tradeoff.

## Backend-Layering Findings

The active env-backend refactor plan is directionally right. The current code still has locomotion/MuJoCo Playground details leaking into generic training infrastructure:

- `jax_rl/training/offpolicy_loop.py` imports `jax_rl.envs.locomotion.curriculum_logging` directly for terrain metrics/images.
- `jax_rl/training/metrics_logger.py` re-exports locomotion curriculum logging for backwards compatibility.
- `jax_rl/training/checkpointing.py` imports `mujoco_playground.registry` and loads `cfg.env_name` to discover DR specs and obs schema.

Why it matters:

- Generic off-policy training is not backend-neutral yet.
- Adding gymnasium/IsaacLab-style envs will either require conditional branches in generic training code or a metadata/logging hook boundary.

Improvement:

- Put optional logging and metadata providers on `EnvBundle` or a small sidecar protocol.
- Let env backends provide `checkpoint_metadata()`, `extra_train_metrics()`, and optional visualization callbacks.
- Keep locomotion curriculum rendering inside the locomotion package, called through that hook.

## Env And Deploy Findings

Good:

- Go2 Warp env saves an `obs_schema` into checkpoint metadata when possible.
- Deploy-side `ObsBuilder` reads checkpoint schema and composes obs by term names.
- Tests cover default 48d layout, schema reordering, dropped accelerometer, unknown terms, and fallback for old checkpoints.
- `PolicyRunner` is pure numpy and avoids a JAX dependency at deployment.

High-risk issue:

- Deploy PD constants currently use `KP_SIM = 35.0`, `KD_SIM = 0.1` for sim mode, marked archived MJX values, while the Warp training env and real deploy path use `Kp=20.0`, `Kd=0.5`. `deploy/sim2sim_direct.py` imports `KP_SIM, KD_SIM`, prints them, and uses them for direct sim2sim.
- Validation: `Go2WarpJoystickFlat.default_config()` sets `Kp=20.0`, `Kd=0.5`, and `action_scale=0.5`; `deploy/sim2sim_direct.py` and `Go2Interface(sim=True)` use `KP_SIM=35.0`, `KD_SIM=0.1`. `save_checkpoint()` does not write PD gains/action scale/contact mode into `meta.json`.

Why it matters:

- Sim2sim is supposed to validate the Warp-trained policy against the Unitree MJCF. Using archived MJX PD gains can produce misleading failures or successes.
- `deploy/go2_constants.py` already has a TODO to read gains from checkpoint metadata.

Improvement:

- Save deploy-critical control metadata under a versioned `meta["control"]` or `meta["deployment"]` key: backend/impl, `Kp`, `Kd`, `action_scale`, `policy_dt`, `physics_dt`, `contact_mode`, `torque_speed_model`, and joint/action order.
- Make deploy/sim2sim read this metadata by default.
- Keep constants only as fallback for old checkpoints, with a loud warning when falling back.
- Add optional CLI overrides for intentional A/B tests, but do not silently use archived MJX gains for Warp checkpoints.
- Add tests with synthetic `meta.json` proving Warp checkpoints choose `20.0/0.5` and old checkpoints warn before falling back.

Medium-risk issue:

- `PolicyRunner.normalize_obs()` uses epsilon `1e-8`, while training obs normalization may use algo-configured eps such as `1e-2` for SAC/FastSAC. If obs normalization is enabled for a deployable checkpoint, pure numpy inference may not match JAX training-time normalization.
- `PolicyRunner.use_obs_norm` is inferred from `norm_count > 0`. Disabled obs normalization intentionally saves identity stats with `count=1` for checkpoint compatibility, so deploy logs can say obs normalization is active even when training did not use it. The action is unchanged for identity stats, but the operator-facing diagnostic is misleading.

Improvement:

- Save `obs_norm_eps` in checkpoint metadata and have `PolicyRunner` use it.
- Save/use an explicit `obs_normalization` boolean for deploy instead of inferring it from count.
- Add a parity test comparing JAX actor output and `PolicyRunner` output on the same checkpoint/params/obs.

Medium-risk issue:

- `PolicyRunner` supports PPO, SAC, and FastSAC. It does not appear to support TD3, FastTD3, FlashSAC, PPOContraction naming beyond what record_video handles. The class docstring says PPO/SAC/FastSAC only, but deployment scripts may accept any checkpoint path.

Improvement:

- Fail early with a clear message listing supported deployment algorithms.
- Or extend `PolicyRunner` to TD3/FastTD3/FlashSAC if those checkpoints are intended for deploy/sim2sim.

Low-risk issue:

- `PolicyRunner._swish()` says it is numerically stable, but it uses `np.where`. NumPy evaluates both branches, so large activations can still emit overflow warnings from `exp()`. The targeted deploy test hit this warning while still passing.

Improvement:

- Implement sigmoid/swish with masked assignment or clipped input so deploy inference is warning-clean.

Potential correctness issue to verify:

- `ActionDelayWrapper.step()` resets its buffer based on `state.done` before calling the inner env step. In the actual `per_step` path, `DomainRandWrapper` is outermost and merges `reset_state.info` for done envs, which probably resets the delay buffer correctly. Existing tests cover `ActionDelayWrapper -> FrameStackWrapper`, but not the real `ActionDelayWrapper -> FrameStackWrapper -> DomainRandWrapper` composition. Add that test so this stays pinned.

High-risk confirmed bug:

- `DomainRandWrapper.step()` builds a fresh randomized model every step for all envs, not only for envs that reset.
- Evidence: `domain_rand.py` pops `_dr_rng`, splits it, calls `_build_dr_model(reset_rng)`, then vmaps `_step_with_model(dr_model, state, action)` for every env before selecting reset state for done envs.
- The surrounding comments describe "per-env randomized model" and reset preparation, but no per-env model is stored in wrapper state. As written, model-domain-randomized parameters may change every environment step for non-done envs.
- Validation: a CPU-only synthetic env recorded `reset_model_value` and `step_model_value` for a non-done episode. They differed immediately after one step, with `max_abs_diff=76.125`.

Why it matters:

- Model DR is temporally inconsistent with per-episode DR assumptions.
- This can make training noisier, hide sim2real assumptions, and make DR schedule/performance interpretation unreliable.

Improvement:

- Persist sampled model-DR replacements in `state.info`, for example under `_dr_model_replacements`.
- On `reset()`: sample replacements, build the randomized model, reset env, and store replacements.
- On `step()`: use stored replacements for the step model; separately sample reset replacements for all envs; after done is known, merge reset replacements into stored replacements only where `done`.
- Add regression tests: model DR stays fixed within an episode, changes on episode reset, and survives wrapper composition with action delay/frame stack.

High-risk confirmed bug:

- `_build_dr_model()` does not compose multiple DR specs targeting the same MJX model field. It starts from the original field for each spec and writes `replacements[spec.field] = new_field`, so later specs overwrite earlier specs for the same field.
- Validation: a CPU-only synthetic env with two `body_mass` specs for disjoint slices produced `[1, 1, 20, 1]` instead of the expected `[10, 1, 20, 1]`.
- Real affected path: `Go2BongoHandstand.get_domain_randomization_spec()` has both `board_mass` and `body_mass` specs targeting `body_mass`, so board mass DR is likely overwritten by robot link mass DR.

Improvement:

- In the same DomainRand refactor, group or sequentially compose specs on a per-field accumulator.
- When a field already has pending replacements, use that pending field as the next spec's base instead of re-reading `getattr(model, spec.field)`.
- Add a regression test for duplicate-field specs on disjoint slices.

## Checkpoint And Artifact Contract Findings

There is no single explicit checkpoint contract across the repo:

- Shared PPO/off-policy/FlashSAC checkpointing writes `meta.json`, `metrics.csv`, `actor_params.npy`, and `orbax/`.
- TD-MPC2 writes `actor_params.npz`, `world_model_params.npz`, and `meta.json` from `scripts/train_tdmpc2.py`.
- PushT writes `actor_params_best.npy` and `actor_params_final.npy` directly from `scripts/train_pusht.py`.
- Deployment `PolicyRunner`, ONNX export, and shared `record_video.py` assume the shared `actor_params.npy` format.

Why it matters:

- Tooling claims can silently exceed artifact compatibility.
- New algorithms have to rediscover which artifact fields are needed for eval, video, deploy, export, and resume.
- Checkpoint metadata already carries critical deploy state like obs schema, but not yet enough to make deploy parity complete.

Improvement:

- Define a versioned checkpoint contract: required files, required `meta.json` fields, optional files, and supported consumers.
- Make each consumer validate the contract up front and fail with a clear message.
- For research-only algorithms, write `meta["artifact_contract"] = "tdmpc2-research-v1"` or similar so generic tools do not pretend to support them.
- Add a tiny synthetic checkpoint fixture for consumer tests.

## Test Suite Observations

Strengths:

- Broad unit coverage for configs, builders, wrappers, replay buffer, docs drift, Go2 env dims, obs schema, reward spec, TD-MPC2 components, FlashSAC blocks, rollout recording, and checkpoint behavior.
- Docs drift tests are unusually useful and protect against repeated stale-doc patterns.

Gaps found so far:

- No test that all public scripts expose `build_parser()` and are included in generated CLI docs.
- No deploy parity test comparing `PolicyRunner` numpy action to JAX actor action.
- No test that checkpoint metadata contains deploy-critical constants like PD gains and action scale.
- No test that "resume" reproduces or intentionally does not reproduce continuous training.
- DomainRand and action-delay composition needs a pipeline-level episode-boundary test.
- Generated CLI drift test is marked slow and only covers a subset of scripts.
- `deploy/test_policy_runner.py::test_policy_runner_loads_and_infers` depends on whatever checkpoint happens to exist under `checkpoints/`; it skips if none. That makes it local-artifact dependent rather than hermetic.
- `pyproject.toml` only defines a `slow` marker and defaults to `-m 'not slow'`. Many GPU/Warp/Go2 tests are not marked slow, so the default run can still hit Warp CUDA OOM.
- Validation: unmarked default-collected tests instantiate `WarpJoystick`, `WarpJoystickCurriculum`, `Go2BongoHandstand`, and MuJoCo Playground wrapper paths. Examples include `test_go2_warp_env.py`, `test_go2_warp_curriculum_env.py`, `test_terrain_curriculum_dr_wrapper.py`, `test_frame_stack_wrapper.py`, `test_training_wrappers.py`, and `test_go2_bongo_env.py`.

Verification run:

- `uv run python -m pytest -q` failed with `200 failed, 500 passed, 46 skipped, 15 deselected, 36 errors` in about 178s.
- The first visible errors were Warp CUDA graph creation OOMs in Go2/Warp reset paths. After that, many later pure JAX tests failed with `RESOURCE_EXHAUSTED`, likely because the device/runtime was already contaminated by the GPU OOM.
- A CPU-only core subset passed cleanly: `JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/test_polyak.py tests/test_replay_buffer.py tests/test_twohot.py tests/test_reward_spec.py tests/test_docs_code_blocks.py -q` gave `259 passed, 46 skipped, 1 warning`.
- A targeted CPU docs/checkpoint/deploy subset passed: `JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/test_docs_drift.py tests/test_cli_utils.py tests/test_checkpoint.py deploy/test_policy_runner.py -q` gave `50 passed, 7 deselected, 2 warnings`.

Interpretation:

- The default test suite is not currently a reliable local signal on this machine/GPU state.
- Core unit/docs tests are healthy when isolated from Warp/GPU resource pressure.
- Warp/Go2 tests need stronger partitioning and resource controls.

Improvement:

- Add pytest markers such as `unit`, `docs`, `gpu`, `warp`, `go2`, and `slow`.
- Make default `pytest` run the hermetic unit/docs subset.
- Add explicit commands for GPU/Warp integration tests, ideally documented in README/CLAUDE.
- Consider forcing CPU for pure unit tests in CI or a `pytest-unit` target.
- Replace local checkpoint-dependent deploy tests with synthetic tiny checkpoints.
- Practical first patch: define `gpu`, `warp`, and `go2` markers in `pyproject.toml`; change default addopts to exclude them; add module-level `pytestmark` to the Warp/Go2 test modules; document `uv run python -m pytest -m warp` as the explicit integration target.

## Complexity Hotspots

Largest Python files in source/scripts/tests, excluding major artifact directories:

- `tests/test_tdmpc2.py`: 1344 lines
- `jax_rl/algos/tdmpc2.py`: 1123 lines
- `jax_rl/envs/manipulation/pusht/pusht.py`: 782 lines
- `jax_rl/envs/terrains/primitives.py`: 614 lines
- `tools/handstand_optimize.py`: 602 lines
- `jax_rl/algos/flash_sac.py`: 574 lines
- `jax_rl/envs/locomotion/go2_warp_joystick.py`: 571 lines
- `scripts/train_tdmpc2.py`: 554 lines
- `scripts/train_ppo_fast.py`: 540 lines
- `scripts/train_pusht.py`: 497 lines
- `scripts/record_video.py`: 486 lines
- `scripts/train_flashsac.py`: 481 lines
- `jax_rl/buffers/jax_replay_buffer.py`: 445 lines
- `scripts/train_ppo.py`: 442 lines
- `jax_rl/envs/locomotion/go2_warp_curriculum.py`: 404 lines
- `scripts/train_ppo_contraction.py`: 401 lines

Interpretation:

- The shared off-policy loop is not the main complexity problem.
- The biggest readability risks are TD-MPC2, standalone research scripts, large environment implementations, and very broad tests.

Improvement:

- Refactor only after behavior is pinned by tests.
- Best first targets: split `scripts/train_tdmpc2.py` parser/context setup from training loop, split TD-MPC2 tests by concern, and extract repeated script boilerplate into existing training helpers where it already matches local patterns.

## Improvement Backlog Draft

P0 correctness/usability:

- Fix `DomainRandWrapper` model-DR temporal behavior.
- Fix duplicate-field DomainRand spec composition.
- Clarify or fix resume semantics.
- Fix deploy/sim2sim PD gain source. Metadata should drive deployment.
- Add JAX-vs-numpy policy inference parity tests.
- Split default tests from GPU/Warp integration tests so `pytest` gives a reliable signal.

P1 docs/API consistency:

- Bring TD-MPC2 and diagnostic scripts into CLI docs or explicitly mark as internal.
- Add `build_parser()` to `train_ppo.py`.
- Fix stale script usage docstrings and `record_video.py` docstring.

P1 architecture:

- Add a single public entrypoint registry for scripts.
- Add checkpoint metadata schema versioning.
- Consolidate metadata writing across off-policy, PPO, FlashSAC, TD-MPC2, and PushT where possible.

P2 ergonomics:

- Add artifact root config/env var for checkpoints, W&B, videos, temp output.
- Add a repo maintenance command or script to summarize local artifacts.
- Make `README` distinguish stable user workflows from research/experimental workflows.

## Alignment Questions

1. Is exact training continuation important, or is current "resume" intended to mean warm-start/fine-tune?
2. Should TD-MPC2 be treated as a first-class public algorithm now, or kept as research-in-progress until benchmarks land?
3. Is real Go2 deployment the top priority, or is this repo primarily a research playground that happens to deploy?
4. Are checkpoints meant to be long-lived artifacts across code changes, or only same-commit run products?
5. Should generated artifacts stay inside the repo by default for convenience, or move outside by default for cleanliness?
