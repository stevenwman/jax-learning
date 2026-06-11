# 2026-06-11 — Go2 env variants-as-data refactor + OSC extraction

Overnight autonomous refactor on branch `go2-osc-impedance` (worktree),
commits `af4bf7f..d15d026` (11 commits). Plan:
`.superpowers/plans/2026-06-11-go2-env-variants.md`. Spec:
`.superpowers/specs/2026-06-11-go2-env-variants-design.md`.

## Motivation (audit findings)

The Go2 Warp env family had grown to 29 named envs via four compounding
pathologies:

1. **Patch-chains** — each new variant was a subclass + a config-factory
   function copying and tweaking the previous one's overrides
   (`go2_warp_osc_joystick.py` → `go2_warp_osc_var_impedance.py` →
   `go2_warp_osc_rough.py`). Answering "what config does env X actually
   train with" required tracing 3–4 files.
2. **God function** — `mjx_backend.py` carried a ~175-line Go2 closure
   block, one hand-written registration per env, with research rationale
   buried in comments there instead of next to the env declarations.
3. **Silent preset fallback (the real bug)** — preset getters only knew
   the older Go2 names. Unknown `Go2Warp*` names fell through to the bare
   base config: **all OSC envs silently trained with NO domain
   randomization (`reset_mode` default, DomainRandWrapper never applied)
   and `eval_every_n_episodes=5000` (≈zero mid-run evals on short runs)**.
   Weeks of OSC runs trained under a different regime than intended.
4. **Half-extraction** — the 2026-06-09 composition refactor moved
   controller *selection* into components, but the OSC mechanics
   (`_run_osc`, `_feet_in_body`, `_compute_nominal_foot_body`) and all
   gains state still lived on the host env, planted there by the
   component at setup.

## What landed (per commit)

| Commit | What |
|---|---|
| `af4bf7f` | **Variants-as-data table** — `jax_rl/envs/locomotion/go2_warp_variants.py`: `EnvVariant` (config callable via `go2_config()` builder + host class name + train overrides + notes), 29 entries, pinned equal to legacy configs by transitional equality tests. |
| `95509a5` | Guard cross-axis `go2_config` kwargs (e.g. OSC gains on a joint-PD env raise); hoist OSC base gains to module constants. |
| `d303cbc` | `mjx_backend.py` Go2 block → a 6-line loop over `GO2_WARP_VARIANTS` + `_resolve_cls`. Research rationale (Kp-sweep design, hard-kick ladder, Jᵀ ablation, zero-shot-physical protocol, rough-hfield notes) migrated into the variants file. |
| `5ca7adc` | All 6 preset getters resolve Go2 names via `_resolve_go2_variant`; **unknown `Go2Warp*` names now RAISE** (`ValueError` with the known-names list). The silent fallback is dead. |
| `47c2df0` | `_resolve_go2_variant` splits algo-vs-train override keys; named error when a variant carries algo overrides but the getter has no algo config (PPO). |
| `b0df7f1` | **Behavior change (see below)** — 21 OSC/physical/rough variants get `_DR_TRAIN = {reset_mode: per_step, eval_every_n_episodes: 500}`. Also fixed stale `--eval-every` help text in the 5 off-policy train scripts. |
| `6596a2b` | **Legacy modules deleted**: `go2_warp_osc_joystick.py`, `go2_warp_osc_var_impedance.py`, `go2_warp_osc_rough.py`. `WarpOscJoystick` / `WarpOscVarImpedance` classes are gone — OSC envs are config presets over `WarpJoystick`. Snapshot pin: `tests/data/go2_warp_variants_snapshot.json`. |
| `fba9bad` | Regen env-presets + CLI reference docs; `gen_env_presets.py` resolves Go2 names through the getters (`_with_go2`) so the 29 variants stay listed. |
| `054ace4` | Snapshot regen one-liner in test docstring + cleanup. |
| `14a2f6f` | **OSC extraction completed** — `_run_osc` / `_feet_in_body` / `_compute_nominal_foot_body` + all gains state now live on the `OSC` component in `go2_warp_components.py`; the host env plants nothing. `mud_eval` callsite updated. |
| `d15d026` | Controller read-only env contract documented; stale comments fixed. |

Net: every Go2 Warp env is now ONE declaration in `go2_warp_variants.py`.
Adding a variant = adding one `EnvVariant(...)` entry.

## ⚠️ BEHAVIOR CHANGE — DR cut date 2026-06-11

**21 OSC / physical-motor / rough variants now default to
`reset_mode="per_step"` (DomainRandWrapper DR ON) +
`eval_every_n_episodes=500`.** Before this, the preset fallback bug meant
OSC envs trained with NO DR and almost no mid-run evals.

**Old OSC runs (anything before 2026-06-11) are NOT comparable to new
ones.** The joint-PD benchmark family (`Flat`, `TorqueSpeed`, `NoAccel`,
`Unitree`, joint HardKick, PosTrackProto) is unchanged; curriculum keeps
its historical per_step-only train dict. If OSC baselines matter for a
comparison, retrain them under the new defaults (TODO).

## Equivalence-gate methodology

- **Configs**: transitional equality tests pinned every variant's config
  byte-equal to legacy factory output BEFORE the legacy modules were
  deleted; then frozen as `tests/data/go2_warp_variants_snapshot.json`.
- **OSC extraction**: GPU/Warp is not bit-reproducible across processes
  (`.temp/REFACTOR_SANITY_CHECKS.md`), so the gate was: measure the
  cross-process Warp noise floor with **identity control** (same code,
  two processes), then show the refactor's old-vs-new trajectory
  deviation sits BELOW that floor. Refactor deviation is
  indistinguishable from run-to-run Warp noise.

## Smoke run (post-refactor end-to-end)

`Go2WarpOscFlatSoftPhysical`, FastSAC, 200k steps, 256 envs, `--eval-every
50` override (log `.temp/logs/smoke_osc_softphysical_1130.log`, ckpt
`checkpoints/20260611_113024_fast_sac_go2warposcflatsoftphysical_seed0`):

- **Best eval 87.5 ± 26.0** (@ 848 eps); final eval 32.3 ± 52.2.
- 929 episodes, online avg return (last 100 eps) 17.3; 17 evals fired
  (the eval cadence actually works now).
- Throughput ~600–690 sps at end (291 s wall) — healthy for OSC +
  physical motor + per_step DR at 256 envs.
- Q diagnostics sane: bias −0.19, RMSE 1.42, corr 0.72.

200k is a smoke (full runs are 5M+); the point is DR + eval cadence +
training all function end-to-end on a pure-preset OSC env.

## Test suite state

**836 passed / 46 skipped** on CPU + GPU OSC tests green. 6 failures are
PRE-EXISTING, not from this work:

1. `tests/test_go2_warp_curriculum_env.py::test_generated_scene_file_written`
   (GPU-marked) — 1 failure.
2. `tests/manipulation/factory/test_factory_action_chain.py` — 2 failures
   (`denormalize` got unexpected kwarg `unidirectional_rot`).
3. `tests/test_pusht_parity.py` — 3 failures (pymunk
   `add_collision_handler` API drift / `gym_pusht` import).

Filed in `.context/TODO.md`.

## Afternoon: first DR-era retrain (VarDampingAxis) + OOM saga

Steven picked `Go2WarpOscVarDampingAxisFlatPhysical` as the first retrain under
the new per_step-DR defaults (5M steps, 256 envs, seed 0, FastSAC).

**OOM at paper config.** The 4.19M-slot buffer (36-d action → ~6.4GB with dual
critic obs) + per_step DR wrapper + Warp does NOT fit on the 16GB GPU at any
`XLA_CLIENT_MEM_FRACTION` (0.75 default fails allocating 2.28GiB, 0.55 fails at
1.12GiB, 0.65 at 576MiB — all at init, 0 steps). Fraction tuning exhausted →
config change required.

**Decision (Steven): `--buffer-size 2097152` (2M) is the DR-era retrain
standard**, keeping 256 envs; launch with `XLA_CLIENT_MEM_FRACTION=0.65`. All
future ladder rungs use the same buffer for HP comparability. 2M still covers
40% of a 5M-step run's experience.

**Run healthy**: obs 72-d (last_act grows to 36), buffer 2,097,152 confirmed in
banner, first eval **97.2 ± 48.3 @ 500 eps**, wandb `go2-osc-impedance`.
Log: `.temp/logs/retrain_vardampaxis_5M_seed0.log`.

**Process note (cwd trap, again):** two launch attempts ran in the MAIN repo
instead of the worktree (no `cd` prefix; cwd does not reliably persist between
shell calls) and died with `Env not found` — masquerading as extra OOM data
points until the traceback was actually read. The `feedback_worktree_cwd`
memory rule exists for exactly this; `cd` EVERY command in worktrees.
