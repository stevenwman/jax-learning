# 2026-05-03 — Splitbelt treadmill env build

## What landed

Built `Go2WarpSplitbelt` end-to-end via subagent-driven-development executing the
plan at `.superpowers/plans/2026-05-02-splitbelt-treadmill-env.md` (which itself
went through 4 audit rounds before execution).

### New files

**Pure Python / JAX modules:**
- `jax_rl/envs/locomotion/splitbelt_schedules.py` — 5 schedule samplers + dispatcher
  (`tied`, `split_constant`, `tied_split_tied`, `random_per_episode`, `continual_phase`).
  Cover A1/A2/A3/A4 protocol families. All pure functions, hermetic.
- `jax_rl/envs/locomotion/splitbelt_geom.py` — `BeltLayout` NamedTuple + `foot_belt_id`
  helper. The §5 risk-callout test from spec.
- `jax_rl/envs/locomotion/splitbelt_analysis.py` — pure-numpy offline gait analyzer:
  step-event detection, step lengths, step-length asymmetry.
- `jax_rl/envs/locomotion/go2_warp_splitbelt.py` — env class. Subclasses
  `Go2WarpEnv`. Overrides `action_size` (return 12, hides belt actuators from policy),
  `reset`, `step`, `_get_obs`, `get_domain_randomization_spec`, `get_control_metadata`.
  Full joystick reward set ported verbatim minus `stand_still`, plus new
  `treadmill_drift`. `_post_init` builds leg-only `_act_to_joint`, foot+torso
  contact sensor IDs, foot-linvel sensor adrs, then mutates belt forcerange and
  re-calls `mjx.put_model` with a smoke assertion (the only working fix per the
  round-3 audit — `Go2WarpEnv.__init__:64-67` clobbers forcerange before put_model).

**XML assets:**
- `jax_rl/envs/locomotion/xmls/treadmill_splitbelt.xml` — robot-agnostic apparatus.
  Two 50m × 0.30m × 0.01m belt slabs on slide joints + velocity actuators (`kv=200`,
  `forcerange=[-200, 200]`) + static `fallback_floor` plane in the gap.
- `jax_rl/envs/locomotion/xmls/go2_warp_splitbelt_scene.xml` — Go2 + treadmill.
  Include order: go2.xml first (so leg joints occupy qpos[7:19], belts qpos[19:21]).
  IMU + foot sensor aliases copied verbatim from `go2_warp_scene_flat.xml`. 17 contact
  pairs (feet × belts/floor + torso × belts/floor). `splitbelt_spawn` keyframe
  (qpos 21 tokens, ctrl 14 tokens).

**Constants:**
- `jax_rl/envs/locomotion/go2_constants.py` — added `LEG_ACTUATOR_NAMES` tuple
  (12 leg actuator names in MJX actuator order, FR/FL/RR/RL).

**Backend / presets:**
- `jax_rl/training/env_backends/mjx_backend.py` — registered `Go2WarpSplitbelt` via
  `pg_locomotion.register_environment(... functools.partial(Cls, task="splitbelt") ...)`
  inside `_register_custom_envs()`.
- `jax_rl/configs/env_presets.py` — added `PRESETS["Go2WarpSplitbelt"]` (PPO,
  reset_mode=per_step, episode_length=1250) and `FAST_SAC_PRESETS["Go2WarpSplitbelt"]`
  (FastSAC, same shape).
- `scripts/record_video.py` — added `splitbelt_hist` collector loop + sidecar
  `splitbelt_traj.npz` emission, gated on `info["splitbelt"]` key (auto-skip for
  non-splitbelt envs).

### Tests

Default lane: **753 passed, 46 skipped, 111 deselected** (baseline 719 + 34 new
splitbelt tests; 4 GPU smoke tests + 2 GPU bundle + 1 GPU+deploy ctrl_metadata
were the deselected delta).

**Hermetic (default lane):**
- `tests/test_splitbelt_schedules.py` — 10 tests
- `tests/test_splitbelt_belt_assignment.py` — 5 tests
- `tests/test_splitbelt_metrics.py` — 4 tests
- `tests/test_splitbelt_obs_schema.py` — 13 tests (incl. structural-drift assertion
  that `[t.name for t in build_obs_groups(env)[group]] == obs_term_names(mode)[group]`,
  hermetic via `SimpleNamespace` fake env)
- `tests/test_env_presets.py` — 2 splitbelt preset-shape tests

**GPU/Warp (deferred — written, not yet run):**
- `tests/test_splitbelt_env_smoke.py` `[gpu, warp, go2]` — reset, step, belt_qvel
  matches schedule, off-belt termination
- `tests/test_splitbelt_bundle.py` `[gpu, warp, go2]` — bundle shape + obs schema
  round-trip via `schema_from_obs_groups`
- `tests/test_splitbelt_control_metadata.py` `[gpu, warp, go2, deploy]` — deploy
  contract drift (`DEFAULT_POSE_POLICY`, `POLICY_TO_SDK`, `ACTION_SCALE`)

## Commits (16, on `new_slate_linen`)

```
3af917d feat(splitbelt): schedule samplers + dispatch (Tasks 1.1-1.6)
b8cab0d feat(splitbelt): foot_belt_id geometry helper + hermetic test
73e7ef1 feat(splitbelt): offline analysis - step events + step length asymmetry
616c585 feat(go2): add LEG_ACTUATOR_NAMES constant for splitbelt actuator filtering
2611707 feat(splitbelt): treadmill XML + Go2 splitbelt scene XML
367e94e feat(splitbelt): env class skeleton + obs_groups dispatch (Tasks 3.1-3.2)
a0ffb04 feat(splitbelt): full env class - _post_init + reset + step + reward (Tasks 3.3-3.5)
061d2d7 test(splitbelt): GPU/Warp env smoke (reset, step, belt qvel, off-belt term)
1d202e2 feat(splitbelt): register Go2WarpSplitbelt in mjx_backend
8362788 test(splitbelt): bundle + control metadata tests
c75873a feat(splitbelt): base PPO + FastSAC presets + hermetic preset test
d3d1aa8 feat(splitbelt): emit splitbelt_traj.npz sidecar from record_video.py
```

(Plus 5 spec/plan commits prior to execution: `4027412`, `5f54eb8`, `e1283c1`,
`b78f893`, `7c1a5bf`, `fc72d10`.)

## What's deferred to next session

1. **Calibration smoke (Task 5.1):** FastSAC + PPO 1M steps each on tied(0.5)
   belts. Go/no-go: eval > 80, PPO `entropy/mean ≥ 0.05`. Then visual verification
   via `record_video.py` (HARD RULE per bongo lesson). Per spec §11.X caveat: also
   grep `reward/pose|reward/feet_air_time|reward/treadmill_drift` to verify
   `treadmill_drift` isn't drowning smoothness terms.

2. **GPU test gate (Task 3.6 + 4.2 + 4.3):** run the three deferred test files
   on a GPU box. The forcerange smoke assertion in `_post_init` will fire on any
   env construction — that catches the `mjx.put_model` re-call regression
   immediately. The off-belt termination test uses a y=2.0 qpos override and
   needs `mjx.forward` to refresh sensors after the override; if it flakes, add
   a forward call.

3. **Per-protocol presets** (A1/A2/A3/A4) — fresh brainstorm/spec/plan cycle.
   Reuse the schedule samplers; add per-protocol algo presets.

4. **History-mode wrapper wiring** — `obs_term_names("history")` returns the
   `blind` layout but `FrameStackWrapper` is not gated in `mjx_backend` based on
   `cfg.obs_mode`. Defer until A3 protocol work.

## Audit-cycle observations

Plan went through 4 rounds of multi-agent audits before execution. Round-by-round
issue counts: R1 → 10 blockers + 5 highs; R2 → 8 cascade blockers (XML order +
`nu=14` axis); R3 → 5 new blockers (forcerange clobber, missing `LEG_ACTUATOR_NAMES`,
missing scene sensors, info-key drift, step_idx clamp); R4 → 1 HIGH + 1 BLOCKER
(`log_splitbelt` gate broke obs modes; keyframe `ctrl=` 13 vs 14 tokens). Pattern:
each fix round introduced ~30-40% of its own next-round issues from speculative
abstractions; ~25% were prior-audit blind spots; ~35% were only inspectable once
plan code became concrete. The `log_splitbelt` gate was speculative bloat that I
introduced in R4 fixes and reverted in R4 final.

Execution itself was uneventful: every plan task landed verbatim or with one
minor adaptation (e.g., env_presets.py's `PRESETS` / `FAST_SAC_PRESETS` actual
shape vs the plan's tuple-key pseudocode). No mechanical bugs caught at execution
that audits hadn't already flagged.

## Lessons / pointers
- Belt actuator forcerange MUST be re-mutated + re-put after `super().__init__`
  because `Go2WarpEnv.__init__:64-67` clobbers it before `mjx.put_model` snapshots
  the model. XML default alone is insufficient; verified empirically in audit.
- `info["command"]`, `info["last_act"]`, `info["last_last_act"]` keys must match
  joystick exactly so the verbatim-copied reward helpers (`_cost_feet_slip`,
  `_cost_feet_height`) read the right field. Earlier plan iterations used
  `info["cmd"]` / `info["last_action"]` and broke first-step `_get_obs`.
- `Go2WarpEnv.action_size` returns `mjx_model.nu` = 14 for splitbelt; must
  override to 12 to hide belt actuators from policy.
- Include order in scene XML matters: `go2.xml` first → leg joints qpos[7:19],
  leg actuators IDs 0-11; if reversed, every hardcoded slice is wrong.
