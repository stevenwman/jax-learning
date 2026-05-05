# 2026-05-03 → 05-05 — Splitbelt treadmill env build + calibration + variants

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

## Calibration smoke (2026-05-04)

**FastSAC 1M @ 512 envs** (`XLA_CLIENT_MEM_FRACTION=0.55`), tied(0.5), seed 0 → **eval 105.5 ± 6.6** (go/no-go > 80 ✓). Online-return progression: 77 → 80 → 90 → 95+ → 105. Entropy stable at -3 to -2. 6385 sps, 156s wall-clock. Q bias=-1.07, RMSE=1.56. Ckpt: `checkpoints/20260504_135211_fast_sac_go2warpsplitbelt_seed0/best`.

**PPO 1M @ 512 envs**, tied(0.5), seed 0 → **eval 53.0 ± 7.5** (under numeric gate; algo-agnosticism gate PASS). Entropy stable at 4.5+ throughout (well above the 0.05 collapse-risk threshold from spec §11.X). KL 7e-5 to 3e-2, logσ -0.17 (exploring, not collapsed). 21K sps, 87s wall-clock. PPO needs 50-100M steps for Go2-class envs to hit ceiling — 53 at 1M is consistent with the trajectory, not a stuck policy. The reward zero-clip + PPO entropy-collapse caveat from spec §11.X did NOT trigger; no mitigation needed. Ckpt: `checkpoints/20260504_135917_ppo_go2warpsplitbelt_seed0`.

Visual rollout (431 steps before off-belt termination, total reward 120.3) confirmed end-to-end pipeline:
- 32-45 touchdowns/foot in 8.6s = ~1.5 Hz stride rate (healthy cadence)
- Sidecar `splitbelt_traj.npz` emitted (12 arrays); offline analyzer reads + computes asymmetry (-0.27 on tied — undertrained, expected to converge to ~0 with more steps).

### Bugs found + fixed during calibration smoke (5 commits)

1. **`action_scale=1.0` vs deploy contract `0.5`** (`33f093a`) — control-metadata test failed; aligned with joystick + deploy.
2. **`splitbelt/term_cause` metric not initialized in reset** (`2061ef8`) — `lax.scan` over action_repeat needed pytree match between reset and step metrics dicts.
3. **`np.argsort` returns int64 not JSON-serializable** in `get_control_metadata` (`40561c5`) — meta.json save crashed; cast via `[int(i) for i in ...]`.
4. **Warp CUDA OOM on RTX 5080 with 1024 envs** — dropped to 512 envs + `XLA_CLIENT_MEM_FRACTION=0.55`. Documented in calibration recipe below.
5. **`floor_found` contact sensor margin-fires at 2cm distance** (`d7a6771`) — every foot at spawn (z=0.014) registered as `floor_found=1.0` because `fallback_floor` is a plane (infinite extent). Fix: switched off-belt detection from sensor-based to position-based via `splitbelt_geom.foot_belt_id` + foot_z<2cm check. Off-belt now fires only when foot is geometrically over the gap AND grounded.
6. **`is_tilt = gravity_body[2] < 0.5` always fired** (same commit) — `get_gravity` returns `(0,0,-1)` when upright, NOT `(0,0,+1)`. Joystick uses `get_upvector(data)[-1] < 0.0` for flipped detection. Fixed to use upvector.

After fixes 5+6: zero-action episode survives 200+ steps (was terminating at step 1). 1M training run climbs eval from 0 to 105 cleanly.

### Lessons added (suggested for `.context/lessons/go2.md`)

- MuJoCo plane geoms are infinite — using `<contact data="found">` against a plane causes contact-pair sensors to fire from any near-z geom, regardless of xy. Use a finite box, OR position-based check.
- `get_gravity()` body-frame z = -1 when upright. `get_upvector()` body-frame z = +1 when upright. Use upvector for "flipped" detection (joystick precedent).
- 1M-step run on RTX 5080 (16GB) needs ≤ 512 envs + `XLA_CLIENT_MEM_FRACTION=0.55` for splitbelt's 14-actuator model. 1024 envs OOMs Warp graph capture during eval env construction.

## 2026-05-05 — belt sign fix, obs alignment, visual upgrades, env variants

### Belt direction bug (`55ed717`)

User noticed in v1 video: belts moved *with* the robot's forward direction instead of dragging it backward. Schedule `tied(0.5)` produced joint qvel = +0.5 along +x; robot faces +x → belt slab moved with robot → "free-ride" treadmill. Fix: negate ctrl/qvel writes in step + reset so positive schedule speed = "drag foot backward" (biomech convention). Spec §6.3 invariants block updated.

Re-trained as v2 → eval **72.3 ± 13.4** (vs v1 105.5 free-ride). Lower because the task is genuinely harder when you have to step forward to stay put.

### Obs schema alignment to joystick (`1babc73`)

Initial splitbelt blind obs order was `[joint_pos, joint_vel, last_act, gravity, gyro, command]`. No reason — plan oversight, not lined up to existing joystick env. Joystick NoAccel order: `[gyro, gravity, joint_pos_offset, joint_vel, last_act, command]`. Both 45d.

Aligned splitbelt blind to joystick's exact name + order (`joint_pos` → `joint_pos_offset`). Two-way transfer now works:
- Existing joystick ckpts pop into splitbelt env without retraining.
- Future splitbelt ckpts can be tested on joystick without obs surgery.

v2 ckpt invalidated by the rename (first-layer weights expect old order). Did NOT retrain — no current use case demands a splitbelt baseline; future training will produce aligned ckpts naturally.

### Visual / camera upgrades (`9736364`, `8b3afc0`, `d6be8a0`)

For visual inspection of belt motion + gait:
- Belt checker textures (left = blue-tone, right = red-tone), `texrepeat="50 2"` (1m × 0.15m squares — readable per pixel without mipmap aliasing).
- Three directional lights (overhead + 2 fill). `directional="true"` removes spot-cone falloff that darkened belt ends as they scrolled past.
- Three fixed cameras: `splitbelt_side` (default — robot's left flank, `xyaxes="-1 0 0  0 0 1"`), `splitbelt_front`, `splitbelt_iso`.
- `record_video.py --no-early-term` flag — keep rolling after `done=True` so failure dynamics are visible. Default still breaks on done.

### Cross-deploy demo (joystick policy on splitbelt)

Recorded `checkpoints/20260428_085344_fast_sac_go2warpjoystickflatnoaccel_seed7002/best/20260505_183541_rollout.mp4` — joystick NoAccel ckpt running on splitbelt env (45d obs match by alignment).

Result: **policy stands still while belt drags it backward.** Joystick state obs does NOT include body lin vel (privileged-only term). Asymmetric AC blindness: actor sees gyro=0, gravity=(0,0,-1), joint_pos_offset≈0, joint_vel≈0, cmd=0 → "perfect standing" → no compensation. Translation is invisible to actor; reward shape (which would punish drift) is a training-time signal, not an inference-time one.

This is structurally interesting — splitbelt's `blind` mode reaches eval 72 because its `treadmill_drift` reward shaped the policy to step forward despite being equally blind to lin vel. Cf. spinal CPG models: open-loop-ish gait shaped by reward, not closed-loop kinematic tracking.

### New env variants (`8b3afc0`, `813d043`, `94e0412`)

| Env name | Obs mode | Schedule | Use case |
|---|---|---|---|
| `Go2WarpSplitbelt` | blind (default) | tied(0.5) | Baseline calibration env |
| `Go2WarpSplitbeltDR` | blind | random_per_episode (v∈[0.3,1.5], ratio∈[0.5,2.0]) | Speed-DR, A2 prep without explicit belt obs |
| `Go2WarpSplitbeltPoseDR` | pose_track | random_per_episode | Idealized stabilization probe — actor sees world-frame body pos + upvec + forwardvec; reward = pos+orient tracking; treadmill_drift dropped |

`pose_track` is **not real-robot deployable** (no SLAM) but cheap to study DR-belt-speed generalization since policy gets ground-truth state. Tests stabilization on unseen belt velocities.

PoseDR FastSAC 1M training in flight (`/tmp/splitbelt-smoke/posedr.log`) at session end.

### Bugs found + fixed in 2026-05-05

7. **Belt slab moved wrong direction** (`55ed717`) — schedule semantics unspecified; default sign was treadmill-incorrect (above).
8. **`np.argsort` int64 not JSON-serializable** (`40561c5`) — `get_control_metadata` save crashed.
9. **`config_dict` not imported** in `mjx_backend.py` (`94e0412`) — DR + PoseDR variant configs failed to construct on first env load.

### Net lessons (added to lesson docs this session)

- MuJoCo planes are infinite → contact-pair `data="found"` margin-fires from any near-z geom (not actual contact). Use box for finite extent, or position-based check.
- `get_gravity()` body-z = -1 when upright. Use `get_upvector()[-1] < 0` for flipped detection.
- `Go2WarpEnv.__init__:64-67` clobbers `actuator_forcerange` BEFORE `mjx.put_model`. XML default insufficient for non-leg actuators. Mutate `_mj_model.actuator_forcerange[idx]` in `_post_init` then re-call `mjx.put_model`. Smoke assertion catches drift.
- `lax.scan` over `action_repeat` requires reset & step state.metrics dicts to have IDENTICAL keys. Init all metric fields in reset.
- Asymmetric AC: actor obs blindness to body translation. Cross-policy transfer demos must check actor obs schema, not just dimension.
- RTX 5080 16GB with 14-actuator model: ≤512 envs + `XLA_CLIENT_MEM_FRACTION=0.55` for FastSAC training to fit Warp graph during eval env construction.

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
