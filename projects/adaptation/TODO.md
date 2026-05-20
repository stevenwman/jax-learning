# Adaptation Project — TODO

Repo-wide TODO is at `.context/TODO.md` (only items spanning multiple
projects). This file is splitbelt / adaptation-specific.

## Open

- [ ] **Retrain PoseDR with cross-belt termination** (2026-05-11). New
  `term_cause=4` lands in `Go2WarpSplitbeltEnv`; current ckpt was
  trained without it. Fresh 1M FastSAC should produce different gait
  (less reliance on cross-foot stance). See journal
  `2026-05-11-usd-and-cross-belt.md` for re-eval table.
- [ ] **Symmetrize splitbelt training distribution** — L-faster vs
  R-faster failure modes differ (cross-belt vs tilt). Mirror-augment
  spawn or flip belt assignment. Per lesson "DR doesn't extrapolate".
- [ ] **G1 splitbelt informed-actor train** (`G1WarpSplitbeltInformed` —
  belt_vel R^2 added to actor `state` obs). Env registered 2026-05-08.
- [ ] **Train PoseDR v3** with broader DR range (`vL ∈ [0.2, 2.0]`,
  `ratio ∈ [0.3, 3.0]`) + heavier orient reward (test the tilt-dominant
  hypothesis from `lessons/splitbelt.md`).
- [ ] **Phase 2 USD render pipeline** — Blender material setup, HDRI
  env light, Cycles. `.usd` export already working (Phase 1 done).
- [ ] **Train splitbelt with `error` obs_mode** — `cmd_track_error +
  drift_xy` in actor. Closes the AC blindness gap. Deployable variant of
  the pose_track idea (no world-pose cheat).
- [ ] **Train PoseDR with `tied_split_tied` schedule** for proper A1
  within-episode adaptation study. Current PoseDR sees mid-episode belt
  change as per-step OOD (terminates at t=776).
- [ ] **Per-protocol algo presets** (A1 / A2 / A3 / A4) — fresh
  brainstorm/spec/plan cycle per spec §11.3.
- [ ] **History-mode wrapper wiring** — `obs_term_names("history")`
  returns blind layout but `FrameStackWrapper` is not gated in
  `mjx_backend` based on `cfg.obs_mode`. Land before A3 protocol study.
- [ ] **Retrain `Go2WarpSplitbelt`** (tied(0.5) baseline) on
  tunneling-fixed env — only if a use case demands. v2 eval 72.3 ±13.4
  was on old env.
- [ ] **Train `Go2WarpSplitbeltDR`** on tunneling-fixed env — same
  caveat. No current ckpt.
- [ ] **Regenerate SEED_PROMPT.md** before next compaction (current
  content is dated 2026-05-05).

## Completed (2026-05-07) — Project framework + physics-metric eval + 3 lessons

Reorganized splitbelt-specific work into `projects/adaptation/` (mirrors
`projects/skill-discovery/` layout): `sweeps/` for orchestrators,
`videos/` + `artifacts/` (gitignored), `lessons/`, `specs/`, `plans/`,
`SEED_PROMPT.md`. Build journal stays in `.context/journals/` (chronological).

Belts widened from y=±0.300 to ±0.500 (each half-width 0.250m, total 1m
span). High-contrast checker textures (texrepeat 200×4, ~25cm tile size,
deep blue↔light blue / dark orange↔light orange). Belt direction unmistakable
in video.

`projects/adaptation/sweeps/eval_physics.py` — cold-hard physics-metric
sweep (lag, sway, termination cause breakdown, survival steps). First
sweep on PoseDR v2 reveals: only tied 0.5 has 0% termination; every
other tested condition has 75-100% term. Tilt is dominant failure mode
(91/144 ep across 9 conditions); 0 torso-falls.

3 new lessons in `lessons/splitbelt.md`:
- Eval reward hides physics failures
- Tilt is the dominant station-keeping failure
- DR doesn't extrapolate, only interpolates

## Completed (2026-05-06) — Foot tunneling fix + PoseDR v2 + OOD sweep

Found PoseDR v1 was exploiting a foot-tunneling physics bug: FL was passing
through belt slab (z=-0.05 for 95% of frames) → "free anchor" inflated
eval to 280.6.

Three fixes landed:
1. Closed vestigial 5cm center belt gap (left/right edges meet at y=0).
2. Added 4 missing cross-belt foot collision pairs (FL×right, FR×left,
   RL×right, RR×left). Spec assumed feet stay on assigned belt; lateral
   drift breaks the assumption.
3. `<pair margin="0.02">` on every foot×belt + foot×fallback_floor pair.
   Compensates for MJX's lack of CCD: contact engages 20mm above belt →
   fast-approach foot decelerated before reaching slab → no tunneling.

PoseDR v2 retrained on fixed env (1M FastSAC, ~2.5min): **eval 378.9 ± 113.4**
vs v1 280.6 ± 207.8. Higher and tighter — real friction-based stationkeeping.
Ckpt: `checkpoints/20260506_195126_fast_sac_go2warpsplitbeltposedr_seed0/best`.

OOD belt-speed sweep (`projects/adaptation/sweeps/eval_ood.py`, 16 ep/v):

| v   | mean  | OOD? |
|-----|-------|------|
| 0.30| 215.6 | (in-dist boundary low) |
| 0.50| **393.2** | peak |
| 1.00| 378.0 | strong |
| 1.50| 125.5 | (in-dist boundary high) |
| 2.00|  65.1 | OOD (17% peak) |
| 2.50|  29.7 | OOD (8% peak) |

Visual: `splitbelt_side_iso` cam (~16° off pure side, shows lateral sway).

A1 probe (`projects/adaptation/sweeps/eval_a1.py`): right touchdown rate
↑ during split phase (0.5/1.0), early termination at t=776 (mid Phase 3).
Mid-episode belt change is per-step OOD for PoseDR (trained on
constant-belt episodes). Built-in `step_length_asymmetry` numerically
broken for cmd=0 stationkeeping (designed for walking gaits) — switched
to touchdown-rate asymmetry.

## Completed (2026-05-05) — Splitbelt env variants + visual upgrades

Belt sign fix (drag-backward biomech convention), obs schema aligned to
joystick NoAccel (cross-deploy works both ways), checker textures +
directional lights + fixed side cam, three env variants registered.
PoseDR v1 trained eval 280.6 (later INVALIDATED 2026-05-06 by tunneling
discovery — see above). Cross-deploy demo: joystick policy on splitbelt
env → robot stands while belt drags. Asymmetric-AC actor blindness:
body lin vel is privileged-only.

## Completed (2026-05-03) — Splitbelt treadmill env (Go2WarpSplitbelt)

Built the splitbelt-treadmill adaptation-benchmark substrate (spec at
`projects/adaptation/specs/2026-05-02-splitbelt-treadmill-env-design.md`).
Two parallel belt slabs on slide+vel actuators over a `fallback_floor`
gap, robot-agnostic apparatus + Go2-specific scene. Schedule samplers
(tied / split_constant / tied_split_tied / random_per_episode /
continual_phase) plus dispatcher cover all four protocol families
(A1 within-episode, A2 context-conditioned, A3 meta-RL, A4 continual).
4 obs modes (blind / informed / error / history); reward = full
joystick set + new `treadmill_drift` term (`stand_still` dropped since
cmd is always 0). Offline gait-asymmetry analyzer in
`jax_rl/envs/locomotion/splitbelt_analysis.py`. PPO + FastSAC base
presets registered.

Tests landed (`JAX_PLATFORMS=cpu uv run python -m pytest -q` —
**753 passed**, +34 over baseline):
- `tests/test_splitbelt_schedules.py` — 10 schedule-sampler hermetic tests
- `tests/test_splitbelt_belt_assignment.py` — 5 foot_belt_id tests
- `tests/test_splitbelt_metrics.py` — 4 offline analyzer tests
- `tests/test_splitbelt_obs_schema.py` — 13 obs-name-layout + structural-drift tests
- `tests/test_env_presets.py` — 2 splitbelt-preset shape tests
- GPU/Warp tests written but deferred (need GPU box):
  - `tests/test_splitbelt_env_smoke.py` — reset/step/belt_qvel/off-belt term [gpu, warp, go2]
  - `tests/test_splitbelt_bundle.py` — bundle shape + obs schema round-trip [gpu, warp, go2]
  - `tests/test_splitbelt_control_metadata.py` — deploy contract drift [gpu, warp, go2, deploy]

Plan at `projects/adaptation/plans/2026-05-02-splitbelt-treadmill-env.md`
(4 audit rounds before exec).
