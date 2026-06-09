# 2026-06-09 — Go2 OSC: physical motor model + zero-shot-physical retrain

Continues `.context/journals/2026-06-08-go2-osc-impedance.md`. Branch
`go2-osc-impedance` (worktree). Driven by the 40 rad/s calf flagged in the
motor-model quantification.

## Quantification (from `.temp/videos/05_motormodel_smoke/`, flat OSC traj)
Per-joint torque-vs-velocity on the flat OSC MVP: hip 11.8 / thigh 17.6 rad/s
(both inside the 30.1 limit), but **calf 40 rad/s — 2× its 20.07 no-load
limit**, 2% of steps over, ~0.3% "motoring" (producing torque above no-load
speed = physically impossible). Problem is concentrated in the cam-driven knee.

## Finding that flipped the premise
"Knee is under-inertia'd → raise armature" was wrong. mjlab go1 (same motor
family) armatures: hip/thigh `rotor·6²=0.004026`, knee `rotor·9²=0.009059`.
**Our uniform 0.01 is already ≥ mjlab's calf 0.009**, and our hip/thigh are
2.5× mjlab's. So the 40 rad/s was never inertia — it's the **missing
torque-speed cap** (which was OFF). Also: mjlab go1 itself uses a position
actuator + armature, no torque-speed curve; the DcMotor 4-quadrant is a
separate, more-realistic class it doesn't use for the quadruped.

## Implemented (TDD, 17 tests in `tests/test_torque_speed_model.py`)
- `go2_warp_base.torque_speed_clip` — mjlab `_clip_effort` 4-quadrant port
  (per-joint stall / velocity_limit / continuous effort; actively brakes
  over-speed). `_apply_torque_speed_limit` delegates to it. Gated by
  `torque_speed_model`.
- `go2_warp_base.physical_armature` — sets mjlab per-joint armature by joint
  name (`*_calf_joint` → knee). Constants in `go2_constants.py`
  (`MOTOR_ROTOR_INERTIA/ARMATURE_HIP/KNEE`). Gated by separate
  `physical_armature` flag (kept independent so the existing `FlatTorqueSpeed`
  ablation env is unaffected). Mutates `dof_armature` before `mjx.put_model` →
  flows into `full_m` → **OSC Λ is armature-aware automatically**.
- Both flags ON for rough envs only (`_add_rough`); flat envs unchanged for
  reproducibility. User chose match-mjlab-exactly for armature.

In-env A/B smoke (same kp=3000 env + terrain seed, only `torque_speed_model`
toggled): calf max |qvel| **44.0 → 20.5 rad/s**, no NaN. The flat policy
terminates earlier with the model on — its gait depended on the unphysical fast
calf → confirms a retrain is needed.

## Retrain — zero-shot from flat-physical (RUNNING, ~6h)
Protocol chosen (user): preserve the original zero-shot claim by making the
motor model consistent across train+eval. 4 new flat-physical envs
`Go2Warp{JoystickFlatPhysical, OscFlatSoftPhysical, OscVarFlatPhysical,
OscVarAxisFlatPhysical}` (flat scene + both physical flags, gains mirror the
rough eval envs). Train fast_sac 5M / 256env / seed 0 → zero-shot eval on
`Go2Warp*RoughUni`. Driver `.temp/scripts/run_physical_retrain.sh` (detached),
logs `.temp/logs/physical_retrain_<env>.log`, wandb `go2-osc-impedance`. First
run (joint-PD) healthy (Return climbing, 0 NaN).

## RESULT — the headline survives the physical motors (cleaner than before)
All 4 retrains done (Returns 261–265, tight). Zero-shot eval on the matching
`*RoughUni` env (physical, varied cmd, identical terrain rough_seed=0, K=8
seeds, max 500; fall = env early-termination = npz len < max):

| controller | falls | mean survival | tracking |
|---|---|---|---|
| joint-PD | **5/8** | 329/500 | 6.40 |
| OSC fixed-soft | **1/8** | 481/500 | 7.59 |
| var per-foot | **1/8** | 443/500 | 6.98 |
| var per-axis | **1/8** | 444/500 | 6.74 |

Under the physical motor model (calf no longer flings to 40 rad/s), joint-PD
still flips **5/8 (63%)** while all three Cartesian-impedance controllers fall
**1/8** — and OSC tracks better (7.59 vs 6.40) + survives longer. Compliance-
helps-on-rough is **confirmed**, in fact cleaner than the original (~3/6).
Variable impedance shows NO survival edge over fixed-soft (same as flat).
Eval script `.temp/scripts/eval_rough_physical.py`, per-seed npzs in
`.temp/eval_rough_phys/`, exemplar clips `.temp/videos/06_rough_physical_eval/`.

NEXT: commit the (still-uncommitted) motor-model impl + this result. Optional
follow-ups: multi-seed TRAINING (not just eval) for error bars; the
"train-on-rough" arm (Option B) if we want inherent-vs-learnable separation.
