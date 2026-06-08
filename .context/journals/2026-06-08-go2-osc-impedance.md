# Go2 Cartesian-Impedance / OSC joystick — MVP (2026-06-08)

## Goal

New Go2 joystick env whose low-level controller is a **per-leg Cartesian
impedance / operational-space controller** (foot = end-effector), instead of
joint-space PD. MVP: fixed impedance, action = 4 foot xyz targets, same
velocity-tracking task/obs/reward. Downstream (later): stiffness in the action
space (variable impedance) + curriculum.

Worktree `go2-osc-impedance`. Spec:
`.superpowers/specs/2026-06-08-go2-osc-impedance-design.md`.

## What was built

- `jax_rl/envs/locomotion/go2_osc.py` — `compute_leg_impedance_torque`. Per-leg
  3-DoF, no nullspace (no redundancy), position-only, body-frame target → world
  error, leg-relative damping, `use_op_space_inertia` flag (Λ-OSC vs Jᵀ),
  **no gravity FF** (pure impedance).
- `jax_rl/envs/locomotion/go2_warp_osc_joystick.py` — `WarpOscJoystick`,
  overrides only `_apply_control`. action_scale 0.12 m, target_mode
  {abs_body, delta_current}.
- `go2_warp_joystick.py` — refactored to expose overridable `_apply_control`
  hook (behavior-preserving; 27 parent tests still pass).
- Registered `Go2WarpOscJoystickFlat`. Tests: 9 CPU (synthetic leg) + 4 GPU.

## Design decisions

- **Foot Cartesian targets, not joint targets** — the brief said "action = just
  joint targets" but the user clarified "just … targets" = *targets only, fixed
  impedance* (vs targets+stiffness later), NOT joint space. Compliance at the
  foot is the whole point.
- **No gravity FF** — on a floating base, `qfrc_bias[leg_dofs]` only cancels
  leg-self gravity (body weight rides through contact, not in qfrc_bias) and
  couples to base attitude. Net benefit ≈ nil → run pure impedance, spring bears
  the load. (Discussed and chosen deliberately with the user.)

## Results (5M FastSAC, wandb run mff6ptxj)

Eval **279.6 ± 4.6**. Trajectory analysis (reward alone never proves gait):
- Forward lock vx=1.0 → actual local vx **1.002** (last 400 steps); world dist
  9.7 m in 10 s = 0.97 m/s. Survives full episode, upright.
- Varied-cmd tracking corr: **vx 0.87 / vy 0.81 / yaw 0.94**.

**BUT the gait is jumpy (a pronk/pogo), quantified from foot-site FK:**

| metric | fwd | varied |
|---|---|---|
| flight phase (0 feet down) | **22%** | **24%** |
| mean feet on ground | 1.28/4 | 1.28/4 |
| max foot lift | 0.18 m | 0.30 m |
| base-z range (nominal 0.27) | 0.29–0.39 | 0.28–0.46 |
| vertical \|vz\| RMS / max | 0.24 / 1.18 | 0.32 / 1.49 m/s |

Spends ~¼ of the time fully airborne, feet flung to 30 cm, body launching to
46 cm. Videos: `.temp/videos/go2_osc_{fwd,varied}.mp4`.

## Why jumpy (working hypotheses — to study before tuning)

1. **Pure impedance + stiff gains = pogo.** kp=[3000,3000,4000] (acceleration
   gains; ω_n≈55–63 rad/s). The vertical spring stores/releases energy with no
   gravity FF to bleed it; horizontal momentum converts to vertical pop (audit
   Agent 4 reproduced this open-loop at base kicks ≥2 m/s).
2. **OSC gives feet low apparent inertia.** Λ normalizes the foot to unit mass,
   so flinging a foot 30 cm is "cheap" — the policy exploits big foot motions to
   track velocity, producing a bounding gait.
3. **Reward barely penalizes bounce.** `lin_vel_z` cost is −0.03/step (weight
   −0.5); `feet_height`/`feet_clearance` were tuned for a PD swing (~0.1 m), so
   30 cm lifts aren't suppressed; `action_rate` penalizes the *pre-scale* raw
   action, so foot-target jerk is under-penalized.

## 5-agent adversarial audit

Math (Jacobian/frame vs mj_jacSite+FD), per-leg index plumbing (name-based +
perturbation), numerics (max reachable ‖Λ‖=2.38 vs 1e4 ridge cap; fp32==fp64)
all verified clean. Findings logged in spec. Behavior-neutral fixes applied:
Λ floating-base caveat comment, stability-critical per-substep-recompute comment
(kp=4000 at 50 Hz diverges; 250 Hz substep keeps ρ≈0.8), delta_current
"can't passively stand" warning, name-based leg-mapping regression test.

## Update — Λ-OSC vs Jᵀ ablation (NULL result)

Hypothesis: OSC's Λ makes feet ~unit-mass → cheap to fling vertically → pogo.
So Jᵀ mode (no Λ, real anisotropic foot inertia, heavy along the leg) should
bounce less. Registered `Go2WarpOscJoystickFlatJt` (use_op_space_inertia=False,
hold-probed N/m gains kp=[1500,1500,2500] kd=[60,60,80]), retrained 5M identical
config (run on wandb go2-osc-impedance).

**Refuted.** Jᵀ eval 279.9 (Λ 279.6 — identical) and the gait is just as jumpy:

| metric (fwd / varied) | Λ-OSC | Jᵀ |
|---|---|---|
| flight phase | 22% / 24% | 27% / 27% |
| max foot lift | 0.18 / 0.30 m | 0.16 / 0.32 m |
| base-z max | 0.39 / 0.46 | 0.39 / 0.46 |
| vert \|vz\| RMS | 0.24 / 0.32 | 0.24 / 0.35 |

The controller's inertia model is **second-order**: the RL policy retrains
around it to the same bouncy gait. The pogo is driven by the **reward** (doesn't
penalize bounce) + **pure-impedance dynamics** (no gravity FF → the stiff spring
stores/releases energy), NOT by Λ. For this task, Λ-OSC vs Jᵀ is a wash.

Next levers (cheapest first): (a) reward — bump `lin_vel_z`, retighten
`feet_height`/`clearance`, fix pre-scale `action_rate`; retrain, measure flight.
(b) gravity / body-weight feedforward (foot_weight mode) so the spring stops
bearing weight via deflection → less stored energy. (a) is a no-controller-change
test of the reward hypothesis; do it first.

## Update — fixed-stiffness sweep (s = 0.25…4 × baseline kp, kd∝√s)

5 trained runs (`Go2WarpOscJoystickFlat{Kp025,Kp05,—,Kp2,Kp4}`), all 5M, seed 0.
Eval: s≤1 → ~280; s=2 → 266; s=4 → 246 (over-stiffening hurts). Gait response
(varied commands, the realistic test):

| s | flight % | max foot-lift | base-z max | yaw corr | energy | fell? |
|---|---|---|---|---|---|---|
| 0.25 | 7 | 0.15 | 0.32 | 0.94 | 109 | no |
| 0.5 | 8 | 0.23 | 0.40 | 0.94 | 166 | no |
| 1 | 24 | 0.30 | 0.46 | 0.94 | 178 | no |
| 2 | 21 | 0.39 | 0.49 | 0.71 | 255 | no |
| 4 | 20 | 0.32 | 0.42 | 0.74 | 278 | no |

Findings: (1) under **varied** commands soft stiffness is dramatically calmer —
**7–8 % flight at s≤0.5 vs ~22 % at s≥1** — at equal tracking and no falls; a
forward-lock-only test had hidden this (flat/noisy flight), so don't judge a
gait from one locked command. (2) Amplitude + effort scale monotonically with
stiffness (foot-lift, base bounce, energy 109→278). (3) Over-stiffening (s≥2)
degrades yaw tracking + eval. **Lower stiffness is a free win down to s≈0.25–0.5**
(calmer, ~60 % energy, equal tracking/robustness). The current default (s=1) is
too stiff. A residual ~7 % flight persists even at s=0.25 → that floor is the
reward, not stiffness. Bounce-hypothesis #1 (spring energy drives pogo) is thus
only partly true: stiffness sets bounce *amplitude/frequency*, but a reward-driven
flight floor remains.

DR note: training perturbs with a FIXED ±0.75 m/s base kick (in-env, every ~7 s,
not part of the DR spec) + per-episode physics DR (friction/mass/damping/…).
Kick magnitude is not randomized, and the audit's pogo-launch threshold is
≥2 m/s — so the sweep's "all robust" only covers mild pushes.

## Update — variable impedance (per-foot scalar, +4): `Go2WarpOscVarImpedanceFlat`

Action grows 12→16: 12 foot targets + 4 per-foot stiffness scalars. Each
a∈[-1,1] maps log-spaced to s∈[0.25,2] scaling that foot's baseline gains
(kp=s·kp_base, kd=√s·kd_base). Controller now takes per-leg gains (3,)-shared or
(4,3)-per-leg; parent reset sizes last_act by `action_size` (regression-safe,
action_size==nu for all fixed envs). obs 48→52 / priv 122→126. Tests: per-leg==
shared-gain equivalence (CPU) + var-env build (GPU). Per-foot-per-axis (+12,
action→24) is the planned follow-up. Not yet trained.

## Status / next

MVP done: builds, trains, walks, tracks, all tests green. NOT tuned. Next:
**study the jumpy/pogo behavior in the existing setting** before any reward/gain
change; then decide whether the bounce is fixable via reward (lin_vel_z,
feet_height) or needs a controller change (gravity FF / lower kp). Variable
impedance (stiffness in action) deferred until the fixed-impedance gait is
understood.
