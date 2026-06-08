# Go2 Cartesian-Impedance / OSC Joystick — Design (2026-06-08)

## Goal

A Go2 velocity-tracking joystick env whose low-level controller is a **per-leg
Cartesian impedance / operational-space controller** (the foot is the
end-effector), instead of the joint-space PD used by every existing Go2 env.

**MVP (this spec):** fixed impedance, policy action = four foot-position
targets, same velocity-tracking task / obs / reward as the joint-PD joystick.
Get a working walking policy.

**Downstream (not this spec):** add per-foot stiffness to the action space
(variable impedance) + a curriculum that lets the policy adapt compliance.

## Design decisions (and the reasoning)

- **Action = foot Cartesian targets, not joint targets.** Early ambiguity in the
  brief ("action space is just joint targets") was resolved with the user:
  "just … targets" meant *targets only, impedance fixed* (vs targets + stiffness
  later), NOT joint space. The compliance lives at the foot (Cartesian), which
  is the whole point of "ee OSC instead of joint space."
- **Per-leg, 3-DoF, no nullspace.** Each leg (hip/thigh/calf) drives one foot.
  A 3-DoF leg on a 3-DoF foot-position task has no redundancy → no nullspace
  term (unlike the 7-DoF Factory arm). Position only — feet are points, no quat.
- **Body-frame target, world-frame error.** Targets are in the trunk
  (`base_link`) frame so the robot can walk (a world-fixed foot target would pin
  it). Target is rotated to world via `R_body`; error is a plain world vector.
  Isotropic-ish gains ⇒ frame of the wrench is immaterial; anisotropic gains are
  treated as world axes (≈ body axes while near-upright). Documented in
  `go2_osc.py`.
- **Damping is leg-relative.** We damp `J_leg·q̇_leg` (leg-induced foot velocity),
  not full world foot velocity — base-induced motion is not leg-controllable.
- **No gravity feedforward (MVP).** On a floating base, `qfrc_bias` sliced to leg
  DoFs only cancels each leg's own link gravity (body-weight support rides
  through contact, absent from `qfrc_bias`) and couples to base attitude. Net
  benefit ≈ nil, so the MVP runs **pure impedance** — the spring bears the load.
  `gravity_ff` config left as an enum (`none` | `joint_bias` | `foot_weight`)
  for a future ablation; `foot_weight` (Jᵀ·per-foot weight share) is the
  quadruped-meaningful one, not the naive `joint_bias` arm port.
- **OSC vs Jᵀ impedance is a flag.** `use_op_space_inertia=True` → full Khatib
  OSC (Λ = (J·M⁻¹·Jᵀ)⁻¹ inertia weighting, foot behaves as unit mass).
  `False` → classic Jacobian-transpose Cartesian impedance with honest N/m
  gains. Free ablation; default True (matches the "OSC" ask).
- **Target modes.** `abs_body` (default): target = nominal_foot + action·scale,
  fixed trunk-frame anchor. `delta_current`: target = current_foot + action·scale
  (integrative, moving anchor). `delta_from_nominal` is just `abs_body` (affine
  reparam) — not a separate mode.

## Gains (tuned, not guessed)

Zero-action hold probe (no gravity FF → spring must hold body weight):
`kp=[800,800,1000]` sagged base 0.27→0.185 m (≈ termination floor 0.18);
`kp=[3000,3000,4000] kd=[110,110,130]` holds 0.270 m, survives σ=0.2 noisy
actions (zmin 0.255, no NaN). `kd ≈ 2·√kp` ≈ critical damping of the unit-mass
OSC loop. `action_scale=0.12 m` (joint env's 0.5 rad is meaningless for feet).

## Files

- `jax_rl/envs/locomotion/go2_osc.py` — controller, `compute_leg_impedance_torque`.
- `jax_rl/envs/locomotion/go2_warp_osc_joystick.py` — `WarpOscJoystick`,
  subclasses `WarpJoystick`, overrides only `_apply_control`.
- `jax_rl/envs/locomotion/go2_warp_joystick.py` — refactored: joint-PD loop
  extracted into overridable `_apply_control` (behavior-preserving).
- `jax_rl/training/env_backends/mjx_backend.py` — registers
  `Go2WarpOscJoystickFlat`.
- `tests/test_go2_osc.py` — synthetic-leg unit tests (CPU, controller math).
- `tests/test_go2_osc_env.py` — Go2 GPU integration (DoF mapping, FK, build).

## Tests (all green)

- Unit (CPU, synthetic 3-DoF leg): equilibrium → τ=0; **OSC mode q̈_foot ==
  kp⊙err exactly** (defining Λ property, impl-independent); Jᵀ-mode sign;
  velocity damping opposes motion; torque clip. (9 cases.)
- Integration (GPU): builds/steps no-NaN; **leg-DoF block isolation** (each foot
  Jacobian nonzero only on its own 3 DoFs — validates the index assumption);
  nominal-foot FK consistency (numpy vs mjx paths). (3 cases.)

## Known limitations (from 5-agent adversarial audit, 2026-06-08)

Verified clean: Jacobian/frame math (vs `mj_jacSite` + finite-diff), per-leg
index plumbing (name-based + perturbation probes, all 4 legs), numerical
robustness (max reachable ‖Λ‖=2.38 vs 1e4 ridge cap; fp32 == fp64).

Carried forward (none block the abs_body MVP):
- **`delta_current` can't passively stand** — zero action has no restoring
  force vs gravity; collapses in ~0.25 s. Documented loudly; abs_body is the
  default. Needs an upward action bias or gravity FF to be usable.
- **Per-substep recompute is stability-critical** — kp=4000 held over a 50 Hz
  step diverges (ρ≈2.8); the 250 Hz substep recompute keeps ρ≈0.8. Commented
  in `_apply_control` so nobody hoists it out of the scan.
- **Λ is leg-block-only on the floating base** — unit-mass decoupling is
  ~10–20% approximate; steady sag ≈18 mm (2× the 1-D estimate). Sign always
  correct, stable. True decoupling needs whole-body inertia (out of scope).
- **Reward terms are PD-tuned** (inherited unchanged): `action_rate` penalizes
  the pre-scale raw action (keeps PD calibration); `feet_clearance`/`energy`/
  `torques` run 2–4× larger under OSC and saturate the calf stall. Retune when
  shaping the OSC gait. Pre-existing landmine inherited: reward lower-clip at 0
  nullifies the `termination=-1` term (bites OSC harder early).
- **Pure-impedance pogo fragility** — open-loop base kick ≥2 m/s launches the
  robot airborne (vs joint-PD surviving to 3 m/s). Inherent to no-FF
  compliance; training kicks (±0.75) are within the safe band.

## Acceptance

MVP done when a 5M FastSAC run on `Go2WarpOscJoystickFlat` produces a policy
that **walks and tracks commands** — verified by eval reward AND recorded video
+ lock-cmd probes (eval reward alone never proves walking, per repo lore).
