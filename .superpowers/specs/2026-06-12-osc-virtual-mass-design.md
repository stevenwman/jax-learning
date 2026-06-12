# OSC Virtual Mass (acceleration-feedback) — design

**Date:** 2026-06-12
**Branch:** `go2-osc-impedance` (unmerged)
**Status:** design — implementation NOT started.

## Motivation

Our Cartesian-impedance / OSC controller lets the policy modulate virtual
**stiffness** (`K`) and **damping** (`D`) per foot / per axis, but **not virtual
mass**. Adding a policy-controlled virtual mass gives the foot a momentum/inertia
knob — distinct from K and D — hypothesised to help the Newton-mud traverse
(e.g. a heavier foot to plow through / resist bogging, a lighter foot to snap out
of mud). This spec adds that knob.

## Final control law (locked by user)

Per foot, in the trunk/world task frame, **bare K, D** (no operational-space
inertia weighting):

```
F_out = A·ẍ + K·err + D·ẋ          τ = Jᵀ·F_out
```

- `err = x* − x`  (foot position error, trunk frame; `x*` = nominal + policy delta)
- `ẋ`  = foot velocity (sensed, `J·q̇_leg`)  — damping term keeps the existing
  damping-opposes-velocity sign: implemented as `− kd·ẋ` (user's `+D·ẋ` is sign
  notation; D is a dissipative coefficient).
- `ẍ`  = foot **acceleration**, **sensed via finite difference** (see below).
- `A`  = **policy-output virtual mass**, per foot × per axis (diagonal). NEW.
- `K`, `D` = the existing variable stiffness/damping (per-axis), unchanged decode.

This is a **direct computed force** (like the current controller): we sum the
terms and send `τ = Jᵀ·F_out`. It does **not** render a target admittance, so it
needs **no external-force (F_ext) estimate** — the contact force's effect rides
in through the sensed `ẍ` (acceleration feedback).

### Decisions locked during design

- **Bare K, D** (`use_op_space_inertia = False`). The validated mud recipe used
  the Λ-weighted form (`= True`); this is a **different controller**, so its
  results are re-validated from scratch — there is intentionally **no
  A=0 == current-controller baseline** and **no A1/A2 (model-based vs model-free)
  comparison**. `A` is the policy output directly (the model-free form).
- **No inertia-cancel clamp / no Λ in the mass term.** Earlier discussion of
  flooring apparent inertia at "cancel foot inertia" is **dropped** per user.
  `A` is simply a bounded policy output (range = config knob, see Risks).
- **ẍ from finite difference** of foot velocity, smoothing added only if needed.

## Closed-loop note (informational)

With bare K,D the realized dynamics are `(Λ_real − A)·ẍ = K·err + D·ẋ + F_ext`,
i.e. `A` shapes the apparent foot inertia `(Λ_real − A)` via acceleration
feedback. Going lighter (`A → Λ_real`) raises the `(Λ_real−A)⁻¹` gain → the known
acceleration-feedback instability; we do NOT clamp it in code (per user), so the
**A range is the stability lever** (see Risks). Heavier-than-natural is
unconditionally stable.

## ẍ (foot acceleration) sourcing

Initial implementation = **control-step finite difference**, held constant across
the substep decimation loop:

```
ẍ = (v_foot_now − v_foot_last) / ctrl_dt
```

- `v_foot_now`: foot world linear velocity from the existing foot-linvel sensor
  (`_foot_linvel_sensor_adr`), read at `apply()` entry, shape (4,3).
- `v_foot_last`: carried in `info["last_foot_vel"]`, updated in `step()` (same
  pattern as `last_torque` / `last_foot_force`).
- `ctrl_dt` = 0.02 (50 Hz). `ẍ` is computed once per control step and broadcast
  across the n_substeps (250 Hz) impedance loop.
- **Smoothing (deferred):** optional EMA on `ẍ` if finite-diff noise destabilises;
  off by default.
- **Higher-fidelity option (deferred):** substep-rate finite-diff (carry prev foot
  vel inside the `lax.scan`). Only if the control-step `ẍ` is too coarse.

## Action layout

Reuses the variable-impedance decode; appends the mass block.

```
per_axis + damping + mass:
[ 12 foot deltas | 12 stiffness s | 12 damping ζ | 12 mass A ]   → action_size 48
                                                   └ NEW (per foot × axis)
```

- `A` decoded via `log_action_scale(action[36:48], a_min, a_max)` → reshape (4,3),
  mirroring the stiffness `s` decode. `a_min, a_max` are config knobs
  (`var_a_min`, `var_a_max`).
- per_foot granularity (scalar A per foot, +4 dims) is a possible variant but the
  default is **per_axis** to match the per-axis stiffness/damping.

## Code seams (where it lands)

All in `jax_rl/envs/locomotion/`:

1. **`go2_warp_components.py`**
   - New controller `VarImpedanceMass(VarImpedance)` (or a `mass_action` flag on
     `VarImpedance`). `action_size` += `_N_STIFFNESS[gran]`.
   - `setup()` reads `var_a_min/var_a_max`.
   - `apply()`: decode `A` from the action tail; compute `ẍ` from
     `info["last_foot_vel"]` + current foot-linvel sensor; pass `A` and `ẍ` into
     `_run_osc`.
   - `_run_osc` / `compute_leg_impedance_torque`: add the task-space mass force
     **before** `Jᵀ` — `F = wrench + A ⊙ ẍ` (with `use_op_space_inertia=False`,
     `F = wrench`, so `F = (K·err − D·ẋ) + A·ẍ`), then `τ = Jᵀ·F`. Extend the
     torque fn with an optional `accel_force=(n_legs,3)` arg (added to `F`, not to
     `tau`, since it's a task-space force).
2. **`go2_warp_joystick.py`**
   - Add `info["last_foot_vel"]` to `reset()` (zeros (4,3)) and update it in
     `step()` after control.
3. **`go2_warp_variants.py`**
   - `go2_config()`: accept `mass_action=True` (+ `var_a` range) → emit
     `osc.mass_action`, `osc.var_a_min/max`. `controller_from_config` routes to
     `VarImpedanceMass`.
   - Register `Go2WarpOscVarMassAxisFlatPhysical` (gait/sanity) + a mud-DR sibling
     `...MudDR4xSlowFirm...` (the Newton traverse test, reusing the slow+firm
     reward profile).
   - Regenerate the variants snapshot; bump the DR-target count pin.

## Eval plan

1. **Sanity:** env builds, `action_size == 48`, one `step()` runs; numerically
   **prove the law** — assert `F_out == A·ẍ + K·err + D·ẋ` for a hand-set
   `(A, K, D, ẍ, err)` on one foot (unit test, GPU).
2. **Flat gait:** train 5M (slow+firm reward), record flat-forward; does the policy
   use mass (heavy stance / light swing)? Inspect per-foot A over the gait cycle.
3. **Newton mud (parity spawn):** traverse metric vs NoAir / winner — does the mass
   knob deepen the traverse?
4. **Stability watch:** monitor for `ẍ`-feedback blow-up; if it appears, tighten
   `var_a_max` and/or enable `ẍ` EMA smoothing.

## Risks / open knobs

- **Acceleration-feedback instability.** Large `A` (apparent inertia → 0) +
  finite-diff `ẍ` noise/delay can diverge. Mitigations (in order): conservative
  `var_a_max`; control-step `ẍ` is delayed-by-one which *helps*; EMA smoothing;
  fall back to substep-rate `ẍ` only if needed. **`var_a_min/var_a_max` need a
  tuning pass — they are the primary stability lever and have no validated
  default yet.**
- **Bare K,D ≠ validated controller.** The proven mud gains (kp 3000–4000) were
  tuned for the Λ-weighted form. Bare K,D may need its own gain re-tune before the
  mass knob is even exercised. Budget a baseline bring-up.
- **Deploy:** `ẍ` (foot accel) is sensed here from sim; on hardware it's a noisy
  double-diff of encoders / a foot IMU. Out of scope for this spec (sim study),
  but the finite-diff + smoothing path mirrors what hardware would do.

## Out of scope (explicitly)

- F_ext estimation / admittance rendering (the "true apparent-inertia" arm) —
  dropped; not needed for this law.
- A1/A2 model-based vs model-free comparison — dropped.
- Feedforward acceleration *command* (policy outputs ẍ_d) — different feature,
  not this one.
