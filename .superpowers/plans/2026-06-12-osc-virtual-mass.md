# OSC Virtual Mass — implementation plan

Spec: `.superpowers/specs/2026-06-12-osc-virtual-mass-design.md`
Branch: `go2-osc-impedance`. Law: `F_out = A·ẍ + K·err + D·ẋ`, `τ = Jᵀ·F_out`,
bare K,D (`use_op_space_inertia=False`), `A` = policy output, `ẍ` = finite-diff
foot velocity.

Each step is independently verifiable; commit per step. CPU tests where possible;
env-build / law-proof tests need the GPU (run when it frees — another job holds it).

---

## Step 1 — config knobs + controller routing
**Files:** `go2_warp_variants.py` (`go2_config`), `go2_warp_components.py`
(`controller_from_config`, `var_action_size`).
- `go2_config(..., mass_action=False, var_a=(A_MIN, A_MAX))`. When
  `mass_action=True` (requires `controller="var_impedance"`), emit
  `osc.mass_action=True`, `osc.var_a_min`, `osc.var_a_max`. Guard: `mass_action`
  with a non-var controller raises (mirror the existing damping_action guard).
- Pick provisional default range `var_a=(0.0, 2.0)` kg — **flagged in spec as a
  tuning knob**, not validated.
- `controller_from_config`: `osc.mass_action` truthy → `VarImpedanceMass`.
- `var_action_size` (or a mass-aware size): `12 + n*(2 if damping else 1) + n`
  when mass_action, `n = _N_STIFFNESS[gran]`.
**Verify:** `go2_config(controller="var_impedance", stiffness_granularity="per_axis",
damping_action=True, mass_action=True)` builds; the cross-axis guard raises on
`mass_action=True` with `controller="osc"`/`joint_pd`. CPU.

## Step 2 — `last_foot_vel` carry-state
**File:** `go2_warp_joystick.py`.
- `reset()` info: `"last_foot_vel": jp.zeros((4, 3))`.
- `step()`: after control, `state.info["last_foot_vel"] =
  data.sensordata[self._foot_linvel_sensor_adr].reshape(4, 3)` (same place as
  `last_torque`/`last_foot_force`). Reshape per the sensor layout.
**Verify:** smoke `reset`+`step` on an existing var-impedance env still runs;
`info["last_foot_vel"]` present, shape (4,3). GPU smoke (cheap).

## Step 3 — `VarImpedanceMass` controller
**File:** `go2_warp_components.py`.
- `class VarImpedanceMass(VarImpedance)`.
- `action_size` += `_N_STIFFNESS[gran]`.
- `setup()`: read `var_a_min/var_a_max`.
- `apply(env, data, action, info)`:
  - `deltas`, `kp`, `kd` exactly as `VarImpedance` (reuse `impedance_gains`).
  - decode `A = log_action_scale(action[tail], a_min, a_max).reshape(4,3)`
    (tail = after the stiffness/damping block).
  - `v_now = data.sensordata[env._foot_linvel_sensor_adr].reshape(4,3)`;
    `v_last = info["last_foot_vel"]`; `acc = (v_now - v_last) / env.dt` (ctrl_dt).
    (Guard `info is None` → `acc = zeros`, like the reward-term guards.)
  - call `self._run_osc(env, data, deltas, kp, kd, info, accel_force=A*acc)`.
**Verify:** unit — decode of a known action tail → expected `A`; `action_size`
== 48 for per_axis+damping+mass. CPU for the pure-decode helper if extractable,
else GPU.

## Step 4 — task-space mass force in the torque law
**Files:** `go2_warp_components.py` (`_run_osc` signature), `go2_osc.py`
(`compute_leg_impedance_torque`).
- `_run_osc(..., accel_force=None)`: thread `accel_force` (4,3, constant across
  substeps) into the substep call.
- `compute_leg_impedance_torque(..., accel_force=None)`: per foot, after
  `wrench = kp_i*err − kd_i*v`, **add the task-space mass force to F before Jᵀ**:
  `F = (Λ@wrench if use_op_space_inertia else wrench)`, then
  `F = F + accel_force[i]` (NOT inside Λ — `A·ẍ` is already a task-space force).
  For this controller `use_op_space_inertia=False` ⇒ `F = wrench + A·ẍ`.
  `τ += Jᵀ·F`. Keep `accel_force=None` → identical to today (zero behavior change
  for OSC/VarImpedance).
**Verify:** **prove-the-law unit test** — single foot, hand-set
`kp,kd,err,v,A,ẍ`, `use_op_space_inertia=False`; assert returned `Jᵀ⁻¹`-projected
force (or directly the intermediate `F`) equals `A·ẍ + kp·err − kd·v` to fp
tolerance. GPU.

## Step 5 — variants + snapshot + count pin
**Files:** `go2_warp_variants.py`, `tests/data/go2_warp_variants_snapshot.json`,
`tests/test_go2_warp_variants.py`.
- Register `Go2WarpOscVarMassAxisFlatPhysical` (mass_action var-impedance, flat)
  and `Go2WarpOscVarMassAxisFlatPhysicalMudDR4xSlowFirm` (mass + 4× mud DR +
  slow+firm reward profile — reuse `_var_muddr4x_slowfirm_config` with
  `mass_action=True`).
- Regenerate snapshot (one-liner in the test docstring).
- Bump the DR-target count pin in
  `test_osc_physical_variants_default_per_step_dr`.
**Verify:** `pytest tests/test_go2_warp_variants.py -k "not mud_field and not
registry"` green (snapshot + count). CPU.

## Step 6 — sanity smoke
- Build `Go2WarpOscVarMassAxisFlatPhysical`; `reset`+`step` with zero action;
  confirm `action_size==48`, reward components present, no NaN. GPU.
- Confirm the new terms are inert when `A·ẍ=0` is NOT assumed — i.e. the
  controller produces finite torques at the home pose.

## Step 7 — training + eval (sequential; needs exclusive GPU)
- Train `Go2WarpOscVarMassAxisFlatPhysical` 5M (`XLA_CLIENT_MEM_FRACTION=0.65
  --buffer-size 2097152 --seed 0 --wandb`). **Watch for ẍ-feedback divergence**
  (NaN / exploding actor loss); if it blows, lower `var_a_max`, re-launch.
- Record flat-forward gait; inspect per-foot `A` over the gait cycle (does the
  policy modulate mass: heavy stance / light swing?).
- Train the mud-DR sibling; Newton traverse eval (parity spawn,
  `record_traverse_maxfwd.py`) vs NoAir/winner.

---

## Notes / coordination
- Another agent is monitoring a separate job; **training + Newton eval are
  GPU-exclusive** — sequence Steps 6–7 around the other job; if they collide,
  ping the user on Slack (per their instruction).
- `cd` into the worktree at the start of every bash call. `uv run python` always.
  No Co-Authored-By in commits. Regenerate snapshot after every variant add.
- Steps 1–5 are CPU-testable except the GPU-build smokes; do all code + CPU
  verification first, batch the GPU steps when the GPU frees.
