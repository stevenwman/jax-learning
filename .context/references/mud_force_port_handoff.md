# Mud force-model port — handoff reference

**Purpose.** There's a *reduced-order analytic* mud force model in the `mud` Isaac
Lab repo — foot wrenches (suction/resistance + shear), gated on foot depth, applied
every step. It's analytic in (foot z, foot velocity) → unlike the Newton MPM, it's
**fully traceable, so it's portable into the MJX/Warp Go2 env for TRAINING**. This
doc is the physics + the source files + how to apply it, so you can conceptualize
the port. Reward / curriculum / eval protocol are out of scope (your call).

Context: the flat-trained Go2 policies bog in mud (Newton MPM eval, `projects/mud_eval/`).
The lever is to train *on* mud. Newton isn't JAX-traceable → use this analytic field
as the trainable proxy, and keep the Newton MPM harness as the **held-out test**.

---

## Source — the model to port (`/home/stevenman/Desktop/Work/Research/mud`, Isaac Lab)

- **`source/mud/mud/tasks/manager_based/mud/forces_cfg.py`** — the force functions.
- **`source/mud/mud/tasks/manager_based/mud/velocity_env_cfg.py`** — registration as
  `EventTerm`s (params + DR ranges + schedule), search `suction_and_resistive_force_term` /
  `shear_force_term` / `reset_mud_depth`.

Mechanism: external **wrench on the `.*_foot` bodies**, `mode="interval"`,
`interval_range_s=(0.005,0.005)` (every sim step), applied **only to feet currently
submerged** (`z_foot < mud_height`), via `instantaneous_wrench_composer.set_forces_and_torques(...)`
+ `write_data_to_sim()`. Force-only, no torque.

### Mud depth (`rand_mud_depth`, fires on reset)
Per-env scalar `env.mud_depth ~ U(depth_range)` = a **flat mud surface height**.
Foot-in-mud gate: `in_mud = z_foot < mud_height`. (Currently fixed 0.22 m for PD
testing; designed to randomize.)

### Term 1 — suction + resistance (vertical, world Z)
```
depth_ratio = ((mud_height − z_foot) / mud_height)²            # quadratic in submersion, ≥0
lumped_coeff ~ U(f_range)   per foot                           # lumped stiffness/damping
sign = −1 if z_vel > 0 else +1                                 # THE TRICK:
z_force = lumped_coeff · depth_ratio · sign · in_mud           #   foot moving UP   → force DOWN = SUCTION (resists pull-out)
                                                              #   foot moving DOWN → force UP   = RESISTANCE (resists sinking)
```
Provenance: a curve from an *intrusion/withdrawal reduced-order robotic test rig in mud*.

### Term 2 — shear drag (xyz, opposes motion)
```
mud_depth   = (mud_height − z_foot).clamp(≥0)
area        = leg_circumference(~U(surface_area)) · mud_depth   # contact area ∝ intrusion depth
shear_stress= area · (log(|v| + 1e-3) + m·|v|)                  # Herschel–Bulkley-ish, m = c1 (viscous coeff)
shear_stress += b   if horizontal speed |v_xy| > 0.3            # yield-stress offset b = c2, only while moving
force       = −clamp(shear_stress, ≥0) · v̂ · in_mud             # opposes the foot's velocity direction
```
`v` = full foot world velocity (xyz); `v̂ = v/|v|`. Provenance: a *boat paper on shear
in mud*, constants approximated from boats traversing mud.

### DR knobs + the ranges as registered
`depth_range=(0.22,0.22)` · `f_range=(14,15)` · `c1=(9,10)` · `c2=(6,7)` ·
`surface_area=(0.1,0.14)`. **Quirk:** shear reads `m=c1[0]`, `b=c2[0]` — only the
*first* tuple element, so c1/c2 aren't actually randomized despite the tuples/comment
(fixed value, or a TODO — your call to fix).

---

## Recommended port → MJX/Warp Go2 env

- **Apply as a foot external Cartesian force.** MJX analog of the Isaac foot wrench is
  `data.xfrc_applied[foot_body]` (world-frame 6-vec; set the linear part, torque 0),
  summed each substep, gated on `z_foot < mud_height`. (Alternative: convert to
  joint torque via `Jᵀ·F` per foot — same foot Jacobian the OSC controller already
  uses — if you'd rather keep everything in qfrc. xfrc is the more direct analog.)
- **Per-foot quantities you need each step:** world position (z → depth) and world
  velocity (z → suction sign; full xyz → shear). Both available in MJX from the foot
  bodies/sites + `cvel` (or finite-diff foot pos). The foot sites + FL,FR,RL,RR leg
  mapping are already worked out in the OSC code (see below).
- **Keep it vmap-safe / traceable:** the source uses masks (`in_mud`, `where`) not
  Python branches — mirror that (multiply by `in_mud`, `jp.where` for the sign /
  yield offset), no data-dependent control flow.
- **DR:** sample `mud_depth` per env (reset) and the coeffs per env/foot — the env
  already has per_step DR infrastructure; slot these in the same way.
- **Apply each PHYSICS substep** (the Isaac 0.005 s interval ≈ the sim step), summed
  with any controller forces — it's an environmental force field, independent of the
  controller (joint-PD or OSC).

## Relevant target-side files (jax_rl)
- Go2 Warp env + components: `jax_rl/envs/locomotion/go2_warp_*.py`,
  `go2_warp_components.py`. **Note:** the env config was recently revamped — defer to
  the current structure; this doc doesn't assume the old layout.
- Foot sites + leg→dof mapping + the foot Jacobian path: `jax_rl/envs/locomotion/go2_osc.py`
  (`compute_leg_impedance_torque`, foot site ids, FL,FR,RL,RR order).
- Held-out test (do NOT train against this — it's the eval): `projects/mud_eval/`
  (Newton MPM triple-mud) + its `HANDOFF.md`. Train on the analytic field, validate
  transfer there.

## Open design questions (for you)
1. **Force application**: foot `xfrc_applied` (recommended, direct analog) vs `Jᵀ·F`
   joint torque. Either is fine; xfrc keeps the mud force separate from control.
2. **Foot velocity source**: body `cvel` vs finite-diff of foot world pos (both the
   suction sign and the shear direction depend on it; pick the cleaner one in your env).
3. **mud_height representation**: flat per-env scalar (start here, matches the source)
   vs a heightfield / graded patch (to mirror the Newton test's thin→thick gradient).
4. **Coefficients**: reuse the Isaac ranges as starting points + DR, OR recalibrate —
   they were tuned for the Isaac robot/foot, so **sanity-check magnitudes against Go2
   foot size + body weight** before trusting them (forces of ~15 N·depth² + shear could
   be too soft/stiff for Go2 at the sampled depth).
5. **The c1/c2 randomization quirk** — fix it (randomize across the range) or keep fixed.
6. **Mud-state obs?** The Isaac env gives the policy NO explicit mud obs — it feels the
   mud purely through dynamics (`last_action` + proprioception). Decide whether to keep
   that (more general / harder) or expose mud depth/contact (easier / less transferable).
7. **Which controllers to train**: the eval result says variable-impedance benefits most
   from soft contact — worth training the OSC/var-impedance controllers on this field,
   not just joint-PD.

## One-line summary
Lift the two analytic foot-force terms (depth²-gated suction/resistance on world-Z +
Herschel–Bulkley shear opposing foot velocity) from `mud/.../forces_cfg.py` into a
foot-`xfrc` field in the MJX Go2 env, DR the depth + coeffs, train; the Newton MPM
harness stays as the held-out transfer test.
