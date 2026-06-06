# OSC / Impedance-Control Wiring Guide

Quick start for an agent who needs to wire operational-space control (OSC) into a new
manipulation env in this repo. Focused on the **controller layer**, not env reward / obs / RL
training details.

Source of truth: `jax_rl/envs/manipulation/factory/controller/{osc.py,action_chain.py}` (~250
lines total, hand-rolled, no external lib).

---

## What OSC is (in one paragraph)

Khatib's operational-space control maps a 6-DoF Cartesian pose error → arm joint torques
through the **operational-space mass matrix** Λ = (J·M⁻¹·J^T)⁻¹, then adds a **dynamically
consistent nullspace** posture term so the redundant arm doesn't drift to bad configurations
while the end-effector tracks. Math:

```
τ_task = J^T · Λ · (Kp·err − Kd·ẋ)
τ_null = (I − J̄·J) · M · (kp_null·(q_def − q) − kd_null·q̇)         where J̄ = M⁻¹·J^T·Λ
τ      = clip(τ_task + τ_null + feedforward, ±torque_limit)
```

For a 7-DoF arm: J is (6, 7), M is (7, 7), Λ is (6, 6). Pose error has 6 components: 3
linear (`tgt_pos − cur_pos`) and 3 angular (axis-angle from quat product `tgt_quat ⊗
conj(cur_quat)`).

---

## What we use (no library)

**Hand-implemented in JAX/MJX-Warp.** No `mujoco_mpc`, no `robosuite`, no `differentiable-robot-model`.

Reasons:
1. MJX-Warp is fp32, fully JIT-traceable — most off-the-shelf OSC code is fp64 NumPy and
   doesn't JIT cleanly on Warp's data layout.
2. We need a *batched* OSC running inside `mjx.step`'s `lax.fori_loop` for action-decimation —
   library wrappers fight that.
3. The whole controller is ~150 lines once you drop variable impedance, joint limits, etc.

The two ingredients you actually need from MJX-Warp:

- `mjx.jac(model, data, point, body_id)` → returns `(jacp, jacr)`, each `(nv, 3)`. **Watch the
  shape**: it's `(nv, 3)`, not `(3, nv)` — we transpose and stack.
- `mjx.full_m(model, data)` → dense (nv, nv) mass matrix. We slice to arm DoFs only.

Site orientation isn't in mjx's public API as a quat — we extract it from `data.site_xmat`
with a numerically stable rotation-matrix → quat conversion (4-case branch on largest
diagonal, see `_site_quat`). Don't write this from scratch; copy it.

---

## Pipeline (end-to-end)

```
            policy raw action (6,) ∈ [-1, 1]                  ← network output
                                  │
                                  ▼  apply_ema(raw, prev_ema, ema_factor)    ◄ smooths jitter
                                  │
                                  ▼  denormalize → (pos_delta, rot_delta)    ◄ scale to bounds
                                  │
                                  ▼  target_pos = clip_to_bounds(fingertip + pos_delta,
                                  │                              anchor=fixed_pos,
                                  │                              bounds=pos_action_bounds)
                                  │  target_quat = target_quat_prev ⊗ rotvec_to_quat(rot_delta)
                                  │
   ┌──────────────────────────────┘
   ▼
   compute_osc_torque(model, data,
                      target_pos, target_quat,
                      site_id=fingertip_site_id,
                      arm_dof_ids, arm_qpos_ids,
                      kp_task=(100,100,100, 30,30,30),     # 3 pos + 3 rot
                      kd_task=(20,20,20, 10.95,10.95,10.95),
                      q_default=NULLSPACE_ARM_QPOS,        # bias pose
                      kp_null=10.0, kd_null=6.32,
                      torque_limit=100.0,
                      feedforward=data.qfrc_bias[arm_dofs]) # gravity comp
   ▼
   ctrl = zeros(nu).at[arm_act_ids].set(tau_arm)
   ▼
   inner_step = mjx.step(model, data.replace(ctrl=ctrl))
   data = lax.fori_loop(0, decimation, inner_step, data)
```

The "action chain" steps (EMA, denorm, clip, rotvec→quat composition) live in
`controller/action_chain.py`. The OSC math lives in `controller/osc.py`. They're glued
together in the env's `step` (see `factory_peg_insert.py:step()` for the cleanest example).

---

## Wiring it for a new env: the checklist

1. **Build / load an MJCF** with an arm under a `<freejoint>`-less hierarchy. Pick a `<site>`
   on the end-effector — that's `fingertip_site_id`. The OSC drives this site to a target
   pose.

2. **Stash these indices on the env at construction time** (look them up once from `mj_model`,
   never search inside `step`):
   - `_fingertip_site_id` from `mj_model.site("fingertip").id`
   - `_arm_dofadr`, `_arm_qposadr`, `_arm_act_ids` — 7-vector of indices for the arm.
     `dofadr ≠ qposadr` when the arm sits behind a freejoint in the kinematic tree.
   - `_init_arm_qpos` — used both as nullspace bias `q_default` and as initial arm pose.

3. **Pick gains.** Start with these, tune from there:
   - `kp_task = (100, 100, 100, 30, 30, 30)` — translation stiff, rotation softer.
   - `kd_task ≈ 2·sqrt(kp_task)` for critical damping (we run slightly under-damped on rot).
   - `kp_null = 10`, `kd_null = 2·sqrt(kp_null) ≈ 6.32`. **Don't skip the nullspace term** —
     without it the 7-DoF Panda happily wanders into wrist-twisted configurations whenever
     the task is locally underdetermined.
   - `torque_limit = 100` Nm — Panda spec limits are 87/87/87/87/12/12/12, so 100 is
     conservative-on-the-loose-end. Tighten if you trust your motor model.

4. **Initial pose via IK.** Don't spawn the arm at random — call `mujoco.mj_inverse` /
   numerical IK once at construction to put the fingertip at a known offset above the task
   asset. We use `_resolve_ik_pose()` for this. Otherwise OSC kicks in from a bad pose and
   the first 50 steps look like a pile of warnings.

5. **Action smoothing.** `ema_factor=0.2` is what we ship. Lower = smoother but laggier.
   **Critical:** reset EMA on episode boundary (`reset_on_done`) — Brax-style auto-reset
   doesn't clear `state.info` for you and stale EMA tanks the first ~10 steps of every
   episode otherwise.

6. **Held-object weld.** If the task involves a grasped object, use a `<weld>` equality
   between the held body and `hand`:
   ```xml
   <equality>
     <weld name="grasp" body1="held_obj" body2="hand"
           relpose="x y z qw qx qy qz" active="true"
           solref="0.001 1" solimp="0.999 0.9999 0.001"/>
   </equality>
   ```
   `relpose` is the held-obj pose in hand frame. Tight `solref`/`solimp` makes the weld
   behave like a rigid grasp. Confirmed during this session: OSC + tight weld easily resists
   100N external force on the held object.

7. **Action clip anchor.** Set `clip_anchor` to the **task target**, not to the current
   fingertip pose. We use `info["fixed_pos"]` = the target seated centroid. This bounds the
   policy's commanded position to a box around the target, which prevents wandering during
   exploration but does *not* prevent reaching the target. For Factory the z-bound is 10cm —
   plenty wide to reach the floor.

---

## Gotchas (the ones that bit us)

1. **`mjx.jac` shape is `(nv, 3)`, not `(3, nv)`.** Off-by-transpose silently produces a
   plausible-looking but wrong wrench. The unit test
   `test_factory_osc_jacobian.py:assert J @ qdot == measured_fingertip_velocity` catches
   this — write it first.

2. **Λ inversion needs damping.** `Lambda = inv(J·M⁻¹·J^T + 1e-4·I)`. Without the ridge,
   cuSolver crashes on near-singular configs (wrist alignments reachable via the IK reset).
   1e-4 is invisible in normal operation, saves you in pathological ones.

3. **fp32 is fine in practice but not in theory.** MJX-Warp is fp32. Λ inversion can ill-
   condition. We've never had to enable `jax_enable_x64` on Factory tasks, but the escape
   hatch is documented in `.superpowers/specs/2026-05-27-factory-mjx-warp-port.md` if you
   see NaN spikes after a reset.

4. **`data.qfrc_bias` ≠ pure gravity.** It also includes Coriolis. As a feedforward this is
   what you want — adds whatever's needed to hold the current configuration.

5. **Don't construct a full-`nu` torque vector inside OSC.** Return `tau_arm` shape `(7,)`
   and let the caller place it into `ctrl[arm_act_ids]`. Lets the same OSC code work
   whether the scene has a free-floating object (extra qvel slots) or not.

6. **OSC is rigid relative to external force.** Confirmed in this session via xfrc_applied
   probes: 100N downward on the held object yields ~3mm transient deflection that snaps
   back when the force is removed. Only 500N sustained for 10s overcomes it. **If the policy
   isn't reaching the floor, it's because it's not commanding a deep enough target — not
   because OSC is too soft.** Don't bump gains looking for a phantom compliance issue.

---

## Reference pointers

| | what | where |
|---|---|---|
| **The math** | Khatib 1987 "A unified approach for motion and force control of robot manipulators" | classical paper, easy to find |
| **Implementation reference** | NVIDIA Factory paper appendix has the exact equations we copied | https://research.nvidia.com/labs/srl/factory/ |
| **Robosuite OSC** | Cleanest open-source reference if you want to cross-check | `robosuite/controllers/osc.py` (PyTorch, fp64) |
| **mjx_env API for Jac/M** | `mjx.jac`, `mjx.full_m` | `mujoco/mjx/_src/support.py` in the mujoco source |
| **Quat conventions** | MuJoCo uses (w, x, y, z). Robosuite uses (x, y, z, w). Don't mix. | mujoco.org docs §Quaternions |
| **Our port spec** | The Factory port plan w/ rationale + tests | `.superpowers/specs/2026-05-27-factory-mjx-warp-port.md` |
| **Lessons** | OSC fp32 quirks, weld grasp tradeoff observations from Factory port | `.context/lessons/manipulation.md` |

---

## What to read first (in this repo)

1. `jax_rl/envs/manipulation/factory/controller/osc.py` — the whole controller, end-to-end.
   ~190 lines including a careful site_xmat → quat conversion you should not rewrite.
2. `jax_rl/envs/manipulation/factory/controller/action_chain.py` — the EMA + clip + rotvec
   composition glue between policy output and OSC. ~65 lines.
3. `jax_rl/envs/manipulation/factory/factory_peg_insert.py` lines around `step()` — the
   minimum end-to-end example showing how OSC is invoked inside an env step's
   `lax.fori_loop` decimation. (Factory GearMesh uses the same pipeline but the env is
   noisier; PegInsert is the cleaner reading path.)
4. `tests/test_factory_osc_*.py` — three tests that pin down Jacobian correctness, pose-
   error sign, and tracking RMS. Re-run after any controller edit.

---

## Tuning quickies (from this session's Factory PegInsert / GearMesh work)

- `kp_task` for the *position* axes (100) is much higher than for *rotation* (30) because
  the gear's tilt error in axis-angle has very different magnitude than its translation
  error in meters — same Kp value would give wildly different effective stiffness.
- `kd_task = (20, 20, 20, 10.95, 10.95, 10.95)` is critically damped for the loaded gear
  inertia (gear ≈ 18g + rotational inertia). Lighter object → drop both proportionally.
- The `pos_action_bounds = (0.02, 0.02, 0.10)` clip box: 2cm xy, 10cm z. Don't tighten z
  bound past the actual reach distance from the policy's typical fingertip pose to the
  task target — you'll silently cap the policy without any error.
- For a new env, sanity-check by running OSC with the policy disabled (`hold_ctrl` path in
  Factory) and `target_pos` swept along a scripted trajectory. RMS error should be <1cm
  over a 30-second sweep. If it isn't, fix that before touching reward.

---

## What this guide explicitly does NOT cover

- How to design the action space / pose-delta scaling for a particular task.
- Reward shaping, observation design, training loop wiring.
- Variable impedance control / admittance control — flagged as future research in the
  Factory port spec; not implemented.
- Whole-body control or quadruped contact-aware controllers — different problem.
- Force-mode control (commanded torque from policy instead of pose-delta). Easy to add but
  requires retraining and a different exploration story.
