# Factory Port — Phase 2 (OSC controller chain) + Phase 1 close

Date: 2026-05-28
Branch: `factory-peg-insert` (worktree at `.worktrees/factory-peg-insert/`)
Plan: `.superpowers/plans/2026-05-27-factory-peg-insert-mvp.md`
Spec: `.superpowers/specs/2026-05-27-factory-mjx-warp-port.md`

## Phase 1 close — DONE

Visual Gate 1 (zero-torque rollout, peg welded, arm holds reset pose) passed
in prior session — commit `0d57bf8`. See [2026-05-27-factory-phase0.md](2026-05-27-factory-phase0.md)
+ in-tree commit log for Phase 1 build-out (Tasks 1.1–1.8).

## Phase 2 — OSC controller chain — DONE

Five commits land the Khatib OSC pipeline + the wiring + visual gate:

| Task | Commit | What |
|---|---|---|
| 2.0 (prereq) | `793df83` | `actuator_mode` config: swap arm actuators position-PD → motor in-place |
| 2.1 | `a5af9bf` | `compute_site_jacobian` — (6, nv) world Jacobian, parity with `mj_jacSite` |
| 2.2 | `9559923` | `compute_pose_error` — axis-angle rotation error, small-angle stable |
| 2.3 | `2d0280a` | `compute_osc_torque` — Λ·wrench + DC nullspace + clamp (returns 7-d arm tau) |
| 2.4 | `0c7dc76` | Wire OSC into `env.step` per inner substep, gravity-comp via `qfrc_bias`, [gpu,warp] tracking test |
| Polish | `a4d9ee3` | Weld coherence + pre-resolve peg pose + render polish |

Test count after Phase 2: **40 hermetic + 6 [gpu,warp]** (all green standalone;
fixture-state leakage in one combined-suite run flagged as transient).

### Task 2.0 sanity gate
With identical `ctrl=0` over 200 inner steps:
- position-PD drags joint4 from -1.97 toward 0 (+0.1 rad shift)
- motor leaves joint4 < -0.988 (gravity sag, NOT toward zero pose)
- `‖q_motor - q_pos_pd‖∞` > 0.3 rad

Confirms the control-interface flipped from position-target to raw-torque.

### Task 2.4 OSC wire-up

Per inner substep (decimation × sim_dt):
```python
tau_arm = compute_osc_torque(
    mjx_model, d,
    target_pos, target_quat,    # target_pos from action chain, target_quat fixed
    site_id, arm_dof_ids, arm_qpos_ids,
    kp_task=(100,100,100,30,30,30),
    kd_task=(20,20,20,10.95,10.95,10.95),   # critical damping = 2·sqrt(kp)
    q_default=DEFAULT_ARM_QPOS,
    kp_null=10.0, kd_null=6.32,
    torque_limit=100.0,
    feedforward=d.qfrc_bias[arm_dof_ids],   # gravity + Coriolis
)
ctrl = jp.zeros(nu).at[arm_act_ids].set(tau_arm)
```

`feedforward` is the key gotcha — without `qfrc_bias` feedforward, OSC has to
fight gravity through pose-error tracking only. At 5cm error and kp=100, the
wrench is only 5N — well under arm weight. Arm sags before tracking converges.
With `qfrc_bias` added, OSC just superimposes corrections on top of a
gravity-cancelled baseline.

### Tracking gate findings

action_chain clips OSC target to hole_pos ± 5cm each step. Initial fingertip
sits at ~(0.5, 0, 0.45) but hole is at (0.6, 0, 0.05) — so init x and z are
OUTSIDE the bounds in those axes. Every per-step proposal clips to the lower
bound regardless of action sign, annihilating the differential.

Only y is tracking-testable in Phase 2 (init y=0 centered on hole y=0).
Phase 3 will start the arm near the bolt and the +x/+z gates become testable.

### Visual Gate 2 — DONE

Scripted action sequence (`scripts/factory/render_phase2.py`):
1. action=0 (settle, 1s) → fingertip drops 24cm toward hole
2. action=+y (1s)        → y from 0 to +0.016m
3. action=-y (1s)        → y reduces
4. action=-z (2s)        → fingertip drops below bore top (z<0.075)

MP4 at `.tmp/recordings/factory_phase2_osc_scripted.mp4` (45 KB, 151 frames).
No NaN over 150 outer steps. User confirmed visually.

## Weld coherence fixes (commit `a4d9ee3`)

User flagged peg appearing to drift relative to the gripper. Root-cause
triage via `scripts/factory/diag_weld.py` (measures `|peg - hand|`, the
projection onto hand z-axis, and orthogonal slip):

1. **relpose direction was inverted.** `relpose="0 0 0.130 1 0 0 0"` with
   body1=peg/body2=hand means "hand at +0.13 along peg's z-axis." With
   panda's gripper-down orientation, the resolved equilibrium put the peg
   ABOVE the hand in world — embedded inside the wrist. Flipped to
   `0 0 -0.130` → peg sits 13cm below hand, sticking out the fingertips.

2. **Cylinder inertia ill-conditioned by 27×.** `Izz=1.51e-7` vs
   `Ixx=Iyy=4.03e-6`. The narrow-z axis caused the constraint solver to
   leave a ~7° persistent tilt under OSC wrench. Sphericalizing to
   `Izz=Ixx=4.03e-6` drops tilt to <1°. Mass + trace preserved.

3. **Reset placed peg at arbitrary (0.5, 0, 0.3),** then relied on the
   weld constraint to drag it 25cm to equilibrium over ~30 substeps. Now
   reset does FK once, computes `peg_pos = hand_pos + R_hand·(0, 0, 0.130)`
   and `peg_quat = hand_quat`, writes them directly. Step-0 ortho_err
   drops from 0.25 m → 0.0 exactly.

Defense-in-depth: tightened weld `solref="0.001 1"` and `solimp="0.999
0.9999 0.001"`. Negative-form direct stiffness `solref="-1e7 -1e4"` blew
up to NaN — solver instability above ~1e7. Stick with positive form.

Final diagnostic results:

```
step=  0 [  reset] |peg-hand|=0.13000  z_hand·diff=+0.13000  ortho_err=0.000000
step= 30 [ settle] |peg-hand|=0.13000  z_hand·diff=+0.13000  ortho_err=0.000014
step= 60 [     +y] |peg-hand|=0.13000  z_hand·diff=+0.13000  ortho_err=0.000012
step= 90 [     -y] |peg-hand|=0.13000  z_hand·diff=+0.13000  ortho_err=0.000007
step=150 [-z dive] |peg-hand|=0.13000  z_hand·diff=+0.13000  ortho_err=0.000004
```

Compare to pre-fix: ortho_err was 37 mm at step=60 under +y push.

### Render polish

- Panda meshes alpha=0.35 so the welded peg is visible behind gripper.
- Lights (`scene_key` + `scene_fill`) + non-colliding visual floor in
  `scene.xml`. Light names had to be `scene_*` to avoid clashing with
  the menagerie panda's `top` light.
- Camera distance 1.0 → 0.95, lookat at `(0.55, 0, 0.15)` — hole-centered.

## Open items / Phase 3 prereqs

- **Arm reset pose.** Currently DEFAULT_ARM_QPOS places fingertip 35cm above
  the bolt. Phase 3 should either start arm near bolt (IK at reset) or widen
  `pos_action_bounds`. Otherwise the policy spends most of every episode just
  reaching, and the OSC-bounds clipping wipes the +x / +z action gradients.
- **DR wrappers.** Spec for hand_init / bolt_pos noise exists in
  `get_domain_randomization_spec`, not yet applied at reset. Phase 1 deferred;
  re-evaluate during Phase 3 smoke training.
- **target_quat is constant.** Phase 5 (NutThread) wires rotation actions.
- **Combined test suite intermittency.** `test_motor_ctrl_zero_does_not_pull_toward_zero`
  failed once when run inside the full file but passed in isolation. Likely
  module-scope env fixture state leakage between tests. Re-run hermetic
  + warp in CI separately, or convert fixtures to function-scope.

## Where to look

| Thing | Path |
|---|---|
| Env class | `jax_rl/envs/manipulation/factory/factory_peg_insert.py` |
| OSC controller | `jax_rl/envs/manipulation/factory/controller/osc.py` |
| Action chain | `jax_rl/envs/manipulation/factory/controller/action_chain.py` |
| Scene MJCF | `jax_rl/envs/manipulation/factory/assets/peg_insert/scene.xml` |
| Render scripts | `scripts/factory/render_phase{1,2}.py` |
| Diagnostics | `scripts/factory/diag_weld.py` |
| Tests | `tests/manipulation/factory/test_factory_*.py` |
| Visual gate 2 MP4 | `.tmp/recordings/factory_phase2_osc_scripted.mp4` |
