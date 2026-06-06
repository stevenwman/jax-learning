# Robot swap — panda → UR5e + Robotiq 2F-85 (deferred design)

Status: **deferred**. Park here, revisit after NutThread + GearMesh envs land
on the current panda Factory port. We'll have more code surface and
opinions about what `compute_osc_torque` should look like by then.

Date: 2026-05-28
Related:
- `.superpowers/specs/2026-05-27-factory-mjx-warp-port.md` (current port)
- `.superpowers/plans/2026-05-27-factory-peg-insert-mvp.md` (PegInsert MVP)
- `.context/journals/2026-05-28-factory-phase2.md` (Phase 2 close)

## Goal

Replace the Franka Panda arm + (implicit, weld-based) gripper with the
**UR5e (6-DOF)** arm and the **Robotiq 2F-85** parallel-jaw gripper, both
sourced from `mujoco_menagerie`. Compose them via the **mjspec** Python
API (MuJoCo ≥3.x) so the scene assembly is programmatic rather than
string-spliced XML.

## Why eventually (not now)

Pro:
- Closer to deployable hardware. UR5e + Robotiq is a common lab setup;
  panda's gripper is decorative pads. Real grasp story (Phase 7+) is
  more credible on Robotiq.
- mjspec composition replaces our `re.sub`-based panda.xml splicer
  (`_build_scene_xml` in `factory_peg_insert.py`). Asset paths, namespaces,
  and joint prefixes auto-resolve. No more manual `<compiler>` strip-out.
- Visually a real gripper holding the peg is more legible in demo videos.

Con (= reasons we stay panda for now):
1. **Lose IsaacLab Factory parity.** Their entire Factory benchmark family
   is panda. Constants we ported (`DEFAULT_ARM_QPOS`, `NULLSPACE_ARM_QPOS`,
   `task_prop_gains`, `franka_fingerpad_length`, `held_asset_relative_pos`
   geometry) are panda-specific. No cross-validation against IsaacLab's
   numbers, no pretrained-weight reuse, no sanity-checking our SAC curves
   against theirs.
2. **6-DOF → no task-space nullspace.** UR5e has 6 joints. OSC's
   nullspace projector becomes `null_proj = I - J_bar·J = 0` for square
   J. Posture regularization disappears. Without it, the arm wanders into
   inverted-elbow / wrist-singular configs during exploration. Need a
   replacement (joint-limit barriers, joint-impedance soft prior, or
   accept the loss).
3. **OSC gain retune.** Our kp_task/kd_task were calibrated for panda's
   mass distribution + critical damping. UR5e is lighter (~30kg vs
   ~18kg panda payload), different link inertias, different reach.
   Expect a half-day of smoke iteration before tracking looks clean.
4. **Robotiq tendon coupling.** 2F-85 has a 5-bar tendon coupling
   between left/right fingers + spring compliance. We're still using
   `<equality><weld/>` for the held-peg attachment (real grasp deferred
   to Phase 7+), so the gripper actuator becomes purely decorative — but
   the weld anchor frame moves from `hand` body to whichever Robotiq
   link the mjspec attaches.
5. **Custom `fingertip_centered` site.** Panda's MJCF has this site at
   `(0, 0, 0.1034)` in hand frame. Robotiq has no such site by default.
   We'd add one between the finger pads in the attached spec.

Net cost estimate: **4-6 hrs** from green-tests-on-panda to green-tests-on-UR5e
+ render + journal entry. Plus subtle-bug risk from changing the
fundamental robot model.

## What changes (concrete, file-level)

### Asset sourcing

Vendor from `mujoco_menagerie/universal_robots_ur5e/` and
`mujoco_menagerie/robotiq_2f85/` into:
```
jax_rl/envs/manipulation/factory/assets/
├── ur5e/                          [new]
│   ├── ur5e.xml + assets/
│   └── LICENSE
└── robotiq_2f85/                   [new]
    ├── 2f85.xml + assets/
    └── LICENSE
```

Drop `franka_panda/` once all three Factory envs are off panda.

### Scene composition via mjspec

Replace `_build_scene_xml()`'s string-splicing in
`jax_rl/envs/manipulation/factory/factory_peg_insert.py`:

```python
import mujoco

def _build_mj_spec() -> mujoco.MjSpec:
    """Compose scene from ur5e + 2f85 + peg + hole_assembly via mjspec.attach."""
    arm = mujoco.MjSpec.from_file(UR5E_XML.as_posix())
    grip = mujoco.MjSpec.from_file(ROBOTIQ_XML.as_posix())
    scene = mujoco.MjSpec.from_file(BASE_SCENE_XML.as_posix())   # peg + hole + lights

    # Attach gripper to UR5e's tool0 (or 'wrist_3_link', whichever menagerie ships).
    tool_frame = arm.body("tool0")
    grip_attached = tool_frame.attach_body(
        grip.worldbody.first_body(),
        prefix="grip_", suffix=""
    )

    # Add fingertip site on the attached gripper between the pads.
    mid = grip_attached.body("base").add_site(
        name="fingertip_centered",
        pos=(0.0, 0.0, 0.18),   # ~tcp depth for 2F-85
        size=(0.005,), rgba=(1, 0, 0, 1),
    )

    # Attach arm to scene at world origin.
    scene.worldbody.attach_body(arm.worldbody.first_body(), prefix="", suffix="")
    return scene
```

Then `mj_model = scene.compile()`. mjspec handles meshdir + asset path
prefixing per sub-spec.

(Docs: https://mujoco.readthedocs.io/en/stable/python.html#mjspec —
verify exact API names; the above is the right shape but method names
shift across 3.x point releases.)

### Joint count + naming churn

- `arm_jnt_names`: `joint1..7` → `shoulder_pan_joint, shoulder_lift_joint,
  elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint` (or
  whatever menagerie ships, possibly with `prefix=` from `attach_body`).
- `arm_act_ids`: 7 → 6. Gripper actuator now has Robotiq's tendon
  schema (1 actuator with mimic), keep it untouched in motor mode.
- `DEFAULT_ARM_QPOS` (7,) → (6,). Compute by hand-tuning a config that
  places the gripper above the hole with TCP pointing down; then run
  the existing `_solve_init_arm_qpos` IK to fine-tune.
- `NULLSPACE_ARM_QPOS` — **delete**. No nullspace on 6-DOF arm.

### OSC adjustments (`controller/osc.py`)

`compute_osc_torque` currently:
```python
null_proj = jp.eye(7) - J_bar @ J_arm                  # (7, 7)
tau_null  = null_proj @ (M_arm @ (kp_null·(q_def - q) - kd_null·qdot))
return clip(tau_task + tau_null + feedforward, ...)
```

For 6-DOF: `null_proj ≈ 0`, so `tau_null ≈ 0`. We can:
- (a) Hard-code `tau_null = 0` for the 6-DOF path; rely on feedforward
  + joint-limit penalties to keep arm sane.
- (b) Replace nullspace posture with joint-impedance soft prior added
  to `tau_task` directly (won't be DC-orthogonal but cheap).
- (c) Add wrist-singularity-aware DLS to the Jacobian to dampen the
  inversion near degenerate configs (we already added a 1e-4 ridge to
  Λ; UR5e wrist singularities likely need larger).

Recommend (a) for v1 + (c) for stability. (b) only if exploration
collapses on joint limits.

### Reset IK

`_solve_init_arm_qpos` is robot-agnostic — only needs joint names,
seed, and a site to reach. Re-targets to UR5e tcp without code changes.
Just update `arm_jnt_names` arg and `seed_qpos`.

### Tests

- `test_factory_obs_schema.py` — obs dims unchanged, passes through.
- `test_factory_actuators.py` — change index from 7 to 6 throughout;
  re-check that "motor mode ctrl=0 sags under gravity" assertion still
  triggers (UR5e mass dist different — may need looser threshold).
- `test_factory_osc_jacobian.py` — load `ur5e.xml` directly, look up
  TCP site by name. Use the existing `_solve_init_arm_qpos` to put the
  arm in a non-singular pose before testing J.
- `test_factory_osc_pose_error.py` — robot-agnostic, no change.
- `test_factory_osc_torque.py` — change `arm_ids = jp.arange(6)`,
  update equilibrium qpos.
- `test_factory_osc_tracking.py` — same shape; differential expectations
  shift because UR5e workspace differs.

### Render

`scripts/factory/render_phase{1,2}.py` — body name set:
```python
panda_bodies = {"link0","link1",...,"hand","left_finger","right_finger"}
```
becomes:
```python
arm_bodies = {"base_link","shoulder_link","upper_arm_link","forearm_link",
              "wrist_1_link","wrist_2_link","wrist_3_link"}
grip_bodies = {"grip_base", "grip_left_pad", "grip_right_pad", ...}
```
(Exact names depend on mjspec prefix choice.) Camera `lookat` may need
re-tuning for UR5e workspace.

### Other Factory envs

NutThread + GearMesh will inherit this change. When they land they
should already expect UR5e geometry — i.e. defer their reset IK targets
+ keypoint frames until after the swap.

## Risks

- **R1 — mjspec API drift.** Method names changed across MuJoCo
  3.0/3.1/3.2 (`attach_body` vs `attach`, `worldbody.first_body()` vs
  `worldbody.body(0)`). Pin to a specific MuJoCo version, sanity-check
  against `mujoco.MjSpec` docs at the locked version.

- **R2 — Warp backend compatibility with mjspec output.** `mjx.put_model`
  is downstream of `MjSpec.compile()`. Should work; verify with a
  smoke test before refactoring everything.

- **R3 — Robotiq tendon coupling on Warp.** The 2F-85 uses
  `<equality><tendon/>` or similar mimic constraints. Confirm Warp's
  narrowphase + constraint solver handles it (panda fingertip used
  `<equality><joint/>` which is supported per Phase 0b spike).

- **R4 — Workspace reach.** UR5e max reach is ~850mm. Hole at world
  (0.6, 0, 0.05) is fine if UR5e base sits near origin. Confirm with a
  reachability plot before committing.

- **R5 — Loss of nullspace might collapse exploration.** No posture
  regularization means SAC could plant arm in a singular wrist
  configuration that the policy can't escape. Mitigation: joint-limit
  penalty in reward, or initial-pose noise that exercises a wider
  joint-config distribution.

## Rollout order (when we revisit)

1. Vendor menagerie assets (UR5e + Robotiq) + license stamping.
2. Smoke test: build a hello-world mjspec scene (UR5e alone, no
   gripper) and confirm `mjx.put_model` + `mjx.step` work on it.
3. Add Robotiq attach to step 2; confirm `mjx.step` still works.
4. Add peg + hole_assembly + weld to gripper TCP frame. Replay
   `diag_weld.py` — ortho_err should stay sub-mm with stiff weld
   solref (we already learned this lesson).
5. Re-target `_solve_init_arm_qpos` to put TCP 2cm above the bore.
6. Adapt `compute_osc_torque` to 6-DOF (drop nullspace, add wrist DLS
   ridge if needed).
7. Re-render Visual Gate 2. User confirms before continuing.
8. Re-tune OSC gains by checking `test_factory_osc_tracking.py` thresholds.
9. Update spec + plan + journal + this file (mark as done).

## Open questions

- Does menagerie's 2F-85 ship with a fingertip site we can reuse, or
  must we add one in mjspec?
- Should we keep the weld attachment for Phase 1-6 (parity with panda
  port) and switch to friction grasp at the same time as the robot
  swap (Phase 7+)? Probably yes — swap is enough churn on its own.
- Is there a UR5e variant in menagerie with built-in 2F-85, or do we
  need to compose ourselves? (If pre-composed, skip mjspec attach
  entirely.)
- Does training-time qvel/qpos statistics change enough that
  `obs_normalization=True` in our SAC preset needs re-warmup?

## Decision deferred until

- PegInsert SAC trained on panda hits ≥0.5 reward/step (Phase 3 done).
- NutThread + GearMesh envs land on panda (Phases 5-6).
- Then re-evaluate: cost of swap vs cost of building real-grasp
  dynamics on top of panda's nonexistent gripper.
