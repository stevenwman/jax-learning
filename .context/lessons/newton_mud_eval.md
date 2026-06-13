# Newton MPM mud eval — lessons

Testing trained jax_rl Go2 policies zero-shot in NVIDIA Newton's MPM two-way-coupled
mud (eval-only; Newton isn't JAX-traceable). Harness: `projects/mud_eval/`. Full
reference: `projects/mud_eval/HANDOFF.md`. Journal: `journals/2026-06-11-newton-mud-eval.md`.

## Collision backend

- **mujoco_warp 0.0.2 (GPU) silently drops robot↔PLANE contacts.** The robot fell
  clean through the flat ground off the mud — both go2.xml AND the original
  go2_description.urdf. It is NOT a model / collision-group / contype issue: the
  masks + graph-coloring permit the pair and the world body isn't excluded.
  **Diagnostic that pinned it:** run `mj_forward` on the solver's OWN cpu mujoco
  (`solver.mj_data`) with the feet on the plane → CPU mujoco generates the
  plane↔foot contacts (ncon>0). So the model is collidable; the GPU narrowphase is
  what drops planes. The vendored example never exposed this because the robot is
  always spawned over the mud (its contact is 100% via the MPM coupler, a separate
  path from mujoco collision).
- **Fix: `SolverMuJoCo(use_mujoco_cpu=True)`** — the CPU mujoco backend resolves
  plane contacts AND keeps the MPM mud coupling (forces apply through
  `xfrc_applied`). Requires graph capture OFF (the CPU path does GPU→CPU copies
  that can't run during a CUDA graph capture → err 906).
- **CPU stepping is NOT slower here** (~0.07 s/frame either way). The MPM solve
  dominates per-frame cost; a single-robot CPU `mj_step` is cheaper than the GPU
  path's per-substep launch/sync overhead. So "GPU rigid solver" buys nothing when
  one robot is coupled to an MPM field — don't assume CPU = slow.
- Lesson meta: when a contact silently doesn't happen, **instrument the narrowphase
  directly** (ncon on a known-penetrating pose) instead of reasoning about masks.
  A user's "I've seen it work" beat my inference that it was a hard limitation.

## Two-way coupling fidelity

- **Co-step the robot and the MPM at the controller rate.** The example decimates:
  robot substeps at 200 Hz but the mud force is HELD across them and the MPM
  integrates once per 20 ms frame (50 Hz). When the foot–medium interaction IS the
  thing you're measuring, that under-resolves it. Fold the MPM step into the robot
  substep loop (exchange forces every sim_dt); scale the coupling kernels by sim_dt
  not frame_dt.
- **Implicit MPM converges faster at finer dt** — stepping it 5× more often (250 Hz
  vs 50 Hz) cost far less than 5× (fewer solver iterations per smaller step). Don't
  pre-write-off finer coupling as too expensive without measuring.

## OSC / impedance control in Newton

- **Get J (foot Jacobian) + M (mass matrix) from the solver's OWN cpu mujoco**
  (`solver.mj_model/mj_data` via `mj_jacSite` + `mj_fullM`) — NOT a shadow model.
  The solver always builds a cpu mjModel at init (`spec.compile()`), so it's the
  same model the sim runs. mujoco_warp 0.0.2 exposes only crb/factor_m/solve_m — no
  `full_m`, no site Jacobian — so the GPU side is a dead end for OSC.
- **Inject torque via `control.joint_f`** (generalized force → `qfrc_applied`). It is
  `None` by default — must allocate `wp.zeros(model.joint_dof_count)` or
  apply_mjc_control skips it. Zero the joint PD (config pd_gains=0) so the OSC torque
  is the only actuation (matches training, where OSC replaces joint PD entirely).
- **Recompute the OSC torque every physics substep** (250 Hz), not once per 50 Hz
  policy step — the impedance loop diverges at 50 Hz for stiff gains. The policy
  emits foot deltas once/frame (held); the controller recomputes J/M/τ each substep.
- Site NAMES don't survive the Newton→mjModel conversion (nsite present but
  `mj_name2id("FL_foot")=-1`) → map foot sites by BODY (the calf bodies), not name.

## Evaluating a jax_rl checkpoint outside the training stack

- **Load the actor artifact directly**: `np.load(actor_params.npy)` + build
  `FastSAC` from `jax_rl.algos.fast_sac` (NOT `jax_rl.training`, whose `__init__`
  side-imports mujoco_playground and would bump mujoco off the pinned version).
  Deterministic eval = `select_action(..., deterministic=True)` = `tanh(mean)`.
- **obs_dim = 36 + action_dim** for these Go2 obs (gyro3+accel3+grav3+jpos12+jvel12+
  last_act[action_dim]+cmd3). The JOINT slices are always 12; `last_act` is the FULL
  action (16/24/… for variable-impedance ckpts). Don't slice joints by action_dim.
- **Newton add_mjcf loads the trained MJCF** (its rigid solver is mujoco_warp). go2.xml
  feet are geom class `"foot"` — add `collider_classes=("collision","foot")` or they're
  missed. The example's `joint_key.index(key)+6` home-posing overflows on go2.xml's
  0-dof `*_foot_joint` → set the home pose directly on `joint_q[7:19]`.

## The result it produced (go2-specific — see lessons/go2.md)

Zero-shot on graded mud, variable-impedance penetrates the thick mud ~45% deeper
than joint-PD and fixed-soft OSC (which tie). Stiffenable compliance helps; fixed
compliance doesn't — confirming the "use a task that demands stiffness modulation"
prediction. But all policies (flat-trained) still bog in thick mud → the next lever
is training on mud / mud-like DR, not the controller.

## Don't let the eval harness GUESS controller params — self-describe + parity-test (2026-06-13)

Porting the OSC/var-impedance/mass controller to the Newton eval (`mud_osc.py`, a
numpy port of the jax `compute_leg_impedance_torque`) bit THREE times — each a
"catapult"/garbage rollout that looked like a physics instability but was a
controller MISMATCH:
1. mass term `A·ẍ` silently dropped (Newton ran without it).
2. `use_op_space_inertia` (Λ vs bare) guessed wrong (`_use_lambda = not _mass`).
3. `var_a_max` / `var_xdd_ema` hardcoded ≠ the trained values.

Root cause was NOT bad math — the jax↔numpy ports were each correct. It was the
harness GUESSING params the ckpt `meta` didn't store. Fixes that actually hold:
- **Self-describe:** `get_control_metadata` emits `meta['control']['osc']` (the full
  controller config). The eval reads it instead of inferring from `action_dim` +
  env-vars. A ckpt now carries everything needed to reproduce its controller.
- **Parity test:** `tests/test_go2_osc.py::test_jax_numpy_osc_parity` feeds identical
  `(J,M,err,v,action)` to both implementations and asserts equal torque (Λ/bare/mass/
  decode) to fp tol — future drift fails CI without forcing a shared class.

**Rule:** when the same controller runs in two code paths (train env + eval harness),
(a) the ckpt must SELF-DESCRIBE its controller config — never infer it at eval; and
(b) a cross-implementation parity test pins the two equal. Inferring controller
identity from `action_dim` is a trap (48-d was ambiguous: bare-mass vs Λ-mass).
