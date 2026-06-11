# 2026-06-11 — Newton triple-mud eval: testing the trained Go2 policies on soft terrain

Branch `go2-osc-impedance` (worktree). Continues the OSC/variable-impedance line
(`2026-06-08`, `2026-06-09`). Goal: take the 2026-06-09 physical-motor retrains
(joint-PD / fixed-OSC / variable-impedance) and **eval them zero-shot in a real
soft-terrain sim** — NVIDIA Newton's MPM two-way-coupled triple-mud — to answer
"does compliance help on mud, and does *stiffenable* compliance help more?"

Full technical reference: **`projects/mud_eval/HANDOFF.md`**. This journal is the
narrative + the result.

## The harness (projects/mud_eval/, self-contained)
Eval-only (Newton isn't JAX-traceable, so we hand-write the step loop). We DON'T
rebuild the policy — `MudJaxPolicy` loads the real jax_rl FastSAC actor from the
checkpoint (`np.load(actor_params.npy)` + `FastSAC(...).select_action`) and a
Newton-state→48d obs adapter, then drops in as the vendored `mpm_go2_multi`
example's `Go2Policy`. Vendored Newton 0.1.3 + a dedicated venv.

Hard-won runtime combo: **warp-lang 1.12.0 / mujoco 3.7.0 / mujoco-warp 0.0.2**
(Newton 0.1.3 needs the pre-per-world-batching mujoco-warp) / torch cu128 / jax 0.9
(actor runs on CPU — fine, no GPU contention with the MPM). jax imported via
`jax_rl.algos.fast_sac` directly to dodge `jax_rl.training`'s mujoco_playground
side-import (would bump mujoco off 3.7.0).

## Four engineering hurdles (each a lesson — see lessons/newton_mud_eval.md)
1. **Model migration** — the example ships `go2_description.urdf`, not the
   `go2.xml` the policy trained on. Newton's `add_mjcf` loads our exact MJCF
   (its rigid solver IS mujoco_warp); a monkeypatch seam dispatches `.xml`→add_mjcf
   + sets the home pose (the example's `joint_key.index+6` posing overflows on
   go2.xml's 0-dof `*_foot_joint`). Feet are class `"foot"` — missed by the
   default collider_classes. Contact geom turned out ~identical to the URDF
   (feet = sphere r=0.022); migration's real value is matched inertials + native OSC.
2. **Co-stepping** — the example decimates the coupling: robot substeps at 200 Hz
   but the mud force is HELD across them and the MPM integrates ONCE per 20 ms
   frame (50 Hz). Since foot-mud interaction is the physics we're measuring, that
   under-resolves it. Folded the MPM step INTO the robot substep loop → robot+mud
   co-step at sim_dt (250 Hz, sim_substeps 5), forces exchanged every substep.
   Cost was *not* the feared 5× (the implicit MPM converges faster at finer dt).
3. **Walkable ground (the big one)** — the robot fell clean through the flat
   ground off the mud (BOTH go2.xml AND the original URDF). Diagnosed: NOT a model
   / collision-group issue — CPU mujoco (`mj_forward` on solver.mj_data) generates
   plane↔foot contacts fine (the masks + colors permit it; world body 0 isn't
   excluded). The **GPU mujoco_warp 0.0.2 collision path silently drops
   robot↔PLANE contacts**. The example never hit this because the robot is always
   spawned over the mud (contact is 100% via the MPM coupler). Fix:
   `SolverMuJoCo(use_mujoco_cpu=True)` — the CPU backend resolves planes, keeps the
   mud coupling (MPM forces apply through `xfrc_applied`), and is **not slower**
   (~0.07 s/frame either way; the MPM solve dominates, one-robot CPU mj_step is
   cheap). Took a user's insistence ("ive seen it work with urdf+plane, check
   contact group") to stop concluding "limitation" and actually instrument it.
4. **OSC in Newton** — the OSC controllers emit foot-position deltas, not joint
   targets, and need J (foot Jacobian) + M (mass matrix). Got both from the
   solver's OWN cpu mujoco (`solver.mj_model/mj_data` via `mj_jacSite` + `mj_fullM`)
   — no shadow model. Per substep: sync state → osc_torque (Khatib Λ, ported from
   `go2_osc.compute_leg_impedance_torque`) → `control.joint_f` (allocated; CPU path
   applies it via `qfrc_applied`), PD zeroed. M3 variable impedance decodes the
   per-foot/per-axis stiffness tail (`impedance_gains_np`, base [3000,3000,4000]).

## THE RESULT — compliant-vs-stiff on graded mud
Thick-first traversal (spawn flat ground y=-1, face +Y, walk into the densest mud),
280 frames. Penetration depth (y into thick mud) + posture (torso z):

| policy | depth | z | |
|---|---|---|---|
| joint-PD | 0.226 | 0.23 | bogs at the thick-mud edge |
| OSC soft (fixed kp 1500) | 0.222 | 0.27 | same depth, springier |
| **var-impedance per-foot** | **0.322** | 0.29 | **~45% deeper, still advancing** |
| **var-impedance per-axis** | **0.341** | 0.28 | deepest |

**Variable impedance wins; fixed compliance does NOT.** Soft-OSC ties joint-PD —
both bog at the edge. Only the *stiffenable* policies (commanding s up to 2× →
kp up to [6000,6000,8000]) push ~0.1 m further into the thick mud, holding the
highest posture. This confirms the 2026-06-09 prediction (lessons/go2.md: "to make
stiffness modulation matter, use a task that demands it: soft/variable contact").

## The standing challenge (where we paused)
Even the var-impedance winner **gets stuck in the thick mud** — it penetrates
further but still bogs and grinds to ~a stop. These are **flat-trained** policies
(physical-motor, flat ground) tested zero-shot on mud; none was trained to expect
soft, sinking terrain. The eval harness now exists and discriminates the
controllers cleanly — the next lever is on the TRAINING side: train on mud (or
mud-like DR: randomized ground compliance / sinking / drag), so the policy learns
to commit stiffness + foot placement for soft terrain rather than transferring a
firm-ground gait. The harness is the measuring stick for that.

## Artifacts
- Code + full reference: `projects/mud_eval/` (HANDOFF.md, mud_{jax_policy,model,
  costep,cpu,osc}.py, record_traverse.py, the gate_*/inspect_* diagnostics).
- Videos: `projects/mud_eval/recordings/traverse{,_osc,_var,_varaxis}_thick.mp4`
  + thin-first ramps + falloff.mp4 (the GPU-collision fall-through proof).
- Commits: walkable-ground fix `0aa5c71`, M2 OSC `38a4e1f`, M3 var `39b05fd`.
