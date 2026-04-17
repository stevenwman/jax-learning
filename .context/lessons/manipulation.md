# Manipulation Lessons

Lessons from building the planar pushing benchmark (`jax_rl/envs/manipulation/push_env.py`).

---

## Cylinder-Box Collisions Require MuJoCo Warp (2026-04-17)

**What happened:** Built a shape-agnostic push env with cylinder pusher + box block geoms. `mjx.put_model(m, impl="jax")` raised:

```
NotImplementedError: (mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_BOX) collisions not implemented.
```

**Root cause:** MJX JAX backend never implemented cylinder-vs-box narrowphase. Only sphere-sphere, sphere-box, sphere-capsule, box-box, and a few others.

**Fix:** Use `impl="warp"`. Warp's CCD (`naccdmax`-sized buffer) handles all geom pairs via GJK+EPA.

**Lesson:** Any env with cylinder pusher or tool interacting with non-sphere objects is Warp-only. This matches [lessons/mjx.md](mjx.md) §"MJX can't load all MJCFs". Don't plan for JAX-backend fallback — cylinder-box comes up naturally in manipulation.

**Workaround if Warp unavailable:** replace cylinder with sphere (radius ≈ height/2). Sphere-box is supported in JAX backend. Loses the flat-top contact profile but OK for most RL.

---

## Contact Penetration: Tighten solref When Soft Defaults Don't Fit (2026-04-17)

**What happened:** Early pusher-block contacts had visible penetration in rendered video — pusher disc overlapped block geoms by several mm for 3–5 frames before separating.

**Root cause:** MuJoCo default contact is `solref="0.02 1"` — 20ms time constant, critically damped. At `sim_dt=0.002` that means ~10 sim steps to resolve a penetration. Between resolutions, visible interpenetration.

**Fix:** Stiffen both geoms:

```xml
<geom solref="0.004 1" solimp="0.98 0.995 0.0005 0.5 2"/>
```

- `solref="0.004 1"` → 4ms time constant ≈ 2 sim steps. Visually clean.
- `solimp="0.98 0.995 ..."` → nearly rigid (default max is 0.95; bump to 0.995).
- Bump `<option iterations="50" ls_iterations="10"/>` to give Newton solver more iterations at the stiffer problem.

**Tradeoff:** stiffer contacts are slightly more expensive per step. Measurable but small (few percent sps hit). Worth it for manipulation where contact is the whole point.

**Lesson:** Default MuJoCo contact is tuned for locomotion (feet on floor, soft enough to avoid hard chatter). Manipulation wants hard contact with fast separation. Always check the contact quality in rendered video before assuming physics is correct.

---

## Shape-Agnostic Obs for Cross-Shape Generalization Benchmarks (2026-04-17)

**Goal:** train policy on push-T, test on push-circle/L/plus. Evaluates whether RL learns "pushing physics" or memorizes T-specific contact.

**Wrong approach (shape-specific obs):** keypoints at block vertices. T has 4 corners, plus has 8, star has 10. Padding wastes capacity and leaks shape identity.

**Right approach (shape-agnostic obs):**

```
[pusher_xy (2), block_xy (2), sin(yaw), cos(yaw) (2),
 target_xy (2), sin(goal_yaw), cos(goal_yaw) (2),
 pusher_vel (2), block_vel (2), last_action (2)] = 16d
```

Same obs space for every shape. Policy only sees block center+angle, not geometry. Contact dynamics must be inferred from how the block *responds* to pushes. Failure mode:

- Pusher contacts block tangent (different per shape). Policy can't see contact point directly.
- Must infer "this block rotated faster than expected → I'm probably hitting an edge, pull back."

This is a harder but more honest generalization test.

**Include `last_action` in obs.** Position-PD actuation has ~35ms time constant vs 20ms control step, so pusher lags the commanded ctrl. Without `last_action`, policy can't distinguish "my last command is still converging" from "I stopped commanding." Adding it helps all action modes but especially position-PD and teleport.

---

## Three Action Modes for RL vs Demonstration-Literature Parity (2026-04-17)

**Literature split for pusher control:**

| Mode | Used by | Mechanism |
|---|---|---|
| Absolute PD | Our default, locomotion community | Policy outputs XY target, PD chases |
| Absolute teleport | gym-pusht / Diffusion Policy | Policy outputs XY target, pusher moves directly toward it at fixed max velocity |
| Delta (velocity) | robomimic, IsaacGym manipulation | Policy outputs ΔXY per step, integrate |

All three are "XY position-like" but feel very different to a policy:

- **Position-PD** is jittery at sharp kp/kd (overshoot on waypoint changes)
- **Velocity/delta** is inherently smooth (policy integrates)
- **Teleport** is smooth via speed-cap; matches the domain where Diffusion Policy baselines live

**Design:** expose all three via `config.action_mode ∈ {"position", "velocity", "teleport"}`. Use same obs space across modes — differences emerge purely in how `ctrl` is computed from `action`. This lets:

- RL policies train on whichever mode fits their action-smoothness assumptions
- Imitation-learning baselines get gym-pusht-equivalent mode for fair comparison
- Ablations isolate "is the dynamics mismatch or the action parameterization responsible for X?"

**Implementation trap:** in teleport mode we don't kinematically set pusher position — that would break contact forces (block wouldn't "feel" the pusher). Instead we step-rate-limit the `ctrl` target before sending to the actuator. Policy sees real pusher XY (lagged), not the commanded target. For strict gym-pusht parity (ctrl == pos each step), bump pusher `kp` very high (≥500) so PD settles within one control step.

---

## Data-Driven Shape Registry Keeps Adding Shapes Cheap (2026-04-17)

Shapes live in a dict of geom specs:

```python
SHAPES = {
    "T":      [{"name": "stem", "type": "box", "size": "0.025 0.075 0.015", "pos": "0 -0.025 0"},
               {"name": "top",  "type": "box", "size": "0.075 0.025 0.015", "pos": "0 0.075 0"}],
    "L":      [...],
    "circle": [{"name": "disc", "type": "cylinder", "size": "0.06 0.015", "pos": "0 0 0"}],
    "plus":   [...],
}
```

XML is built programmatically by `build_xml(shape)` — inserts the geoms into a template that provides pusher, walls, lights, target ghost. Adding a new shape = one entry in `SHAPES`. No XML file proliferation, no copy-paste drift.

**Mirror the same geom spec to the target ghost** (transparent, `contype=0 conaffinity=0`). Visual target match without collision interference.

**Lesson:** for tasks that vary by one axis (here: block shape), drive the variation from Python data, not per-variant XML files. The template stays canonical, variation lives in one place, and shape-name is a first-class env parameter.
