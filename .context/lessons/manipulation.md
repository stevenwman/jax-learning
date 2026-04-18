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

---

## Slide-Joint Body Pos Is an Offset, Not a Starting Position (2026-04-17)

**What happened:** Wrote `qpos = [pusher_x, pusher_y, block_x, block_y, block_yaw]` in reset(), expecting pusher/block to appear at those world coordinates. On render, pusher was visibly *outside* the wall on the left — despite qpos values being inside the valid range.

**Root cause:** Pusher body had `<body pos="-0.15 0 0.015">` in the MJCF. Slide joints are **additive to body pose**: world_x = body_pos_x + qpos[0]. So qpos=-0.19 produced world_x = -0.15 + -0.19 = -0.34, past the wall at ±0.3. Same for block (body_pos 0.05). Visible confusion: policy obs said pusher at -0.19, physics had it at -0.34.

**Fix:** Set body pos to origin (`<body pos="0 0 0.015">`) so qpos directly = world XY. No offset bookkeeping.

**Lesson:** When using slide joints to express "XY position" of a body, always anchor the body at origin. Any nonzero `<body pos>` becomes a hidden offset baked into every qpos read/write. Check with `d.xpos[body_id]` vs `d.qpos[joint_qposadr]` during a forward pass — they should match for a pure slide body. Silent visual-only bug otherwise.

---

## Zero-Action Attractor in Velocity/Teleport Modes (2026-04-18)

**What happened:** On push-T with shaped reward, position-PD mode converged to eval +143 (47% success). Velocity-delta and teleport modes plateaued at -85 to -90 despite same obs, same reward, same training budget. Shaping tweaks (r_block_vel 2→10, added r_pusher_vel_toward_block) marginally helped vel/tele but never closed the gap.

**Diagnosis (with per-component reward + velocity logging):** pusher_vel_mag 0.80 m/s in pos mode vs 0.06 m/s in vel/tele. Even with 4× cap bump (vel 0.01→0.04 m/step, tele 0.25→1.5 m/s), pusher used only 7-13% of the cap. Policies learned small actions and stayed there.

**Root cause:** in vel/teleport modes, `action=0 → ctrl=current → pusher holds still`. Self-consistent attractor. Policy has no local gradient to explore "what if I moved aggressively" — any aggressive action reverts to hover next step because the new ctrl again equals current pusher position. In pos-PD mode, `action=0 → ctrl=[0,0] ≠ current` so PD always yanks pusher toward origin at kp×err force. Accidentally forces motion even at zero action. Exploration-friendly by construction.

**Implications:**
- Position-PD is the surprisingly-good default for pushing tasks trained from scratch with RL.
- Velocity/teleport match imitation-literature conventions (gym-pusht, DP) but those use expert demos as exploration crutch — not RL-from-scratch.
- Don't assume equivalent action spaces have equivalent trainability. The attractor structure matters.

**Fix options (none fully closed gap in our tests):**
1. Bypass PD entirely for vel mode: directly set `qvel[pusher]=action*scale`. Removes attractor but breaks contact force coupling.
2. Large action bonus term `+α||action||` to counter the hover equilibrium. Fragile.
3. Accept: pos-PD for RL, reserve vel/tele for BC/imitation comparisons.

**Lesson:** before blaming HPs, check if your action parameterization has a fixed-point at `action=0`. If yes, and the fixed-point is inside the workspace, expect cautious policies. Add diagnostic `pusher_vel_mag / block_vel_mag / pusher_to_block` to metrics when A/B-ing action modes — the numbers tell you whether policy is action-limited or structural.

---

## Reward Shaping Strength Is a Dial, Not a Monotonic Knob (2026-04-18)

**What happened:** Bumping `r_block_vel` scale from 2.0 → 10.0 tanked pos mode from eval +143 → -1.7. Bigger shaping reward made training worse.

**Mechanism:** at 10.0, the reward became: _achievable only if you shove the block around_. Angle-matching (`r_angle`) stopped mattering in relative terms — drowned by block_vel gradients. Policy learned to hustle the block back-and-forth for r_block_vel rather than patiently rotating to the target angle. Returns got mostly-positive-but-not-solving: task-reward noise but benchmark-low success rate (47% → 3.5%).

**Reverting `r_block_vel` to 2.0** restored eval +143 at cost of losing the new `r_pusher_vel` term (net −25 from the perturbation, better than -1.7 but still below +143).

**Lesson:** when shaping, the ratio between shaping terms and ground-truth (pos+angle error) must stay bounded. A shaping term that dominates becomes the policy's real objective — and if it's not aligned with the actual task, you optimize the wrong thing. Tuning heuristic: keep shaping reward magnitude ≤ 0.5× max task reward. If a bigger shaping term helps early exploration, anneal it down over training (curriculum on reward weights).

---

## Always Log Per-Component Reward + Velocity Magnitudes When Shaping (2026-04-18)

**What happened:** Spent several iterations blindly tweaking reward term weights without visibility into which component was dominant. Added env-side metrics (`r_pos`, `r_angle`, `r_approach`, `r_block_vel`, `r_pusher_vel`, `pusher_vel_mag`, `block_vel_mag`, `pusher_to_block`) then ran a quick diag rollout on each checkpoint. Immediately saw pusher_vel 24× smaller in vel/tele vs pos. One diag run answered several open hypotheses.

**Pattern:**
1. Emit every reward component as a scalar metric from env (in `state.metrics`).
2. Emit proxy metrics for "is the policy even _using_ the action space?" (`pusher_vel_mag`, distance to relevant bodies).
3. Rollout 200 steps on the trained checkpoint, print mean of each metric.
4. Diff metrics across configs instead of only comparing eval reward.

**Lesson:** per-component reward logging is essentially free (µs per step) and turns "why is this worse" from a multi-hour A/B into a 10-line diff. Bake it into env design from day 1 for any env with more than 2 reward terms.

---

## Vendoring Old Static Benchmarks Beats Pip Dependency (2026-04-18)

**What happened:** `gym-pusht` from Hugging Face broke on `pymunk>=7` (`Space.add_collision_handler` removed). pip install pulled pymunk 7.2; env crashed at `reset()`. Downgrading pymunk unblocked it, but now our env set has a brittle dependency on a floating pip version of a 2-year-old static benchmark.

**Fix:** Copied `gym_pusht/envs/pusht.py` + `pymunk_override.py` + LICENSE (Apache 2.0) into `jax_rl/envs/manipulation/pusht/`. Added `reward_mode` kwarg for swappable rewards (`coverage` | `sparse` | `shaped` | `approach`) without forking any upstream logic. Packed 206 LeRobot expert demos into `demos/pusht_demos.npz` (0.29 MB). Parity test: 100 random steps → byte-exact match with pip `gym-pusht` in coverage mode.

**When to vendor vs pin:**
- **Vendor** when: upstream is static (no ongoing development), upstream is small (<1k LOC), or you need to extend the API (new kwargs). Push-T ticks all three.
- **Pin** when: upstream is actively maintained, large, or security-sensitive. Vendoring loses automatic fixes.

**Lesson:** a 700-line frozen benchmark from a 2023 paper is better vendored than pip'd. The pinning battle is lost before it starts — some dep will force you to upgrade pymunk/numpy/torch eventually, and old benchmarks don't follow. Copying in the code + LICENSE is the cheapest form of reproducibility insurance.
