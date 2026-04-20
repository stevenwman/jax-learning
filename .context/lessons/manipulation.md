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

---

## TimeLimit Is NOT Applied by Direct Env Construction (2026-04-19)

**What happened:** Trained SAC on vendored PushTEnv for 8 runs, best sto coverage 14%. Hit a "ceiling." Kept tuning HPs (reward scale, batch, UTD, entropy, network, action repeat, frame stack, etc.) without progress. The ceiling turned out to be a missing env wrapper.

`gym.make("gym_pusht/PushT-v0")` registers via `EnvSpec(max_episode_steps=300)` and auto-wraps with `TimeLimit(300)`. But our training script did direct construction: `PushTEnv(obs_type=..., reward_mode=...)`. This **does not apply TimeLimit**. Failed episodes (which is 100% of them, since success is rare) ran indefinitely in training, never terminating or truncating.

**Why this destroys SAC:** with no termination, the Q target `r + γV(s')` bootstraps a value assuming the episode continues forever. For push-T where non-success contact_gated rewards average slightly negative per step, the critic learns to estimate an unbounded negative sum. Q values drifted to -1400+ early in some runs, oscillated for 200k+ steps before stabilizing.

**Diagnostic signatures we should have caught earlier:**
- `ep_r_avg = -3100` when per-step reward is bounded in [-0.3, 1.5] — **physically impossible** for one episode. Means episode was cumulative across many would-be resets.
- Q1 values at -50 to -1400 range when Q should be bounded ≤ ~150 under any normal reward structure.
- Massive Q1 loss spikes (2000+) early in training.

**Fix (one line):**
```python
env = gym.wrappers.TimeLimit(env, max_episode_steps=300)
```
After this: **peak coverage 14% → 88%.** 6× improvement in one fix. All prior HP tuning was running on broken infrastructure.

**Lesson:** gym env wrapper stack is implicit via `gym.make()`. Direct construction silently drops wrappers. Always either (a) use `gym.make()`, or (b) assume direct construction gives you nothing and apply every wrapper explicitly (TimeLimit, OrderEnforcing, etc.). Read `env.spec.max_episode_steps` to verify TimeLimit is active.

---

## Verify Infrastructure Before Tuning HPs (2026-04-19)

**What happened:** 8 consecutive training runs across 6 reward modes and 15+ HP combinations, all plateauing at ~14% coverage. I kept proposing "maybe larger batch, maybe higher UTD, maybe different shaping" without questioning whether the env was even running correctly. Only after user pushback ("metrics look impossible, find evidence before fixing") did I check the underlying mechanics and find the missing TimeLimit in 5 minutes.

**Costs of rapid pivoting without verification:**
- 6+ hours of wall-clock training time on a broken infra
- Incorrect conclusions ("push-T has a ~14% RL ceiling")
- Documented misleading "lessons" about shaping that would have steered future agents wrong
- Generic "try more things" instead of **"this metric is wrong, why"**

**The tell I missed:** `ep_r_avg = -3100` is not a tuning signal — it's a reality check failure. Per-step reward is bounded in [-0.3, 1.5], episode is nominally 300 steps, so ep return is in [-90, +450]. The number -3100 can only mean multiple episodes without reset. I should have asked "can this number even happen?" before reaching for another HP.

**Rule:** when a metric goes out of physically-possible range, stop tuning. The bug is in infrastructure (env, buffer, reward, wrappers), not HPs. **Specifically:**
1. Compute min/max possible value for every scalar you log.
2. When you observe a value outside that range, halt and find the bug.
3. HP tuning assumes the metric reflects what you think it does. If it doesn't, tuning is chasing noise.

Applied retroactively, this would have caught the TimeLimit bug at run 1 instead of run 9.

---

## Pymunk CoG Offset → `block.position` ≠ Block Goal Pose (2026-04-19)

**What happened:** Calibration script tried to set the T at the goal pose with `block.position = (256, 256), block.angle = π/4`. Coverage came out at 0.298, not 1.0. Debugging revealed `block.center_of_gravity = (0, 45)` in body frame. When you set `body.position = X` for a body with non-zero CoG, X is the **body origin location**, not the CoG location. World CoG = body.position + rotate(CoG, angle). Setting position then angle (per `_set_state` order) made the block end up at world (287.8, 269.2) — 32 pixels from where I asked.

**True identity state** (gives coverage 0.9995): `[agent_x, agent_y, 224.2, 242.8, π/4]` for `reset_to_state`. Discovered by 0.5-px grid search around (256, 256).

**Implications for any code using `block.position`:**
- Distance shaping `||block.position - goal_pose[:2]||` is measuring distance between body origin and goal *body origin*, NOT geometric centers. Off by ~30 px on the constant CoG offset, so still gives a useful gradient — but interpret with care.
- For exact pose matching (e.g. setting up a calibration test), use `pymunk_to_shapely(body, shapes).centroid` to get the actual world centroid.
- For RL training, the bias is a constant offset, so the policy learns to compensate. Not a hard bug, but a footgun for diagnostics.

**Lesson:** any pymunk body with non-zero `center_of_gravity` behaves "weirdly" under direct position+angle assignment because rotation pivots around CoG, not origin. For T-shapes, hexagons, asymmetric polygons — read `body.center_of_gravity` first; never assume `body.position` = world geometric center.

---

## Coverage Metric Is Geometrically Sensitive — 2 px Drops 5% (2026-04-19)

**What happened:** Trained policy achieved 84% mean coverage. User looked at 5 rendered episodes — visually they all looked very close to goal. Wrote calibration tool that directly perturbs block pose by known amounts and measures coverage. Result:

| Coverage | x-translation | yaw |
|----------|---------------|-----|
| 0.999    | 0 px          | 0°  |
| 0.95     | 1.9 px        | 2.5°|
| 0.90     | 3.8 px        | 5.1°|
| 0.85     | 5.7 px        | 7.8°|
| 0.80     | 7.8 px        | 10.5°|
| 0.70     | 12.0 px       | 16.3°|

The T has thin bars (~15 px wide). A 2 px translation (0.4% of arena) loses 5% coverage because the thin top bar shifts off the target's top bar. **Coverage is calibrated for fine-grained alignment** — visually similar poses can differ by 10-20% coverage.

**Implication:** when reporting RL results on push-T, an 80-85% mean coverage policy is "close to perfect" visually but quantitatively the metric is unforgiving. Don't conclude "policy is bad" from low coverage without checking pose error directly.

**Calibration script:** `tools/pusht_coverage_calibration.py`. Outputs labeled figure showing what each coverage level looks like.

---

## Tunable `success_threshold` for Sparse-Reward Tractability (2026-04-19)

**What happened:** Default `success_threshold=0.95` (DP convention) is unreachable by humans (max LeRobot demo = 0.9489) and by pure RL (peak ~0.89). Sparse reward mode `1 if coverage > 0.95 else 0` therefore gives literally zero signal during from-scratch training.

**Fix:** added `success_threshold` kwarg to `PushTEnv`. Defaults to 0.95 (parity), can be lowered for tractable sparse RL:
```python
env = PushTEnv(reward_mode="sparse", success_threshold=0.85)
```

**Reporting convention reminder:** push-T literature reports **max coverage per episode**, not binary success rate. DP scores ~0.91-0.95, BC LSTM ~0.55-0.74, ours ~0.84-0.89. Don't compare against 0.95 termination flag.

---

## The 95% Success Threshold Is Above Human Expert Performance (2026-04-19)

**What happened:** Our trained SAC policy hit 84% mean coverage, 0% success rate (threshold = 0.95). Looked like failure but evidently the policy was getting very close. Dug into the bundled LeRobot expert demos (206 human teleops): **max coverage across all 25,650 frames = 0.9489.** Not a single frame in the expert dataset crosses the 0.95 threshold. Humans playing the game teleoperated fail the "success" check too.

**Implication:** For push-T (gym-pusht default), success threshold = 0.95 is functionally unreachable for expert humans via teleoperation. Published RL results that cross it (e.g. DPPO) use BC pretraining + RL fine-tuning to exploit gradient-based refinement beyond demo quality.

**Our policy:**
- Mean det coverage: 0.84 — about 89% of human teleop peak
- Peak sto coverage: 0.889 — about 94% of human teleop peak
- Final pose error: ~7 px position, ~1.4° angle
- **Functionally solves the task within human-achievable range.**

**Lesson:** before treating "0% success" as training failure, check whether the success metric is actually achievable by any baseline including humans. On imitation benchmarks in particular, thresholds may be calibrated around a BC baseline's ceiling rather than a hard physical achievability line. Report coverage distribution and compare to the bundled demos, not just the binary success flag.

---

## Don't Eyeball Metrics from Rendered Video (2026-04-19)

**What happened:** User sent 5 screenshots of trained-policy eval episodes. I confidently described image 4 as "position close but ~30° yaw off → ~50% coverage" based on visual inspection. User challenged ("image 4 has almost perfect yaw"). Ran env-side diagnostic: actual angle error for ep 4 was **4.66°**, not 30°. Coverage was 0.858 (85.8%), not 50%.

**My 30° eyeball claim was off by 6×.** Coverage claim was off by 1.5×. All based on "looking at the picture."

**Why this matters:** I was steering the user's decision ("add more angle shaping, angle is the bottleneck") on invented data. Could have pushed the whole training session in a wrong direction. The correct diagnosis was: angle is fine (<5° across all eps), position is the binding constraint (~7 px short).

**Rule:** env metrics are scalars. Don't reconstruct them from pixels. If you need final-state metrics to diagnose a run, compute them from the env state (pymunk `body.position`, `body.angle`) or from the info dict, not from screenshots. 2 lines of Python. Don't save the user from "is the policy succeeding" by staring at videos — it's the kind of diagnosis that looks authoritative but isn't. Write the diagnostic script.

---

## Contact-Gated + Frame Stack + Action Repeat + Keypoint Obs — Combined Recipe (2026-04-19)

**What worked** (after TimeLimit fix, eval 84% coverage):

1. **Obs**: `environment_state_agent_pos` (18d: 8 T-keypoints × 2 + agent_xy) + `FrameStack(3)` → 54d obs. Includes implicit velocity + past action history.
2. **Obs normalization**: rescale pixel coords [0, 512] to [-1, 1] via linear wrapper. Without, Q explodes early. Standard pre-SAC preprocessing.
3. **Action repeat = 2**: each policy decision held for 2 env steps. Halves effective control frequency to 5 Hz but commits to directional pushes long enough for contact to matter.
4. **Reward `contact_gated`**: gated approach (when not touching) + gated directional velocity (when touching) + angle-delta (rotation progress) + large success bonus. See `jax_rl/envs/manipulation/pusht/pusht.py` for full formula.
5. **SAC HPs**: `target_entropy_scale=2.0`, `batch=1024`, `UTD=2`, `lr=1e-4`, `gamma=0.995`, `tau=0.005`, `reward_scale=0.1`, `grad_clip_norm=1.0`, `buffer=500k`.
6. **Network**: (256, 256) actor, (256, 256) critic — vanilla SAC Q-head, NOT FastSAC C51.

**Lesson:** push-T RL from scratch requires ~6 stacked design decisions to work. Each individually could be ablated (and we did ablate several). The full stack matters because the task's exploration landscape is genuinely difficult. Frame stack alone doesn't fix it; action repeat alone doesn't fix it; contact-gated shaping alone doesn't fix it. The combination plus TimeLimit hit 84% coverage. Published pure-RL results are rare on this env for a reason — it takes a carefully-tuned pipeline.

---

## FastSAC C51 Critic Is Wrong Choice for Bounded-Reward Manipulation (2026-04-19)

**What happened:** Early training attempts used `FastSAC` (C51 distributional critic, atoms over `[v_min, v_max]` range). Training diverged because `v_max` defaults to 20 in the paper preset, but cumulative contact_gated reward over 300 steps can hit ~150. Critic atoms don't cover actual Q range → critic is "blind" beyond v_max → policy can't improve past that ceiling.

**Fix:** switch to vanilla SAC (scalar Q, unbounded). Same task works immediately (once TimeLimit is also fixed).

**When to use each (updated from earlier locomotion-centric lessons):**
- **FastSAC / C51**: good for **unbounded** locomotion rewards, paper-tuned for humanoid / Go2 scale. Needs `v_min`/`v_max` sized to actual discounted Q.
- **Vanilla SAC**: better for **bounded, short-horizon, shaped-reward** tasks like push-T. No atom-range landmine.

**Lesson:** distributional critics are optimization tools, not reward-range magic. Always sanity-check that `[v_min, v_max]` covers `reward_min * horizon` to `reward_max * horizon` under your gamma and episode length. If it doesn't, the distributional critic is actively worse than a scalar one.

---

## Log-Barrier Coverage Reward Beats Linear by +8pp (2026-04-20)

**What happened:** Baseline `contact_gated` used `r_coverage = coverage_clip` — linear in `[0, 1]`. Policy plateaued around 85% cov because marginal reward was flat (1pp gain = 0.01 reward regardless of coverage level). Geometry of coverage metric is nonlinear: 30→60% is gross-motor, 90→95% is pixel-precision. Linear underpays precision work.

**Fix:** `coverage_shape="log_barrier"` → `r_coverage = -log(1 - clip(cov/threshold, 0, 1) + ε)` with `ε=0.01`.

```
cov    linear   log_bar
0.5    0.53     0.73
0.7    0.74     1.30
0.9    0.95     2.77
0.95+  1.00     4.60 (ceiling at ε=0.01)
```

Marginal reward near goal ∝ `1/(1-cov+ε)` — at cov=0.9, gradient is 9× steeper than at cov=0. Policy actually pursues the last few percent.

**Evidence:**
- v9 linear (full stack): 0.852 sto
- baseline_logbar (same stack, log_bar only): **0.933 sto** (+8pp, det std 5× tighter)

**Cost:** Q magnitudes grow ~3-5× (ep_r_avg from 130 → 500). No instability observed at `reward_scale=0.1, grad_clip_norm=1.0`. Monitor Q1 trajectory for early-training blowup signs.

**When NOT to use:** if coverage metric is already well-distributed (e.g. dense geometric progress), log_barrier adds instability without gain. Best applied to metrics where the last few % are disproportionately hard (IoU, pose overlap, SSIM).

**Lesson:** match reward curvature to metric curvature. Linear reward on a geometrically-nonlinear metric caps learning where marginal task difficulty exceeds marginal reward. Log-barrier is the standard interior-point shape for this.

---

## Action Repeat Is the Single Most Critical Knob on Push-T RL (2026-04-20)

**What happened:** ablation stripping `action_repeat=2 → 1` collapsed performance from 0.933 → 0.522 sto (−41pp). Largest effect of any single knob in the ablation study (vs keypoints −5pp, frame_stack −1pp, reward shape +8pp).

**Why:** Push-T needs sustained directional force on the block to initiate/continue contact pushes. At AR=1 (10 Hz policy), any small policy oscillation reverses the pusher direction mid-push. With AR=2 (5 Hz policy), each action commits for 2 env steps → the block actually accumulates velocity. Also: halves decision count → less Q-target noise → more stable learning.

**Evidence:**
| AR | Det | Sto |
|---|---|---|
| 2 | 0.867 | 0.933 |
| 1 | 0.326 | 0.522 |

Q1 range during training: AR=2 stable ~+90, AR=1 oscillates into negative. Alpha (SAC entropy coef) also bounced 0.04 with AR=1 vs stable 0.02 with AR=2.

**Lesson:** AR is not cosmetic — it's **algorithmic**. For contact-rich manipulation, policy-rate > control-rate matters more than obs richness or reward shaping. Always test K∈{2, 4, 8} before optimizing anything else. Action chunking (Q-chunking NeurIPS 2025) is the generalization.

---

## Minimal Shape-Agnostic Config Matches Full Stack (2026-04-20)

**What happened:** cross-shape benchmarks need shape-agnostic obs (keypoints have shape-specific dim). Hypothesis: swapping 18d keypoints → 5d state + dropping frame_stack would drop performance substantially. Tested on push-T:

| Config | Det | Sto |
|---|---|---|
| Full: keypoints(18d) + FS=3 + AR=2 + log_bar | 0.867 | 0.933 |
| **Minimal: state(5d) + FS=1 + AR=2 + log_bar** | **0.906** | **0.939** |

Minimal *beat* full stack. First success event (cov=0.9511, terminated at step 27) observed only on minimal config. Single seed per run — noise possible — but direction clear.

**Why it might win:** FS=3 on 5d state = 15d with correlated dims (same 5 obs shifted in time). Under log_barrier which gives strong signal, the extra 10 dims add input noise without useful velocity signal (vels already recoverable). With FS=1, the critic fits a simpler input distribution → lower-variance Q → tighter policy.

**Implication for cross-shape:** use `state + FS=1 + AR=2 + log_bar + contact_gated` as the cross-shape baseline. 5d obs is shape-agnostic (agent_xy + block_xy + block_yaw), no redefinition per shape.

**Lesson:** don't assume richer obs = better. When the reward signal is strong (log_bar), minimal obs often outperforms padded obs. Test the minimal config before porting complexity.

---

## Bigger Success Bonus Doesn't Raise Ceiling on Unreachable Thresholds (2026-04-20)

**What happened:** tested `success_bonus: 50 → 200` on full-stack log_bar config. Expected: bigger terminal pull → policy reaches higher coverage. Actual: det 0.867 → 0.914 (+5pp tighter), sto 0.933 → 0.933 (unchanged).

**Why:** with `success_threshold=0.95` and peak coverage observed during training ~0.94, the policy never crossed threshold → never sampled the bonus. Making an unobserved terminal larger doesn't change learning.

Det tightened because Q near goal has less variance (policy converges to the same near-threshold trajectory), but the stochastic ceiling is set by the unreachable threshold, not the bonus magnitude.

**Lesson:** before tuning success bonus, verify policy actually hits successful terminations during training. If terminal is never sampled, its magnitude is irrelevant to learning — only to offline analysis. Alternative: lower threshold until terminations happen during training (tradeoff: caps learning at the threshold).

---

## Pymunk Shape Construction — Decompose Concave Letters Into Convex Rings (2026-04-20)

**What happened:** needed 4 new block shapes (ellipse, iso-triangle, letter S, letter U) for cross-shape benchmark. pymunk requires **convex** `Poly` geoms; letters S / U are concave. Early attempts at S used rotated rectangles along a bezier centerline — produced visible "fins" at hook tips (tangent rotates fast near tight curvature → quads poke perpendicular to curve).

**Fix:** build concave letters as sets of **annular sectors** (fat C's). Each sector decomposed into wedge quads via:

```python
def _ring_polys(center, inner_r, outer_r, theta_start, theta_end, n):
    thetas = np.linspace(theta_start, theta_end, n + 1)
    return [[
        (cx + inner_r * cos(t0), cy + inner_r * sin(t0)),
        (cx + outer_r * cos(t0), cy + outer_r * sin(t0)),
        (cx + outer_r * cos(t1), cy + outer_r * sin(t1)),
        (cx + inner_r * cos(t1), cy + inner_r * sin(t1)),
    ] for t0, t1 in zip(thetas[:-1], thetas[1:])]
```

- **Letter S**: two 270°-arc rings, rot-180 symmetric, overlapping in middle strip → smooth uniform-curvature S with 18 quads total. Thickness `outer_r - inner_r` controls stroke width; `inner_r` controls hook-interior radius (must be ≥ pusher radius for reachability).
- **Letter U**: 180° half-ring + 2 rectangles (arms).
- **Ellipse / triangle**: single convex `Poly` each (no decomposition needed).

Every wedge is tangent to a circle → uniform curvature across the shape, no kinks or tangent-mismatch artifacts.

**Design knobs:**
- `n_per_ring` = 10 (30°/wedge) looks clean; 8 acceptable, 6 chunky.
- Scale shape so `inner_r > pusher_radius + margin` (pusher 15 px, inner_r ≥ 22 → 7 px margin).
- For letter hooks: trim the END wedge of each ring (reduce `theta_end` by ~30°) for tapered tip instead of squared terminal.

**Lesson:** when pymunk's convexity constraint bites, reach for annular sectors before bezier ribbons. Circle-tangent decompositions produce much cleaner curves per poly count than curve-sampling approaches.

---

## Shapely TopologyException on Overlapping Convex Pieces (2026-04-20)

**What happened:** after adding multi-piece block shapes (S = 18 overlapping quads, U = 10+2 polys), `_get_coverage` crashed:

```
shapely.errors.GEOSException: TopologyException: side location conflict
at 343.55936998684041 274.20135325161755. This can occur if the
input geometry is invalid.
```

**Root cause:** `pymunk_to_shapely` builds a `sg.MultiPolygon` from the convex pieces and calls `.intersection(goal_geom).area`. When pieces overlap (S has two overlapping rings by design), the MultiPolygon has self-intersections at overlap boundaries. GEOS rejects the geometry as invalid.

**Fixes (in order of robustness):**
1. **Union first, then intersect:** replace `MultiPolygon` with `unary_union(polygons).buffer(0)` — heals overlap seams into a single valid polygon. `buffer(0)` is a known GEOS idiom for fixing self-intersecting geometries.
2. **Catch + stub:** try/except around `.intersection` → return 0. Fine for vibes-only tests where coverage doesn't drive learning. Used in `tools/record_pusht_shapes.py`.
3. **Redesign shape**: decompose such that convex pieces don't overlap. Doubles the convex-piece count and breaks the elegant ring construction.

**Lesson:** MultiPolygon is not the same as "polygon with holes" in GEOS. Overlapping convex pieces → invalid MultiPolygon. When porting T-specific coverage metric to multi-shape envs, wrap union with `buffer(0)` or catch the exception explicitly. Don't assume valid geometry from valid pymunk shapes.

---

## Zero-Shot Cross-Shape Transfer Is a Floor, Not a Baseline (2026-04-20)

**What happened:** took `minimal_logbar` policy trained on T (5d state obs, 0.939 sto on T) and ran it on 4 new shapes (ellipse, triangle, S, U) without retraining. Coverage results:

| Shape | Coverage |
|---|---|
| tee (in-distribution) | 0.868 |
| ellipse | 0.121 |
| triangle | 0.025 |
| U | 0.126 |
| S | ~0 (coverage stubbed) |

**Why it fails:** the 5d state obs is `(agent_xy, block_xy, block_yaw)`. No shape information. Policy learned pushing strategies specific to T-geometry (e.g. which side to approach given the T's asymmetric CoM-vs-bbox). On a different shape, the same approach strategy pushes the wrong surface — block rotates unpredictably or moves perpendicular to goal.

**Implication:** cross-shape generalization needs either:
1. **Shape-aware obs** — pass shape ID or shape-specific keypoints. Kills 5d shape-agnostic property.
2. **Domain randomization during training** — sample shapes per episode, force policy to learn "push-block-to-pose" as a shape-invariant skill rather than a T-specific one.
3. **Contact-based obs** — pusher's contact history on the block's current surface (robot-relative, shape-invariant). Similar to what humans use.

**Lesson:** zero-shot transfer from T-only training is the **floor** for cross-shape generalization, not a working baseline. Useful as a sanity check that infrastructure is correct (policy runs, renders, doesn't crash across all shapes), but not as a claim of generalization. Next step: DR training on mixed shapes.
