# Push-T Obs / Action / Config Design Lessons

Design choices for the planar pushing benchmark — obs schema, action modes, shape registry, vendoring, and the working full-stack recipe. Physics-engine details in `pusht_physics.md`; rewards in `pusht_rewards.md`; eval/metrics in `pusht_eval.md`.

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

## Vendoring Old Static Benchmarks Beats Pip Dependency (2026-04-18)

**What happened:** `gym-pusht` from Hugging Face broke on `pymunk>=7` (`Space.add_collision_handler` removed). pip install pulled pymunk 7.2; env crashed at `reset()`. Downgrading pymunk unblocked it, but now our env set has a brittle dependency on a floating pip version of a 2-year-old static benchmark.

**Fix:** Copied `gym_pusht/envs/pusht.py` + `pymunk_override.py` + LICENSE (Apache 2.0) into `jax_rl/envs/manipulation/pusht/`. Added `reward_mode` kwarg for swappable rewards (`coverage` | `sparse` | `shaped` | `approach`) without forking any upstream logic. Packed 206 LeRobot expert demos into `demos/pusht_demos.npz` (0.29 MB). Parity test: 100 random steps → byte-exact match with pip `gym-pusht` in coverage mode.

**When to vendor vs pin:**
- **Vendor** when: upstream is static (no ongoing development), upstream is small (<1k LOC), or you need to extend the API (new kwargs). Push-T ticks all three.
- **Pin** when: upstream is actively maintained, large, or security-sensitive. Vendoring loses automatic fixes.

**Lesson:** a 700-line frozen benchmark from a 2023 paper is better vendored than pip'd. The pinning battle is lost before it starts — some dep will force you to upgrade pymunk/numpy/torch eventually, and old benchmarks don't follow. Copying in the code + LICENSE is the cheapest form of reproducibility insurance.

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
