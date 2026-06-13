# Go2 Locomotion Lessons

---

## action_scale Caps Peak Tracking Speed — Don't Assume Smaller Is Safer (2026-04-28)

**Hypothesis (entering):** halving `action_scale` from 0.5 → 0.25 should
make first-hardware deploys safer (smaller joint deltas per step), at
worst trading off some tracking performance.

**What actually happened:** the 0.25 policy plateaued at eval **273**
after 100M steps, with persistent eval std ±58. Same pipeline at
`action_scale=0.5` plateaued at **288** by 28M with std ±5.9.

| scale | total | best | final | std | eval-points to plateau |
|---|---|---|---|---|---|
| 0.25 | 100M | 273.5 | 242.5 | ±58 | never settled |
| 0.50 |  50M | 288.1 | 283.3 | ±5.9 | ~14M |

**Why:** the env samples `cmd_vx ∈ [-1.5, 1.5]`. At scale=0.25, max joint
delta per step is half — peak achievable forward velocity is also lower,
so the policy can't satisfy `cmd_vx > ~1.0` no matter how good its gait.
The reward function penalizes the gap → wide eval variance. At 0.5 the
policy can saturate the command space; gait converges; std collapses.

**Diagnostic confirmed:** rendering the 0.25 policy with
`--varied-cmds 50 --cmd-max 1.0 1.0 1.2` (linvel capped at ±1) showed
visibly cleaner tracking. Re-running with default cmd range showed the
robot lagging on high-speed commands.

**Rule of thumb:** `action_scale` is a peak-velocity ceiling, not just a
"smoothness knob". Set it from your *desired tracking range*, not from
"how aggressive do I want the policy to look on first hardware run".
Conservative caps belong in the deploy command profile (clamp `cmd_vx`
at the start), not in `action_scale`.

**Numbers for reference:** `cmd_vx_max / action_scale` ≈ 3 was the
breakpoint here (1.5 / 0.5). Below that ratio, tracking saturates.

---

## When Porting Reward Weights Between Robots, Rebalance — Don't Copy (2026-03-25)

**What happened:** Go2 PPO trained for 100M+ equivalent steps across 20+ seeds without ever walking. The env was structurally correct. Reward WEIGHTS were copied from Go1 (tracking_lin_vel=1.0, tracking_ang_vel=0.5). Go2 never walked.

**Root cause:** Go1's dynamics make walking easy enough that tracking=1.0 suffices. Go2 is heavier with different joint geometry — pose reward (~450) dominated tracking (~130). Crouching was optimal.

At 10x tracking (tracking_lin_vel=10.0, tracking_ang_vel=5.0): pose still ~450 but tracking ~4500. Walking dominates.

**Wrong hypotheses chased first (all real bugs, none sufficient alone):**
- Calf torque fix (24→45.43 Nm): eval ~15 → ~12-14, no walking
- Entropy collapse tuning: higher entropy_coef made things worse
- Obs normalization, network init, double damping, action scale: no effect
- Height termination alone: eval dropped to 4-7

**The actual fix:** 10x tracking + height termination (base_z < 0.18m) + calf torque fix. Seed 2100: eval 233 at 50M steps.

**Rule of thumb:** After any port, run 5M steps, extract reward breakdown, verify: `tracking_reward / (pose_reward + tracking_reward) > 0.5`. If not, increase tracking weights.

---

## Always Verify MJCF Actuator Limits Against Hardware Specs (2026-03-25)

Go2 couldn't lift its feet. Menagerie's `go2_mjx.xml` sets calf `forcerange="-24 24"` but real Go2 calf torque is **45.43 Nm** — 53% of real.

**Fix:** Override `actuator_forcerange` for calf joints to [-45.43, 45.43] in `go2_base.py`.

**Lesson:** Cross-check EVERY actuator limit against hardware URDF/spec sheet. Menagerie has TWO Go2 models with DIFFERENT actuator specs.

---

## Env Parity Is Not Just Reward Math (2026-03-24)

Go2 plateaued at eval ~15 while Go1 hit 27 with identical reward weights.

**Root causes found:**
1. Missing CCD iterations (Go1=20, Go2=4 default)
2. Incomplete privileged state (116d vs 122-130d)
3. Soft foot contacts (solimp 0.015 vs 0.9)
4. Calf torque limited to 24 Nm vs real 45.43 Nm
5. Reward balance: tracking too weak vs pose

Physics fixes (1-4) were necessary but not sufficient. Reward rebalancing (5) was the final blocker.

---

## Entropy Collapse Was a Symptom, Not the Cause (2026-03-25)

Go2 PPO entropy collapsed to -7.6 by 9M steps. Three entropy_coef values tested: 0.01 (eval ~12-14), 0.02 (eval ~4-5), 0.05 (eval ~2-4).

Higher entropy_coef made training WORSE. The real problem was reward landscape: pose dominated tracking. Fix was 10x tracking reward, not entropy tuning.

---

## Compare Full Env Implementation, Not Just Config (2026-03-25)

Matched Go1's config exactly. Go2 still plateaued. Diffing a working reference found:
1. Go1 uses "feetonly" collision
2. No height termination in our env
3. Reward balance: pose dominated tracking

**Lesson:** Diff the FULL implementation — collision geometry, termination conditions, reward balance. Config can be identical and the env still broken.

---

## Menagerie MJCF Contact Physics Differ From Playground (2026-03-24)

Menagerie Go2 uses solimp=0.015 (marshmallow-soft), Playground Go1 uses solimp=0.9 (firm). Fixed by overriding in go2_base.py. Also: Go2 has ALL body geoms active for collision while Go1 uses feetonly — thighs touching floor encourage crouching.

---

## Custom Locomotion Envs — Playground Integration

### Subclassing MjxEnv Works Cleanly
Subclass `MjxEnv`, override `_get_obs()`, `_get_reward()`, `_get_termination()`. Composable reward dict weighted by config.

**Key gotcha:** Menagerie MJCF is missing sensors the env needs. Created scene XML that `<include>`s Menagerie model and adds sensors.

### Go2 vs Go1 Naming Differences
Body: `base` (Go1: `trunk`). Foot sites: `FL_foot` (Go1: `FL`). Joint order: FL/FR/RL/RR (Go1: FR/FL/RR/RL). PD override works identically.

---

## Training MJCF != Deployment MJCF — Unify Before Expecting Transfer (2026-03-26)

**What happened:** Built full sim2sim pipeline (DDS, headless simulator, numpy policy inference). Robot stands up perfectly at 0.27m via the FSM. Policy takes over, robot falls immediately.

**Root cause:** We train on Menagerie `go2_mjx.xml` (with custom overrides: solimp=0.9, calf torque=45.43Nm, feetonly-ish contacts). We deploy to unitree_mujoco's `go2.xml` which has different actuator models (pure torque motors vs position-controlled), different contact parameters, different damping. These aren't two copies of the same sim — they're two completely different physics environments that happen to model the same robot.

**What proved the pipeline is correct:** The stand-up FSM works perfectly (cosine interpolation to default pose, robot reaches 0.27m). Joint remapping is correct (FL/FR/RL/RR ↔ FR/FL/RR/RL). DDS round-trip works. Policy inference produces valid actions. The failure is purely physics mismatch.

**Lesson:** "Sim2sim" only works if both sims use the same MJCF and physics config. Training on one model and deploying to another is really sim-to-different-sim — same gap as sim-to-real, just with known physics on both sides. Before expecting policy transfer, unify the MJCF: either train on the deployment model, or make the deployment model match the training model.

**Options going forward:**
1. Train directly on unitree_mujoco's `go2.xml` (cleanest for deployment, but need MJX compatibility)
2. Make our MjxEnv load unitree_mujoco's MJCF (keep training pipeline, match deployment physics)
3. Domain randomization across both models (most robust, most work)

Full comparison table: `.context/go2/mjcf_comparison.md`

---

## Actuator Type Mismatch Is the #1 Sim2Sim Failure Mode (2026-03-26)

**What happened:** Policy stands up correctly (FSM works), falls the instant it takes control.

**Root cause:** Our MJX env uses `general` actuators with `biastype="affine"` — PD control is baked INTO the actuator. `ctrl[i] = position_target`, and MuJoCo internally applies `force = Kp*(ctrl-q) - Kd*qvel`. The unitree_mujoco model uses `motor` actuators — `ctrl[i] = raw_torque`. The DDS bridge computes PD externally and writes torque.

Although the PD math is the same, the integration timing differs: training PD is inside the physics substep (applied per-substep), deploy PD is computed once at DDS rate (50Hz) and held constant across substeps. This creates a 5-substep lag in the effective control response.

**Also:** Joint damping is 5x different (0.5 training vs 0.1 deploy), foot contact model differs (condim 3 vs 6, different friction), friction cone differs (pyramidal vs elliptic).

**Lesson:** When building a sim2real pipeline, the deployment simulator's physics must be auditable against the training env. Don't assume "same robot model = same physics." Audit: actuator type, damping, contact params, solver settings, timestep. Full comparison template in `.context/go2/mjcf_comparison.md`.

---

## MJX and CPU MuJoCo Diverge Over Time on the Same Model (2026-03-27)

**What happened:** Policy trained on MJX walks for 3 seconds on CPU MuJoCo (same MJCF, same overrides via go2_cpu.py), then falls. Same policy in MJX runs indefinitely.

**Root cause:** MJX (JAX/XLA GPU) and mj_step (C CPU) are numerically different implementations. Same algorithm, same model, but floating point accumulation diverges over ~150 policy steps. The policy isn't robust enough to handle the drift.

**Not a bug — a robustness gap.** The policy works on CPU for 3 seconds. It's not a catastrophic mismatch (like the MJCF difference which caused instant failure). It's gradual drift that a more robust policy could ride out.

**Deeper investigation:** Not float32 vs float64 (tested — truncating to f32 made zero difference). Not solver settings (all match). The divergence is bursty, not smooth — step 2 shows obs_diff=19.8 while step 8 is only 3.5. This correlates with contact state changes (making/breaking foot contacts). The MJX and CPU contact solvers produce slightly different forces at the boundary, and those differences compound through the policy feedback loop.

**Fix path:** Wider domain randomization + random external forces (velocity kicks) during training. This is standard in SOTA quadruped pipelines (legged_gym, walk-these-ways) for exactly this reason — making policies robust to physics perturbations covers the MJX/CPU gap as a side effect.

---

## Matching Gain Values Is Not Enough — Actuator Integration Timing Matters (2026-03-26)

**What happened:** Matched Kp=35 and Kd=0.1 between training and deploy. Retrained PPO (eval 219). Sim2sim: robot stands up correctly, then explodes when policy takes over. Trajectory analysis: joint velocities hit ±95 rad/s (training sees ±5).

**Root cause:** Same PD gains, different integration timing. Training uses `general` actuators — MuJoCo applies PD at every physics substep (5× per ctrl_dt, every 0.004s). Deploy computes PD externally at 50Hz (every 0.02s) and writes constant torque. Between updates, joints accelerate freely for 4 substeps with no feedback. The velocity spike creates out-of-distribution obs (normalized jvel = 9.2 vs expected ±2), the policy outputs garbage, and the feedback loop escalates.

**Update:** PD rate was NOT the root cause. Implemented PD at physics rate (every mj_step) — jvel still hits ±65. The `general` vs `motor` actuator type produces fundamentally different dynamics even at the same rate with the same gains.

**Lesson:** When transferring between MuJoCo models, the actuator TYPE matters more than the gains or PD rate. `general` with `biastype="affine"` and `motor` with external PD are NOT equivalent — they interact with MuJoCo's integrator differently. Every major Go2 RL pipeline (unitree_rl_gym, unitree_rl_lab, walk-these-ways) uses `motor` + external PD. If you want sim2sim or sim2real transfer, train with the same actuator model the deployment target uses.

---

## Domain Rand Covers Parameter Ranges, Not Model Structure (2026-03-26)

**What happened:** Implemented Go2 domain randomization (Playground Go1 pattern): friction U(0.3,1.2), mass variation, COM jitter. Trained PPO with DR to eval 237. Sim2sim still failed.

**Why DR didn't help:** DR randomizes friction coefficient VALUES but both training envs use `condim=3` (basic friction model). The unitree_mujoco model uses `condim=6` (full 3D friction with rolling + spinning). Same coefficient, completely different force computation. DR can't bridge structural model differences.

**What we then tried:** Matched ALL physics at runtime in sim2sim_direct.py — condim, friction, cone, dt, solimp. Still failed. The two MJCFs (Menagerie go2_mjx.xml vs unitree_mujoco go2.xml) differ in body inertias, mesh geometry, and joint configurations that can't be overridden at runtime.

**Lesson:** Domain randomization bridges parameter uncertainty (friction values, mass, COM position). When sim2sim fails, compare the actual MJCFs — solver settings, collision geometry, and default parameters like damping matter more than you think. But also: the MJCF model itself must match between training and deployment. Running the training MJCF on CPU mj_step produced walking; running the unitree MJCF with matched parameters did not.

---

## Read the XML Before Running Numerical Tests (2026-03-26)

**What happened:** Spent hours running sim2sim tests, matching parameters one at a time, concluding the models were "fundamentally different." Could have found the root cause in 5 minutes by grepping the XML for `damping`.

**Root cause found by XML audit:**
- Menagerie `go2_mjx.xml`: `<joint damping="2" armature="0.01"/>` (default class)
- unitree_mujoco `go2.xml`: `<joint damping="0.1" armature="0.01" frictionloss="0.2"/>` (default class)

20x damping difference in a single XML attribute. Everything else (body positions, meshes, IMU site, joint names) is identical between the two models.

**Sensors verified correct:** Different sensor ordering and names, but both read the same physical joints. Our SDK_TO_POLICY remapping handles it. Confirmed by comparing `jointpos joint="FL_hip_joint"` in both XMLs — same joint, different sensordata index.

**Lesson:** When two MuJoCo models produce different dynamics, grep the default class definitions FIRST. That's where global parameters like damping, armature, and frictionloss are set. Don't run sim2sim experiments before reading the 5 lines of XML that define the physics.

---

## Pure Cartesian Impedance / OSC on a Floating Base is Inherently Jumpy (2026-06-08)

**What happened:** Built a Go2 joystick env driven by per-leg Cartesian
impedance / OSC (foot = end-effector, action = foot xyz targets) instead of
joint PD, with NO gravity feedforward (pure impedance — the spring bears body
weight). Trained 5M FastSAC → eval 279.6, tracks commands well (fwd vx
1.0→1.002; varied-cmd corr vx 0.87 / yaw 0.94). Eval reward looked great.

**But the gait is a pronk/pogo:** trajectory FK showed **22–24% flight phase**
(all four feet off the ground), mean 1.28/4 feet down, feet flung to 0.30 m,
base launching 0.27→0.46 m, vertical velocity RMS 0.24–0.32 m/s. Eval reward
hid all of this — classic "surviving/tracking ≠ walking well."

**Why (the reusable physics):**
1. Pure impedance with no gravity FF stores energy in the stiff vertical spring
   and has nothing to bleed it → it pogos. Horizontal momentum converts to
   vertical pop on a floating base.
2. ~~OSC's Λ normalizes the foot to ~unit apparent mass, so flinging a foot is
   "cheap" → bounding.~~ **TESTED AND REFUTED** (2026-06-08): retrained the Jᵀ
   variant (no Λ, real anisotropic foot inertia, heavy along the leg) — gait was
   just as jumpy (flight 22%→27%, eval 279.6→279.9, all metrics within noise).
   The controller's inertia model is second-order; the RL policy retrains around
   it. Bounce is reward + dynamics, not Λ. Meta-lesson: a controller-level
   inertia change is easily absorbed by an RL policy — ablate it by *retraining*,
   not by eyeballing the open-loop controller.
3. PD-tuned rewards don't suppress the new dynamics: `lin_vel_z` cost was tiny
   (−0.03/step), `feet_height`/`feet_clearance` were tuned for ~0.1 m PD swings,
   and `action_rate` penalizes the PRE-scale raw action so foot-target jerk is
   under-penalized.

**Lessons:**
- **Characterize gait by flight-phase % + foot-lift + base-z bounce from
  trajectory FK, never by eval reward.** A high reward with 22% flight is a
  pogo, not a walk. Save the `_traj.npz` from record_video and compute it.
- **No-gravity-FF impedance forces high stiffness** (the spring must hold body
  weight via deflection), and high stiffness + low foot apparent inertia is a
  recipe for bounce. If you want a calm gait, the levers are: add a gravity /
  body-weight feedforward (so gains can drop), lower kp, or re-weight
  `lin_vel_z`/`feet_height` for the OSC swing — study before guessing which.
- **Tune impedance gains with a zero-action hold probe BEFORE training.** The
  first default (kp=[800,800,1000]) sagged the base to the 0.18 m termination
  floor under zero action; [3000,3000,4000] holds 0.27 m. Cost: one 150-step
  rollout. Catches the collapse before wasting a 30-min train.
- **Inherited reward specs are silently mis-calibrated for a new action space.**
  When you reuse a joint-PD env's rewards for a Cartesian-target action,
  per-term magnitudes shift 2–4× and some penalties (action_rate computed
  pre-scale) keep the old calibration. Audit term magnitudes vs the old env.

---

## Variable Impedance: Measure the Commanded Stiffness, Not the Reward (2026-06-08)

**What happened:** Added per-foot commanded stiffness to the OSC action space —
scalar (+4, action 16) and per-foot-per-axis (+12, action 24). Each stiffness
dim maps log to s∈[0.25,2] scaling that foot/axis baseline gain (kd∝√s). Trained
5M each on flat Go2 velocity tracking.

**Eval reward was a red herring.** Scalar eval 281.8 (≈ baseline 279.6), per-axis
276.7 (slightly *lower*). By reward alone, per-axis looks like a regression. But
extracting the *commanded* stiffness over a rollout told the real story:

- **Per-axis learned vertical-stiff / tangential-soft from reward alone**: mean
  s_z 0.60 > s_xy 0.51, z/xy > 1 for every foot, AND vertical stiffness ramps up
  during stance (s_z stance > swing, all feet). That's textbook load-bearing
  impedance modulation — stiff along the load axis, compliant in shear, phase-
  gated. The coarser per-foot scalar only showed a weak, leg-heterogeneous
  version of this.
- The policy also independently **leaned soft** (mean s≈0.47), landing at the
  same soft sweet spot the fixed-stiffness sweep found.

**Lessons:**
- **For variable-impedance / any "extra control DoF" study, measure what the
  policy *commands*, not just the task reward.** A finer action space can reveal
  a clean, physically-meaningful strategy while *lowering* reward (extra
  exploration cost on a task that doesn't need the DoFs). Reward ranks it worse;
  the commanded-signal analysis ranks it more interpretable.
- **Flat velocity tracking barely exercises impedance modulation** — the
  behavior is real but mild. To make stiffness modulation *matter*, use a task
  that demands it: rough terrain, soft/variable contact, or large/ randomized
  disturbances (note the training kick is a FIXED ±0.75 m/s, well below the
  ≥2 m/s pure-impedance failure threshold — randomize/ramp it to stress
  compliance).
- **Derive kd from commanded kp** (kd∝√kp, ζ≈1) instead of adding damping action
  dims — halves the added params and keeps the loop critically damped as the
  policy varies stiffness.
- **Variable impedance MATTERS on soft contact — confirmed on Newton MPM mud**
  (2026-06-11; `journals/2026-06-11-newton-mud-eval.md`, `lessons/newton_mud_eval.md`).
  Zero-shot on graded mud, the variable-impedance ckpts penetrate the thick
  (densest) mud ~45% deeper than joint-PD and **fixed**-soft OSC, which TIE (both
  bog at the edge, y≈0.22; var per-foot y0.32, per-axis y0.34). Only the
  *stiffenable* policies (commanding s up to 2× → kp up to [6000,6000,8000]) push
  through, holding the highest posture. This is the "use a task that demands
  stiffness modulation: soft/variable contact" prediction realized — fixed
  compliance ≈ stiff PD; adaptive stiffening wins. **Caveat:** all are FLAT-trained
  (physical motor) tested zero-shot; even the winner still bogs/grinds to a stop in
  thick mud. The controller can't fix what training didn't prepare for → training on
  mud / mud-like DR (randomized ground compliance, sinking, drag) is the next lever,
  with the Newton harness as the measuring stick.

---

## Silent Preset Fallback Trained OSC Envs Without DR for Weeks (2026-06-11)

**What happened:** the preset getters in `env_presets.py` only knew the
older Go2 names. Unknown `Go2Warp*` names (every OSC/physical/rough env)
fell through silently to the bare base config — so all OSC envs trained
with **no DomainRandWrapper** (`reset_mode` default) and
`eval_every_n_episodes=5000` (≈zero mid-run evals). Nothing crashed;
training "worked"; the regime was just wrong for weeks. Discovered only
during the 2026-06-11 variants audit.

**Fix (two rules):**
1. **Preset lookups for a known env family should RAISE on unknown
   names.** All 6 getters now route `Go2Warp*` through
   `_resolve_go2_variant`, which raises `ValueError` (with the known-names
   list) instead of falling back. A fallback default is correct for
   genuinely open-ended env names; for a closed family it is a silent
   misconfiguration machine.
2. **Variants-as-data kills the patch-chain traceability problem.** One
   declaration per env (`EnvVariant`: config knobs + host class + train
   overrides + notes) in `go2_warp_variants.py`; backend registration and
   preset resolution both loop the same table. "What does env X train
   with" is now one table lookup instead of tracing 3–4 subclass /
   config-factory files — which is exactly how the no-DR bug stayed
   invisible.

**Cut date:** OSC runs trained before 2026-06-11 are NOT comparable to
runs after (no DR vs per_step DR + eval-every-500). Journal:
`journals/2026-06-11-env-variants-refactor.md`.

## Virtual-mass OSC + the RR-hang (2026-06-12)

**RR-hang is var-impedance-systematic, but reward-fixable.** Every Cartesian-
impedance/OSC policy (4 reward profiles + the new acceleration-feedback mass
controller) parks the back-right foot: contact raster (replay qpos → CPU
`mj_forward` → `*_floor_found` sensors, duty% over a steady window) shows RR ~0%
while FR/FL/RL are 36–63%. Joint-PD is symmetric (49–57%). CRITICAL: a per-leg
joint-CYCLE count is misleading — RR's joints cycle 44× while its foot never
touches; you must measure CONTACT duty, not joint motion. So it's a learned
3-legged optimum the richer var-impedance action space can reach, not a hard
foot-geometry lock.

**Fix = `gait_participation` reward (anti-leg-park).** Penalize
steps-since-all-four-feet-last-completed-a-contact (scale −2.0, comparable to
feet_slip −0.6 weighted). On the mass controller it lifted RR 0%→32% contact —
all four feet plant — with eval unchanged (276 vs 283). Earlier reward tries
(RMA-minimal, NoAir) didn't fix it because they only *removed* swing-shaping;
they lacked an explicit participation penalty. RR still a touch light (32 vs
44–58); a larger weight evens it further at some cost to the other terms.

**Virtual mass via acceleration feedback:** `F = A·ẍ + K·err + D·ẋ` (bare,
`use_op_space_inertia=False`), policy commands per-axis A, ẍ = control-step
finite-diff of foot world velocity (held across substeps). Needs NO external-
force estimate — the contact force's effect rides in through the sensed ẍ. The
torque-delta proof: passing `A·ẍ` adds exactly `Jᵀ·(A·ẍ)` (`test_accel_force_
enters_as_jt_a_xdd`). STABLE on flat (trains to eval 283, policy actively
modulates A); CATAPULTS on Newton MPM mud — the sharper contact transients spike
ẍ and A·ẍ (a_max=2.0) flings the robot 2 m up + flips ±53°. Clamp a_max→0.5 →
no catapult. The mass *range* is the accel-feedback stability lever; a_max=2.0
(an unvalidated spec guess) is too large for the rough substrate. Newton transfer
needs the Newton-side `mud_osc` ported too (48-d parse + bare law + A·ẍ via
control-step ẍ in the costep) — done, commit e7bc97c.

## Finite-diff across a step that differenced a value with itself (ẍ≡0 bug, 2026-06-12)

The virtual-mass controller needs foot acceleration `ẍ = (v_now − v_prev)/dt`.
For ~weeks-equivalent it computed **exactly 0** because of WHICH two samples it
differenced:
- `v_prev` (`last_foot_vel`) was stored at the END of `step()` (post-control physics).
- next step read `v_now` at its START — but `state.data` flows UNCHANGED across the
  step boundary, so start-of-step-N+1 == end-of-step-N bit-for-bit.
- → `v_now` and `v_prev` were the same instant → diff ≡ 0. The `A·ẍ` term was DEAD;
  the controller was secretly just bare-K/D var-impedance.

**Why it stayed invisible:**
1. Velocity tracking still trained fine (eval 283) — the headline metric never sees a
   zeroed auxiliary term; the policy just ignores the dead channel.
2. The PARALLEL eval path (Newton `mud_osc`) computed ẍ a DIFFERENT, correct way
   (its own prev-vel tracking) → the term went live ONLY at eval, driven by an `A` the
   policy never learned → looked like an "instability" (catapult), was really a
   train/eval mismatch.

**Fix:** update `v_prev` at the START of the step (inside the controller, right when
`v_now` is read), so consecutive samples are one control-step apart.

**Reusable rules:**
- A finite difference across a step is only valid if the two samples are from
  DIFFERENT times. If state flows unchanged across the boundary, "store prev at end,
  read current at start" aliases prev==current → silent 0. Store `prev` at the SAME
  phase you read `current`.
- INSTRUMENT zeroable terms directly (`print |ẍ|`), don't infer from the headline
  metric — a dead term hides behind a policy that compensates via other channels.
- When a quantity is computed in two places (train env + eval harness), a discrepancy
  surfaces ONLY at transfer and masquerades as a physics/stability problem.

## Virtual-mass VERDICT on Newton mud — negative (2026-06-13)

After fixing the ẍ≡0 bug (mass term was live for the first time) and taming it
(Λ-weighted + var_a=(0,0.5) + ẍ EMA 0.3), trained on the analytic-mud ROM forces
and evaluated on held-out Newton MPM mud: **marginal + shallower, doesn't earn its
keep.** 6 reps → ~33% catapult (max z 2.3-7.8 m, flips, collapses); the 4 clean runs
reach only y≈0.3 (stuck at thick-mud entry, don't clear) vs the non-mass winners
NoAir y-2.42 / slow+firm y-0.16 (stable clears). Mechanism: **sim-to-sim ẍ gap** — the
analytic mud the policy trains on is smooth; MPM contact ẍ is spiky, so A·ẍ blows up at
contact even tamed. Note ROM-mud TRAINING *does* help depth (flat zero-shot bogs y1.62 →
ROM-trained y0.3), but the mass term's instability caps it. Takeaway: an acceleration-
feedback term is only as good as the ẍ distribution it trained on; to use it on a spiky
substrate, train on spiky ẍ (or drop it — the non-mass recipe is the mud champion). The
thread's lasting wins were the ẍ≡0 fix and the OSC parity infra, not the mass controller.
