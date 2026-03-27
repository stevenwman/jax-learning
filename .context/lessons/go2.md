# Go2 Locomotion Lessons

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

**Lesson:** Domain randomization bridges parameter uncertainty (friction values, mass, COM position). It does NOT bridge structural differences between MuJoCo models (different meshes, different body trees, different inertias). If two models don't produce the same dynamics even with identical contact/actuator/solver params, the MJCF itself is different and you need to train on the target model directly.
