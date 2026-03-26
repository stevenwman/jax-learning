# Go2 PPO Debugging Trail

**Started:** 2026-03-23
**Status:** RESOLVED — 10x tracking reward + height termination + calf torque fix. Seed 2100: eval 233 @ 50M steps, base height 0.31m, actual locomotion confirmed.

---

## Symptom

PPO trains on Go2JoystickFlat without crashing, but the robot either stands still or shuffles feet without lifting them. Eval returns plateau at 10-50 regardless of hyperparameters. The policy never learns to track velocity commands.

## Diagnostic Evidence

Velocity tracking test (5M step checkpoint, reward ~15):
```
cmd=[+1.45, +0.64, -0.08]  vel=[-0.30, +0.20]  # Going BACKWARDS
```
Robot receives forward command, drifts backward. Actions are non-zero (norm ~1.2) — it's trying, but hasn't learned locomotion.

---

## Hypotheses and Results

### H1: Reward scaling too low (reward_scaling=1.0)
**Result: PARTIALLY CONFIRMED**

Per-step rewards are ~0.01-0.03 after `* dt`. VLoss = 0.00 — value function has no signal.

- `reward_scaling=1.0`: VLoss = 0.00, eval ~15 (standing)
- `reward_scaling=10.0`: VLoss = 0.05-0.20, eval ~15-17 (still standing, but value function learning)

Reward scaling alone didn't fix walking — it just gave the value function signal. The policy still doesn't walk.

**Note:** Playground's Brax PPO also uses `reward_scaling=1.0` for Go1 and it works. This suggests our PPO may handle small rewards differently, or the Go1 env produces larger per-step rewards. Needs investigation.

### H2: Reward weights too harsh — penalties dominate tracking reward
**Result: PARTIALLY CONFIRMED**

Original legged_gym weights (dof_pos_limits=-10, lin_vel_z=-2) penalized too aggressively. Robot learned to stand still to avoid penalties.

Switched to Playground Go1 Joystick weights (milder penalties). Standing reward dropped from ~15 to ~10, but walking still didn't emerge.

Then amplified tracking rewards (tracking_lin_vel: 1.0→4.0, tracking_ang_vel: 0.5→2.0):
- Eval jumped from ~15 to ~35-50
- Robot started **shuffling** (moving joints, feet sliding on ground)
- Never learned to **lift feet** despite feet_air_time reward

Also tried bumping feet_air_time: 0.1→0.5. No improvement — 50M steps, reward oscillated at 35-50.

### H3: Missing obs information — no joint velocity or full gyro
**Status: TESTING (active hypothesis)**

Critical discovery comparing obs spaces:
- **Go1 Joystick (Playground): 48 dims** — includes joint_vel (12), full gyro (3), linvel (3)
- **Our Go2: 31 dims** — missing joint_vel (12), had only yaw_rate (1 of 3 gyro dims)

**Missing joint_vel is the most likely root cause.** Without knowing how fast joints are moving, the policy can't:
- Sense joint momentum (critical for swing phase timing)
- Learn smooth gait transitions
- Implement effective damping behavior

Both joint_vel and full gyro are available on real hardware (encoders + IMU), so there's no sim-to-real reason to exclude them.

**Fix:** Added joint_vel (12) + full gyro (3→replacing 1 yaw_rate) = 45 dims. Training launched 2026-03-23.

### H4: PPO hyperparameters not matched to Playground
**Status: PARTIALLY ADDRESSED — Kp/action_scale now matched**

Playground Go1 config vs ours:
| Param | Playground Go1 | Our Go2 |
|---|---|---|
| num_envs | 8192 | 1024 (GPU limited) |
| total_timesteps | 200M | 50M |
| network (policy) | (512, 256, 128) | now matched |
| network (value) | (512, 256, 128) | now matched |
| Kp | 35 | now matched (was 20) |
| action_scale | 0.5 | now matched (was 0.25) |
| reward_scaling | 1.0 | 1.0 (matched) |
| value obs | privileged_state (123d) | same as policy — **NOT matched** |

Still unmatched: privileged value obs, env count (1024 vs 8192, GPU constraint).

### H5: First-principles gradient analysis — reward_scaling was a red herring
**Status: CONFIRMED**

Deep dive into Brax PPO source (`brax/training/agents/ppo/losses.py`) revealed:

**Advantage normalization makes policy gradient scale-invariant:**
- Both Brax (line 174) and our PPO normalize advantages: `adv = (adv - mean) / (std + eps)`
- This means `reward_scaling` does NOT affect policy gradient magnitude
- Policy gradient ∝ normalized_advantage × ∇log_prob — independent of reward scale
- `reward_scaling` only affects value targets (and thus VLoss magnitude)
- Our VLoss=0.00 was a display rounding issue, not a learning failure
- **Conclusion:** chasing reward_scaling was wrong. The 4x tracking reward improvement (seed 99, eval 36→50) came from changing the reward LANDSCAPE (making walking more rewarding than standing), not from scaling.

**The real Brax differences that affect learning:**

| # | Disparity | Effect on gradient | Severity |
|---|---|---|---|
| 1 | **Missing linvel (3 dims)** | Policy can't sense body velocity directly. Must infer from joint position changes across timesteps — impossible without frame stacking. | **HIGH** |
| 2 | **Privileged critic obs (123d vs 45d)** | Brax critic sees ground truth: unnoised sensors, foot contacts (4), foot velocities (12), actuator forces (12), external perturbation forces (3+1). Better value estimates → lower variance advantages → cleaner policy gradients. | **HIGH** |
| 3 | **Minibatch size: 640 vs 5120** | 1024 envs × 20 steps / 32 minibatches = 640. PG uses 8192 × 20 / 32 = 5120. 8x noisier gradient estimates. However, this can be compensated by training longer (more updates) — the data-to-update ratio matters more than batch size alone. | **MEDIUM** — compensatable with more steps |
| 4 | **vf_coefficient=0.5 in Brax** | Brax multiplies VLoss by 0.5 (losses.py line 194). We don't. Our value gradients are 2x larger relative to policy gradients. Minor effect. | **LOW** |
| 5 | **Obs ordering** | Different but irrelevant for MLP. | **NONE** |

**Key insight on env count (1024 vs 8192):** More envs increases minibatch size (less gradient noise) BUT the same effect is achievable by training longer. The data-to-update ratio (how many env transitions per gradient step) is what matters, not the number of parallel envs. With 1024 envs we just need proportionally more wall-clock time to get equivalent training signal.

---

## Run Log

| Seed | Envs | Steps | reward_scaling | Obs | Net | Eval | Notes |
|------|------|-------|----------------|-----|-----|------|-------|
| 42 | 1024 | 5M | 10.0 | 31d | (32,32,32,32) | 15.0 | Standing, VLoss>0 |
| 123 | 1024 | 20M | 10.0 | 31d | (128,128,128,128) | 12.4 | Oscillating 10-17 |
| 99 | 512 | 20M | 10.0 | 31d | (512,256,128) | 36.8 | Shuffling, no foot lift. 4x tracking reward |
| 111 | 512 | 50M | 10.0 | 31d | (512,256,128) | 35.1 | Same with feet_air_time=0.5, no improvement |
| 88 | 512 | 20M | 10.0 | 31d | (512,256,128) | ~15 | PG weights + rs=10, stagnated |
| 200 | 1024 | 50M | 1.0 | 31d | (512,256,128) | ~8 | PG weights + rs=1, regressing |
| 300 | 1024 | 50M | 1.0 | **45d** | (512,256,128) | 14.1 | + joint_vel + full gyro. No improvement — obs alone can't fix it with rs=1.0 |
| 400 | 512 | 50M | 1.0 | **48d+116d** | (512,256,128) asym | 9.0 → 17.9 | Full PG parity. Brax PPO on Go2 got eval 17.9 at 50M. |
| brax-go1 | 512 | 50M | 1.0 | 48d+116d | (512,256,128) asym | 21.7 | Brax PPO on Go1. Matches published curve. |
| 600 | 512 | 50M | 1.0 | 48d+116d | (512,256,128) asym | ~11 (killed at 27M) | vloss 0.25x + full-batch adv norm. Gap narrowed. |
| 700 | 512 | 50M | 1.0 | 48d+116d | (512,256,128) asym | RUNNING | Go1 A/B test with fixed PPO. 110k sps. |
| 1400 | 1024 | 50M | 1.0 | 48d+116d | (512,256,128) asym | ~12-14 | Calf torque fix (45.43 Nm). Entropy collapsed to -7.6 by 9M. |
| 1500 | 1024 | 50M | 1.0 | 48d+116d | (512,256,128) asym | killed at 7M | entropy_coef=0.05. Too much exploration, eval stuck at 2-4. |
| 1600 | 1024 | 50M | 1.0 | 48d+116d | (512,256,128) asym | ~4-5 | entropy_coef=0.02. Slightly better than 0.05 but worse than 0.01. |
| 1700 | 1024 | 50M | 1.0 | 48d+116d | (512,256,128) asym | ~11-12 | action_scale=0.5 + torque fix. Robot falling, same plateau. |
| 1900 | 1024 | 50M | 1.0 | 48d+116d | (512,256,128) asym | ~12 | Brax obs norm matching test. Same plateau — obs norm not the cause. |
| 2000 | 1024 | 50M | 1.0 | 48d+116d | (512,256,128) asym | ~4-7 | Height termination (base_z<0.18) alone. Policy crouches at exactly 0.18m. |
| **2100** | **1024** | **50M** | **1.0** | **48d+116d** | **(512,256,128) asym** | **233** | **10x tracking (lin=10.0, ang=5.0) + height term + torque fix. WINNER. Base height 0.31m, locomotion confirmed.** |

### H6: Our PPO implementation differs from Brax PPO in a way that matters
**Status: PARTIALLY FIXED — vloss scaling (0.25x) and full-batch advantage normalization matched to Brax. Gap narrowed from 2x to ~1.3x on Go2 but still investigating on Go1.**

Brax PPO on Go2: eval 17.9 at 50M steps (10 min wall-clock)
Our PPO on Go2: eval 9.0 at 50M steps (47 min wall-clock)

Two distinct problems:

**1. Wall-clock: 5x slower (confirmed root cause: Python collect loop)**

Profiling revealed our Python-level rollout loop kills GPU pipeline throughput:

```
Pure env.step (20 steps, 512 envs):   0.111s → 92,000 sps
Python collect loop (20 steps):       9.729s → 1,053 sps
Overhead ratio: 87x
```

Each Python-level operation (select_action, buffer.add, norm_update, critic_obs_buf.at[].set) forces JAX to synchronize the GPU pipeline. Brax avoids this by putting the entire collect→update cycle in a single `jax.lax.scan` — zero Python-GPU sync during training.

The 32k sps we see in training (vs 1k in profiling) comes from JAX's async dispatch partially hiding the overhead when the update step is included, but it's still 2.5x slower than Brax's ~82k sps.

**Fix:** JIT the entire collect phase into a `lax.scan`. This requires:
- Moving `select_action` inside the scan body
- Pre-allocating buffers as JAX arrays (not Python-level RolloutBuffer)
- Running obs normalization inside the scan
- Carrying (env_state, ts, norm_state, rng) as scan carry

This is a significant refactor of `train_ppo.py` but would give ~2.5-5x speedup.

**2. Sample efficiency: 2x worse at same step count**

Even accounting for wall-clock, Brax gets eval 17.9 vs our 9.0 at the same 50M steps. This suggests a real learning difference — possibly:
- Brax's obs normalization (Welford running stats, applied inside JIT) vs ours
- Brax's `vf_coefficient=0.5` vs our 1.0
- Brax's learning rate schedule vs our fixed/annealing
- Subtle GAE/value target differences

Needs further investigation.

### H7: Menagerie Go2 has marshmallow-soft foot contacts
**Status: FIXED in go2_base.py. Not yet validated in training.**

Comparing Go2 (Menagerie) vs Go1 (Playground) foot collision properties:

| Property | Go2 (Menagerie) | Go1 (Playground) |
|---|---|---|
| solimp | 0.015 1 0.031 (marshmallow) | 0.9 0.95 0.023 (firm) |
| condim | 6 (full friction cone) | 3 (basic) |
| foot size | 0.0175 | 0.023 (31% larger) |

Go2's extremely soft contacts prevent crisp ground reaction forces needed for push-off during walking. Fix: override in `go2_base.py` to match Go1's firm contacts.

### H8: Training budget insufficient
**Status: CONFIRMED**

Go1 Playground paper plot shows Brax PPO reaching ~25 reward at 100M steps. Our Brax PPO on Go1 reached 21.7 at 50M — exactly on the published curve. Both Go1 and Go2 need 100-200M steps to converge.

---

## Key Lessons (so far)

1. **Always compare obs spaces against reference implementations.** We lost hours tuning rewards when the real problem was missing proprioceptive information (joint_vel, full gyro). The env design doc excluded joint_vel based on a wrong assumption about sim-to-real requirements.

2. **VLoss = 0.00 means the value function has no signal.** This is a red flag. Either reward_scaling is too low or rewards are too small. Playground's Brax PPO may handle this differently.

3. **Python-level rollout loops kill JAX throughput.** Each Python operation between JIT'd calls forces GPU pipeline sync. A 20-step Python loop with per-step select_action + buffer.add gives 87x overhead vs pure env.step. The fix is `lax.scan` for the entire collect phase.

4. **Always check collision physics against reference implementations.** Menagerie's Go2 uses solimp=0.015 (marshmallow), Playground's Go1 uses solimp=0.9 (firm). Same reward math, completely different physics behavior.

5. **Verify training budget against published results before debugging.** We spent hours investigating reward weights when the Playground paper shows Go1 Joystick needs 100M+ steps to reach ~25 reward. Our 50M step runs were simply undertrained.

3. **Reward amplification (tracking 1.0→4.0) broke the standing-still local optimum** but created a new one (shuffling). Reward shaping alone can't compensate for missing observation information.

---

## Active Hypotheses (2026-03-25)

### H9: Calf torque too low (Menagerie MJCF bug)
**Status: FIXED, eval improved from ~15 → ~14 at 20M but still no walking**

Menagerie `go2_mjx.xml` sets all actuators to `forcerange=[-24, 24]`. Real Go2 calf is 45.43 Nm (from Unitree URDF). The calf was running at 53% of real torque. Fixed in `go2_base.py`.

The fix improved force capability but the policy still converges to shuffling — likely because entropy collapses before the policy explores foot-lifting.

### H10: Entropy collapse prevents exploration of foot-lifting
**Status: PARTIALLY CONFIRMED**

- `entropy_coef=0.01`: entropy collapses to -7.6 by 9M steps, logσ=-1.4. Policy locks into shuffling.
- `entropy_coef=0.02`: entropy decays slower but eval stuck at 4-5 (worse — policy can't commit).
- `entropy_coef=0.05`: too much exploration, eval stuck at 2-4.

Sweet spot not found. The problem may not be entropy alone but interaction with other factors.

### H11: action_scale=0.3 too restrictive for walking stride
**Status: TESTING (seed 1700)**

With action_scale=0.3, the policy can only command ±0.3 rad from default. Calf effective range is 0.6 rad (34% of joint range). Go1 uses 0.5 → 1.0 rad (52% of range). A walking gait needs ~0.8 rad of calf swing — barely achievable at 0.3, comfortable at 0.5.

Previously avoided 0.5 because "Go2 overshoots" — but that was before the torque fix. Testing now.

### H12: Default pose mismatch
**Status: RULED OUT**

Go2 hip_abduction = 0.0, Go1 = ±0.1. Minor (5.7°), wouldn't explain training gap. Thigh and calf defaults identical.

### H13: Feet/joint ordering mismatch corrupts rewards
**Status: RULED OUT**

Go2 orders FL/FR/RL/RR, Go1 orders FR/FL/RR/RL. But all foot reward functions use `jp.sum()` over all feet — order-independent. Privileged state ordering doesn't matter because the critic learns from data.

### H14: init_noise_std too low
**Status: NOT TESTED**

Flax defaults initialize with small weights → near-deterministic initial policy. MJX reference repo uses init_noise_std=1.0 for PPO. Higher initial noise = more diverse early trajectories. Test after H11.

### H15: Command sampling bias
**Status: NOT TESTED**

If Go2's command distribution produces vx≈0 too often, standing is optimal. Need to log command stats during training to verify. Low priority — same command config as Go1.

### H16: No height-based termination allows crouching local optimum
**Status: CONFIRMED — necessary but not sufficient alone**

Trajectory analysis confirmed the robot crouches at 0.17m (vs Brax standing at 0.31m). Our termination only fires on full flip (upvector_z < 0). The robot can sit on the ground indefinitely earning pose reward. alexeiplatzer terminates at base_z < 0.18m — directly prevents crouching.

Source: [alexeiplatzer/unitree-go2-mjx-rl](https://github.com/alexeiplatzer/unitree-go2-mjx-rl)

**Fix applied:** Added `base_z < 0.18` termination to `_get_termination()`.

**Result (seed 2000):** Eval dropped to 4-7. Height termination alone made things WORSE — episodes end immediately and the policy still crouches, just hovering just above 0.18m. Required in combination with 10x tracking reward to work correctly.

### H17: Body collision geoms still enabled (Go1 uses feetonly)
**Status: NOT YET APPLIED**

Go1 Playground uses a `feetonly` XML variant where ALL body collision geoms (hip, thigh, calf, base) are disabled — only 4 foot spheres can contact the floor. Our Go2 includes the full Menagerie MJCF with ALL collision geoms active. Extra body-floor contacts during training create confusing reward signals and may encourage conservative crouching to avoid body collisions.

Source: [MuJoCo Playground Go1 feetonly XML](https://github.com/google-deepmind/mujoco_playground), alexeiplatzer also disables non-foot contacts.

**Fix:** Create `go2_mjx_feetonly.xml` or override `contype=0 conaffinity=0` on non-foot geoms in `go2_base.py`.

### H18: Missing motor target clipping to joint limits
**Status: NOT YET APPLIED (low priority — solved via reward rebalancing)**

Our env sends `default_pose + action * action_scale` directly to MuJoCo without clipping to joint limits. alexeiplatzer clips: `motor_targets = jp.clip(motor_targets, lower, upper)`. Without clipping, the PD controller can target positions outside joint limits, causing saturated actuator forces and discontinuous dynamics.

Source: [alexeiplatzer/unitree-go2-mjx-rl base.py](https://github.com/alexeiplatzer/unitree-go2-mjx-rl)

### H19: Reward balance — tracking too weak relative to pose reward
**Status: CONFIRMED — THE ACTUAL FIX**

Reward breakdown at 12-14 eval (all previous seeds):
- Pose reward: ~450 per episode
- Tracking (lin_vel): ~130 per episode
- Feet clearance penalty: ~-89

Crouching was mathematically optimal. Walking would gain ~130 more tracking but cost energy + risk instability. At tracking_lin_vel=1.0 (Go1 weight), the robot maximized reward by crouching.

**Fix:** tracking_lin_vel=10.0, tracking_ang_vel=5.0. Now: pose ~450, tracking ~1300+. Walking dominates.

**Why Go1 works at 1.0:** Go1's dynamics make walking emerge more easily (lighter robot, different default stance). The same weight that balances reward correctly for Go1 underpowers tracking for Go2.

**Result (seed 2100):** eval 233 at 50M steps. Base height 0.31m. Robot locomotes and tracks velocity commands. Trajectory data confirmed.

---

## Hypotheses That Were WRONG or Incomplete

| Hypothesis | Result | Why Wrong |
|---|---|---|
| H1: reward_scaling | Red herring | Advantage normalization makes PG scale-invariant |
| H6: PPO implementation | Partially fixed, not the blocker | Both our PPO and Brax hit same plateau on same env |
| H9: Calf torque | Real bug, not sufficient | Eval went 15→12-14 but no walking |
| H10: Entropy collapse | Symptom not cause | Higher entropy_coef made things WORSE |
| H11: action_scale=0.5 | No effect alone | Same plateau with both 0.3 and 0.5 |
| H14: init_noise_std | Not tested, irrelevant | Network init not the bottleneck |
| Obs normalization | No effect | Seed 1900 (Brax norm) = same plateau |
| MJX reference config | Made things WORSE | alexeiplatzer's weights: eval ~8 |
| Height termination alone | Made things WORSE | Eval 4-7, robot crouches at 0.18m |

4. **PD gains (Kp) and action_scale affect reward magnitude.** Kp=35/action_scale=0.5 (Playground) produces larger movements and thus larger tracking rewards per step than Kp=20/action_scale=0.25 (Unitree official). This may explain why Playground can use reward_scaling=1.0.
