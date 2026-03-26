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
