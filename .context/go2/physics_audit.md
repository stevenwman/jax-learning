# Go2 Physics Audit — Why Won't It Walk?

**Date:** 2026-03-25
**Status:** IN PROGRESS — damping fix applied, training running

## The Problem
Go2 plateaus at eval ~15 while Go1 reaches 27 with identical reward weights.
The robot wiggles/shuffles but never lifts feet or tracks forward velocity.

## Hypotheses Tested

### H1: Double Damping (CONFIRMED BUG, FIXED)
**Finding:** Go2 had effective Kd=1.0 instead of intended 0.5.

Menagerie Go2 uses `<general biastype="affine" biasprm="0 -50 -0.5">` actuators.
The `-0.5` in biasprm[2] is a Kd term. Our code also set `dof_damping[6:] = 0.5`.
Both produce velocity-proportional damping, so they stack: effective Kd = 1.0.

Go1's Menagerie uses `<position>` actuators which have biasprm[2]=0.
So Go1 only gets Kd from dof_damping = 0.5.

**Fix attempt 1 (seed 1200):** Set `dof_damping[6:] = 0`, keep actuator Kd only.
Initial eval dropped to 0.1 — robot collapsed without passive damping. Was killed
prematurely. ACTUALLY: reviewing full log shows it recovered — 6.3 at 4M, 10.7 at 8.7M
and still climbing when killed. The robot learned to stabilize, just needed more time.
**Lesson: don't kill runs after 2M steps. Wait for at least 10M.**

**Fix attempt 2 (seed 1300):** Keep `dof_damping[6:] = 0.5`, zero `actuator_biasprm[:, 2] = 0`.
Effective Kd = 0.5 from dof_damping only. Running.

**Caveat (Steven's pushback):** Position control should overcome damping — it just affects
speed of reaching targets. Double damping alone may not explain feet not lifting.
Need training run to validate.

### H2: Force Limits / Torque-to-Weight (MENAGERIE BUG — ROOT CAUSE)
Menagerie's `go2_mjx.xml` has a bug: parent class sets `forcerange="-24 24"` for ALL
joints, and the `knee` subclass doesn't override it.

**Real hardware specs (from Unitree URDF):**
- Go2 calf: **45.43 Nm** (not 24!)
- Go1 calf: **35.55 Nm**
- Go2 calf is actually STRONGER than Go1's

**In the sim:** Go2 calf was limited to 24 Nm — 53% of real torque.
At 15.2 kg body weight, this meant the calf couldn't generate enough force
to lift feet during a walking stride. The policy correctly learned shuffling
was optimal because lifting feet was physically impossible.

**Fix:** Override `actuator_forcerange` for calf actuators to [-45.43, 45.43]
in `go2_base.py`. Seed 1400 running to validate.

### H3: Effective Action Range (CONFIRMED DIFFERENCE)
With action_scale=0.3 (MJX reference config):
- Go2 thigh: 15% of joint range, calf: 34%
- Go1 thigh (scale=0.5): 19%, calf: 52%

Go1 has 67% more usable calf range. Reverted Go2 to action_scale=0.5.

### H4: Physical Foot Lift Capability (CONFIRMED — BOTH CAN WALK)
Manual gait command (no policy) test:
- Go2: max foot clearance 0.20m
- Go1: max foot clearance 0.32m (1.6x higher)
- **Go2 CAN physically lift feet.** The physics are not the blocker.

### H5: CCD Iterations (FIXED, NO IMPACT)
Go2 used default ccd_iterations=4, Go1 uses 20.
Fixed in go2_base.py. Seed 1000 showed no improvement — not the cause.

### H6: Privileged State Incomplete (FIXED, NO IMPACT)
Go2 critic got 116d, Go1 gets ~130d (missing accelerometer, xfrc_applied).
Fixed. Seed 1000 showed no improvement — not the cause.

## Current Status
- Seed 1200 running: damping fix + Go1 PG weights + action_scale=0.5
- If damping fix helps → the combined effect of Kd + force limits + action range was the blocker
- If damping fix doesn't help → need obs history or fundamentally different reward shaping for Go2

## Key Numbers for Reference
| Metric | Go2 | Go1 |
|---|---|---|
| Mass | 15.2 kg | 12.7 kg |
| Calf max torque | 24 Nm | 35.5 Nm |
| Calf torque/weight | 1.58 | 2.79 |
| Effective Kd (before fix) | 1.0 | 0.5 |
| Effective Kd (after fix) | 0.5 | 0.5 |
| Manual foot clearance | 0.20m | 0.32m |
| action_scale | 0.5 | 0.5 |
| Best eval (100M steps) | ~15 | 27.4 |
