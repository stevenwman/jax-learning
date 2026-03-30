# MuJoCo Warp Lessons

---

## Warp CCD Overflow — naccdmax Sizing for Complex Geometry (2026-03-29)

**What happened:** Training Go2WarpJoystickFlat at 1024 envs produced 8.6 million lines of `CCD overflow - please increase naccdmax to N` warnings. Training ran at 65k sps instead of ~91k, and contacts were silently being dropped each step.

**Root cause:** Unitree's go2.xml has full collision geometry (cylinders + boxes on every leg) — many more geom pairs than Menagerie's sphere-only go2_mjx.xml. Warp's CCD (continuous collision detection) broadphase buffer was too small at the default size. The max overflow value hit 1251, meaning that many collision candidates were truncated per step.

**Fix:** Set `naccdmax=4000` in `default_config()` and pass it through `make_data()`. Also bumped `ccd_iterations=100` and `njmax=100` to suppress related warnings.

**Playground 0.2.0 API change:** The parameter was renamed from `nconmax` to `naconmax`, and `naccdmax` is new (didn't exist in the MJX-only API). When upgrading Playground, check `make_data()` signature for renamed/new parameters.

**Lesson:** When using Warp with complex collision geometry at high env counts, you MUST size `naccdmax` appropriately. Start with 2x the max overflow value you see. The performance impact of overflow is significant (~30% sps loss) because Warp retries/logs per overflow per env per substep. The warnings are not just noise — they indicate dropped contacts that affect physics fidelity.

---

## Warp OOMs in Python Loops — Must JIT Physics Steps (2026-03-29)

**What happened:** A simple 500-step PD test loop (`for i in range(500): data = mjx.step(model, data)`) OOM'd at step ~100 on an RTX 5080 (16GB). The same 500 steps ran instantly when wrapped in `jax.lax.scan`.

**Root cause:** Warp allocates collision buffers (collision_pair, CCD arrays, EPA arrays) inside each `mjx.step()` call. In a Python loop, each call creates new allocations that accumulate on the GPU — Warp's allocator doesn't free them between calls. JIT compilation via `lax.scan` compiles the entire loop into one fused XLA computation where Warp allocates once and reuses.

**This doesn't affect MJX (JAX backend)** because JAX manages memory natively through XLA's buffer pool. It's a Warp-specific issue due to the JAX↔Warp FFI boundary.

**Lesson:** Never step Warp physics in a Python loop — always JIT via `lax.scan` or `jax.jit`. This includes debug/test scripts, recording scripts, and any single-env rollout. If you need to inspect intermediate states, save them inside the scan carry and extract after.

---

## Warp forcerange=[0,0] Means Unlimited — Unitree XML Missing Force Limits (2026-03-29)

**What happened:** First Warp Go2 training run got eval 2.2 (vs MJX's 244). Video showed joints contorting wildly — robot collapsed in <25 steps every episode.

**Root cause:** Unitree's go2.xml sets `ctrlrange` on motor actuators but NOT `forcerange`. MuJoCo defaults `forcerange=[0,0]` which means **unlimited force**. Our external PD controller (`tau = Kp*(target-q) + Kd*(0-dq)`) could produce arbitrarily large torques with no clamping. The MJX env explicitly sets `forcerange` to match motor limits (±23.7 hip, ±45.43 knee) in `go2_base.py`, but we skipped this in `go2_warp_base.py` assuming unitree's XML was correct.

**Fix:** Add to `go2_warp_base.py`:
```python
for i in range(self._mj_model.nu):
    self._mj_model.actuator_forcerange[i] = self._mj_model.actuator_ctrlrange[i]
```

**After fix:** Eval jumped from 2.2 → 14.1. Training returns reached 50+ mid-run (vs stuck at 2 before). Robot stays up for hundreds of steps instead of 24.

**Lesson:** When using third-party MJCFs with motor actuators, ALWAYS check `forcerange` — `ctrlrange` and `forcerange` are independent. `ctrlrange` clips the control input, `forcerange` clips the output force. If `forcerange=[0,0]`, forces are unclamped regardless of `ctrlrange`. This is especially dangerous with external PD controllers where the torque can spike on large position errors. Cross-reference `actuator_ctrlrange` and `actuator_forcerange` for every actuator when debugging joint contortion.

---

## Warp Inherits XML Solver Settings — Check iterations, cone, eulerdamp (2026-03-29)

**What happened:** Even after the forcerange fix, Warp Go2 eval (14.1) was still far below MJX (244). Policy showed KL→0, Clip→0 at end of training = stopped learning.

**Root cause (identified, not yet fixed):** Unitree's go2.xml has very different solver settings from Menagerie's go2_mjx.xml, and we didn't override them:

| Setting | MJX env | Warp env | Impact |
|---|---|---|---|
| solver iterations | 1 | 100 | 100x stiffer contacts |
| ls_iterations | 5 | 50 | 10x line search |
| friction cone | pyramidal | elliptic | Different friction model |
| eulerdamp | disabled | enabled | Implicit velocity damping |

These create fundamentally different contact dynamics. The reward weights (tracking_lin_vel=10, orientation=-5, etc.) were tuned on MJX's soft 1-iteration pyramidal solver. The 100-iteration elliptic solver is much stiffer — the robot's foot contacts behave differently, making the same reward landscape much harder to optimize.

**Lesson:** When porting an env to a new MJCF, don't just check actuators and geometry — audit `<option>` solver settings. `iterations`, `ls_iterations`, `cone`, and `eulerdamp` can make the same robot feel like a completely different physical system. These are NOT overridden by `go2_base.py`-style runtime patches because our MJX env inherits them from `go2_mjx.xml` which was already tuned for MJX.

---

## Joint Order ≠ Actuator Order — THE Root Cause of Warp Go2 Failure (2026-03-29)

**What happened:** Warp Go2 env couldn't stand, couldn't recover from kicks, joints slammed to limits instantly. Spent hours debugging PD gains (Kp=20 vs 35), solver settings (iterations 1 vs 100), entropy tuning — none of it mattered. Best eval was 211 with the robot squatting motionless.

**Root cause:** In unitree's go2.xml, `qpos[7:]` is in body-tree order (**FL**, FR, RL, RR) but `ctrl` is in actuator order (**FR**, FL, RR, RL). The PD controller read position from `data.qpos[7:]` and wrote torque to `data.ctrl` — sending FL_hip's torque to FR_hip's actuator and vice versa. Every correction the PD made was applied to the wrong leg. The robot was fighting itself.

**Why it didn't affect MJX:** Menagerie's go2_mjx.xml has actuators in the SAME order as body tree (both FL-first). Joint order = actuator order = no mismatch.

**Why it was so hard to find:**
- The robot DID partially learn (eval 211) because PPO is robust enough to find a static strategy even with cross-wired legs
- PD-hold tests showed "working" behavior for the first 10 steps because the initial error is near zero and the cross-wired torques are also near zero
- The collapse happened gradually (exponential divergence from small perturbation), looking like a physics/solver issue rather than a wiring bug
- We went down rabbit holes: solver iterations (1 vs 100), PD gains (Kp 20 vs 35), entropy coefficient, reward weights — all red herrings

**Fix:**
```python
# Build mapping: act_to_joint[actuator_idx] = joint_idx
for i in range(nu):
    act_to_joint[i] = model.actuator_trnid[i, 0] - 1

# In PD substep: remap before writing to ctrl
tau_joint = kp * (target - qpos[7:]) + kd * (0 - qvel[6:])
tau_act = tau_joint[act_to_joint]  # ctrl[a] = tau_joint[act_to_joint[a]]
data = data.replace(ctrl=tau_act)
```

**After fix:** Robot stands at 0.212m, survives velocity kicks, recovers from being airborne. PD-hold test runs 500 steps (10 seconds) with alternating horizontal and upward kicks — stable throughout.

**Lesson:** When using a third-party MJCF, NEVER assume joint order equals actuator order. MuJoCo's body tree defines qpos/qvel order (by XML body hierarchy), but actuator order is defined by the `<actuator>` block (arbitrary). ALWAYS verify by printing both orderings. If they differ, every operation that reads qpos and writes ctrl needs an explicit remapping. This is the single most impactful bug of the entire Warp migration — everything else was noise.

---

## PD Gains Must Match Solver Stiffness — Kp/Kd From One Solver Don't Transfer (2026-03-29)

**What happened:** Warp Go2 with Kp=35, Kd=0.1 (copied from MJX env, originally from Playground Go1) couldn't stand. PD hold test showed calves losing against gravity — pos_error grew monotonically from 0 → 0.36 rad over 28 steps, then oscillation spiked to 10+ rad/s at step 51 and robot collapsed.

**Root cause:** Two interacting issues:
1. **Kd=0.1 is too low** for 100-iteration solver. At 10 rad/s velocity, damping = 0.1 × 10 = 1 Nm — negligible. Oscillation built unchecked.
2. **Kp=35 wasn't enough** for stiff contacts. With 100 solver iterations (vs MJX's 1), contacts are rigid and full robot weight bears directly on joints. Kp=35 × 0.36 rad error = 12.6 Nm — insufficient to fight gravity through the knee.

**Fix:** Use unitree_rl_gym's PD gains: **Kp=20, Kd=0.5**. These were designed for this exact solver configuration (100 iterations, elliptic cone). Lower Kp (softer position tracking) but 5x higher Kd (kills oscillation). Robot holds at 0.203m for 10+ seconds — stable.

**Diagnostic method:** `tools/pd_hold_test.py` — JIT'd zero-action PD hold with per-step pos_error, velocity, and torque printout. Key signals:
- Monotonically growing pos_error = PD losing against load
- Velocity spikes = oscillation from underdamping
- Torque near limits = Kp too low for the error magnitude

**Lesson:** PD gains are tightly coupled to solver settings. Kp/Kd tuned on a 1-iteration pyramidal solver (MJX) cannot be reused on a 100-iteration elliptic solver (Warp/unitree). Always use PD gains from the same physics configuration. For Go2: Kp=35/Kd=0.1 for MJX, Kp=20/Kd=0.5 for Warp/unitree.
