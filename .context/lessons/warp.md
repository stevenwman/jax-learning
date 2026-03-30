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
