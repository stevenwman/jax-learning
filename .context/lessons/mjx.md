# MJX Physics Lessons

---

## MJX Physics NaN at Scale — Not an Algo Bug

**Problem:** FastTD3 on HumanoidRun NaN'd at random step counts (315k, 600k, 1.2M) with identical configs.

**Root cause:** MJX physics solver produces NaN obs stochastically (contact solver failure, singular mass matrix). With 1024 envs, at least one crashes. NaN enters replay buffer → Q trains on NaN → cascade.

**How we found it:** Different step counts each run = stochastic input failure, not algo divergence. Debug sync points masked the bug by changing timing.

**Fix (three layers):**
1. NaN-safe env step: NaN in obs → zero obs, zero reward, done=True (auto-resets)
2. Action NaN guard: NaN in action → zero action before env.step
3. C51 log_prob clamp: `jnp.maximum(log_softmax(...), -30.0)`

**Also guard Inf:** MJX produces Inf from velocity overflow (separate from NaN). Add `isinf()` alongside `isnan()`. Confirmed: FastDSAC survived 54.3M steps (past 53M crash) after adding Inf guard.

**Lesson:** When NaN happens at random step counts, check inputs (env output) before gradients. Physics engines crash at scale — guard the boundary with both NaN and Inf checks.

---

## GPU OOM at Scale Is Usually Not a Leak

**Problem:** Training OOM'd at 67-85M steps. Assumed memory leak.

**Root cause:** No leak. RTX 5080 (16GB) starts at 95% capacity with 1024 envs. XLA command buffers (outside JAX's pool) slowly accumulate.

**Key insight:** `jax.devices()[0].memory_stats()` only reports JAX-managed memory. XLA graphs, command buffers, and CUDA driver overhead live outside this pool. `nvidia-smi` shows the real total. Always check both.

**Fixes:**
- `XLA_CLIENT_MEM_FRACTION=0.7` — leave room for XLA/driver overhead
- `XLA_FLAGS=--xla_gpu_enable_command_buffer=` — disable CUDA graph caching

**Lesson:** When debugging GPU OOM, start with `nvidia-smi` monitoring. If CUDA memory is flat from startup, there's no leak — the GPU is just too small.

---

## MJX Eval Recompilation — Root Cause Identified, Upstream Issue

**Problem:** `jit(while)` and `jit(scan)` inside MJX physics solver recompile ~2x per eval call. Over 100 evals → ~200 extra compilations → CUDA command buffers accumulate → OOM.

**Diagnostic trail (systematic isolation):**
1. Env-only: 2 while compiles (startup) — stable
2. Env + algo updates: 2 while compiles — algo doesn't cause recompilation
3. **Env + 5 evals: 12 while compiles** — each eval adds ~2
4. Identified `weak_type=True` on `.data.time` (MuJoCo Issue #2306) — fixing it didn't help
5. Recompilation is internal to MJX's step — upstream issue

**Mitigation:** `XLA_CLIENT_MEM_FRACTION=0.7` leaves headroom. ~5% overhead. Set in all train scripts via `env_setup.py`.

**Lesson:** When debugging recompilation, isolate components systematically. Don't assume the obvious suspect is the full answer — verify the fix works.

---

## MJX→CPU Transfer: Every Obs Dimension Must Match (2026-03-27)

**What happened:** Policy trained on MJX (eval 233+) failed immediately on CPU env — robot fell within a few steps. The CPU env was supposedly "the same" MJCF with the same overrides.

**Root cause:** `go2_cpu.py` line 128 returned `np.zeros(3)` for local_linvel instead of reading the actual `local_linvel` velocimeter sensor. Three of 48 obs dimensions were always zero during deployment but carried real velocity data during training. The policy relied on linvel feedback for balance and got silence instead.

**Fix:** One line — read `sensordata[linvel_adr:linvel_adr+3]` instead of `np.zeros(3)`.

**After fix:** CPU rollout walks smoothly — arguably nicer than MJX (deterministic physics, no stochastic solver noise).

**Lesson:** When building a deployment/CPU env to mirror a training env, verify EVERY obs dimension produces real values. Zeroing "optional" sensors is NOT safe — if the policy was trained on them, it depends on them. The safest approach: run both envs on the same initial state, print obs side-by-side, diff every dimension. A single zeroed dimension can destroy transfer.

**Broader pattern:** This is the same class of bug as the record_video.py obs normalization issue (policy trained with norm, recording without). Any preprocessing mismatch between training and deployment causes silent failure. Audit the FULL obs pipeline: sensors → extraction → normalization → policy.

---

## MuJoCo Friction Uses Max-Combine — Randomize Foot Geoms, Not Just Floor (2026-03-28)

**What happened:** Domain randomization randomized floor friction U(0.2, 2.0) but the policy still failed on different surfaces. With aggressive range U(0.05, 4.5), training collapsed to eval 0.

**Root cause:** MuJoCo combines friction between colliding geoms using **element-wise max** (not multiply like PhysX). If foot friction is 0.6 and floor is 0.05, effective friction = max(0.6, 0.05) = 0.6. Floor-only randomization has no effect when foot friction caps it.

**Fix:** Randomize ALL geom friction (feet + floor + body) uniformly. Range [0.3, 1.5] (moderate). [0.05, 4.5] from WTW was designed for PhysX multiply-combine — too extreme for MuJoCo max-combine.

**Lesson:** DR ranges from Isaac Gym/Isaac Lab/PhysX papers are NOT directly portable to MuJoCo. The friction combining rule changes effective ranges dramatically. Always check the simulator's contact model before copying DR configs.

---

## Sim2sim Between Different MJCFs Is Harder Than MJX→CPU (2026-03-28)

**What happened:** Policy trained on Menagerie Go2 MJCF works perfectly on our CPU env (go2_cpu.py, 10s+ walking). Same policy fails within 2s on unitree_mujoco's Go2 MJCF, despite matching all overridable parameters (damping, friction, force limits, timestep, contacts).

**Root cause (investigated exhaustively):** The two MJCFs describe the same robot but with:
- Different solver defaults (pyramidal/1-iter vs elliptic/100-iter)
- Different collision geometry types (capsule vs cylinder on calf bodies)
- Different geom counts (57 vs 65)

Zero-torque test showed 2-3x joint velocity divergence after a single physics step. These are irreducible MJCF authoring differences — same robot, different model files, different dynamics.

**What we tried (all failed on unitree model):**
- Matching solver iterations (1) — explodes (elliptic cone needs iterations)
- Matching cone type (pyramidal) — bouncing chaos at 1 iteration
- Matching foot geom sizes — no effect
- Matching contact params — no effect
- All of the above combined — still unstable

**What worked:** Domain randomization during training (friction, mass, damping, motor strength) + velocity kicks. Policy eval dropped from 233 → 200 but transfers to CPU env cleanly. Still fails on unitree_mujoco though — the MJCF gap is too large for parameter-level DR.

**Lesson:** Sim2sim between your own MJX and CPU envs is easy (same MJCF). Sim2sim between different MJCFs of the "same" robot is nearly as hard as sim2real. If you need to deploy on a different simulator's model (unitree_mujoco, Isaac), either train on THEIR model directly or accept the gap and use aggressive DR + real-world fine-tuning.

---

## MJX Has Unsupported Collision Primitives — Can't Load All MJCFs (2026-03-28)

**What happened:** Tried loading unitree_mujoco's Go2 MJCF into MJX to train directly on their model and eliminate the sim2sim gap. MJX threw `NotImplementedError: (mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_BOX) collisions not implemented.`

**Root cause:** Unitree's MJCF uses **cylinder** geoms (type=5) for calf collision bodies. Our Menagerie MJCF uses **capsules** (type=2) for the same parts. MJX only supports a subset of MuJoCo's collision primitives — cylinder-box is not one of them.

**This is a hard MJX limitation**, not a solver or iteration issue. No amount of parameter matching can fix it.

**Options:**
1. Replace cylinders with capsules in a modified unitree XML — makes it MJX-compatible but changes collision dynamics
2. Use MuJoCo Warp instead — likely supports full collision primitive set (closer to CPU MuJoCo)
3. Stay on Menagerie MJCF + DR — which already works for MJX→CPU transfer

**Lesson:** Before attempting to load a third-party MJCF into MJX, check which geom types are used. Run `mjx.put_model()` as a smoke test — it will immediately tell you if unsupported collision pairs exist. MJX's supported collisions as of 2026: sphere, capsule, ellipsoid, box, and plane (not all pairs). Cylinder is NOT supported.

---

## MuJoCo Warp CCD Overflow — naccdmax Sizing for Complex Geometry (2026-03-29)

**What happened:** Training Go2WarpJoystickFlat at 1024 envs produced 8.6 million lines of `CCD overflow - please increase naccdmax to N` warnings. Training ran at 65k sps instead of ~91k, and contacts were silently being dropped each step.

**Root cause:** Unitree's go2.xml has full collision geometry (cylinders + boxes on every leg) — many more geom pairs than Menagerie's sphere-only go2_mjx.xml. Warp's CCD (continuous collision detection) broadphase buffer was too small at the default size. The max overflow value hit 1251, meaning that many collision candidates were truncated per step.

**Fix:** Set `naccdmax=2000` in `default_config()` and pass it through `make_data()`. Overflow count dropped from 8.6M to ~1430 (occasional spikes). sps went from 65k → 91k at 1024 envs.

**The `ccd_iterations` warning** (`opt.ccd_iterations, currently set to 20, needs to be increased`) also appears but doesn't produce per-step spam. This is a Warp-specific advisory that the CCD solver wants more iterations for complex geometry. Monitor for contact artifacts — if feet clip through floor, increase it.

**Playground 0.2.0 API change:** The parameter was renamed from `nconmax` to `naconmax`, and `naccdmax` is new (didn't exist in the MJX-only API). When upgrading Playground, check `make_data()` signature for renamed/new parameters.

**Lesson:** When using Warp with complex collision geometry at high env counts, you MUST size `naccdmax` appropriately. Start with 2x the max overflow value you see. The performance impact of overflow is significant (~30% sps loss) because Warp retries/logs per overflow per env per substep. The warnings are not just noise — they indicate dropped contacts that affect physics fidelity.
