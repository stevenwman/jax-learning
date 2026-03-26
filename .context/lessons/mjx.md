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
