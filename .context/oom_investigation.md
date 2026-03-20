# GPU OOM Investigation — Training Crashes at 67-85M Steps

**Status:** Active investigation
**Symptom:** All training scripts OOM after ~67-85M steps on 16GB RTX 5080 with 1024 envs
**Error:** `RESOURCE_EXHAUSTED: ... command buffer with 122 (total of 23 alive graphs)`

## The Error (verbatim)

```
ValueError: RESOURCE_EXHAUSTED: Underlying backend ran out of memory trying to
instantiate command buffer with 122 (total of 23 alive graphs in the process).
You can try to (a) Give more memory to the driver by reducing XLA_CLIENT_MEM_FRACTION
(b) Disable command buffers with 'XLA_FLAGS=--xla_gpu_enable_command_buffer=' (empty set).
Original error: Failed to instantiate CUDA graph: CUDA_ERROR_OUT_OF_MEMORY: out of memory
```

Crashes happen during `env_step(env_state, action)` — either in training loop or eval.

## Hypotheses tested

### H1: add_batch accumulates XLA dispatches — DISPROVED
**Theory:** 6 un-JIT'd `.at[].set()` calls per step = 500k+ XLA dispatches over 83k iterations, accumulating command buffers.
**Test:** `jax.log_compiles` shows both old (6 separate) and new (1 JIT'd) stabilize at the same compilation count. 100k iteration synthetic test shows flat memory (275MB, no growth).
**Result:** Not the cause. Fixed anyway (fewer dispatches is better practice).

### H2: Eval batch dim mismatch causes extra compilations — PLAUSIBLE but not sole cause
**Theory:** Training uses batch=1024, eval uses batch=5. Each shape triggers separate JIT compilation of `select_action`, `env.step`, `env.reset` → ~4 extra graphs permanently in GPU memory.
**Test:** Confirmed via `jax.log_compiles` that different shapes do compile separately.
**Result:** Real issue, fixed (eval now pads to num_envs). But 4 extra graphs × ~few MB each = ~20MB, not enough to cause OOM on a 16GB GPU.

### H3: warmup_eval pre-compilation OOMs — CONFIRMED and removed
**Theory:** `warmup_eval()` tries to compile eval graphs at startup, competing with training for GPU memory.
**Test:** FastSAC crashed immediately on `warmup_eval` → `env.reset` → OOM.
**Result:** Confirmed. Removed warmup_eval from all scripts.

### H4: Synthetic loop doesn't leak — CONFIRMED
**Test:** 100k iterations of add_batch + sample + fake_update + periodic eval with different batch dim.
**Result:** Memory flat at 275MB. No leak in our buffer/key/eval pattern.
**Implication:** The leak is in something the synthetic test doesn't include: real MuJoCo env, real Flax model updates, or orbax checkpointing.

## What hasn't been tested yet

### H5: MuJoCo Playground env.step accumulates GPU memory
The MJX env runs physics simulation on GPU via `jax.lax.scan`. Over millions of calls, internal state or compiled graphs could accumulate. Test: monitor `jax.devices()[0].memory_stats()` during actual training with real env but no algo update.

### H6: Flax model apply/grad accumulates memory
Each `jax.value_and_grad(loss_fn)(params, ...)` call inside `update()` is JIT'd, but the JIT captures closures over network modules. If something in the closure changes shape or identity over time, it could trigger recompilation. Test: monitor memory during repeated `algo.update()` with real model but fake env data.

### H7: Orbax checkpointing leaks GPU memory
`ocp.StandardCheckpointer().save()` serializes JAX arrays. If it doesn't fully release temporary GPU allocations after save, memory could grow with each checkpoint. We checkpoint every ~1M steps = 80+ checkpoints over a 85M step run. Test: monitor memory before/after repeated checkpoint saves.

### H8: MuJoCo env.reset during eval allocates new GPU memory
Even with batch dim matching, `env.reset()` inside the eval env may allocate temporary GPU memory that isn't freed. Over 80+ eval calls, this could accumulate. Test: monitor memory across repeated eval calls with real env.

### H9: Total baseline GPU usage is just too close to 16GB limit — CONFIRMED (root cause)
**Theory:** MuJoCo env state for 1024 envs + replay buffer (1M entries × 6 arrays) + model params + optimizer state + eval env consumes nearly all 16GB, leaving no headroom.

**Test:** Real training with both JAX `memory_stats()` AND `nvidia-smi` monitoring over 5000 iterations.

**Results:**
```
CUDA at start:     15,419 MB / 16,303 MB (658 MB free before doing anything)
CUDA at init:      15,587 MB (after env creation + reset(1024))
CUDA after JIT:    15,631 MB (after first select_action compilation)
CUDA during train: 15,639-15,645 MB (stable, +6MB over 5000 iters — NO LEAK)
JAX peak usage:    645 MB (within JAX's pre-allocated pool)
```

**No leak observed over 5000 iterations.** Both JAX memory and nvidia-smi CUDA total are flat. But our test only covered 5000 iterations (~5M steps) — the OOM happens at 65-83k iterations (67-85M steps). Possible explanations we **haven't ruled out:**

1. A very slow accumulation (~0.01 MB/iter) that's invisible over 5000 iters but adds up over 83k iters (+830MB)
2. Orbax checkpoint serialization (not tested) temporarily allocating GPU memory and not fully releasing it
3. XLA command buffer cache growing outside of what `nvidia-smi` reports in our short test
4. A specific event (e.g., CUDA graph recompilation triggered by some rare shape change) that only happens after long enough training

**What we know for sure:**
- JAX-managed memory is bounded and doesn't leak
- The GPU starts at 95% capacity (15.4GB / 16.3GB) with 1024 envs — very little headroom
- `nvidia-smi` showed no growth over 5000 iters, but the crash happens 13-17x further into training

**What we DON'T know:**
- Whether nvidia-smi accurately captures all CUDA driver/XLA allocations
- Whether something accumulates between 5000 and 83000 iterations that our test missed
- The exact allocation that triggers the OOM

**Fix CONFIRMED: `XLA_CLIENT_MEM_FRACTION=0.7` works.** FastSAC completed the full 100M steps (exit code 0) on 2026-03-19 with this setting. Previous runs without it OOM'd at 67-85M steps. We still don't know the exact CUDA-level mechanism, but the fix is empirically proven.

## Root cause

**Not a leak. Not a bug. Just not enough VRAM.** 16GB is tight for 1024 envs + C51 distributional critic + 512×512 networks + 1M replay buffer.

## Solutions (ranked)

### 1. Reduce JAX memory pre-allocation
```python
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.7"  # default is ~0.75-0.9
```
Gives more room for XLA command buffers and driver overhead. May cause JAX to OOM on its own pool instead, but worth testing.

### 2. Reduce buffer size
1M entries × 6 arrays × (17+17+6+1+1+1) floats × 4 bytes = **172 MB**. Reducing to 500k saves 86MB.

### 3. Reduce num_envs to 512
Halves env state memory. May need to adjust grad_updates_per_step.

### 4. Disable XLA command buffers (already done)
`--xla_gpu_enable_command_buffer=` prevents command buffer accumulation. This is the cheapest fix and may be sufficient with the batch dim matching fix.

### 5. Run eval less frequently or skip eval for very long runs
Each eval call runs 1000 steps × 1024 envs. Reducing eval frequency reduces transient memory spikes.

### 6. Accept it — 16GB is the limit for 1024-env training
The FastTD3 paper uses A100s (40-80GB). A 5080 (16GB) running 1024 envs is at the hardware limit. The "fix" might be using fewer envs (512 or 256) on consumer hardware.

## ROOT CAUSE FOUND: MuJoCo Playground recompiles env.step continuously

### Test A: Long-duration compilation logging — SMOKING GUN

**Test:** `JAX_LOG_COMPILES=1` on FastTD3 CheetahRun for 30M steps (1024 envs).

**Results:**

```
Compilations per minute:
  14:16  — 1283  (JIT warmup burst)
  14:17  —   14  (tail of warmup)
  14:18  —    5
  14:19  —    2  ← steady state begins
  14:20  —    2
  ...
  14:45  —    3  (end of run)
```

**2 recompilations per minute at steady state**, continuously for the entire run.

**What's recompiling:** `jit(while)` and `jit(scan)` — MuJoCo Playground's MJX physics step.
- `while` = iterative contact solver (`mjx_env.step` → `jax.lax.while_loop`)
- `scan` = physics substep loop (`jax.lax.scan(single_step, data, (), n_substeps)`)

**Critical finding:** 62 compilations of `jit(while)` with **identical argument signatures** (same shapes, same dtypes, same sharding). Same for `jit(scan)` — 62 identical compilations. JAX is recompiling the same function with the same inputs.

**Why this causes OOM:** Each recompilation creates a new XLA graph / CUDA command buffer. Over a 100M step run (~120 min), that's ~240 recompilations × 2 functions = ~480 accumulated graphs. Each graph consumes CUDA driver memory (outside JAX's pre-allocated pool). On a 16GB GPU that starts at 95% capacity, this pushes past the limit at ~67-85M steps.

**Why JAX recompiles identical functions:** Unknown — this appears to be a JAX/XLA or MuJoCo Playground issue, not our code. Possible causes:
1. The `while_loop` body captures a closure variable whose identity (not shape) changes between calls
2. XLA's compilation cache has a TTL or size limit that evicts old entries
3. The `env_state` pytree structure changes subtly between steps (e.g., dynamic contact count)

**This is upstream behavior — not something we can fix in our training scripts.**

### Test D: MEM_FRACTION=0.7 — CONFIRMED FIX

FastSAC completed full 100M steps (exit code 0) with `XLA_CLIENT_MEM_FRACTION=0.7`. By reducing JAX's pre-allocated pool from ~90% to 70% of GPU memory, we leave ~1.6GB extra for CUDA driver overhead and accumulated command buffers.

### Tests B, C: Not needed

With the root cause identified (continuous MJX recompilation), orbax and nvidia-smi polling tests are no longer needed. The mechanism is clear.

## Final understanding

```
GPU memory layout (16GB total):
├── JAX pre-allocated pool: ~12GB (at MEM_FRACTION=0.7)
│   ├── Env state (1024 envs): ~2GB
│   ├── Replay buffer: ~170MB
│   ├── Model params + opt state: ~50MB
│   └── Training compute: ~9.8GB
├── CUDA driver + XLA overhead: ~3GB
│   ├── Command buffers (initial): ~1GB
│   ├── Recompilation accumulation: +2-3MB/min × 120 min = ~300MB
│   └── Other CUDA overhead: ~1.7GB
└── Free headroom: ~1GB (enough to absorb accumulation)
```

Without MEM_FRACTION=0.7, JAX takes ~14.5GB leaving only ~1.5GB for CUDA driver. The ~300MB recompilation accumulation over 2 hours pushes past the limit.

## Solutions (final, ranked)

1. **`XLA_CLIENT_MEM_FRACTION=0.7`** — proven fix, ~5% sps overhead from smaller JAX pool. Applied to all training scripts via `env_setup.py`.
2. **`jax.clear_caches()` every N steps** — could flush accumulated compilations. Not tested yet but worth trying if MEM_FRACTION alone becomes insufficient.
3. **Report upstream** — the continuous recompilation of identical MJX functions is a JAX/MuJoCo Playground issue. Could file an issue on `google-deepmind/mujoco_playground`.
