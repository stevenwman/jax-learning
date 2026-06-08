# mjx-Warp eval-call memory leak (Factory GearMesh OOM debug)

Date: 2026-06-04
Branch: `factory-peg-insert`
Prior: `.context/journals/2026-06-03-factory-6dof-hover-trap.md` (hover trap solved via altitude-weighted r_align)

## TL;DR

After fixing the GearMesh hover trap on 2026-06-03, every attempt to train Factory GearMesh past ~800k steps hit `RuntimeError: Failed to allocate 3072 bytes on device 'cuda:0'` during eval. Spent the day characterizing the failure. Found it is **deterministic, eval-call-only, and rooted in mjx-Warp's JAX FFI layer** — each call to `evaluate()` leaks +1 jax Array (~1.7 MB on the GearMesh model) that **cannot be reclaimed via `gc.collect()` or `jax.clear_caches()`**. Workaround: ship with `eval_every_n_episodes=1500` (3 evals over 2M, well under the empirical 14-eval cliff).

## Symptom

| Run | Step at OOM | Error |
|---|---|---|
| v17/iter0 (CoACD) | 798k | `Failed to allocate 3072 bytes` |
| iter 3 (CoACD, exclusive GPU) | 798k | same |
| hybrid SDF runs | first eval | EPA buffer (74 MB) — separate issue |

OOM **always** lands during eval (`env.reset` or `forward_vmap` inside `evaluate()`). Train steps never OOM, even at the same policy state where eval crashes. Multi-tenancy ruled out: iter 3 hit the same step with no other GPU users.

## Investigation

### Hypotheses tested

| Hypothesis | Result |
|---|---|
| Slow per-eval memory leak | Confirmed but small (~1.7 MB/call) |
| CUDA mempool ratchet | Plausible mechanism; introspection API returned zeros |
| Pool fragmentation | Possible but inferred |
| Multi-tenant contention | Ruled out (iter 3 OOMed on exclusive GPU at same step) |
| First-eval JIT compile too big | Ruled out (1st-13th evals succeed) |
| JIT retracing (different shapes) | **Ruled out** — `step_cache_len` stayed at 1 across 20 calls |
| Increasing contact count (better policy → more contacts) | Inconsistent with train not OOMing |
| Per-call jax Array leak | **Confirmed** — `live_arrays` grows monotonically +1 per evaluate() call |

### Decisive probe (`.tmp/cache_probe.py`)

Minimal harness: load `FactoryGearMesh`, call `evaluate()` 20 times with zero-action policy, dump VRAM + cache state per call.

```
i  gpu_MB delta  step_cache_len  live_arrays
1  12719  +410   1               221
5  12727  +0     1               225
10 12733  +2     1               230  ← gc.collect()+jax.clear_caches() here
11 12739  +6     1               231  ← cleanup did NOT release
15 12743  +2     1               235
20 12753  +4     1               240
```

Findings:
- `step_cache_len` constant at 1 → **no retracing**
- `live_arrays` grows +1 per call → **JAX retains one array per evaluate() invocation**
- `gc.collect()` + `jax.clear_caches()` + `del r` at i=10 **did not** drop either VRAM or `live_arrays` → not a Python reference issue
- ~1.7 MB per call in the minimal probe; real training likely loses more per call due to actor/critic and replay buffer entanglement

The leaked array is held somewhere inside the mjx-Warp FFI layer (callback registry, stream-ordered allocator's per-call workspace, or Warp's CUDA event tracking). Reproduces with stock mjx-Warp + FactoryGearMesh — we did not patch upstream.

### Eval budget calculation

- Leak: ~1.7 MB/call (probe) → likely 5–10 MB/call in real training with full SAC state
- Free margin at training start (FactoryGearMesh, num_envs=64): ~1.7 GB
- Empirical OOM cliff: **14 evals** (deterministic step 798k)
- Safe budget: keep `total_evals < 12` to leave headroom for the eval-time transient

## Fix (shipped)

`jax_rl/configs/env_presets.py:FLASH_SAC_PRESETS["FactoryGearMesh"]`:
```python
total_timesteps=2_000_000,
eval_every_n_episodes=1500,   # ~3 evals over 2M, safely under 14-eval cliff
```

## What we did NOT fix

- The underlying mjx-Warp leak still exists. Any run with `>14 evals` will OOM regardless of env or workload.
- Possible upstream fixes (none attempted):
  - Patch mjx-Warp's FFI to release per-call workspace after `forward()` returns
  - Run eval in a subprocess (fork-and-exit per eval — kills all per-process GPU state)
  - Switch to mjx-native backend (no Warp FFI) — costs throughput

## Dead ends (recorded so we don't relitigate)

- **Hybrid mesh-SDF flanking gears** to reduce CoACD part count: SDF×CYLINDER pair is not implemented in Warp's SDF dispatcher (`collision_sdf.py:405 wp.printf("ERROR: SDF type not implemented")`). Adding contype/conaffinity bitmasks to exclude that pair silences the spam, but trained policy at Return 824 vs CoACD's 1369 at same step — converges slower. Reverted.
- **`XLA_PYTHON_CLIENT_MEM_FRACTION`/`PREALLOCATE`**: capping JAX prealloc just OOMs sooner. Confirmed the 12 GB "post-put_model" footprint is entirely JAX prealloc (drops to ~3 GB at 0.10 fraction).
- **Bumping naccdmax/njmax**: njmax bumps from 1024→32768 fixed `nefc overflow` in SDF mode but unrelated to the eval-time leak.

## Where to look

| Thing | Path |
|---|---|
| Cache probe (decisive) | `.tmp/cache_probe.py` |
| Iter 3 VRAM trace | `.tmp/logs/gearmesh_iter3_vram.log` |
| Iter 3 train log (showing OOM at 798k) | `.tmp/logs/gearmesh_auto_njmax_iter0.log` |
| Updated preset | `jax_rl/configs/env_presets.py:576-595` |
| Reverted env defaults | `jax_rl/envs/manipulation/factory/factory_gear_mesh.py:88-103` |

## Open issues

1. The leak almost certainly affects all mjx-Warp envs — Factory just exposes it because eval is long (`lax.scan(450)`) and the env is contact-heavy. Worth verifying on PegInsert at high eval cadence.
2. If subprocess-per-eval is acceptable as a workaround, `jax_rl/utils/eval.py` could fork with `os.fork()` + `multiprocessing` and IPC the scalar metrics back. Significant infra work.
3. Upstream issue to mjx / Warp warranted; minimal repro is in `.tmp/cache_probe.py`.
