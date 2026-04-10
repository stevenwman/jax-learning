# AutoResetWrapper & Domain Randomization: Lessons Learned

**Date:** 2026-04-07 to 2026-04-08

> **Historical note:** The legacy DR v1 path (`go2_randomize.py`, `bongo_randomize.py`,
> `DomainRandomizationVmapWrapper`, `--domain-rand` flag) and the `syncd` reset mode
> in `DomainRandWrapper` were removed on 2026-04-09. Only `per_step` mode remains.
> This doc is preserved as historical context for the design decisions.

## TL;DR

Built DRv2Wrapper (now renamed to `DomainRandWrapper` in `domain_rand.py`) to replace AutoResetWrapper + DomainRandomizationVmapWrapper. Two modes: per_step and syncd. **Per_step is the winner for Go2** — 6% throughput cost but better sample efficiency (199 vs 167 avg_return at same wall clock). Syncd has 2x raw throughput but waste from dead envs kills effective sample efficiency and integrating it with the training loop (tracker, buffer, logging) is a nightmare.

Current DR v1 only provides 256 fixed physics configurations for the entire training run. DRv2 (now `DomainRandWrapper`) enables per-episode re-randomization. Next step: wire DR specs into Go2 env class.

## Key Findings

### 1. Current DR is weak — only 256 fixed configs

With `full_reset=False` (the default), domain randomization happens once at init. 256 envs × 1 randomization = 256 unique physics configs, frozen forever. Same limitation in Brax. Per-episode DR doesn't exist in v1.

### 2. full_reset=True vs False is wildly inconsistent across envs

Benchmark (256 envs, RTX 5080):

| Env | full_reset=False | full_reset=True | Diff |
|-----|----------------:|----------------:|-----:|
| Go2 Warp Joystick | 30,534 sps | 37,893 sps | **+24.1%** |
| Go2 Bongo | 45,969 sps | 38,374 sps | -16.5% |
| CartpoleBalance | 298,391 sps | 172,700 sps | -42.1% |
| WalkerWalk | 52,259 sps | 48,005 sps | -8.1% |
| CheetahRun | 284,362 sps | 2,482 sps | **-99.1%** |
| HumanoidRun | 53,354 sps | 49,009 sps | -8.1% |

Go2 is the only env where full_reset=True is faster. CheetahRun is catastrophically slow — root cause still unexplained.

### 3. The "compute both paths" constraint is fundamental to JAX

Under jax.vmap, jax.lax.cond is lowered to jnp.where — both branches always execute for all envs. No selective reset possible. This is an XLA/compiler constraint.

| Framework | Selective reset? | Per-episode DR? | How? |
|-----------|:---:|:---:|-----|
| Isaac Lab (PyTorch) | Yes, O(k) | Yes | tensor[env_ids] scatter-write |
| Brax (JAX) | No, O(N) | No — fixed at init | where_done, no full_reset option |
| MuJoCo Playground (JAX) | No, O(N) | Only with full_reset=True | where_done + optional full_reset |

### 4. Physically inconsistent state causes 10-19x solver blowup

Replacing qpos with zeros on Go2 (legs clip through body) while keeping contact buffers from a previous step → constraint solver explosion. **Always use default pose / keyframe for reset, never zeros.**

| Env | Zeroed qpos | Valid qpos | Ratio |
|-----|--------------------:|-------------------:|------:|
| Go2 Warp | 21,944 µs | 2,194 µs | **10x** |
| Go2 Bongo | 28,483 µs | 1,522 µs | **19x** |
| Cartpole | 335 µs | 368 µs | 0.9x |

### 5. Dead envs do NOT slow down physics

Tested syncd mode: SPS constant at ~36k regardless of waste fraction (0% to 100% dead envs). MuJoCo Warp solver cost is the same for fallen vs standing robots.

### 6. Cumulative SPS is misleading — always compare converged SPS

Initial syncd showed ~470 sps vs legacy ~3,950. This was comparing syncd's cumulative SPS (includes JIT compilation) vs legacy's converged SPS. Legacy also starts at ~368 sps. Both converge to ~3,400 sps. Wasted 2+ hours debugging a non-existent "4x slowdown."

**Lesson:** Never compare SPS numbers from different training phases. JIT warmup dominates the first few minutes.

### 7. Per-step GPU→CPU sync kills async execution

`np.asarray(env_state.info["_drv2_episode_done"])` every step forces GPU→CPU synchronization, breaking async GPU pipeline. This alone caused a real ~8x slowdown. Must defer GPU reads to log-time only, or use CPU-side tracking arrays.

### 8. Syncd mode: 2x raw throughput but impractical for training

| Benchmark | Legacy sps | Syncd sps | Per_step sps |
|-----------|----------:|---------:|------------:|
| Raw env.step (isolated) | 47,800 | 104,000 | 59,150 |
| Full training loop | ~3,400 | ~3,400 | ~3,200 |

Syncd has 2x raw env throughput, but:
- **Waste:** 60-95% of env-steps are post-done (dead envs waiting for batch reset). Early training is worst.
- **Tracker integration:** `done` is sticky in syncd mode, breaks EpisodeTracker (floods with fake zero-reward episodes). Requires `_newly_done` masking + `episode_rewards` reset on batch reset.
- **Buffer pollution:** Post-done transitions (reward=0, garbage obs/actions) fill the replay buffer. Unknown impact on learning.
- **Logging mess:** Episode counts, avg_return, waste fraction all need special handling.
- **Even with "reset when all done" optimization**, waste is still high because envs die at different times.

### 9. Per_step mode: the practical winner for Go2

Per_step DRv2 (now `DomainRandWrapper` in `jax_rl/envs/wrappers/domain_rand.py`) = functionally equivalent to full_reset=True but in a single wrapper. For Go2:
- **~4% throughput cost** vs legacy (3,791 vs 3,950 sps)
- **Better final eval:** 280.1 vs 270.3
- **Better avg_return:** 265.9 vs 248.8
- **Fresh ICs every episode** — no frozen 256 configs
- **Clean state.info reset** — no stale step_count, last_act, etc.
- **No waste, no tracker hacks, no buffer pollution**
- **Foundation for per-episode DR** — just add DR specs

Full 5M step comparison (Go2WarpJoystickFlat, FastSAC, seed 0, 256 envs, wandb: drv2-comparison):

| | avg_return | eval | sps | wall clock |
|--|----------:|-----:|----:|----------:|
| Legacy (AutoReset full_reset=False) | 248.8 | 270.3 | 3,950 | 21 min |
| Per_step (DomainRandWrapper) | 265.9 | 280.1 | 3,791 | 22 min |

### 10. forward() cost breakdown

| Env | forward() | env.step() | Ratio |
|-----|----------:|-----------:|------:|
| Go2 Warp | 2,439 µs | 9,326 µs | 26% |
| Cartpole | 338 µs | 360 µs | 94% |
| CheetahRun | 628 µs | 644 µs | 98% |
| HumanoidRun | 892 µs | 3,118 µs | 29% |

For complex envs (Go2, Humanoid), forward() is ~25-30% of step cost. For simple envs, forward() ≈ step() (physics is trivial, overhead dominates). This means per_step mode's "extra reset" cost is proportionally smaller for complex envs where it matters most.

## Things we want per-episode variation on (>256 unique)

1. Initial qpos (joint positions with noise)
2. Initial qvel (joint velocities)
3. Ground friction (geom_friction)
4. Joint damping (dof_damping)
5. Joint armature (dof_armature)
6. Joint friction loss (dof_frictionloss)
7. Body masses (body_mass)
8. Torso COM jitter (body_ipos)
9. Motor strength (actuator_gainprm)
10. Kp scale (PD proportional gain)
11. Kd scale (PD derivative gain)
12. Initial command (vx, vy, yaw_rate)

All 12 are frozen at init with current v1 setup. Per_step DRv2 enables re-randomizing all of them every episode.

## Meta-lessons

### Benchmarking discipline
- **Always compare converged SPS**, not cumulative. JIT warmup is 10-100x slower.
- **Test across multiple envs.** Go2 being the only env faster with full_reset=True would have been invisible without cross-env benchmarking.
- **Don't feed garbage input.** Zeroed qpos on Go2 caused a 10x solver blowup that looked like a fundamental forward() cost.
- **GPU sync in measurement code affects results.** `np.asarray()` or `block_until_ready()` in the measurement loop changes what you're measuring.

### Integration is harder than the algorithm
- DRv2 env stepping works perfectly in isolation. The 80% of effort was integrating with: EpisodeTracker (sticky done), replay buffer (post-done transitions), wandb logging (episode count inflation), and effective step counting.
- Syncd mode required changes to: wrapper, training loop, tracker, buffer add, logging, batch reset timing, and waste tracking. Each change introduced new bugs.

### The JAX tax is real but manageable
- Can't do selective reset (O(N) always). Accept it.
- Per_step mode pays ~6% overhead for Go2. Worth it for per-episode DR.
- Gradient updates dominate (80% of training time), so env-side optimizations have diminishing returns.

### 11. CheetahRun per_step is NOT catastrophically slow

Late-night test: DRv2 per_step on CheetahRun ran at **6,309 sps** — not the 2,482 sps from the full_reset=True benchmark. The earlier 99% slowdown was either a benchmark artifact or a JIT graph difference specific to AutoResetWrapper, not a fundamental per_step issue.

`Eps 0, Return NaN` was NOT a bug — CheetahRun never sets done (running task), first episode completes at step 1000. The run was killed at step 712. DRv2 per_step may work for all envs, not just Go2. Needs confirmation past step 1000.

## Decision: Per_step for Go2 (confirmed), potentially all envs

Per_step `DomainRandWrapper` (formerly DRv2) is the path forward for Go2 sim-to-real. It's slightly slower but gives us:
- Per-episode IC variation
- Per-episode DR (once specs are wired in)
- Clean episode boundaries
- Better sample efficiency (shown in training comparison)

Other envs may also work with per_step — CheetahRun ran at 6,309 sps (not catastrophic). Needs confirmation past step 1000.

## Open Questions

1. ~~**CheetahRun full_reset catastrophe:**~~ LIKELY NOT REAL. DRv2 per_step ran at 6,309 sps. Original 99% slowdown was probably a benchmark artifact. Need full run past step 1000 to confirm episodes complete.
2. **Per-episode DR training validation:** DR specs wired into Go2 and smoke tested (4 envs). Need full 5M training run to confirm learning is not degraded.
3. **DRv2 as universal replacement:** If CheetahRun confirms working, DRv2 per_step could replace legacy for all envs. Would simplify the codebase (one wrapper stack instead of two).
