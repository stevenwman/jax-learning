# Frame Stack Validation Checklist

Reference doc for A/B testing frame stacking. Each item is a concern to verify — results become lessons.

## Edge Cases — Resolved

### 1. record_video.py incompatibility — FIXED
- **Problem:** record_video.py loads env via `pg_registry.load()`, not `env_setup.py`. No FrameStackWrapper applied. Frame-stacked checkpoint (obs_dim=144) would feed raw 48d obs to 144d network.
- **Fix:** record_video.py now reads `n_frame_stack` from `meta.json` → `train_config` and applies `FrameStackWrapper` before rollout. Commit `334dec6`.
- **Lesson:** Any env transformation applied during training must also be applied in every inference consumer (eval, recording, deploy). Pattern: read wrapper config from checkpoint metadata.

### 2. Auto-reset frame staleness — FIXED
- **Problem:** Brax `AutoResetWrapper` only resets `pipeline_state` and `obs` on done. `state.info["frame_stack"]` was left stale — containing frames from the dead episode. Next episode's first N-1 policy inputs were contaminated.
- **Root cause:** AutoResetWrapper (Brax source, lines 129-158) caches `first_pipeline_state` and `first_obs` during `reset()`, replays via `jp.where(done, cached, new)` in `step()`. Does NOT touch `state.info`.
- **Fix:** `FrameStackWrapper.step()` uses `jp.where(state.done, tiled_stack, shifted_stack)` — on done, re-tiles the stack from current obs so next episode starts clean. JIT-safe (no Python branch). Commit `334dec6`.
- **Lesson:** Any per-env state stored in `state.info` that should reset at episode boundaries must be explicitly handled — Brax auto-reset won't do it for you. Use `jp.where(done, ...)` pattern.

### 3. Replay buffer memory — FIXED (sample-time reconstruction)
- **Before:** Naive stacked buffer stored 144d obs + 144d next_obs = **5.08 GB** at 4M entries.
- **After:** Sample-time reconstruction stores raw 48d obs only (no next_obs). Buffer = **0.77 GB** (83% reduction). 4M buffer fits easily on 16GB GPU.
- **How:** `FrameStackConfig` on `JaxReplayBuffer`. Stores newest 48d frame, reconstructs 3-frame stack at sample time by looking back `k * num_envs` indices. Episode boundaries handled via `jp.where(done, tile, actual)`. `next_obs` derived from `buffer[i + num_envs]`.

## Edge Cases — Deferred (low priority)

### 4. Obs normalization interaction
- **Concern:** `--obs-norm` computes running statistics over 144d stacked obs. Frames 0-2 have different temporal distributions (frame 0 = current, frame 2 = 2 steps old). Joint velocities especially drift over 2 steps. Normalizing the full 144d blends these distributions, biasing the statistics.
- **Correct approach:** Normalize per-frame — apply norm to each 48d slice independently using the same running stats. The stats are computed on raw 48d obs (same distribution regardless of frame position), then applied to each slice of the stack separately.
- **Why deferred:** Off-policy Go2 doesn't use `--obs-norm`. Only relevant for vision RL with `--obs-norm`.

### 5. sim2sim with frame stacking
- **Concern:** Deploy `ObsBuilder` mirrors frame stacking in numpy. Need to verify FIFO matches JAX wrapper.
- **Why deferred:** Needs a frame-stacked checkpoint first. Will test after A/B training.

## A/B Comparison Plan

**Env:** Go2WarpJoystickFlat, 1024 envs, FastSAC
**Budget:** 18M steps (matches previous best of 276.5 eval)
**Seeds:** 2 per condition minimum
**Buffer:** 4M (sample-time reconstruction keeps buffer at 0.77 GB regardless of stacking)

| Run | Config | Purpose |
|-----|--------|---------|
| A (baseline) | Previous run: seed 6001, eval 276.5 @ 17.3M steps | No stacking (48d) — already done |
| B (stacked) | `--algo fast_sac --env Go2WarpJoystickFlat --frame-stack 3 --seed 8000` | 3-frame (144d) |

**Metrics to compare:**
- Eval return curve (sample efficiency)
- Wall-clock time per 1M steps (overhead from larger obs)
- Peak GPU memory (`nvidia-smi`)
- sim2sim transfer quality (if both train successfully)

**Hypothesis:** Frame stacking should help the policy infer velocity/dynamics from observation history, potentially allowing removal of privileged velocity info from actor obs in the future. But the 3x obs dim increases network input, which could hurt sample efficiency on the current 18M budget.

**Key question:** Does the temporal info in stacked frames provide signal the policy can't already get from `last_action` (12d, already in obs)?

## A/B Results (2026-04-01)

| Steps | Baseline (seed 6001, no stack) | Frame-stacked (seed 8001, 3x) |
|-------|-------------------------------|-------------------------------|
| ~1M | 175.8 | 138.3 |
| ~5M | 215.8 | 268.2 |
| ~9M | 273.8 | 267.8 |
| ~14M | 274.8 | 280.1 |
| ~17-19M | 276.5 | 265.2 |
| Final eval | **276.5** (best @ 17M) | **271.3** ± 8.3 (@ 20M) |

Also ran seed 8000 (no mid-training evals): final eval 274.6 ± 6.0.

**Conclusion:** Frame stacking has no measurable effect on Go2 FastSAC. The policy already has `last_action` (12d) in the 48d obs, which provides sufficient temporal context. The extra 96d of stacked history doesn't add useful signal for locomotion.

**This matches the literature:** locomotion benefits from GRU/RNN temporal context (ANYmal, DeFM) or learned terrain estimators (DreamWaQ, WTW), not raw frame stacking. Frame stacking helps vision/manipulation (DrQ-v2, CURL) where consecutive pixel frames encode motion.

**Frame stacking infrastructure is still valuable for:**
- Vision RL (CNN on stacked pixel frames — future)
- Environments without `last_action` in obs
- Benchmarking against DrQ-v2 style methods
