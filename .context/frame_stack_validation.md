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

### 3. Replay buffer memory — MEASURED, OK with reduced buffer
- **Numbers:** Raw (48d) buffer at 4M entries = **1.86 GB**. Stacked (144d) = **5.08 GB** (2.7x).
- **Full budget:** 5.08 GB (buffer) + ~2.3 GB (baseline GPU) + ~2-3 GB (JIT/model/envs) ≈ 10-11 GB. Fits on 16GB GPU but tight.
- **Mitigation:** Use `--buffer-size 2097152` (2M) for A/B testing. Halves buffer to ~2.5 GB.
- **Future option:** Store raw 48d in buffer, reconstruct stacks at sample time (DrQ-v2 pattern). 3x cheaper but more complex. Only worth it if 2M buffer hurts learning.

## Edge Cases — Deferred (low priority)

### 4. Obs normalization interaction
- **Concern:** `--obs-norm` computes running statistics over 144d stacked obs. Frames 0-2 have different temporal distributions. Stats would blend them.
- **Why deferred:** Off-policy Go2 doesn't use `--obs-norm`. Only relevant if we enable it for frame-stacked training.

### 5. sim2sim with frame stacking
- **Concern:** Deploy `ObsBuilder` mirrors frame stacking in numpy. Need to verify FIFO matches JAX wrapper.
- **Why deferred:** Needs a frame-stacked checkpoint first. Will test after A/B training.

## A/B Comparison Plan

**Env:** Go2WarpJoystickFlat, 1024 envs, FastSAC
**Budget:** 18M steps (matches previous best of 276.5 eval)
**Seeds:** 2 per condition minimum
**Buffer:** 2M (to fit in VRAM alongside other GPU users)

| Run | Config | Purpose |
|-----|--------|---------|
| A (baseline) | `--algo fast_sac --env Go2WarpJoystickFlat --buffer-size 2097152` | No stacking (48d) |
| B (stacked) | `--algo fast_sac --env Go2WarpJoystickFlat --frame-stack 3 --buffer-size 2097152` | 3-frame (144d) |

**Metrics to compare:**
- Eval return curve (sample efficiency)
- Wall-clock time per 1M steps (overhead from larger obs)
- Peak GPU memory (`nvidia-smi`)
- sim2sim transfer quality (if both train successfully)

**Hypothesis:** Frame stacking should help the policy infer velocity/dynamics from observation history, potentially allowing removal of privileged velocity info from actor obs in the future. But the 3x obs dim increases network input, which could hurt sample efficiency on the current 18M budget.

**Key question:** Does the temporal info in stacked frames provide signal the policy can't already get from `last_action` (12d, already in obs)?
