# Fast Algorithm Literature Mismatches

**Status:** Documented 2026-03-20. Updated 2026-03-21 after source code audits. Most critical mismatches now FIXED.

Two papers, two sets of mismatches. Sources verified against actual source code (holosoma repo for FastTD3/FastSAC, anonymous repo for FastDSAC) and paper tables.

---

## Paper 1: FastTD3 / FastSAC — Seo et al. 2025

**Paper:** "Learning Sim-to-Real Humanoid Locomotion in 15 Minutes" (arXiv:2512.01996)
**Source code:** https://github.com/amazon-far/holosoma
**Config file:** `src/holosoma/holosoma/config_types/algo.py` → `FastSACConfig`
**Network file:** `src/holosoma/holosoma/agents/fast_sac/fast_sac.py`

### Critical Mismatches (all FIXED as of 2026-03-20)

| Param | Paper (holosoma code) | Was | Now | Status |
|---|---|---|---|---|
| **tau** | `0.125` | `0.005` | `0.125` | **FIXED** |
| **Network architecture** | 3-layer tapered: actor 512→256→128, critic 768→384→192 | Flat 2-layer: (512, 512) | Tapered 3-layer | **FIXED** |
| **Activation** | SiLU (swish) | ReLU | SiLU | **FIXED** |
| **C51 atoms** | `101` | `51` | `101` | **FIXED** |
| **v_min/v_max** | `[-20.0, 20.0]` | varies | `[-20.0, 20.0]` | **FIXED** |
| **Policy delay (FastSAC)** | `policy_frequency=4` | 1 (no delay) | 4 | **FIXED** |
| **Obs normalization** | Sample-time with `EmpiricalNormalization` | Disabled | Sample-time via `--obs-norm` flag | **FIXED** |

### Moderate Mismatches (partially addressed)

| Param | Paper (holosoma code) | Our value | Status |
|---|---|---|---|
| **Grad clipping** | Disabled (`max_grad_norm=0.0`) | `clip_by_global_norm(1.0)` | Still different — we add clipping |
| **LR schedule** | Constant | Cosine decay 3e-4 → 3e-5 | Still different — we add decay |
| **Grad updates/step** | `num_updates=8` | `12` | Still different |
| **Alpha optimizer** | AdamW with β=(0.9, 0.95) | AdamW (fixed from plain Adam) | **FIXED** |
| **log_std bounds** | min=-5.0, max=0.0 | min=-5.0, max=1.0 | Close, max differs |
| **Critic hidden dim** | `768` (wider than actor 512) | Tapered 768→384→192 | **FIXED** |

### Verified Matches

| Param | Paper | Ours | Status |
|---|---|---|---|
| Learning rate | 3e-4 | 3e-4 | OK |
| Adam β2 | 0.95 | 0.95 | OK |
| Weight decay | 0.001 | 0.001 | OK |
| Batch size | 8192 | 8192 | OK |
| Gamma (locomotion) | 0.97 | 0.97 | OK (fixed 03-19) |
| Alpha init | 0.001 | 0.001 | OK |
| Target entropy | 0.0 | 0.0 | OK |
| Max std | 1.0 (log_std_max=0.0) | 1.0 | OK |
| Q aggregation | avg | avg | OK |
| Twin critics | 2 | 2 | OK |
| Mixed noise (FastTD3) | U[0.01, 0.05] | U[0.01, 0.05] | OK (fixed 03-20) |

---

## Paper 2: FastDSAC — arXiv:2603.12612

**Paper:** "FastDSAC: Unlocking the Potential of Maximum Entropy RL in High-Dimensional Humanoid Control"
**Source code:** Downloaded from https://anonymous.4open.science/r/FastDSAC_ICML-5194, local copy at `/tmp/fastdsac_paper/FastDSAC/`
**Hyperparams:** `fast_sac/hyperparams.py` (source code) + Paper Table 1

### MAJOR LESSON: Paper describes "Gaussian NLL" but code uses Huber loss

The paper says "Gaussian distributional critic" and shows KL-divergence equations (Eq 8). We implemented
Gaussian NLL: `(y-μ)²/σ² + log(σ²)`. This NaN'd on every HumanoidRun attempt.

The paper's **actual source code** (`train_fastdsac_torch_enhanced.py`) uses:
- **Huber loss** (delta=50) for mean prediction error
- **Bounded per-sample ratio weighting** clamped to [0.1, 10]
- **EMA of batch std** (`mean_std`) for ratio computation
- **No `1/variance` anywhere** — division is by `(q_std_detach + bias)`
- **`z.clamp(-3, 3)`** on target distribution samples

See `LESSONS.md` > "FastDSAC: Paper Says 'Gaussian NLL' but Code Uses Huber Loss" for full diagnostic trail.

### Config Status (as of 2026-03-21, after source code audit)

| Param | Paper source code | Our value | Status |
|---|---|---|---|
| **Critic loss** | Huber (delta=50) + bounded ratio | Huber (delta=50) + bounded ratio | **FIXED** |
| **tau** | 0.1 | 0.1 | **FIXED** |
| **Batch size** | 32,768 | 32,768 | **FIXED** |
| **Buffer size** | 51,200 (HumanoidBench) / 10,240 (Playground) | 51,200 | **FIXED** |
| **Alpha init** | 0.001 | 0.001 | **FIXED** |
| **Weight decay** | 0.1 | 0.1 | **FIXED** |
| **Activation** | GELU | GELU | **FIXED** |
| **Actor dims** | 512→256→128 (tapered) | 512→256→128 | **FIXED** |
| **Critic dims** | 1024→512→256 (tapered) | 1024→512→256 | **FIXED** |
| **Reward scale** | 0.2 | 0.2 | **FIXED** |
| **Policy delay** | 2 | 2 | **FIXED** |
| **Grad updates/step** | 2 (per 128 envs) | 16 (per 1024 envs, matching UTD) | **FIXED** |
| **Learning starts** | 10 iterations | 10,240 samples (10 * 1024 envs) | **FIXED** |
| **num_envs** | 128 | 1024 | Different but scaled via UTD |
| **DEM temperature** | Per-task: 0.5-10.0 | Global 1.0 | TODO — not per-task |
| **Log std bounds** | min=-5.0, max=0.0-1.0 | min=-5.0, max=1.0 | Close |
| **AMP (bf16)** | Enabled | Not available (JAX) | N/A |

### Remaining Questions
- Does 1024 envs with UTD-matched grad updates behave the same as 128 envs with 2 updates?
- HumanoidRun (dm_control) vs HumanoidBench — different reward scales, obs dimensions
- The paper's `reward_scale=0.2` may be too aggressive for dm_control HumanoidRun (rewards ~0.002/step → scaled to 0.0004)

---

## Fix Status Summary

### FastTD3/FastSAC — all critical fixes applied
1. ~~tau=0.125~~ **FIXED**
2. ~~Tapered 3-layer network~~ **FIXED**
3. ~~SiLU activation~~ **FIXED**
4. ~~Policy delay=4 for FastSAC~~ **FIXED**
5. ~~Obs normalization~~ **FIXED** (sample-time normalization)
6. ~~C51 atoms=101~~ **FIXED**

### FastDSAC — rewritten from source code
1. ~~Critic loss (Huber, not NLL)~~ **FIXED**
2. ~~Alpha init~~ **FIXED**
3. ~~Buffer/batch size~~ **FIXED**
4. ~~Weight decay, activation, network dims~~ **FIXED**
5. ~~UTD ratio matching~~ **FIXED**
6. DEM per-task temperature — TODO
