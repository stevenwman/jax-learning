# Fast Algorithm Literature Mismatches

**Status:** Documented 2026-03-20. Needs resolution before HumanoidRun benchmarking.

Two papers, two sets of mismatches. Sources verified against actual source code (holosoma repo) and paper tables.

---

## Paper 1: FastTD3 / FastSAC — Seo et al. 2025

**Paper:** "Learning Sim-to-Real Humanoid Locomotion in 15 Minutes" (arXiv:2512.01996)
**Source code:** https://github.com/amazon-far/holosoma
**Config file:** `src/holosoma/holosoma/config_types/algo.py` → `FastSACConfig`
**Network file:** `src/holosoma/holosoma/agents/fast_sac/fast_sac.py`

### Critical Mismatches

| Param | Paper (holosoma code) | Our value | File | Severity |
|---|---|---|---|---|
| **tau** | `0.125` | `0.005` | `fast_td3_config.py:9`, `sac_config.py:15` | **CRITICAL** — 25x slower target update. With 8 grad steps/env step, tau=0.125 means target network closely tracks online. Our tau=0.005 causes massive lag. |
| **Network architecture** | 3-layer tapered: actor 512→256→128, critic 768→384→192 | Flat 2-layer: (512, 512) for both | `fast_td3_config.py:28`, `sac_config.py:28` | **CRITICAL** — fundamentally different network shape and capacity |
| **Activation** | SiLU (swish) — `nn.SiLU()` in fast_sac.py | ReLU | `fast_td3_config.py:29`, `sac_config.py:29` | **HIGH** |
| **C51 atoms** | `101` | `51` | `fast_td3_config.py:16` | **HIGH** — half the distributional resolution |
| **v_min/v_max** | `[-20.0, 20.0]` | varies per env ([-10, 50] to [-10, 150]) | `env_presets.py` | **HIGH** — different support range |
| **Policy delay (FastSAC)** | `policy_frequency=4` | 1 (no delay) | `fast_sac.py` (algo) | **HIGH** — paper delays actor updates to every 4th critic update, like TD3 |
| **Obs normalization** | `obs_normalization=True` with `EmpiricalNormalization` (normalize at **sample time**, raw obs in buffer, eps=1e-2) | Disabled (`make_identity_norm_state`) | `train_fast_sac.py`, `train_fast_td3.py` | **HIGH** — paper normalizes obs at sample time; we skip entirely. Note: we tried normalizing pre-storage and it exploded (see LESSONS.md). Paper's approach stores raw obs, normalizes after sampling — safe but not yet implemented. |

### Moderate Mismatches

| Param | Paper (holosoma code) | Our value | File | Severity |
|---|---|---|---|---|
| **Grad clipping** | `max_grad_norm=0.0` (disabled) | `clip_by_global_norm(1.0)` | `train_fast_td3.py:66`, `train_fast_sac.py:67` | MEDIUM — we add clipping paper doesn't use |
| **LR schedule** | Constant (no schedule) | Cosine decay 3e-4 → 3e-5 | `train_fast_td3.py:63`, `train_fast_sac.py:64` | MEDIUM — we add decay paper doesn't use |
| **Grad updates/step** | `num_updates=8` | `12` | `fast_td3_config.py:25` | MEDIUM |
| **Alpha optimizer** | AdamW with β=(0.9, 0.95) | Plain Adam with default β | `train_fast_sac.py` | MEDIUM |
| **log_std bounds** | `log_std_min=-5.0`, `log_std_max=0.0` | `log_std_min=-20.0`, `log_std_max=2.0` (via PolicyHeadConfig) | `networks_config.py:29-30` | MEDIUM — wider range allows more extreme exploration |
| **Buffer size** | `buffer_size=1024` (per env, effective = 1024 × num_envs) | `1,000,000` (global) | `fast_td3_config.py:22` | MEDIUM — different semantics, may be equivalent depending on num_envs |
| **Critic hidden dim** | `768` (vs actor 512) | Same as actor (512) | All fast algo configs | MEDIUM — paper uses wider critic than actor |

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
**Source code:** https://anonymous.4open.science/r/FastDSAC_ICML-5194 (anonymous, may be down)
**Hyperparams:** Paper Table 1 + shared hyperparameters section

### Critical Mismatches

| Param | Paper (Table 1 + text) | Our value | File | Severity |
|---|---|---|---|---|
| **Batch size** | `32,768` | `8,192` | `fast_dsac_config.py:25` | **CRITICAL** — 4x smaller batch |
| **Buffer size** | `5,120-51,200` per task | `1,000,000` | `fast_dsac_config.py:22` | **CRITICAL** — 20-200x larger buffer. Paper uses tiny buffers for fresh data. |
| **Learning starts** | `1,000` | `25,000` | `fast_dsac_config.py:23` | **HIGH** — paper starts learning 25x earlier |
| **Alpha init** | `0.001` (HumanoidBench) / `0.01` (MuJoCo PG) | `1.0` (log_alpha=0) | `fast_dsac.py:253` | **HIGH** — missing entirely. FastDSACConfig has no alpha_init field. |
| **Critic loss** | Decomposed KL gradient (Eq 8): separate mean/variance gradient terms with custom formulation | Standard Gaussian NLL: `(y-μ)²/σ² + log(σ²)` | `fast_dsac.py:165-170` | **HIGH** — the variance gradient differs. Paper: `((y_z-Q)²-σ²)/σ³`. NLL: `(y-μ)²/σ⁴ - 1/σ²`. |
| **Log std bounds** | `log_std_min=-10`, `log_std_max=0.5-1.0` per task | `log_std_min=-20`, `log_std_max=2.0` | `networks_config.py` | **HIGH** — much wider bounds |

### Moderate Mismatches

| Param | Paper | Our value | File | Severity |
|---|---|---|---|---|
| **DEM temperature** | Per-task: 0.5-10.0 (Table 1) | Global `1.0` | `fast_dsac_config.py:19` | MEDIUM — not per-task |
| **Gradient clipping** | Not mentioned | `clip_by_global_norm(1.0)` | `train_fast_dsac.py` | MEDIUM — we add it |
| **LR schedule** | Constant (not mentioned) | Cosine decay 3e-4 → 3e-5 | `train_fast_dsac.py` | MEDIUM — we add it |
| **LayerNorm** | HumanoidBench: enabled. MuJoCo PG/IsaacLab: disabled | Default `True` for all tasks | `fast_dsac_config.py:30`, `env_presets.py` | MEDIUM — wrong for MuJoCo PG tasks |
| **Gamma** | 0.97 (MuJoCo PG), 0.99 (HumanoidBench) | 0.99 default | `env_presets.py` | MEDIUM — wrong for MuJoCo PG tasks |

### Verified Matches

| Param | Paper | Ours | Status |
|---|---|---|---|
| Learning rate | 3e-4 | 3e-4 | OK |
| Alpha LR | 3e-4 | 3e-4 | OK |
| Tau (soft update) | 0.005 | 0.005 | OK |
| Target entropy | 0.0 | 0.0 | OK |
| Variance eps | 1e-6 | 1e-6 | OK |
| Weight decay | 1e-4 | 1e-4 | OK |
| Adam betas | (0.9, 0.95) | (0.9, 0.95) | OK |
| Activation | ReLU | ReLU | OK |
| Twin critics | 2 | 2 | OK |
| Beta max | 2.0 | 2.0 | OK |
| Beta min (default) | 0.01 | 0.01 | OK |
| DEM weight formula | softmax(l*β/τ) * N | softmax(l*β/τ) * action_dim | OK |

---

## What Likely Matters Most (for HumanoidRun)

### From FastTD3/FastSAC paper:
1. **tau=0.125** — this is the #1 suspect. Fast target updates are essential for stability at high UTD ratio.
2. **Tapered 3-layer network** — the halving pattern (512→256→128) is a specific architectural choice that affects gradient flow.
3. **SiLU activation** — SiLU has smoother gradients than ReLU near zero; matters for high-dim tasks.
4. **Policy delay=4 for FastSAC** — this is a TD3-style trick applied to SAC. Major algorithmic change we're missing entirely.
5. **Obs normalization** — running normalization helps with high-dim obs (67-dim for humanoid).

### From FastDSAC paper:
1. **Tiny buffer (51K)** — forces the agent to learn from recent data. Our 1M buffer dilutes fresh transitions.
2. **Batch size 32K** — 4x more data per gradient step means better gradient estimates.
3. **Alpha init 0.001** — starting with high alpha (1.0) wastes early training on max-entropy exploration.
4. **Decomposed critic loss** — the variance gradient is fundamentally different from standard NLL.

---

## Recommended Fix Priority

1. **tau** — single config change, huge potential impact
2. **Network architecture** — needs new tapered MLP encoder or head changes
3. **Policy delay for FastSAC** — needs algo change
4. **Alpha init for FastDSAC** — single config change
5. **Buffer size / batch size for FastDSAC** — config changes
6. **SiLU activation** — config change if encoder supports it
7. **Obs normalization** — complex, contradicts our off-policy lesson
8. **Critic loss formulation** — algo rewrite for FastDSAC
