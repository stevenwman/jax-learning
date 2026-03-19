# FastTD3 / FastSAC Implementation Plan

Paper: [FastTD3](https://arxiv.org/abs/2505.22642), [FastSAC Humanoid](https://arxiv.org/abs/2512.01996)
GitHub: [younggyoseo/FastTD3](https://github.com/younggyoseo/FastTD3)

## Core Idea

Same TD3/SAC algorithm structure, but:
1. **C51 distributional critic** instead of scalar Q — predicts return distribution, not just mean
2. **Large batch sizes** (8K-32K) with massive parallel envs (4K-16K)
3. **Q averaging** instead of min — reduces underestimation bias at scale
4. **Low replay ratio** (2-8 updates per step) — large batches already mix well
5. **LR cosine decay** (3e-4 → 3e-5)

Result: humanoid locomotion in <3 hours on single GPU, sim-to-real in 15 minutes.

## What Changes vs Vanilla TD3/SAC

| Component | Vanilla | Fast | Implementation impact |
|-----------|---------|------|----------------------|
| Critic output | Scalar Q | 51-atom distribution | New `DistributionalQHead` |
| Critic loss | MSE on scalar | Cross-entropy on projected atoms | New C51 projection + loss |
| Q aggregation | min(Q1, Q2) | avg(Q1, Q2) | Config flag, trivial |
| Batch size | 256-512 | 8K-32K | Config only |
| Replay ratio | 1-8 | 2-4 (lower than SAC!) | Config only |
| LR | Constant 3e-4 | Cosine 3e-4 → 3e-5 | optax schedule |
| Exploration noise | 0.1 | 0.2-0.4 | Config only |
| LayerNorm | Optional | Required | Already have it |
| Actor loss | -mean(Q1(s,a)) | -mean(E[Z1(s,a)]) | Extract scalar from distribution |

**Everything else is identical**: Polyak targets, policy delay, target noise smoothing, replay buffer, truncation masking.

## C51 Distributional Critic — The Core Change

### What it is
Instead of Q(s,a) → scalar, predict Z(s,a) → categorical distribution over 51 atoms.

```
Support: z_i = V_min + i * Δz, i ∈ [0, 50]
         V_min = -10, V_max = 10, Δz = 0.4

Network output: logits of shape (batch, 51)
Probabilities:  p(s,a) = softmax(logits)
Expected Q:     Q(s,a) = Σ p_i * z_i  (dot product with support)
```

### Why it helps
Scalar Q collapses the full return distribution to a single number. With large batches, this throws away information. The distribution:
- Captures multimodality (multiple possible outcomes)
- Provides better gradient signal (richer loss landscape)
- Stabilizes learning with large batch sizes

### Bellman update (the hard part)
Standard Bellman: `Q_target = r + γ * Q(s', a')`
Distributional: `Z_target = r + γ * Z(s', a')` — but shifted atoms don't align with support.

**C51 Projection**: redistribute shifted target atoms back onto the fixed support via linear interpolation.

```
For each target atom z_j with probability p_j:
  1. Compute shifted atom: ẑ_j = r + γ * z_j
  2. Clip to [V_min, V_max]
  3. Find neighbors: b = (ẑ_j - V_min) / Δz, l = floor(b), u = ceil(b)
  4. Split probability: m[l] += p_j * (u - b), m[u] += p_j * (b - l)
```

Loss: cross-entropy between projected target distribution `m` and predicted `softmax(logits)`.

### Twin Q with averaging
For actor loss and target computation:
```python
# Average the expected Q values (not the distributions)
q1 = jnp.sum(jax.nn.softmax(q1_logits) * support, axis=-1)
q2 = jnp.sum(jax.nn.softmax(q2_logits) * support, axis=-1)
q_avg = 0.5 * (q1 + q2)
```

For critic loss, each Q network has its own distributional target — no averaging in distribution space.

## New Files

1. **`jax_rl/networks/heads/q_distributional.py`** (~60 lines)
   - `DistributionalQHead(nn.Module)`: concat(obs, action) → MLP → Dense(num_atoms)
   - Same structure as QHead but final layer outputs 51 logits instead of 1 scalar
   - `q_value(logits, support)` static method for expected Q

2. **`jax_rl/utils/distributional.py`** (~80 lines)
   - `project_distribution(target_probs, target_support, v_min, v_max, num_atoms)` → projected probs
   - `categorical_td_loss(q_logits, projected_target)` → cross-entropy loss
   - `make_support(v_min, v_max, num_atoms)` → atom values array

3. **`jax_rl/configs/fast_td3_config.py`** (~30 lines)
   - Extends TD3Config with: num_atoms, v_min, v_max, q_aggregation, lr_end

4. **`jax_rl/algos/fast_td3.py`** (~250 lines)
   - Same structure as td3.py but critic uses distributional loss
   - Actor loss uses expected Q from distribution

5. **`train_fast_td3.py`** — copy train_td3.py, add LR decay schedule

FastSAC: same distributional critic, applied to SAC. Separate `fast_sac.py` after FastTD3 validates.

## Reuse from Existing Code

- `MlpEncoder` — identical
- `DeterministicHead` / `GaussianHead` — identical
- `ReplayBuffer` — identical (numpy, works at any batch size)
- `Polyak soft update` — identical
- `Truncation masking` — identical
- `train_td3.py` structure — copy + modify
- `select_action` — identical (deterministic + noise)

## Config

```python
@dataclass
class FastTD3Config:
    # TD3 core
    tau: float = 0.005
    policy_delay: int = 2
    target_noise_std: float = 0.2
    noise_clip: float = 0.5
    exploration_noise_std: float = 0.2

    # C51 distributional
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    q_aggregation: str = "avg"      # "avg" (FastTD3) or "min" (vanilla)

    # Training (scaled for parallel envs)
    buffer_size: int = 1_000_000
    min_buffer_size: int = 25_000
    batch_size: int = 8_192         # start conservative, scale to 32K
    grad_updates_per_step: int = 4

    # Network
    hidden_dim: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    q_layer_norm: bool = True       # required for stability at scale

    # LR decay
    lr_end: float = 3e-5            # cosine decay target
```

## Benchmark Targets
- CheetahRun: ≥700 at 5M steps, faster wall-clock than vanilla TD3
- WalkerWalk: ≥950 at 5M steps
- HumanoidRun: ≥200 at 5M steps (the real test — vanilla TD3 only got 4.3)

## Implementation Order
1. `DistributionalQHead` + `distributional.py` (C51 math)
2. Unit test: verify projection is correct (atoms shift, probs sum to 1)
3. `fast_td3.py` algorithm
4. `train_fast_td3.py`
5. Smoke test on CartpoleBalance
6. Benchmark on CheetahRun, WalkerWalk, HumanoidRun
7. FastSAC (reuse distributional critic, add entropy)

## SimbaV2 — Deferred
Hyperspherical normalization (L2 norm on features+weights instead of LayerNorm). +11% over baseline SAC. Interesting but not critical — FastTD3 with standard LayerNorm already achieves SOTA. Revisit after FastTD3/FastSAC are validated.

## Key Risk: C51 Projection Numerics
The projection step is the most error-prone part. Common pitfalls:
- Off-by-one in atom indexing
- Probability mass leaking (not summing to 1 after projection)
- Numerical instability at support boundaries (V_min, V_max)
- Wrong gamma application with done/truncation masking

Mitigation: write a standalone test that verifies projection on known inputs before integrating into the training loop.
