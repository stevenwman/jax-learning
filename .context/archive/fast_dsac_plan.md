# FastDSAC Implementation Plan

**Paper:** FastDSAC: Unlocking the Potential of Maximum Entropy RL (arXiv:2603.12612)
**Status:** Implementation planned
**Goal:** Replace our failed FastSAC (C51 + SAC, plateaued at 375) with FastDSAC (continuous Gaussian critic + DEM)

## What FastDSAC changes from FastSAC

### 1. Gaussian Distributional Critic (replaces C51)

**Network:** `GaussianQHead` — concat(obs, action) → MLP → two outputs:
- `mean`: scalar Q-value estimate
- `variance`: `softplus(raw_output)` for positivity

**Loss (Gaussian NLL with decomposed gradient):**
```
L_critic = E[(y_q - Q_ψ)² / σ²_ψ + log σ²_ψ]
```

Decomposed gradient (Eq. 8 from paper):
```
∇_ψ J ≈ ω · E[
  -(y_q_min - Q_ψ) / (σ²_ψ + ε) · ∇_ψ Q_ψ       # mean gradient
  -((y_z_sample - Q_ψ)² - σ²_ψ) / (σ³_ψ + ε) · ∇_ψ σ_ψ  # variance gradient
]
```

Where:
- `y_q_min = r + γ(min(Q1', Q2') - α log π(a'|s'))` — conservative target (same as SAC)
- `y_z_sample ~ N(Q_target, σ²_target)` — sample from target distribution
- `ω = E_batch[σ²_ψ]` — running mean of critic variance (gradient scaling)
- `ε = 1e-6`

### 2. Dimension-wise Entropy Modulation (DEM)

**Actor outputs three heads from shared features:**
- `mean` [batch, action_dim] — action means (existing)
- `log_std` [batch, action_dim] — base log-scale (existing)
- `dem_logits` [batch, action_dim] — DEM weights (NEW)

**Weight computation:**
```python
w_i = softmax(dem_logits * β_e / τ) * action_dim
# w_i sums to action_dim (budget constraint: mean(w) = 1)
```

**Std modulation:**
```python
σ_i = w_i * exp(log_std_i)  # DEM redistributes exploration budget
```

- `τ` (temperature): controls sparsity. Higher → more uniform. Start with 1.0.
- `β_e`: per-env scaling factor, fixed at init. `β_e ~ U[β_min, β_max]` where β_min=0.01, β_max=2.0.
- No separate DEM loss — weights learn through actor gradient via reparameterization trick.

### 3. Other changes from FastSAC

- **Target entropy = 0** (not -dim(A)). Prevents alpha collapse.
- **AdamW** with β=(0.9, 0.95), weight_decay=1e-4 (not plain Adam)
- **No C51 projection** — removes all distributional.py / DistributionalQHead dependencies
- **No variance clipping** on critic (unlike DSAC-T)

## New files

1. `jax_rl/algos/fast_dsac.py` — FastDSAC algorithm (init, update, select_action)
2. `jax_rl/networks/heads/q_gaussian.py` — GaussianQHead (mean + variance output)
3. `jax_rl/configs/fast_dsac_config.py` — FastDSACConfig dataclass
4. `train_fast_dsac.py` — training script

## Modified files

5. `jax_rl/networks/heads/gaussian.py` — add optional `dem_logits` output head
6. `jax_rl/configs/env_presets.py` — add `get_fast_dsac_preset()`

## What we reuse unchanged

- `jax_rl/buffers/jax_replay_buffer.py` — same replay buffer
- `jax_rl/utils/eval.py` — same eval (with warmup_eval)
- `jax_rl/networks/encoders/mlp.py` — same MLP encoder
- Polyak target update, tanh squashing, alpha auto-tuning logic

## Config (FastDSACConfig)

```python
@dataclass
class FastDSACConfig:
    # SAC core
    tau: float = 0.005
    target_entropy: float = 0.0  # NOT -dim(A)
    alpha_lr: float = 3e-4

    # Gaussian distributional critic
    critic_variance_eps: float = 1e-6

    # DEM
    dem_temperature: float = 1.0
    beta_min: float = 0.01
    beta_max: float = 2.0

    # Training
    buffer_size: int = 1_000_000
    min_buffer_size: int = 25_000
    batch_size: int = 8_192  # paper uses 32K but we're on single 5080
    grad_updates_per_step: int = 12

    # Network
    hidden_dim: tuple[int, ...] = (512, 512)
    activation: str = "relu"
    q_layer_norm: bool = True

    # Optimizer
    weight_decay: float = 1e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.95

    # LR decay
    lr_end: float = 3e-5
```

## Preset (CheetahRun)

```python
_FAST_DSAC_BASE_CFG = TrainConfig(
    total_timesteps=100_000_000,
    num_envs=1024,
    episode_length=1000,
    lr=3e-4,
    gamma=0.99,
    num_eval_episodes=5,
    ppo=None,
)
```

## Success criteria

- FastDSAC on CheetahRun should significantly exceed FastSAC's 375 plateau
- Target: approach or exceed FastTD3's 880
- Alpha should NOT collapse to near-zero (target_entropy=0 prevents this)
- Critic variance should stabilize (not diverge or collapse)
