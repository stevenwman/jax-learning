# FlashSAC JAX Port — Design Spec

**Date:** 2026-04-07
**Paper:** Kim et al., "FlashSAC: Fast and Stable Off-Policy RL for High-Dimensional Robot Control" (arXiv:2604.04539, April 2026)
**Reference impl:** https://github.com/Holiday-Robot/FlashSAC (PyTorch)
**Goal:** Faithful JAX/Flax port of FlashSAC for benchmarking against our FastSAC on Go2 and dm_control tasks.

---

## Context

FlashSAC is a follow-up to Seo et al.'s FastSAC (2025). Both target fast off-policy RL at scale, but FlashSAC adds explicit stability mechanisms (weight/feature/gradient norm bounding) that enable deeper networks and fewer gradient updates. The paper claims order-of-magnitude wall-clock speedup over PPO on sim-to-real humanoid locomotion, and consistent wins over FastTD3 on high-dimensional tasks.

Our current FastSAC achieves eval 276.5 on Go2WarpJoystickFlat @ 18M steps. FlashSAC's stability improvements (particularly for high-dim tasks) could push this further and reduce tuning burden via their unified entropy target.

**Decision:** Port FlashSAC as a standalone algorithm + training script. Do not modify existing FastSAC or `train_offpolicy.py`. Validate via A/B comparison.

---

## New Files (5 files, 0 modifications)

| File | Purpose |
|------|---------|
| `jax_rl/networks/flash_blocks.py` | FlashSACEmbedder, FlashSACBlock, UnitRMSNorm |
| `jax_rl/configs/flash_sac_config.py` | FlashSACConfig dataclass |
| `jax_rl/algos/flash_sac.py` | FlashSAC algo class (TrainingState, update, init, select_action) |
| `jax_rl/utils/reward_scaling.py` | RewardNormState, update_reward_stats(), scale_reward() |
| `train_flashsac.py` | Standalone training script |

---

## 1. Network Architecture (`flash_blocks.py`)

### FlashSACEmbedder
```
input (D_in) → BatchNorm(D_in, momentum=0.01) → Dense(D_in → D_hidden) [orthogonal, no bias]
```
- BatchNorm normalizes raw inputs (obs for actor, concat(obs,action) for critic)
- Dense: orthogonal init, no bias — weight rows projected to unit norm post-step

### FlashSACBlock (inverted residual, expansion=4)
```
residual = x
x → Dense(D → 4D) [orthogonal, no bias]
  → BatchNorm(4D, momentum=0.01)
  → ReLU
  → Dense(4D → D) [orthogonal, no bias]
  → BatchNorm(D, momentum=0.01)
  → ReLU
x = x + residual
```
- All Dense layers: orthogonal init, no bias
- All BatchNorm: momentum=0.01, eps=1e-5

### UnitRMSNorm
```
x → (x / rms(x)) * scale    [eps=1e-6]
```
- Applied after final block, before policy/value head
- Scale parameter normalized to ‖scale‖₂ = √D post-step

### Full Actor
```
FlashSACEmbedder(obs_dim → hidden_dim)
→ FlashSACBlock × num_blocks
→ UnitRMSNorm(hidden_dim)
→ mean_head: UnitDense(hidden_dim → action_dim) + free bias param
→ logstd_head: UnitDense(hidden_dim → action_dim) + free bias param
   log_std = log_std_min + (log_std_max - log_std_min) * 0.5 * (1 + tanh(raw))
```
- `log_std_min = -10.0`, `log_std_max = 2.0` (hardcoded, matching reference)
- UnitDense = Dense with orthogonal init, no built-in bias, kernel weight-normalized post-step
- Bias is a separate free parameter (NOT weight-normalized)

### Full Critic (×2 for double Q)
```
input: concat(obs, action)
FlashSACEmbedder(obs_dim + action_dim → hidden_dim)
→ FlashSACBlock × num_blocks
→ UnitRMSNorm(hidden_dim)
→ UnitDense(hidden_dim → num_atoms) + free bias param    [C51 logits]
```

**Deliberate simplification:** We use two separate critic networks (like our FastSAC) rather than the reference's ensembled `EnsembleFlashSACBlock` which fuses two Q networks into batched `(2, B, D)` operations. In JAX, two separate `apply()` calls are clean and JIT-friendly. The ensemble optimization is a PyTorch throughput trick — functionally equivalent, and we can add `nn.vmap` later if profiling shows it matters.

### BatchNorm in JAX/Flax
Uses `flax.linen.BatchNorm` with `use_running_average` parameter:
- Training: `model.apply({params, batch_stats}, x, train=True, mutable=['batch_stats'])` → returns `(output, {'batch_stats': updated_stats})`
- Eval: `model.apply({params, batch_stats}, x, train=False)` → returns output only

Batch stats (running mean/var) stored in TrainingState as separate fields.

---

## 2. Weight Normalization

Post-optimizer-step function applied to all network parameters:

**Dense kernels:** Project each row to unit L2 norm.
```python
kernel = kernel / max(‖kernel‖₂ along axis=-1, 1e-8)
```

**BatchNorm scale+bias:** Project joint (scale, bias) vector to ‖·‖₂ = √D.
```python
sqsum = sum(scale² + bias²)
factor = √D / √(sqsum + 1e-8)
scale, bias = scale * factor, bias * factor
```

**RMSNorm scale:** Project to ‖scale‖₂ = √D.
```python
sqsum = sum(scale²)
factor = √D / √(sqsum + 1e-8)
scale = scale * factor
```

Implemented as `normalize_weights(params)` using `jax.tree_util.tree_map_with_path`. Path matching rules:
- Leaf name `kernel` → unit-norm rows (all Dense layers in embedder + blocks + heads)
- Leaf name `scale` under a `BatchNorm` parent → joint (scale, bias) normalization to √D
- Leaf name `scale` under `UnitRMSNorm` → scale-only normalization to √D
- Leaf name `bias` under `BatchNorm` → handled jointly with scale (see above)
- All other leaves (optimizer state, free bias params on heads, etc.) → untouched

The mean/std head biases and the C51 value head bias are **free parameters** — not weight-normalized. Only their kernel matrices are normalized.

---

## 3. Training State

```python
@flax.struct.dataclass
class TrainingState:
    # Core (same shape as FastSAC)
    actor_params: Any
    actor_opt_state: optax.OptState
    q1_params: Any
    q2_params: Any
    q_opt_state: Any
    target_q1_params: Any
    target_q2_params: Any
    log_alpha: jnp.ndarray
    alpha_opt_state: optax.OptState
    key: jax.Array
    update_count: jnp.ndarray

    # BatchNorm running stats
    actor_batch_stats: Any
    q1_batch_stats: Any
    q2_batch_stats: Any
    target_q1_batch_stats: Any
    target_q2_batch_stats: Any

    # Reward normalizer
    reward_norm_state: Any    # RewardNormState pytree

    # Noise repetition
    noise_state: Any          # NoiseState pytree
```

---

## 4. Update Logic (critic loss — cross-batch pattern)

FlashSAC concatenates obs + next_obs into a 2B batch for shared BatchNorm statistics. This ensures the critic's running stats are computed over a combined distribution of current and next states.

**Update order matches reference:** actor → temperature → critic → target EMA (per step).

```
Actor update (every policy_delay steps):
  1. Concat: [obs; next_obs] → 2B batch through actor (train=True for BN stats)
  2. Take first half for loss computation
  3. loss = mean(alpha * log_prob - min(Q1, Q2))
  4. Optional BC regularization: loss += bc_alpha * |Q|.mean() * MSE(action, batch_action)
  5. Apply weight normalization to updated actor params

Temperature update (every policy_delay steps, after actor):
  - loss = alpha * (entropy - target_entropy)

Critic update (every step, uses freshly-updated actor):
  1. Sample next_action from updated actor (no grad, train=False)
  2. Concat: obs_all = [obs; next_obs], act_all = [action; next_action]  → shape (2B, ...)
  3. Target critic forward on obs_all (train=True, mutable) → split, take next-state half
  4. C51 projection (FlashSAC-specific, NOT reusing existing project_distribution):
     - target_bin_values = reward + gamma^n_step * (bin_values - alpha*log_prob) * (1-done)
     - This differs from FastSAC which adjusts the reward instead of the support
  5. Online critic forward on obs_all (train=True, mutable) → split, take current-state half
  6. Cross-entropy loss between projected target and online logits
  7. Apply weight normalization to updated critic params

Target update (after critic):
  - Polyak: target_params = tau * online + (1-tau) * target  [tau=0.01]
  - Copy online batch_stats to target (use fresh running stats)
```

**Note on `jax.lax.cond` + BatchNorm:** The policy_delay branching (do actor update vs skip) requires both branches to return pytrees with identical structure. The "skip" branch must return dummy batch_stats of the same shape. This is the same pattern as FastSAC's `_skip_actor_alpha_update` but extended to include batch_stats — pass through the existing batch_stats unchanged.

**Note on 2B cross-batch:** BatchNorm running stats are intentionally computed over the combined 2B distribution (current + next states). This is a deliberate design choice from the paper, not an implementation convenience — it prevents train/inference distribution mismatch in the running statistics.

---

## 5. Reward Scaling (`reward_scaling.py`)

Pure-functional JAX implementation of adaptive reward normalization.

```python
@flax.struct.dataclass
class RewardNormState:
    G_r: jnp.ndarray       # (num_envs,) running discounted return per env
    G_r_max: jnp.ndarray   # scalar: all-time max |G_r|
    G_mean: jnp.ndarray    # scalar: Welford running mean
    G_var: jnp.ndarray     # scalar: Welford running variance
    G_count: jnp.ndarray   # scalar: sample count
```

**Per env step** (called in train script collection loop):
```python
G_r = gamma * (1 - done) * G_r + reward
G_r_max = max(G_r_max, max(|G_r|))
# Welford update on G_r
```

**Per training batch** (called inside critic loss):
```python
r_scaled = reward / max(√(G_var + eps), G_r_max / G_max)
```
where `G_max = 5.0` — guarantees scaled returns fit within critic support [-5, 5].

---

## 6. Exploration: Noise Repetition

```python
@flax.struct.dataclass
class NoiseState:
    noise: jnp.ndarray      # (num_envs, action_dim)
    count: jnp.ndarray      # (num_envs,) steps since last resample
    repeat_n: jnp.ndarray   # (num_envs,) current repeat length
```

During training collection:
```python
reinit = (count == 0) | (count >= repeat_n)
new_noise = random.normal(key, (num_envs, action_dim))
new_n = zeta_sample(key, mu=2.0, max_n=16)  # inverse CDF, precomputed

noise = where(reinit[:, None], new_noise, old_noise)
repeat_n = where(reinit, new_n, old_repeat_n)
count = where(reinit, 1, count + 1)

action = tanh(mean + std * noise)
```

Only active during training. Eval uses deterministic `tanh(mean)`.

Zeta CDF is precomputed once at init: `P(k) ∝ k^{-2}` for k=1..16, normalized and cumsum'd.

---

## 7. Unified Entropy Target

```python
target_entropy = 0.5 * action_dim * log(2 * pi * e * sigma_target²)
```
where `sigma_target = 0.15`. Fixed across all tasks — no per-env tuning.

---

## 8. Config Defaults

```python
@dataclass
class FlashSACConfig:
    # Architecture
    num_blocks: int = 2
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    expansion: int = 4
    num_atoms: int = 101
    v_min: float = -5.0
    v_max: float = 5.0

    # Training
    tau: float = 0.01
    policy_delay: int = 2
    batch_size: int = 2048
    buffer_size: int = 1_000_000
    min_buffer_size: int = 10_000
    grad_updates_per_step: int = 1
    n_step: int = 1                        # n-step returns (default single-step)

    # Temperature
    alpha_init: float = 0.01
    sigma_target: float = 0.15
    alpha_lr: float = 3e-4

    # Actor regularization
    bc_alpha: float = 0.0                  # behavioral cloning weight (0 = disabled)

    # Reward scaling
    normalize_reward: bool = True
    G_max: float = 5.0

    # Exploration
    noise_zeta_mu: float = 2.0
    noise_zeta_max: int = 16

    # LR schedule (warmup → cosine decay)
    lr_init: float = 3e-4                  # initial LR (before warmup)
    lr_peak: float = 3e-4                  # peak LR (after warmup)
    lr_end: float = 1.5e-4                 # final LR (after decay)
    lr_warmup_frac: float = 1e-6           # fraction of total steps for warmup
    lr_decay_frac: float = 1.0             # fraction of total steps for decay

    # Weight norm
    weight_norm: bool = True
```

---

## 9. Training Script (`train_flashsac.py`)

Standalone script. Copies boilerplate from `train_offpolicy.py` (CLI, logging, eval, checkpointing) but with a clean collection + gradient loop:

```
Collection loop:
  1. select_action_with_noise(state, obs, key) → action, new_noise_state
  2. env.step(action) → next_obs, reward, done, ...
  3. update_reward_stats(reward_norm_state, reward, done, gamma)
  4. buffer.add(transition)

Gradient loop (every step):
  for _ in range(grad_updates_per_step):  # default 1
    batch = buffer.sample(batch_size)
    batch['reward'] = scale_reward(reward_norm_state, batch['reward'])
    state, metrics = algo.update(state, batch)
```

CLI mirrors `train_offpolicy.py` flags where applicable: `--env`, `--seed`, `--total-timesteps`, `--num-envs`, etc.

---

## 10. Reused Components

| Component | Source | Notes |
|-----------|--------|-------|
| C51 support/q-value | `jax_rl/utils/distributional.py` | `make_support`, `logits_to_q` (reused). `project_distribution` NOT reused — FlashSAC places entropy inside the gamma term, needs a new `flash_project_distribution` in the algo file |
| Gaussian sampling | `jax_rl/networks/distributions.py` | `sample_gaussian` |
| Replay buffer | `jax_rl/buffers/` | Existing uniform replay buffer |
| Env creation | `jax_rl/envs/` | `make_env` factory |
| Logging / eval | `train_offpolicy.py` | Copy boilerplate |

---

## 11. Verification Plan

### Smoke test (CPU, no GPU needed)
1. `uv run python train_flashsac.py --env CartpoleBalance --total-timesteps 50000 --num-envs 4` — verify it runs without error
2. Check metrics are logged (critic_loss, actor_loss, entropy, alpha, reward_scale)
3. Verify weight norms stay bounded (log max kernel norm per step)

### Unit tests
- `test_flash_blocks.py`: verify FlashSACBlock output shape, residual connection, BatchNorm state updates
- `test_reward_scaling.py`: verify scale_reward bounds outputs within [-G_max, G_max]
- `test_weight_norm.py`: verify normalize_weights projects kernels to unit norm

### A/B benchmark (GPU required)
```bash
# FastSAC baseline
uv run python train_offpolicy.py --env CheetahRun --algo fast_sac --total-timesteps 5000000 --seed 100

# FlashSAC
uv run python train_flashsac.py --env CheetahRun --total-timesteps 5000000 --seed 100
```
Compare: eval_return vs wall-clock, eval_return vs env_steps, training stability (critic loss variance).

### Go2 benchmark (GPU required)
```bash
uv run python train_flashsac.py --env Go2WarpJoystickFlat --total-timesteps 100000000 --seed 6000
```
Target: match or exceed FastSAC's 276.5 eval.
