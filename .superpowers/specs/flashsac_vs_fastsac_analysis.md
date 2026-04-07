# FlashSAC vs FastSAC: Comparative Analysis & Benchmarking Plan

**Document Date:** 2026-04-07  
**Paper:** Kim et al., "FlashSAC: Fast and Stable Off-Policy RL for High-Dimensional Robot Control" (2026)  
**FlashSAC Repo:** https://github.com/Holiday-Robot/FlashSAC  
**FastSAC Repo:** This project (`jax_rl/algos/fast_sac.py`)

---

## Executive Summary

FlashSAC is a follow-up to Seo et al.'s FastSAC (2025). Both solve the same problem (speed + stability in off-policy RL) but with different trade-offs:

| Aspect | FastSAC | FlashSAC |
|--------|---------|----------|
| **Framework** | JAX | PyTorch + IsaacLab |
| **Core Innovation** | Tapered networks + policy delay | Inverted residual blocks + norm bounding |
| **Stability Mechanism** | Policy delay every 4 critic updates | Explicit weight/feature/gradient norm constraints |
| **Entropy Target** | Fixed α_init=0.001, target_entropy_scale varies | Unified σ_target=0.15 (no per-task tuning) |
| **Exploration** | Standard OU noise | Zeta-distributed noise repetition |
| **Distributional Critic** | C51 with 51 atoms | C51 with configurable atoms (default ~256) |
| **Reward Normalization** | Vanilla scaling | Adaptive scaling (Eq. 6: bounds returns by running variance) |
| **Batch Norm** | Post-RMS norm only | Pre-activation batch norm + post-RMS norm |
| **Weight Norm** | Not discussed | Explicit constraint (projects to unit sphere) |
| **Tested On** | Go2 humanoid, CheetahRun, HumanoidRun, etc. | 60+ tasks across IsaacLab, DMControl, MuJoCo, etc. |
| **Best Eval** | Go2: 276.5 @ 18M steps | Humanoid climbing: 4 hrs vs PPO 20 hrs |
| **Wall-Clock Scaling** | Good (large batch, fewer updates) | Excellent (order of magnitude faster sim-to-real) |

---

## Technical Comparison

### 1. Architecture

#### FastSAC (JAX)
```
Actor:
  input [obs_dim] → linear(512/256/128) + ReLU → linear(256/128/64) + ReLU → mean/logstd heads
  
Critic (C51):
  input [obs_dim + action_dim] → linear(hidden_dim) + activation → DistributionalQHead
  - DistributionalQHead: outputs logits over 51 atoms
  - Cross-entropy loss between predicted and projected target distribution
```

**Key: Tapered networks (512→256→128)**

#### FlashSAC (PyTorch)
```
Actor:
  input [obs_dim] → Embedder(linear + batch_norm + ReLU) 
                 → [FlashSACBlock]^num_blocks (inverted residual with batch norm)
                 → UnitRMSNorm → NormalTanhPolicy head

Critic:
  input [obs_dim + action_dim] → EnsembleFlashSACEmbedder
                               → [EnsembleFlashSACBlock]^num_blocks (batch norm + residuals)
                               → EnsembleUnitRMSNorm → EnsembleCategoricalValue (C51)
                               
Key: Inverted residual blocks (expand→ReLU→project+residual) + pre-activation batch norm
```

**Key: Residual architecture with dual batch norm (pre-activation + post-RMS)**

#### Architectural Differences
| Component | FastSAC | FlashSAC |
|-----------|---------|----------|
| **Block Type** | Linear layers | Inverted residual blocks |
| **Batch Norm** | Post-RMS only | Pre-activation + post-RMS |
| **Condition Number** | ~10-14 | ~2-4 (Figure 9 shows 5-10x reduction) |
| **Depth** | 2-3 layers | 4-6 layers (configurable) |
| **Width** | Tapered (512→256→128) | Fixed hidden_dim per block |
| **Network Scaling** | Limited by stability | Explicit norm bounding enables larger models |

**FlashSAC Advantage:** The inverted residual blocks + norm constraints allow 6-layer networks without instability. FastSAC's tapered networks are simpler but hit a capacity ceiling.

---

### 2. Training Loop

#### FastSAC (JAX - `train_offpolicy.py`)
```python
for step in range(total_steps):
    # Collect transitions (parallel envs)
    trajectories = env.step(policy_fn(state))
    buffer.add(trajectories)
    
    # Gradient updates (UTD = 8)
    for _ in range(8):
        batch = buffer.sample(batch_size=8192)
        # Critic update
        q1_loss = cross_entropy(Q1(s,a), projected_target_dist)
        q2_loss = cross_entropy(Q2(s,a), projected_target_dist)
        # Actor update (every 4 critic steps)
        actor_loss = E[α * logπ(a|s) - min(Q1, Q2)(s,a)]
        # Temperature update
        alpha_loss = -α * (logπ + target_entropy)
```

**Characteristics:**
- UTD ratio: 8 (fixed)
- Batch size: 8192 (configurable)
- Policy delay: every 4 critic updates (fixed)
- No explicit norm bounding
- Reward scaling: standard (divide by max observed return)

#### FlashSAC (PyTorch - `flash_rl/agents/flashSAC/update.py`)
```python
for step in range(total_steps):
    # Collect transitions (parallel envs via IsaacLab)
    trajectories = env.step(policy_fn(state))
    buffer.add(trajectories)
    
    # Gradient updates (UTD = 2, but multiple gradient calls)
    for _ in range(updates_per_step):
        batch = buffer.sample(batch_size=2048)
        # Critic update
        q_loss = cross_entropy(Q(s,a), projected_target_dist) + weight_norm_penalty
        # Actor update (every N steps, configurable)
        actor_loss = E[σ_target * logπ - Q] + weight_norm_penalty
        # Temperature update (fixed entropy target)
        alpha_loss = -(log_alpha) * (entropy - target_entropy)
        
        # Weight normalization (explicit constraint)
        for param in networks:
            param /= ||param|| + ε  # project to unit sphere
```

**Characteristics:**
- UTD ratio: ~2 (very infrequent updates, fewer steps)
- Batch size: 2048 (one-quarter of FastSAC)
- Policy delay: configurable (default 4, can be higher)
- **Explicit weight norm constraint** (new)
- **Adaptive reward scaling** (Eq. 6: r̄ = r / max(√(σ²_G + ε), G_max/G_max))
- **Pre-activation batch norm** in blocks (new)

**Key Differences:**
1. **FlashSAC is even more aggressive on update frequency** — only 2 updates per 1024 env steps vs FastSAC's 8
2. **Norm bounding is explicit** — FlashSAC manually projects weights after each step; FastSAC relies on optimizer momentum
3. **Reward scaling is adaptive** — FlashSAC bounds returns dynamically; FastSAC uses fixed scaling
4. **Entropy target is unified** — FlashSAC σ_target=0.15 for all tasks; FastSAC has per-task α_init

---

### 3. Hyperparameter Defaults

#### FastSAC (from `env_presets.py`)
```python
_FAST_SAC_BASE_ALGO = FastSACConfig(
    tau=0.125,                     # target update rate (fast, 0.125 vs SAC's 0.005)
    target_entropy_scale=0.0,      # H_target = 0 (no entropy regularization by default)
    alpha_lr=3e-4,
    alpha_init=0.001,              # start near-zero
    max_std=1.0,                   # actor std cap (log_std_max = 0 → std = 1)
    batch_size=8_192,
    grad_updates_per_step=8,       # 8 gradient updates per environment step
    hidden_dim=(512, 256, 128),    # tapered
    critic_hidden_dim=(768, 384, 192),
    activation="swish",
    q_layer_norm=True,
    policy_delay=4,                # actor every 4th critic update
)

_FAST_SAC_BASE_CFG = TrainConfig(
    total_timesteps=100_000_000,
    num_envs=1024,
    episode_length=1000,
    lr=3e-4,
    reward_scaling=1.0,
    gamma=0.97,                    # 0.97 for locomotion (not 0.99)
    num_eval_episodes=5,
    handle_truncation=True,
)
```

#### FlashSAC (from `flash_rl/agents/flashSAC/agent.py`)
```python
@dataclass
class FlashSACConfig:
    # Actor
    actor_num_blocks: int         # 4-6 (configurable)
    actor_hidden_dim: int         # 256-512
    actor_bc_alpha: float         # for behavioral cloning regularization
    actor_noise_zeta_mu: float    # Zeta distribution mean for noise repeat
    actor_noise_zeta_max: int     # max repeat length
    actor_update_period: int      # every N critic steps

    # Critic
    critic_num_blocks: int        # 4-6
    critic_hidden_dim: int        # 512-768
    critic_num_bins: int          # typically 256 (vs FastSAC's 51)
    critic_min_v: float           # support bounds
    critic_max_v: float
    critic_target_update_tau: float  # ~0.005 (slow target update)

    # Temperature
    temp_initial_value: float     # α_init (auto-tuned)
    temp_target_sigma: float      # σ_target = 0.15 (unified)
    temp_target_entropy: float    # computed from σ_target

    # Training
    buffer_max_length: int        # 10M (large buffer)
    sample_batch_size: int        # 512-2048
    learning_rate_*: float        # warmup + cosine decay

    gamma: float                  # 0.97 (same as FastSAC)
    n_step: int                   # n-step returns (default 1)
```

**Key Hyperparameter Differences:**

| Param | FastSAC | FlashSAC |
|-------|---------|----------|
| **Batch Size** | 8192 | 512-2048 |
| **Grad Updates Per Step** | 8 | ~2 (configurable) |
| **Num Atoms** | 51 | 256 (4x more) |
| **Actor Blocks** | 3 (tapered) | 4-6 (uniform width) |
| **Critic Blocks** | 3 (tapered) | 4-6 (uniform width) |
| **Target Update τ** | 0.125 (fast) | 0.005 (slow) |
| **Entropy Target** | Per-task α_init | Unified σ=0.15 |
| **Noise** | Standard Gaussian | Zeta-distributed repetition |
| **Buffer Size** | 4M | 10M |
| **Policy Delay** | 4 | 4-8 (configurable) |

**Interpretation:**
- FastSAC: high throughput, frequent updates (8×), large batch (8K), fast target updates (τ=0.125)
- FlashSAC: high stability, infrequent updates (2×), medium batch (2K), slow target updates (τ=0.005), more expressive critic (256 atoms)

---

### 4. Stabilization Mechanisms

#### FastSAC
1. **Policy Delay:** Update actor every 4th critic update (standard SAC trick)
2. **Clipped Double Q:** min(Q1, Q2) for conservative value estimates
3. **Layer Norm on Critic:** q_layer_norm=True prevents activation saturation
4. **Entropy Temperature:** Auto-tuned via Lagrangian objective
5. **Large Batch Size:** 8192 smooths gradient estimates

**Design Philosophy:** Scale the system (larger batch, bigger model) while being conservative with updates (policy delay, double Q).

#### FlashSAC
1. **Weight Normalization:** Project weight vectors to unit sphere after each step
2. **Pre-Activation Batch Norm:** Normalize before ReLU in each block
3. **Post-RMS Norm:** Normalize before value head
4. **Inverted Residual Blocks:** Gradient flow stabilization (expand→nonlinearity→project)
5. **Adaptive Reward Scaling:** Dynamically bound returns by running variance
6. **Noise Repetition:** Temporal correlation (Zeta-distributed) for coherent exploration
7. **Unified Entropy Target:** Fixed σ_target=0.15 (no per-task tuning)

**Design Philosophy:** Explicitly control activation distributions and gradient flow throughout training via norm constraints, enabling larger models without instability.

**Comparison:**
- FastSAC: **implicit stability** via conservative updates and large batch smoothing
- FlashSAC: **explicit stability** via norm bounding at every layer

---

## Benchmarking Plan

### Objective
Compare FlashSAC and FastSAC on **high-dimensional locomotion and manipulation** to determine:
1. Which converges faster (wall-clock time)?
2. Which achieves higher asymptotic performance?
3. Which is more stable (lower variance across seeds)?
4. How do they scale with model capacity?

### Test Suite

#### Phase 1: Direct Comparison (Same Env, Different Algos)
**Environment:** Go2WarpJoystickFlat (our current SOTA)

| Config | FastSAC | FlashSAC |
|--------|---------|----------|
| **Total Steps** | 100M | 100M |
| **Num Envs** | 1024 | 1024 (match throughput) |
| **Batch Size** | 8192 | 2048 × 4 sub-batches (to match throughput) |
| **Wall-Clock Budget** | 2 hrs | 2 hrs |
| **Seeds** | 3 (6000, 6001, 6002) | 3 (same seeds) |
| **Eval Interval** | Every 1M steps | Every 1M steps |

**Command:**
```bash
# FastSAC (baseline — already have results)
uv run python train_offpolicy.py \
  --env Go2WarpJoystickFlat --algo fast_sac \
  --total-timesteps 100000000 --num-envs 1024 \
  --seed 6000

# FlashSAC (TBD — need implementation)
uv run python train_offpolicy.py \
  --env Go2WarpJoystickFlat --algo flashsac \
  --total-timesteps 100000000 --num-envs 1024 \
  --seed 6000
```

#### Phase 2: Stability Under Distribution Shift
**Environment:** CheetahRun (lower-dim, easier to train; good for ablation)

| Config | Details |
|--------|---------|
| **Steps** | 5M (shorter run, faster feedback) |
| **Seeds** | 5 (to measure variance) |
| **Metrics** | {mean eval, std eval, wall-clock time, sample efficiency} |

#### Phase 3: Reward Normalization Interaction (Optional)
Since FlashSAC uses adaptive reward scaling, test interaction with our per-frame obs-norm:

| Config | Details |
|--------|---------|
| **Env** | CheetahRun with frame-stack=3 |
| **Flag** | Compare `--obs-norm` on/off with both algos |
| **Question** | Does FlashSAC's reward norm + our obs-norm compound or interfere? |

#### Phase 4: Vision-Based (Optional)
**Environment:** DMControl vision task (e.g., Walker Run)

| Config | Details |
|--------|---------|
| **Agent Obs** | Vision (84×84 RGB) |
| **Critic Obs** | Vision (same) |
| **Baseline** | DrQ-v2 |
| **Question** | Does FlashSAC's norm bounding help with vision? |

---

### Metrics to Track

For each run, log:
1. **Learning Curve:** eval_return vs env_steps (plot with shaded std over seeds)
2. **Wall-Clock Time:** eval_return vs wall-clock seconds
3. **Sample Efficiency:** steps to reach 90% of peak performance
4. **Stability:** coefficient of variation (std / mean) across seeds
5. **Training Dynamics:** 
   - Critic loss trajectory
   - Actor loss trajectory
   - α (entropy coefficient) evolution
   - Weight norm statistics (max, mean)
6. **Gradient Health:**
   - Gradient norm per layer (FlashSAC: should be bounded)
   - Condition number of Hessian (if feasible)

---

### Expected Outcomes

**Hypothesis A: FlashSAC wins on high-dim tasks** (dexterous, humanoid)
- Paper shows 2-5× speedup on manipulation
- Our Go2 may benefit from explicit norm bounding

**Hypothesis B: FastSAC wins on moderate-dim tasks** (CheetahRun, WalkerWalk)
- Tapered networks are efficient
- Fewer blocks = less overhead
- Our Go2 results (276.5) already competitive

**Hypothesis C: FlashSAC is more robust to hyperparameter changes**
- Fixed σ_target vs per-task α_init
- Unified entropy target should reduce tuning burden

---

### Implementation Considerations

#### Option 1: Implement FlashSAC in JAX (Porting Cost: Medium)
**Pros:**
- Integrate into `jax_rl/algos/flashsac.py`
- Reuse existing training infra (train_offpolicy.py)
- Direct comparison on same codebase

**Cons:**
- Port PyTorch to JAX (batch norm, weight norm, zeta noise)
- Debug numerical differences

**Files to Modify:**
- Create `jax_rl/algos/flashsac.py` (copy FastSAC, add norm bounding)
- Create `jax_rl/networks/heads/q_categorical.py` (C51 with 256 atoms)
- Add to `env_presets.py` FLASHSAC_PRESETS

#### Option 2: Benchmark Against Their PyTorch Implementation (Lower Cost)
**Pros:**
- Use their reference implementation (no porting risk)
- Ensure faithful comparison

**Cons:**
- Separate Python environments (JAX vs PyTorch)
- Different hardware/library overheads
- Harder to isolate FlashSAC's improvements

**Recommendation:** Start with **Option 2** (external benchmark) to validate the paper's claims on Go2, then **Option 1** (JAX port) if the results are promising.

---

### Success Criteria

| Criterion | Success | Failure |
|-----------|---------|---------|
| **Wall-Clock** | FlashSAC <2 hrs to reach 200 reward on Go2 | >3 hrs |
| **Peak Eval** | FlashSAC ≥ 270 (within 5% of FastSAC's 276) | <250 |
| **Stability** | StdDev across seeds <10% of mean | >15% |
| **Generalization** | Single config works for Go2 + CheetahRun | Needs per-task tuning |

---

## Decision Points

1. **JAX Port or External Benchmark?**
   - Recommend: External benchmark first (2-3 days), then JAX port if promising (1 week)

2. **Include Vision Tasks?**
   - Recommend: Phase 4 optional; prioritize locomotion/manipulation first

3. **Hyperparameter Sensitivity Study?**
   - Recommend: Phase 3 (obs-norm interaction) if FlashSAC wins on Phase 1

4. **Long-Horizon Runs (>20M steps)?**
   - Recommend: Skip for MVP; focus on wall-clock efficiency first

---

## Related Work to Review

- Seo et al. (2025) FastSAC/FastTD3 paper — check what they learned
- Paper's ablations (§6) show explicit decomposition of each norm component
- Inverted residual blocks are borrowed from MobileNet (Howard et al. 2017)

---

## Notes for Future Self

- FlashSAC's code is clean (PyTorch, well-organized)
- They use `torch.compile` for JIT (PyTorch 2.0+)
- Repo has IsaacLab integration already; can benchmark Go2 directly
- Paper says they don't task-specific tune; test this claim on Go2

---

**Next Step:** Decide between JAX port (Option 1) or external benchmark (Option 2).
