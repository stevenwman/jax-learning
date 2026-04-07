# JAX RL Framework — Design Document

## Overview

A modular, JAX-native reinforcement learning library built on Flax Linen, designed to make **efficient robot learning accessible**.

This framework is both a **learning vehicle** and a **lab tool**. Building it is an opportunity to develop deep fluency with JAX, Flax Linen, and RL algorithms by implementing them from scratch. The resulting codebase will be shared across the lab, so every design decision prioritizes interpretability — a new lab member should be able to read a single algorithm file and understand what's happening without chasing abstractions across ten modules.

### Why this framework?
- **Learnable**: Every algorithm is implemented from fundamentals, heavily commented, with clear mappings to the papers. No magic, no "just trust the base class."
- **Accessible**: Clean code the whole lab can read, modify, and extend — not research spaghetti
- **Efficient**: JAX-native with FastTD3/SAC for training in minutes, not hours
- **Robot-first**: Built for the sim-to-real pipeline — train fast in sim, deploy on hardware

### Core design principles
1. **Fast code, clear explanations** — use every JAX trick that improves performance, but comment the non-obvious ones so the lab can learn from them. The code should be efficient *and* educational.
2. Understand the code → modify it → run experiments → deploy on robot
3. Modular encoder+head architecture — swap components without rewriting algorithms
4. Sim-to-real as first-class citizen, not afterthought

### Target workflow
```
Simulation (MuJoCo Playground)
    → Train with PPO/SAC/FastTD3
    → Domain randomization + curriculum
    → Distill to deployable policy (BC)
    → Export for hardware
    → Run on real robot
```

---

## Priority Order

### Algorithms (RL)
| Priority | Algorithm | Notes |
|----------|-----------|-------|
| ⭐⭐⭐ | PPO | On-policy baseline, start here |
| ⭐⭐ | SAC | Off-policy, entropy-regularized |
| ⭐⭐ | FastTD3 | Off-policy, deterministic |
| ⭐ | TD3 | Standard implementation |
| ⭐ | FastSAC | Optimized SAC variant |

### Algorithms (BC / Imitation)
| Priority | Algorithm | Notes |
|----------|-----------|-------|
| ⭐⭐ | MLP BC | Simple behavioral cloning baseline |
| ⭐ | Flow Matching | Generative policy |
| ? | Diffusion | Lower priority, evaluate later |

### Networks
| Priority | Type | Notes |
|----------|------|-------|
| ⭐⭐ | MLP | GaussianMLP, DeterministicMLP, QNetwork |
| ⭐ | CNN | For pixel observations |
| ⭐ | Transformer | Future, for sequence modeling |

### Environments
| Priority | Source | Notes |
|----------|--------|-------|
| ⭐⭐ | dm_control (via MuJoCo Playground) | JAX-native, jittable |
| ⭐ | Custom envs | For sim-to-real research |
| ⭐ | Curriculum / Domain Rand | Wrappers, copy patterns from RSL |

### Extra
- FM + RL (flow matching policies in RL loop) — future work

---

## FastTD3 / FastSAC Implementation Notes

Reference: [FastTD3 paper](https://arxiv.org/abs/2505.22642), [FastSAC humanoid paper](https://arxiv.org/abs/2512.01996), [GitHub](https://github.com/younggyoseo/FastTD3)

These are high-performance variants of TD3/SAC optimized for massively parallel simulation. Key results: HumanoidBench tasks in <3 hours, humanoid locomotion sim-to-real in 15 minutes on single RTX 4090.

### Key Modifications (over vanilla TD3/SAC)

| Modification | Details |
|--------------|---------|
| **Parallel simulation** | Thousands of envs (up to 16k), leverage JAX/IsaacLab vectorization |
| **Large batch updates** | Batch size up to 8K (vs typical 256) |
| **Distributional critic** | C51 instead of standard Q-value (quantile regression too expensive at scale) |
| **Layer normalization** | Critical for stability in high-dimensional control |
| **Twin Q averaging** | Average twin Q-values instead of min (clipped double-Q) — reduces underestimation |
| **High replay ratio** | More gradient steps per simulation step, off-policy data reuse |
| **Observation normalization** | Running mean/std normalization |

### FastSAC/FastTD3 — Seo et al. 2025 (arXiv:2512.01996)

**Critical hyperparameters (from paper, easy to miss):**
- gamma=0.97 for locomotion (NOT 0.99)
- AdamW with β2=0.95, weight_decay=0.001 (NOT plain Adam)
- Mixed noise schedule: σ ~ U[0.01, 0.05] per step (NOT fixed 0.1-0.2)
- FastSAC: α_init=0.001, max_std=1.0, target_entropy=0

### FastDSAC — arXiv:2603.12612 (different research group)
- Gaussian distributional critic (replaces C51 — no quantization artifacts)
- Dimension-wise Entropy Modulation (DEM) — per-dim exploration weights
- Population diversity — per-env β scaling for exploration heterogeneity
- Target entropy = 0

### Network architecture
```
Actor:  MLP (512, 512), ReLU, LayerNorm
        + optional DEM logits head (FastDSAC)
Critic: MLP (512, 512), ReLU, LayerNorm
        Output: C51 logits (FastTD3/FastSAC) or (mean, variance) (FastDSAC)
```

Network dims are per-algo config (SACConfig.hidden_dim, TD3Config.hidden_dim, etc.), NOT in TrainConfig. PPO is the exception — its network dims are in TrainConfig (policy_hidden_dim, value_hidden_dim) for historical reasons.

### Hyperparameters (paper recipe — Seo et al. 2025)
```python
batch_size: 8192
learning_rate: 3e-4 (with cosine decay to 3e-5)
gamma: 0.97 (locomotion) / 0.99 (whole-body tracking)
tau: 0.005
policy_delay: 2 (TD3)
noise: U[0.01, 0.05] (mixed per step)
optimizer: AdamW(β2=0.95, weight_decay=0.001)
noise_std: 0.1-0.2
num_envs: 4096-16384
```

### SimbaV2 Integration
FastTD3 + SimbaV2 (hyperspherical normalization) is recommended for best results:
- Faster training
- Better asymptotic performance
- See: [SimbaV2 paper](https://arxiv.org/abs/2502.15280)

### Implementation priority
1. Standard TD3/SAC first (for correctness)
2. Add distributional critic (C51)
3. Add layer normalization
4. Scale batch size + parallel envs
5. Tune replay ratio

---

## Technical Decisions

### Stack
- **JAX** for autodiff, jit, vmap, scan
- **Flax Linen** for neural networks (not NNX — see rationale below)
- **Optax** for optimizers (including `optax.contrib.muon` for future Muon/matrix-whitening experiments — already in optax 0.2.6, drop-in compatible with our decoupled optimizer pattern)
- **Orbax** for checkpointing
- **Wandb** for logging

### Why Flax Linen over NNX?

NNX is the newer, more Pythonic Flax API and is recommended for new projects in general. However, **Linen is the better choice for high-throughput RL** for several specific reasons:

1. **Performance on small models**: `nnx.jit` traverses the object graph in pure Python on every call, adding overhead that primarily affects small-to-medium models — exactly the 256-512 hidden dim MLPs we use. Benchmarks show ~3x overhead for NNX on MLP workloads vs. Equinox/Linen. Mitigations exist (`nnx.cached_partial`, functional training loops via `nnx.split`/`nnx.merge`) but at that point you're writing Linen-style code with extra indirection.

2. **Ecosystem alignment**: Brax training (the reference PPO/SAC used by MuJoCo Playground) uses Flax Linen. FastTD3, CleanRL-JAX, and RSL-RL all use functional params-as-pytrees patterns. Linen means we can lift code directly from these references.

3. **RL-native pattern**: Linen's `model.apply(params, obs)` maps cleanly onto RL's "params-in, metrics-out" training loop. Params are plain pytrees that flow through `jax.jit`, `jax.grad`, `jax.vmap` with zero Python overhead inside the compiled path. No graph traversal, no hidden state mutation.

4. **Immutability = safety**: Linen's stateless design makes it impossible to accidentally mutate shared state across parallel environments or between actor/critic networks — a common RL bug source. The explicit params threading is more verbose but removes an entire class of errors.

**Note**: Linen is not being deprecated. The Flax team has stated it will continue to be supported, and major projects (MaxText, MaxDiffusion, Brax) still rely on it.

**Learning benefit**: Linen's explicit functional pattern (init → params → apply) forces you to understand how JAX actually works — pytrees, pure functions, JIT compilation boundaries. NNX hides this behind Pythonic sugar. For a codebase that's meant to teach the lab JAX fundamentals, the explicitness is a feature.

### Config System
- **Dataclasses** for type safety and IDE support
- Pure Python, no framework dependency — Hydra is a Phase 7+ consideration only if experiment management becomes a bottleneck
- Nested configs (AlgoConfig contains NetworkConfig)

```python
@dataclass
class NetworkConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "relu"

@dataclass
class PPOConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    policy_head: PolicyHeadConfig = field(default_factory=PolicyHeadConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Note: optimizer config (LR, schedule, grad clipping) lives outside PPO —
    # optimizers are constructed externally and passed to PPO.__init__
```

### Replay Buffer
- JAX-native for off-policy algorithms
- Fixed size, jittable sampling
- On-policy uses simple rollout buffer

**Off-policy replay buffer design decisions:**

- **Fully on-device (GPU)**: Buffer lives entirely in GPU memory, matching the Brax SAC pattern. No CPU↔GPU transfer during training. This is the key to high throughput — the entire collect→sample→update loop stays on-device and can be compiled into a single jitted function.
- **Memory layout**: Struct-of-arrays (SoA) — separate arrays for obs, action, reward, next_obs, done, truncation. This is more efficient for batched sampling than array-of-structs because each `jnp.take` pulls a contiguous slice. Fields stored as `jax.Array` of shape `(buffer_size, *field_shape)`.
- **Circular buffer with uniform sampling**: Fixed-size pre-allocated arrays with a write pointer that wraps. Sampling is `jax.random.randint` for indices, then `jnp.take` — fully jittable, no Python control flow. Start simple with uniform random sampling.
- **Capacity**: Sized to fit in GPU memory. For typical RL (obs_dim ~100, action_dim ~20, float32): 1M transitions ≈ 1.5 GB. At 16K parallel envs this fills fast — may need 100K-500K capacity with high replay ratio instead of 1M.
- **Prioritized replay**: Deferred. Uniform sampling is sufficient for TD3/SAC/FastTD3. PER adds significant complexity (sum-tree on device) for marginal gains in the massively parallel setting where data is abundant.
- **N-step returns**: Support configurable n-step return computation at insertion time (FastTD3 uses this).

### Environment Interface
Two backends with common interface:

**JAX-native (MuJoCo Playground / MJX)**
- Entire rollout jittable
- vmap over parallel envs
- 10-100x faster for simple envs
- 1-3 min JIT compile time

**MJWarp backend (preferred):**
MuJoCo Warp uses NVIDIA Warp instead of pure JAX for physics, providing massive speedups (up to 152x for locomotion, 313x for manipulation vs MJX on RTX 4090). Key points:
- Integrated into MJX via `impl='warp'` — same JAX API, faster backend
- Now default in MuJoCo Playground (as of Aug 2025)
- NVIDIA GPU required (CUDA 12.4+)
- Not differentiable via Warp (fine for model-free RL, rules out differentiable physics)
- May degrade for scenes >60 DoFs (humanoids are borderline)

**Alternatives considered:**
- *mjlab*: Isaac Lab API + MJWarp — well-designed env orchestration (composable managers for obs, rewards, domain rand, curriculum) but PyTorch-native. Doesn't fit our JAX stack for algorithms, but excellent design reference for Phase 5 env composition.
- *Isaac Lab*: Full-featured but heavy (Omniverse runtime), PhysX backend

**Gymnasium fallback**
- For envs not yet ported to JAX
- Python loop with jitted policy
- SubprocVecEnv for parallelism

---

---

## Architecture & Implementation Details

The detailed architecture specs, interface definitions, and phase checklists that were originally here have been superseded by:

- **README.md** — project structure, examples, benchmark results
- **TODO.md** — current priorities and task backlog
- **AGENT_HANDOFF.md** — working patterns, architecture decisions, debugging lessons
- **go2/sim_to_real_plan.md** — Go2 deployment pipeline
- **vision_rl_design.md** — CNN encoder, ManiSkill integration

The original detailed specs are preserved in `.context/archive/rl_framework_plan_full.md` for historical reference.
