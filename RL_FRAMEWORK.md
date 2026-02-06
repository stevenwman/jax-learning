# JAX RL Framework — Design Document

## Overview

A modular, JAX-native reinforcement learning library built on Flax NNX, designed to make **efficient robot learning accessible**.

### Why this framework?
- **Accessible**: Clean code you can actually read and modify, not research spaghetti
- **Efficient**: JAX-native with FastTD3/SAC for training in minutes, not hours
- **Robot-first**: Built for the sim-to-real pipeline — train fast in sim, deploy on hardware

### Core design principles
1. Understand the code → modify it → run experiments → deploy on robot
2. Modular encoder+head architecture — swap components without rewriting algorithms
3. Sim-to-real as first-class citizen, not afterthought

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

### FastSAC-specific notes
- Same recipe as FastTD3 but with entropy regularization
- Can be unstable due to entropy maximization in high-dim action spaces
- Restrict stochasticity (smaller std bounds) helps stability
- Generally slower than FastTD3 but still much faster than vanilla SAC

### Network architecture (from paper)
```
Actor:  MLP with LayerNorm, hidden_dim=256-512, 1-2 blocks
Critic: MLP with LayerNorm, hidden_dim=512, 2 blocks, C51 distributional output
```

### Hyperparameters (typical)
```python
batch_size: 8192
learning_rate: 3e-4 (with cosine decay to 3e-5)
gamma: 0.99 (0.95 for simple tasks)
tau: 0.005
policy_delay: 2 (TD3)
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
- **Flax NNX** for neural networks (not Linen)
- **Optax** for optimizers
- **Orbax** for checkpointing
- **Wandb** for logging

### Package Management
- Start with **uv** (fast, simple)
- Fall back to **conda** if JAX/CUDA versioning gets messy
- Pin versions in `pyproject.toml` as issues come up
- Document any version-specific gotchas

### Config System
- **Dataclasses** for type safety and IDE support
- Structured for future Hydra compatibility
- Nested configs (AlgoConfig contains NetworkConfig)

```python
@dataclass
class NetworkConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "relu"

@dataclass
class PPOConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    lr: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
```

### Replay Buffer
- JAX-native for off-policy algorithms
- Fixed size, jittable sampling
- On-policy uses simple rollout buffer

### Environment Interface
Two backends with common interface:

**JAX-native (MuJoCo Playground / MJX)**
- Entire rollout jittable
- vmap over parallel envs
- 10-100x faster for simple envs
- 1-3 min JIT compile time

**Gymnasium fallback**
- For envs not yet ported to JAX
- Python loop with jitted policy
- SubprocVecEnv for parallelism

---

## Architecture

### Directory Structure

```
jax_rl/
├── configs/
│   ├── base.py              # BaseConfig dataclass
│   ├── ppo.py               # PPOConfig
│   ├── sac.py               # SACConfig
│   └── networks.py          # NetworkConfig variants
│
├── networks/
│   ├── protocols.py         # Encoder, QHead, PolicyHead protocols
│   ├── encoders/
│   │   ├── mlp.py           # Standard MLP encoder (configurable LayerNorm)
│   │   ├── cnn.py           # Conv encoder for pixels
│   │   └── simba.py         # SimbaV2 (hyperspherical norm) — future
│   ├── heads/
│   │   ├── q_scalar.py      # Standard Q output
│   │   ├── q_distributional.py  # C51 distributional Q
│   │   ├── gaussian.py      # Gaussian policy (mean + log_std)
│   │   ├── deterministic.py # Deterministic policy (TD3 actor)
│   │   ├── discriminator.py # Skill classifier q(z|s) — USD, future
│   │   └── value.py         # State value V(s) — PPO
│   ├── builders.py          # Compose encoder + head from config
│   └── distributions.py     # TanhNormal, rsample helpers
│
├── algos/
│   ├── base.py              # BaseAlgorithm ABC
│   ├── ppo.py               # PPO implementation
│   ├── sac.py               # SAC implementation
│   ├── td3.py               # TD3 implementation
│   └── bc/
│       ├── mlp.py           # Behavioral cloning
│       └── flow_matching.py # Flow matching policy
│
├── buffers/
│   ├── rollout.py           # On-policy buffer (PPO)
│   └── replay.py            # Off-policy buffer (SAC, TD3), JAX-native
│
├── envs/
│   ├── base.py              # EnvWrapper protocol
│   ├── mjx_adapter.py       # MuJoCo Playground adapter
│   ├── gymnasium_adapter.py # Gymnasium fallback
│   └── wrappers/
│       ├── normalize.py     # Observation/reward normalization
│       ├── curriculum.py    # Curriculum learning wrapper
│       └── domain_rand.py   # Domain randomization wrapper
│
├── training/
│   ├── trainer.py           # Orchestrates train loop, eval, logging
│   ├── logger.py            # Wandb integration
│   └── checkpoint.py        # Orbax save/load
│
└── utils/
    ├── rng.py               # RNG key management helpers
    ├── tree.py              # Pytree utilities
    └── metrics.py           # Compute returns, GAE, etc.
```

### Key Interfaces

#### Encoder Protocol
```python
class Encoder(Protocol):
    def __call__(
        self, 
        obs: jax.Array, 
        action: jax.Array | None = None,
        context: jax.Array | None = None,
    ) -> jax.Array:
        """
        Returns feature vector.
        - action: needed for Q encoders (s,a) -> features
        - context: goal, skill, or task embedding (for goal-conditioned, USD, multi-task)
        """
        ...
```

#### Encoder Config (with context fusion)
```python
@dataclass
class EncoderConfig:
    obs_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "relu"
    norm: str | None = "layer"          # None, "layer", "spectral"
    norm_placement: str = "pre"          # "pre" or "post" activation
    
    # Context handling (goal, skill, task)
    context_dim: int | None = None
    context_fusion: str = "concat"       # "concat", "film", "cross_attn"
```

**Fusion methods:**
- `concat`: Simple concatenation, works for MLPs
- `film`: Feature-wise Linear Modulation — context scales/shifts features
- `cross_attn`: Cross-attention between obs and context tokens (for transformers)

#### Q Head Protocols
```python
class QHead(Protocol):
    def __call__(self, features: jax.Array) -> jax.Array:
        """Returns scalar Q-value."""
        ...
    
class DistributionalQHead(Protocol):
    def __call__(self, features: jax.Array) -> jax.Array:
        """Returns (batch, num_atoms) logits for C51."""
        ...
    
    def q_value(self, features: jax.Array) -> jax.Array:
        """Returns expected Q-value from distribution."""
        ...
```

#### Policy Head Protocol
```python
class GaussianHead(Protocol):
    def __call__(self, features: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Returns (mean, log_std)."""
        ...

class DeterministicHead(Protocol):
    def __call__(self, features: jax.Array) -> jax.Array:
        """Returns action directly (for TD3)."""
        ...
```

#### Composed Networks (built from encoder + head)
```python
class Policy(Protocol):
    def __call__(
        self, obs: jax.Array, context: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (mean, log_std) or (action, None) for deterministic."""
        ...
    
    def sample(
        self, obs: jax.Array, key: PRNGKey, context: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (action, log_prob)."""
        ...

class QFunction(Protocol):
    def __call__(
        self, obs: jax.Array, action: jax.Array, context: jax.Array | None = None
    ) -> jax.Array:
        """Returns Q-value (scalar, even for distributional — returns expected value)."""
        ...
    
    def distributional(
        self, obs: jax.Array, action: jax.Array, context: jax.Array | None = None
    ) -> jax.Array | None:
        """Returns distribution logits if distributional, else None."""
        ...
```


#### Algorithm Interface
```python
class BaseAlgorithm(ABC):
    @abstractmethod
    def update(self, batch: Batch) -> dict[str, float]:
        """Perform gradient update, return metrics"""
        ...
    
    @property
    @abstractmethod
    def state(self) -> AlgoState:
        """Return serializable state for checkpointing"""
        ...

class OnlineAlgorithm(BaseAlgorithm):
    """RL algorithms that collect from environment"""
    @abstractmethod
    def collect(self, env, env_state, key) -> tuple[Batch, EnvState, PRNGKey]:
        """Collect rollout/samples"""
        ...

class OfflineAlgorithm(BaseAlgorithm):
    """BC, offline RL — data comes from dataset, no collect method"""
    pass
```

**BC → RL Finetuning:**
Encoder+head design enables weight transfer naturally:
```python
# Pretrain with BC
bc = BC(policy=policy)
train_offline(bc, dataset)

# Finetune with RL — same policy network, weights carry over
sac = SAC(policy=bc.policy, ...)
train_online(sac, env)
```

#### Algorithm Config Pattern (Q aggregation, etc.)
```python
@dataclass
class TD3Config:
    # ... network configs ...
    q_aggregation: str = "min"      # "min" (standard) or "avg" (FastTD3)
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5

@dataclass
class SACConfig:
    # ... network configs ...
    q_aggregation: str = "min"      # "min" or "avg"
    init_temperature: float = 1.0
    learnable_temp: bool = True
```

#### Environment Interface
```python
class JaxEnv(Protocol):
    def reset(self, key: PRNGKey) -> EnvState:
        ...
    
    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        ...
    
    @property
    def observation_size(self) -> int: ...
    
    @property
    def action_size(self) -> int: ...
    
    @property
    def action_bounds(self) -> tuple[jax.Array, jax.Array]: 
        """(low, high) for continuous actions. Policies output tanh, scaled to bounds."""
        ...
```

**Action handling:** Policies output in [-1, 1] (tanh), env wrapper scales to actual bounds. This is standard practice and simplifies the policy.

### Trainer Pattern (Hybrid)

Algorithms own the math (collect, update). Trainer owns infrastructure:

```python
class Trainer:
    def __init__(self, algo, env, config, logger, checkpointer):
        self.algo = algo
        self.env = env
        self.config = config
        self.logger = logger
        self.checkpointer = checkpointer
    
    def train(self, num_steps: int):
        env_state = self.env.reset(self.key)
        
        for step in range(num_steps):
            # Algo handles collection
            batch, env_state, self.key = self.algo.collect(
                self.env, env_state, self.key
            )
            
            # Algo handles update
            metrics = self.algo.update(batch)
            
            # Trainer handles infrastructure
            self.logger.log(metrics, step)
            
            if step % self.config.eval_freq == 0:
                eval_metrics = self.evaluate()
                self.logger.log(eval_metrics, step)
            
            if step % self.config.checkpoint_freq == 0:
                self.checkpointer.save(self.algo.state, step)
```

---

## Implementation Phases

### Phase 1: Foundation
- [ ] Config dataclasses (EncoderConfig, PPOConfig, etc.)
- [ ] MLP encoder (with configurable LayerNorm + context fusion)
- [ ] Scalar Q head
- [ ] Value head V(s)
- [ ] Gaussian policy head
- [ ] Deterministic policy head
- [ ] Network builder (compose encoder + head)
- [ ] Distribution utilities (TanhNormal, rsample)
- [ ] Rollout buffer
- [ ] Wandb logger

### Phase 2: First Algorithms (standard versions)
- [ ] PPO implementation
- [ ] TD3 implementation (standard, min Q)
- [ ] SAC implementation (standard, min Q)
- [ ] MuJoCo Playground adapter
- [ ] Test on dm_control (CartpoleBalance, HalfCheetah)
- [ ] Verify scores match reference (CleanRL/Brax)
- [ ] Basic README + example script (ongoing, not deferred)

**PPO Implementation Details (from RSL-RL / Huang et al. 2022):**
- Clipped surrogate loss + clipped value loss (optional)
- Entropy bonus
- GAE with proper timeout handling (bootstrap on timeout, don't bootstrap on true termination)
- Advantage normalization (per minibatch optional)
- Gradient clipping
- Adaptive LR schedule: if KL > target × 1.5 → decrease LR, if KL < target / 1.5 → increase LR
- Random early termination at init to decorrelate parallel env rollouts

### Phase 3: Fast Variants
- [ ] Distributional Q head (C51)
- [ ] FastTD3 (C51 + avg Q + large batch + LayerNorm)
- [ ] FastSAC (same recipe)
- [ ] JAX-native replay buffer (for high throughput)
- [ ] Benchmark: compare wall-clock time vs standard

### Phase 4: Behavioral Cloning
- [ ] MLP BC
- [ ] Flow matching policy
- [ ] Dataset loading utilities

### Phase 5: Sim-to-Real Pipeline
- [ ] Normalization wrappers
- [ ] Curriculum learning wrapper
- [ ] Domain randomization wrapper
- [ ] Observation groups (actor vs critic asymmetric obs)
- [ ] Custom env template (for your robot)
- [ ] Policy export (JAX → ONNX or saved weights for deployment)

**Asymmetric Obs Note:**
Actor and critic can receive different observations (e.g., critic sees privileged sim info).
- PPO: Well-established, critic is just baseline
- SAC/TD3: Less studied — critic gradient ∇a Q may conflict when privileged info varies but actor obs is identical. Open question, good ablation study.

**Deployment Options (TBD):**
- JAX weights → load in Python on robot (if Python available)
- ONNX export → run on edge device
- JAX2TF → TFLite for embedded
- Unitree SDK, robot arm controllers, etc. — depends on target hardware

### Phase 6: Goal-Conditioned & Unsupervised RL
- [ ] Goal-conditioned wrappers (HER-style relabeling)
- [ ] DIAYN (skill discriminator + SAC)
- [ ] METRA (metric-aware abstraction)
- [ ] Factorized USD (if needed for real robot)

**USD Architectural Notes:**
When implementing USD, we'll need:
- **Discriminator head**: q(z|s) classifier (new head type, not Q-value)
- **Contrastive encoder**: for METRA temporal distance learning
- **Intrinsic reward hook**: reward = f(discriminator, encoder) instead of env reward
- **Algorithm wrapper pattern**: USD wraps base algo (e.g., DIAYN wraps SAC)

The `context` argument in Encoder already supports skill/goal conditioning — this was the key architectural decision.

### Phase 7: Polish & Extensions
- [ ] CNN encoder
- [ ] SimbaV2 encoder (hyperspherical norm)
- [ ] RND (Random Network Distillation) for curiosity-driven exploration
- [ ] Symmetry augmentation (for legged robots)
- [ ] Recurrent policies (LSTM/GRU)
- [ ] Hydra config integration (optional)
- [ ] Multi-GPU support
- [ ] Full documentation site (basic docs are ongoing from Phase 2)
- [ ] Tutorial notebooks

### Phase 8: Hyperparameter Tuning (low priority)
- [ ] Optuna integration (simple Bayesian opt + pruning)
- [ ] W&B Sweeps fallback (zero new deps)
- [ ] PBT wrapper (population-based training — adapts hyperparams mid-run)

**Notes:**
- Optuna is easiest to start: define objective function, let it search
- PBT is better for RL long-term (no restarts, exploits good configs)
- W&B Sweeps if we want to avoid new dependencies
- Key hyperparams to tune: lr, gamma, batch_size, network width/depth, entropy_coef (PPO), tau (SAC/TD3)

---

## Reference Implementations

- **CleanRL**: Single-file reference for algorithm correctness
- **Brax**: JAX-native RL, good for PPO/SAC patterns
- **RSL-RL**: Robotics-focused, PPO + distillation, observation groups
- **MuJoCo Playground**: Env interface and training scripts
- **37 PPO Details**: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

---

## Quickstart Vision

When Phase 2 is done, someone should be able to:
```bash
python train.py --algo ppo --env HalfCheetah --num_envs 4096
# → Training logs to wandb
# → Checkpoint saved
# → ~10 min to decent policy on single GPU
```

When Phase 5 is done:
```bash
python train.py --algo fasttd3 --env MyQuadruped --domain_rand --curriculum
python distill.py --teacher checkpoint.pkl --student deployable_policy
python export.py --policy deployable_policy --format onnx
# → Ready to run on hardware
```

---

## Development Principles

- **Start flat** — one file per component during learning, refactor into packages as they stabilize
- **Test incrementally** — verify each component before building on it
- **Correctness first** — match CleanRL scores before optimizing for speed
- **Document as you go** — if you figured something out, write it down
- **Accessibility matters** — if the code is hard to understand, simplify it