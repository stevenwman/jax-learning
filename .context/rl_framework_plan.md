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
- **Flax Linen** for neural networks (not NNX — see rationale below)
- **Optax** for optimizers
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
│   ├── types.py             # TrainingState, Params type aliases
│   ├── encoders/
│   │   ├── mlp.py           # Standard MLP encoder (nn.Module, configurable LayerNorm)
│   │   ├── cnn.py           # Conv encoder for pixels
│   │   └── simba.py         # SimbaV2 (hyperspherical norm) — Phase 3
│   ├── heads/
│   │   ├── q_scalar.py      # Standard Q output
│   │   ├── q_distributional.py  # C51 distributional Q
│   │   ├── gaussian.py      # Gaussian policy (mean + log_std)
│   │   ├── deterministic.py # Deterministic policy (TD3 actor)
│   │   ├── discriminator.py # Skill classifier q(z|s) — Phase 6 (USD)
│   │   └── value.py         # State value V(s) — PPO
│   ├── builders.py          # Compose encoder + head from config, init params
│   └── distributions.py     # TanhNormal, rsample helpers
│
├── algos/
│   ├── base.py              # BaseAlgorithm (stateless, functional)
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

#### Encoder (Linen Module)
```python
class MLPEncoder(nn.Module):
    """MLP encoder with configurable LayerNorm and context fusion."""
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "relu"
    norm: str | None = "layer"
    context_dim: int | None = None
    context_fusion: str = "concat"

    @nn.compact
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

All network modules follow this pattern: `nn.Module` subclass with `@nn.compact`, called via `model.apply(params, obs, ...)`. The module itself is stateless — a template that defines the computation graph. Params are initialized once via `model.init(key, dummy_obs)` and threaded explicitly through the training loop.

#### Encoder Config (with context fusion)

Config dataclasses are used to construct Linen modules. The config is not the module — it's a recipe for building one:

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

# Builder pattern: config -> Linen module + initialized params
def build_policy(config: PPOConfig, key: PRNGKey, obs_dim: int, action_dim: int):
    model = GaussianPolicy(encoder_config=config.encoder, action_dim=action_dim)
    params = model.init(key, jnp.zeros(obs_dim))
    return model, params
```

**Fusion methods:**
- `concat`: Simple concatenation, works for MLPs
- `film`: Feature-wise Linear Modulation — context scales/shifts features
- `cross_attn`: Cross-attention between obs and context tokens (for transformers)

#### Q Head Modules
```python
class ScalarQHead(nn.Module):
    """Standard Q output: features -> scalar."""
    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        return nn.Dense(1)(features).squeeze(-1)

class DistributionalQHead(nn.Module):
    """C51 distributional Q: features -> (batch, num_atoms) logits."""
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        """Returns (batch, num_atoms) logits."""
        return nn.Dense(self.num_atoms)(features)

    def q_value(self, logits: jax.Array) -> jax.Array:
        """Expected Q-value from distribution."""
        support = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * support, axis=-1)
```

#### Policy Head Modules
```python
class GaussianHead(nn.Module):
    """Gaussian policy: features -> (mean, log_std)."""
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, features: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = nn.Dense(self.action_dim)(features)
        log_std = nn.Dense(self.action_dim)(features)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

class DeterministicHead(nn.Module):
    """Deterministic policy: features -> action (for TD3)."""
    action_dim: int

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        return nn.tanh(nn.Dense(self.action_dim)(features))
```

#### Composed Networks (built from encoder + head)

In Linen, composed networks are modules that contain sub-modules. The key pattern: `model.apply(params, obs)` — the model object is a stateless template, params are a separate pytree.

```python
class GaussianPolicy(nn.Module):
    """Encoder + GaussianHead, composed as a single Linen module."""
    encoder_config: EncoderConfig
    action_dim: int

    @nn.compact
    def __call__(
        self, obs: jax.Array, context: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (mean, log_std)."""
        features = MLPEncoder(**asdict(self.encoder_config))(obs, context=context)
        return GaussianHead(self.action_dim)(features)

class QFunction(nn.Module):
    """Encoder + QHead, composed as a single Linen module."""
    encoder_config: EncoderConfig
    distributional: bool = False

    @nn.compact
    def __call__(
        self, obs: jax.Array, action: jax.Array, context: jax.Array | None = None
    ) -> jax.Array:
        """Returns Q-value (scalar). For distributional, returns expected value."""
        features = MLPEncoder(**asdict(self.encoder_config))(obs, action=action, context=context)
        if self.distributional:
            head = DistributionalQHead()
            logits = head(features)
            return head.q_value(logits)
        return ScalarQHead()(features)
```

**Usage pattern (Linen functional style):**
```python
# 1. Create module (stateless template)
policy = GaussianPolicy(encoder_config=enc_cfg, action_dim=env.action_size)

# 2. Initialize params (once)
policy_params = policy.init(rng_key, dummy_obs)

# 3. Forward pass (inside jitted training loop)
mean, log_std = policy.apply(policy_params, obs)

# 4. Sampling (pass RNG explicitly)
action, log_prob = policy.apply(policy_params, obs, rngs={'sample': sample_key},
                                method=policy.sample)
```

This is the same pattern Brax uses. Params are plain pytrees — they flow through `jax.jit`, `jax.grad`, and `jax.vmap` with zero overhead.


#### Algorithm Interface

Algorithms operate on `TrainingState` — a flax dataclass containing all params and optimizer state as pytrees. This makes the entire training loop jittable.

```python
@flax.struct.dataclass
class TrainingState:
    """All mutable state for training, as a single pytree."""
    policy_params: Params
    critic_params: Params
    target_critic_params: Params
    optimizer_state: optax.OptState
    normalizer_state: RunningStatisticsState
    env_steps: int

class BaseAlgorithm(ABC):
    """Stateless algorithm — all state lives in TrainingState."""
    
    @abstractmethod
    def init(self, key: PRNGKey, env: JaxEnv) -> TrainingState:
        """Initialize all params and optimizer state."""
        ...
    
    @abstractmethod
    def update(self, state: TrainingState, batch: Batch) -> tuple[TrainingState, dict[str, float]]:
        """Perform gradient update, return new state and metrics."""
        ...

class OnlineAlgorithm(BaseAlgorithm):
    """RL algorithms that collect from environment."""
    @abstractmethod
    def collect(
        self, state: TrainingState, env: JaxEnv, env_state: EnvState, key: PRNGKey
    ) -> tuple[Batch, EnvState, PRNGKey]:
        """Collect rollout/samples using current policy params."""
        ...

class OfflineAlgorithm(BaseAlgorithm):
    """BC, offline RL — data comes from dataset, no collect method."""
    pass
```

**Key difference from the NNX pattern**: Algorithms don't hold mutable state. `update()` takes a state in and returns a new state out — pure functional, fully jittable. This matches how Brax structures its training loops.

**BC → RL Finetuning:**
Encoder+head design enables weight transfer naturally. Since params are plain pytrees, transferring weights is just copying the right subtree:
```python
# Pretrain with BC
bc_state = bc_algo.init(key, env)
bc_state = train_offline(bc_algo, bc_state, dataset)

# Finetune with RL — extract policy params, initialize SAC state with them
sac_state = sac_algo.init(key, env)
sac_state = sac_state.replace(
    policy_params=bc_state.policy_params  # weights carry over
)
sac_state = train_online(sac_algo, sac_state, env)
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
@flax.struct.dataclass
class EnvState:
    """All env state as a pytree. Mirrors brax.envs.State."""
    pipeline_state: Any          # Physics engine internal state (MJX data)
    obs: jax.Array               # Current observation
    reward: jax.Array            # Reward from last step
    done: jax.Array              # True on terminal states (bool)
    truncation: jax.Array        # True on timeout (bool) — distinct from done
    info: dict[str, jax.Array]   # Extra info (reward components, contacts, etc.)
    metrics: dict[str, jax.Array]  # Logging metrics (episode return, length, etc.)

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
    
    @property
    def episode_length(self) -> int:
        """Max steps per episode. Needed for GAE timeout handling."""
        ...
    
    @property
    def auto_reset(self) -> bool:
        """Whether env auto-resets on done/truncation (MuJoCo Playground does)."""
        ...
```

**Key design decisions:**
- **done vs truncation**: Critical for correct GAE. `done=True` means the episode ended naturally (don't bootstrap). `truncation=True` means the episode hit the time limit (bootstrap the value). PPO implementations that conflate these will learn incorrect value functions.
- **info dict**: Carries per-step metadata — reward components, contact forces, privileged observations. Needed for logging, curriculum learning, and asymmetric actor-critic.
- **auto-reset**: MuJoCo Playground environments auto-reset when done. The adapter must handle this correctly — the "next obs" after a done is the first obs of the new episode, not a terminal observation.
- **Decimation / action repeat**: For envs with decimation > 1 (e.g., locomotion: physics at 200Hz, policy at 50Hz), `step()` applies the same action for multiple physics substeps internally. CartpoleBalance has decimation=1 so this is transparent initially, but locomotion envs will have decimation of 2-4. The env handles this — the algorithm always sees one step per action.
- **Action handling:** Policies output in [-1, 1] (tanh), env wrapper scales to actual bounds. This is standard practice and simplifies the policy.

### Trainer Pattern (Functional)

Algorithms own the math (collect, update) as pure functions. Trainer owns infrastructure. All mutable state flows through `TrainingState` — no hidden mutation.

```python
class Trainer:
    def __init__(self, algo, env, config, logger, checkpointer):
        self.algo = algo
        self.env = env
        self.config = config
        self.logger = logger
        self.checkpointer = checkpointer
    
    def train(self, num_steps: int):
        key = jax.random.PRNGKey(self.config.seed)
        key, init_key, env_key = jax.random.split(key, 3)
        
        # Initialize all state as pytrees
        training_state = self.algo.init(init_key, self.env)
        env_state = self.env.reset(env_key)
        
        for step in range(num_steps):
            key, collect_key = jax.random.split(key)
            
            # Algo handles collection (pure function)
            batch, env_state, collect_key = self.algo.collect(
                training_state, self.env, env_state, collect_key
            )
            
            # Algo handles update (pure function, returns new state)
            training_state, metrics = self.algo.update(training_state, batch)
            
            # Trainer handles infrastructure
            self.logger.log(metrics, step)
            
            if step % self.config.eval_freq == 0:
                eval_metrics = self.evaluate(training_state)
                self.logger.log(eval_metrics, step)
            
            if step % self.config.checkpoint_freq == 0:
                self.checkpointer.save(training_state, step)
```

**Note**: The inner loop (collect + update) can be wrapped in `jax.lax.scan` for maximum throughput, compiling the entire training loop into a single XLA program — this is what Brax does.

---

## Implementation Phases

### Phase 1: Foundation
- [x] Config dataclasses (EncoderConfig, PPOConfig, etc.)
- [x] TrainingState `flax.struct.dataclass` (params + optimizer state as pytree)
- [x] MLP encoder (nn.Module with configurable LayerNorm + context fusion)
- [ ] Scalar Q head
- [x] Value head V(s)
- [x] Gaussian policy head
- [ ] Deterministic policy head
- [x] Network builder (compose encoder + head from config, init params)
- [x] Distribution utilities (TanhNormal, rsample)
- [x] Observation normalization (running mean/std, Welford running stats)
- [x] Rollout buffer
- [ ] Wandb logger

### Phase 2: First Algorithms (standard versions)
- [x] PPO implementation (3 variants: eager, jit, scan — scan is 542x faster than eager)
- [ ] TD3 implementation (standard, min Q)
- [ ] SAC implementation (standard, min Q)
- [x] MuJoCo Playground adapter
- [x] Test on dm_control — CartpoleBalance validated (≥995), CheetahRun validated (826 at 20M, target was ≥700 at 60M)
- [ ] Verify scores match reference (see benchmark table below) — CartpoleBalance passes, CheetahRun passes
- [x] Basic README + example script
- [x] Orbax checkpointing (timestamped dirs, meta.json, metrics CSV)
- [x] Video recording (two-phase: scan rollout on GPU, render on CPU)
- [x] Determinism verified (bit-identical across 3 runs for env + full training)
- [x] Optimizer decoupled from PPO (externally constructed, supports LR annealing)

**PPO Implementation Details (from RSL-RL / Huang et al. 2022):**
- Clipped surrogate loss + clipped value loss (optional)
- Entropy bonus
- GAE with proper timeout handling (bootstrap on timeout, don't bootstrap on true termination)
- Advantage normalization (per minibatch optional)
- Gradient clipping
- Adaptive LR schedule: if KL > target × 1.5 → decrease LR, if KL < target / 1.5 → increase LR
- Random early termination at init to decorrelate parallel env rollouts

**Evaluation Protocol:**
- Use **deterministic policy** for eval (mean action for Gaussian, direct output for deterministic)
- Run **10 full episodes** per eval, report mean and std of episode reward
- Eval frequency: every N env steps (configurable, default ~every 1% of total training)
- Track: mean reward, std reward, min/max reward, mean episode length
- For off-policy (SAC/TD3): also log critic loss, actor loss, entropy (SAC), Q-values
- For on-policy (PPO): also log approx KL, clip fraction, explained variance, entropy

**Benchmark Targets (Phase 2 validation — MuJoCo Playground DM Control, 5 seeds, A100):**

DM Control rewards are normalized 0–1000. These targets are approximate — within ~10% of reference means is a pass.

| Algo | Env | Target Reward | Env Steps | Reference |
|------|-----|--------------|-----------|-----------|
| PPO | CartpoleBalance | ≥950 | 10M | MuJoCo Playground paper |
| PPO | CartpoleSwingup | ≥800 | 20M | MuJoCo Playground paper |
| PPO | CheetahRun | ≥700 | 60M | MuJoCo Playground paper |
| PPO | HopperStand | ≥800 | 60M | MuJoCo Playground paper |
| PPO | WalkerRun | ≥600 | 60M | MuJoCo Playground paper |
| SAC | CartpoleBalance | ≥950 | 2M | MuJoCo Playground paper |
| SAC | CartpoleSwingup | ≥800 | 5M | MuJoCo Playground paper |
| SAC | CheetahRun | ≥700 | 5M | MuJoCo Playground paper |
| TD3 | CheetahRun | ≥700 | 5M | CleanRL / Spinning Up |

**Fast Variant Benchmark Targets (Phase 3 — wall-clock comparison):**

| Algo | Env | Metric | Target | Reference |
|------|-----|--------|--------|-----------|
| FastTD3 | CheetahRun | Wall-clock to 700 reward | ≤5 min (single GPU) | FastTD3 paper |
| FastTD3 vs TD3 | CheetahRun | Speedup | ≥5x wall-clock | Internal ablation |
| FastSAC vs SAC | CheetahRun | Speedup | ≥3x wall-clock | Internal ablation |

### Phase 3: Fast Variants
- [ ] SimbaV2 encoder (hyperspherical normalization — recommended for FastTD3)
- [ ] Distributional Q head (C51)
- [ ] FastTD3 (C51 + avg Q + large batch + LayerNorm + optional SimbaV2)
- [ ] FastSAC (same recipe)
- [ ] JAX-native replay buffer (for high throughput)
- [ ] Benchmark: compare wall-clock time vs standard (see table above)

### Phase 4: Behavioral Cloning
- [ ] Gymnasium adapter (Python-loop fallback for CPU envs like Robomimic)
- [ ] MLP BC
- [ ] Flow matching policy
- [ ] Dataset loading utilities (HDF5/Robomimic format)

### Phase 5: Sim-to-Real Pipeline
- [ ] Reward normalization wrapper
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

**Design reference for Phase 5:** Study mjlab's manager-based term composition pattern (observations, rewards, events, curriculum, domain randomization as modular registered functions with lifecycle hooks: on_reset, on_step, on_interval). Their architecture is the cleanest example of composable env orchestration. We don't need their full 8-manager system, but the pattern of small self-contained terms registered with lifecycle hooks is a good model for our wrappers. See: https://mujocolab.github.io/mjlab/main/source/architecture_overview.html

### Phase 6: Goal-Conditioned & Unsupervised RL

This is the primary motivation for the framework. Each algorithm has concrete milestones:

**Milestone 6a — Goal-Conditioned RL:**
- [ ] Goal-conditioned wrappers (HER-style relabeling)
- [ ] Validate: goal-conditioned SAC on a reaching/navigation task
- Success criteria: agent reliably reaches commanded goals in MuJoCo env

**Milestone 6b — DIAYN:**
- [ ] Skill discriminator head q(z|s)
- [ ] Intrinsic reward hook (reward = discriminator log-prob)
- [ ] DIAYN wrapping SAC
- [ ] Validate: discover diverse locomotion skills on Ant/HalfCheetah
- Success criteria: visually distinct skills, discriminator accuracy >80%

**Milestone 6c — METRA:**
- [ ] Contrastive encoder for temporal distance learning
- [ ] Metric-aware skill abstraction
- [ ] Validate: learn skills with metric structure (not just discriminability)
- Success criteria: skills form meaningful embedding space, reproduce METRA paper results on standard benchmarks

**Milestone 6d — Factorized USD:**
- [ ] Factorized skill discovery (if needed for real robot)
- [ ] Validate: transfer discovered skills to downstream tasks
- Success criteria: skill pretraining improves sample efficiency on downstream task vs training from scratch

**USD Architectural Notes:**
When implementing USD, we'll need:
- **Discriminator head**: q(z|s) classifier (new head type, not Q-value)
- **Contrastive encoder**: for METRA temporal distance learning
- **Intrinsic reward hook**: reward = f(discriminator, encoder) instead of env reward
- **Algorithm wrapper pattern**: USD wraps base algo (e.g., DIAYN wraps SAC)

The `context` argument in Encoder already supports skill/goal conditioning — this was the key architectural decision.

### Phase 7: Polish & Extensions
- [ ] CNN encoder
- [ ] RND (Random Network Distillation) for curiosity-driven exploration
- [ ] Symmetry augmentation (for legged robots)
- [ ] Recurrent policies (LSTM/GRU)
- [ ] Hydra config integration (optional — only if experiment management becomes a bottleneck; pure dataclasses are sufficient until then)
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

- **Brax** (`brax/training`): JAX-native RL, PPO/SAC — **uses Flax Linen**, primary reference for our training patterns
- **CleanRL**: Single-file reference for algorithm correctness — excellent for learning since each algo is one readable file
- **FastTD3**: High-throughput TD3/SAC — functional JAX patterns, directly compatible
- **RSL-RL**: Robotics-focused, PPO + distillation, observation groups
- **MuJoCo Playground**: Env interface and training scripts (trains via Brax PPO)
- **mjlab**: Manager-based env design + MuJoCo Warp — reference for Phase 5 env composition patterns (PyTorch-native, not for algorithms)
- **37 PPO Details**: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

### Learning Resources (JAX / Flax / RL)
- **JAX 101**: https://jax.readthedocs.io/en/latest/jax-101/ — start here for JAX fundamentals (pytrees, jit, vmap, grad)
- **Flax Linen quickstart**: https://flax-linen.readthedocs.io/ — Linen module patterns, init/apply, train state
- **Spinning Up in Deep RL**: https://spinningup.openai.com/ — best conceptual intro to PPO, SAC, TD3 with math
- **The 37 PPO Details**: essential reading before implementing PPO — documents every implementation trick that affects performance

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

- **Full structure from day one** — every component lives at its final address; no big refactors later
- **Test incrementally** — verify each component before building on it
- **Correctness first** — match CleanRL scores before optimizing for speed
- **Document as you go** — if you figured something out, write it down
- **Readability is a feature** — if the code is hard to understand, simplify it. A lab member picking up this codebase should be productive within a day.

### Code Style for Interpretability

The audience for this code is *the lab* — people who know Python and ML but may not know JAX or RL internals. Every file should be self-contained enough that someone can read it top-to-bottom and learn something.

**Commenting philosophy:**
- Every algorithm file starts with a docstring linking to the paper and summarizing the key equations
- Non-obvious JAX patterns get inline comments explaining *why* (e.g., "# jax.lax.stop_gradient here because we don't want critic gradients flowing into the actor")
- Each loss function includes the equation number from the paper it implements
- Type annotations everywhere — they're free documentation

**Naming:**
- Prefer full names over abbreviations (`policy_params` not `pi_p`, `critic_loss` not `q_loss`)
- Variable names should match paper notation where it helps (`log_pi` for log π(a|s), `advantage` for Â)
- Config fields are self-documenting (`clip_eps` not `ce`, `entropy_coef` not `ec`)

**JAX-specific readability:**
- Use `jax.lax.scan`, `jax.lax.cond`, `vmap`, etc. wherever they improve performance — but always include a comment explaining what the operation does in plain English and what the naive version would look like
- When using `vmap`, comment what axis is being mapped over and the shape transformation
- Keep `jit` boundaries at the highest level possible (whole train step, not individual sub-functions) — this is both faster and easier to debug
- Include shape annotations in comments for non-obvious tensor operations (e.g., `# (batch, num_atoms) -> (batch,)`)
- For debugging: keep a `DEBUG` flag or config option that disables `jit` so you can use standard Python debugging tools. Fast by default, debuggable when needed.
**File structure per algorithm:**
```
# ppo.py
"""
Proximal Policy Optimization (PPO)

Paper: https://arxiv.org/abs/1707.06347
Reference: Brax PPO, CleanRL ppo_continuous_action.py

Key equations:
  L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]  (Eq. 7)
  where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

Implementation notes:
  - Uses GAE for advantage estimation (see utils/metrics.py)
  - Observation normalization via running statistics (Phase 1 foundation)
  - Timeout vs termination handled separately in GAE (see EnvState.truncation)
"""
```