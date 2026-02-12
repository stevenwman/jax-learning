# JAX RL Framework

A modular, JAX-native reinforcement learning library built on Flax NNX, designed to make **efficient robot learning accessible**.

## Design Philosophy

- **Accessible**: Clean code you can actually read and modify
- **Efficient**: JAX-native with support for FastTD3/SAC
- **Robot-first**: Built for the sim-to-real pipeline
- **Modular**: Encoder+head architecture - swap components without rewriting algorithms
- **Type-safe**: Protocols for clean interfaces, dataclasses for configs

## Project Structure

```
jax_rl/
├── configs/              # Configuration dataclasses
│   ├── networks.py       # EncoderConfig, PolicyHeadConfig, ValueHeadConfig
│   └── ppo.py            # PPOConfig
│
├── networks/             # Neural network components
│   ├── protocols.py      # Encoder, PolicyHead, ValueHead protocols
│   ├── encoders/
│   │   └── mlp.py        # MLPEncoder with optional LayerNorm
│   ├── heads/
│   │   ├── gaussian.py   # GaussianHead for stochastic policies
│   │   └── value.py      # ValueHead for state values
│   ├── builders.py       # Actor/Critic builders
│   └── distributions.py  # Distribution utilities (TanhNormal, etc.)
│
├── algos/                # Algorithm implementations
│   └── ppo.py            # PPO with separate actor/critic
│
├── buffers/              # Replay/rollout buffers
│   └── rollout.py        # RolloutBuffer with GAE
│
└── [future]
    ├── envs/             # Environment adapters
    ├── training/         # Trainer, logger, checkpointing
    └── utils/            # Helper functions
```

## Current Implementation Status

### ✅ Phase 1 - Foundation (Complete)
- [x] Config dataclasses with proper defaults
- [x] Network protocols (Encoder, PolicyHead, ValueHead)
- [x] MLPEncoder with LayerNorm support
- [x] GaussianHead and ValueHead
- [x] Actor/Critic builders
- [x] Distribution utilities (sample_gaussian, log_prob, entropy)
- [x] RolloutBuffer with GAE computation

### ✅ Phase 2 - PPO Implementation (Complete)
- [x] PPO algorithm class
- [x] Clipped surrogate objective
- [x] Separate actor/critic optimizers
- [x] Entropy bonus
- [x] Gradient clipping
- [x] Optional value clipping
- [x] Advantage normalization

### 🔲 Phase 3 - Next Steps
- [ ] Environment adapters (MuJoCo Playground, Gymnasium)
- [ ] Training loop / Trainer class
- [ ] Wandb logger
- [ ] Example training script
- [ ] Test on dm_control environments

## Quick Example

```python
from flax import nnx
from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.algos import PPO

# Configure PPO
config = PPOConfig(
    encoder=EncoderConfig(
        obs_dim=17,
        hidden_dim=(256, 256),
    ),
    policy_head=PolicyHeadConfig(
        action_dim=6,
        squash=True,
    ),
    num_envs=4096,
    num_steps=32,
)

# Initialize PPO
rngs = nnx.Rngs(0)
ppo = PPO(config, obs_dim=17, action_dim=6, rngs=rngs)

# Select actions
action, log_prob, value = ppo.select_action(obs, key)

# Update from rollout
metrics = ppo.update(batch)
```

## Key Architectural Decisions

### 1. Protocols for Modularity
All network components follow protocols, making it easy to swap implementations.

### 2. Separate Actor/Critic
No `value_coef` mixing - each network has its own optimizer and loss.

### 3. Encoder + Head Composition
Networks are composed from modular pieces.

### 4. Configurable Everything
Dataclasses for type-safe configs.

## Testing

Run the test suite to verify the implementation:
```bash
python test_ppo_setup.py
```

## Design Documentation

See `.context/rl_framework_plan.md` for the full design document.

## Dependencies

```bash
pip install jax flax optax distrax
```

## Next: Training Loop

The next step is to implement:
1. Environment adapters (MuJoCo Playground with MJWarp backend)
2. Trainer class (collect -> update -> log loop)
3. Wandb integration
4. Example training script for CartPole or HalfCheetah

---

Built with ❤️ for robot learning research.
