# JAX RL Framework

A modular, JAX-native reinforcement learning library built on Flax Linen, designed to make **efficient robot learning accessible**.

This framework is both a **learning vehicle** and a **lab tool** — every algorithm is implemented from fundamentals with clear mappings to the papers. A new lab member should be able to read a single algorithm file and understand what's happening without chasing abstractions across ten modules.

## Design Philosophy

- **Learnable**: Heavily commented, clear mappings to papers. No magic.
- **Accessible**: Clean code the whole lab can read, modify, and extend
- **Efficient**: JAX-native with FastTD3/SAC for training in minutes, not hours
- **Robot-first**: Built for the sim-to-real pipeline
- **Modular**: Encoder+head architecture — swap components without rewriting algorithms

## Project Structure

```
jax_rl/
├── configs/
│   ├── networks_config.py   # EncoderConfig, PolicyHeadConfig, ValueHeadConfig
│   └── ppo_config.py        # PPOConfig
│
├── networks/
│   ├── encoders/
│   │   └── mlp.py           # MLPEncoder with optional LayerNorm
│   ├── heads/
│   │   ├── gaussian.py      # GaussianHead for stochastic policies
│   │   └── value.py         # ValueHead for state values
│   ├── builders.py          # Actor/Critic Linen modules
│   └── distributions.py     # Gaussian sampling, log_prob, entropy
│
├── algos/
│   └── ppo.py               # PPO (Linen functional style)
│
├── buffers/
│   └── rollout.py           # RolloutBuffer with GAE
│
└── [future]
    ├── envs/                # MuJoCo Playground adapter, Gymnasium fallback
    ├── training/            # Trainer, Wandb logger, Orbax checkpointing
    └── utils/               # RNG helpers, pytree utils, metrics
```

## Current Status

### Phase 1 - Foundation (done)
- [x] Config dataclasses
- [x] MLPEncoder with LayerNorm
- [x] GaussianHead and ValueHead
- [x] Actor/Critic builders (Linen modules)
- [x] Distribution utilities (sample_gaussian, log_prob, entropy)
- [x] RolloutBuffer with GAE

### Phase 2 - PPO (~90%)
- [x] PPO algorithm class (Linen functional: TrainingState in, TrainingState out)
- [x] Clipped surrogate objective
- [x] Separate actor/critic optimizers
- [x] Entropy bonus, gradient clipping, advantage normalization
- [ ] Jitted training loop (collect + update via `jax.lax.scan`)
- [ ] MuJoCo Playground env adapter
- [ ] Wandb logger
- [ ] Validate on CartpoleBalance / CheetahRun

## Quick Example

```python
import jax
import jax.numpy as jnp
from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.algos.ppo import PPO, TrainingState

# Configure
config = PPOConfig(
    encoder=EncoderConfig(obs_dim=17, hidden_dim=(256, 256)),
    policy_head=PolicyHeadConfig(action_dim=6, squash=True),
    num_envs=4096,
    num_steps=32,
)

# Initialize (Linen style: params are separate pytrees)
ppo = PPO(config, obs_dim=17, action_dim=6)
key = jax.random.PRNGKey(0)
training_state = ppo.init(key)

# Select actions
obs = jnp.zeros((4096, 17))
action, log_prob, value = ppo.select_action(training_state, obs, key)

# Update from rollout batch
new_state, metrics = ppo.update(training_state, batch, key)
```

## Why Flax Linen?

Linen's explicit functional pattern (`model.init(key, x)` -> params, `model.apply(params, x)`) maps cleanly onto RL's "params-in, metrics-out" training loop. Params are plain pytrees that flow through `jax.jit`, `jax.grad`, `jax.vmap` with zero Python overhead. This matches Brax, FastTD3, and CleanRL-JAX. See the [design doc](.context/rl_framework_plan.md) for the full rationale.

## Dependencies

```bash
pip install jax flax optax
```

## Design Documentation

See [.context/rl_framework_plan.md](.context/rl_framework_plan.md) for the full design document including architecture, implementation phases, benchmark targets, and reference implementations.
