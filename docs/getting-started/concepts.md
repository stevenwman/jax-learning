# Concepts

This page explains the key abstractions in jax-learning. Understanding these will help you navigate the codebase and customize training for your own tasks.

## Three-layer architecture

jax-learning uses a three-layer architecture:

```
Environment  →  Training Script  →  Algorithm
```

- **Environment** defines the physics, observations, and rewards.
- **Training Script** is the glue — it creates the environment, builds wrappers, instantiates the algorithm, and runs the training loop.
- **Algorithm** only sees observation arrays and reward scalars. It never knows about the environment directly.

This means you can swap algorithms without touching the environment, or swap environments without touching the algorithm.

## Environments

Environments are subclasses of `MjxEnv`. Each defines a MuJoCo model, observation function, and reward function.

For the Go2 quadruped, observations are dictionaries:

```python
{
    "state": jnp.array(shape=(48,)),              # what the real robot can see
    "privileged_state": jnp.array(shape=(116,)),   # extra sim info (friction, etc.)
}
```

### Two backends

| Backend | Engine | Best for |
|---------|--------|----------|
| **MJX** | JAX-native MuJoCo | Simple envs (Cartpole, Cheetah, Humanoid) |
| **Warp** | MuJoCo Warp | Go2 tasks — supports cylinder collisions and the exact Unitree MJCF model |

!!! tip "When to use which"
    If you're working with Go2 environments, use Warp. For standard benchmarks (Cheetah, Humanoid, etc.), MJX works well.

## Algorithms

Five algorithms, each a self-contained class with no inheritance hierarchy:

| Algorithm | Type | Key trait |
|-----------|------|-----------|
| **PPO** | On-policy | Fast (~110k steps/sec), uses `lax.scan` for vectorized rollouts |
| **SAC** | Off-policy | Entropy-regularized, good sample efficiency |
| **TD3** | Off-policy | Twin critics, delayed policy updates |
| **FastSAC** | Off-policy | Distributional C51 critic, large batch training |
| **FastTD3** | Off-policy | Distributional C51 critic, large batch training |

!!! note "On-policy vs off-policy"
    **On-policy** (PPO) collects fresh experience every iteration and discards it after one update. Simple and stable, but needs many environment steps. **Off-policy** (SAC, TD3, Fast variants) stores experience in a replay buffer and reuses it across many updates — more sample-efficient, but trickier to tune.

## Configs

Every training run is defined by two config objects:

- **`TrainConfig`** — Shared settings: `env_name`, `num_envs`, `total_timesteps`, `lr`, etc.
- **Algo-specific config** — `PPOConfig`, `SACConfig`, `FastSACConfig`, `FastTD3Config`, etc.

**Presets** in `env_presets.py` return `(TrainConfig, AlgoConfig)` tuples with known-good hyperparameters:

```python
from jax_rl.configs.env_presets import cheetah_fast_td3

train_cfg, algo_cfg = cheetah_fast_td3()
```

This keeps training scripts short — you pick a preset and override only what you need.

## Wrappers

Wrappers transform environments without modifying the environment class. They're applied via `build_wrapper_pipeline(cfg)`:

| Wrapper | Purpose |
|---------|---------|
| `VmapWrapper` | Vectorizes the environment across `num_envs` parallel instances |
| `EpisodeWrapper` | Handles episode truncation (max steps) |
| `AutoResetWrapper` | Automatically resets environments when episodes end |
| `FrameStackWrapper` | Stacks consecutive observations for temporal context |
| `ActionDelayWrapper` | Adds latency to simulate real-world communication delays (sim-to-real) |

## RewardSpec and ObsSpec

Rewards and observations are built from composable terms, so you can add or remove components without editing the environment class.

**Reward terms:**

```python
RewardTerm(name="tracking_lin_vel", fn=reward_tracking_lin_vel, weight=1.0)
RewardTerm(name="action_rate", fn=reward_action_rate, weight=0.01)
```

**Observation terms:**

```python
ObsTerm(name="joint_pos", fn=obs_joint_pos, noise_scale=0.01)
ObsTerm(name="joint_vel", fn=obs_joint_vel, noise_scale=0.05)
```

Each environment defines lists of these terms. At runtime, they're automatically composed into the full reward signal and observation vector.

## Training scripts

Three entry points cover all use cases:

| Script | Use |
|--------|-----|
| `train_ppo_fast.py` | On-policy training (PPO) |
| `train_offpolicy.py --algo sac\|td3\|fast_sac\|fast_td3` | Off-policy training |
| `record_video.py` | Load a checkpoint and render a video |

## Next steps

- [Quickstart](quickstart.md) — Train your first policy
- [Train Locomotion](../tutorials/train-locomotion.md) — Train a Go2 quadruped to walk
- [API Reference](../api/algos.md) — Detailed algorithm documentation
