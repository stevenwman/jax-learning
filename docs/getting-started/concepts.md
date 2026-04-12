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

!!! tip "New to RL or JAX?"
    The [Glossary](../glossary.md) defines every term used on this page — JIT, vmap, lax.scan, on-policy, off-policy, GAE, and more.

This means you can swap algorithms without touching the environment, or swap environments without touching the algorithm.

## Environments

Environments are subclasses of `MjxEnv`. Each defines a MuJoCo model, observation function, and reward function.

For the Go2 quadruped, observations are dictionaries:

```python
{
    "state": jnp.array(shape=(48,)),              # what the real robot can see
    "privileged_state": jnp.array(shape=(122,)),   # extra sim info (friction, contacts, etc.)
}
```

Privileged observations include extra information the real robot can't measure (like ground-truth contact forces or friction coefficients) — useful for training a stronger critic, but invisible at deployment. See [Asymmetric Critic](../tutorials/asymmetric-critic.md) for how this is used.

### Physics backends

| Backend | Engine | Status | Used for |
|---------|--------|--------|----------|
| **Warp** | MuJoCo Warp | **Primary** | Go2 tasks — cylinder collisions, exact Unitree MJCF, sim2real validated |
| **MJX** | JAX-native MuJoCo | Benchmarks only | DM Control Suite (Cartpole, Cheetah, Humanoid). Go2 MJX env is archived. |

## Algorithms

Six algorithms, each a self-contained class with no inheritance hierarchy.

!!! tip "Which should I use?"
    - **New to RL?** Start with **PPO** on `CartpoleBalance` — it's the most forgiving and trains in under a minute.
    - **Training a robot (e.g. Go2 locomotion)?** Use **FastSAC**. It's the current default for Go2 and reaches a working policy in ~8 minutes.
    - **Want maximum sample efficiency at large scale?** Try **FlashSAC** or **FastTD3** (1000+ parallel envs, tens of millions of steps).
    - **Everything else** (SAC, TD3) is useful for algorithm comparisons and smaller-scale experiments.

| Algorithm | Type | Key trait |
|-----------|------|-----------|
| **PPO** | On-policy | Fast (~110k steps/sec), uses `lax.scan` for vectorized rollouts |
| **SAC** | Off-policy | Entropy-regularized, good sample efficiency |
| **TD3** | Off-policy | Twin critics, delayed policy updates |
| **FastSAC** | Off-policy | Distributional C51 critic, large batch training |
| **FastTD3** | Off-policy | Distributional C51 critic, large batch training |
| **FlashSAC** | Off-policy | Inverted residual blocks, BatchNorm, weight norm, adaptive reward scaling |

!!! note "On-policy vs off-policy"
    **On-policy** (PPO) collects fresh experience every iteration and discards it after one update. Simple and stable, but needs many environment steps. **Off-policy** (SAC, TD3, Fast variants) stores experience in a replay buffer and reuses it across many updates — more sample-efficient, but trickier to tune.

## Configs

Every training run is defined by two config objects:

- **`TrainConfig`** — Shared settings: `env_name`, `num_envs`, `total_timesteps`, `lr`, etc.
- **Algo-specific config** — `PPOConfig`, `SACConfig`, `FastSACConfig`, `FastTD3Config`, etc.

**Presets** in `env_presets.py` return `(TrainConfig, AlgoConfig)` tuples with known-good hyperparameters:

```python
from jax_rl.configs.env_presets import get_fast_td3_preset

train_cfg, algo_cfg = get_fast_td3_preset("CheetahRun")
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
RewardTerm(name="tracking_lin_vel", fn=reward_tracking_lin_vel)
RewardTerm(name="action_rate", fn=reward_action_rate)
```

**Observation terms:**

```python
ObsTerm(name="joint_pos", fn=obs_joint_pos, noise_scale=0.01)
ObsTerm(name="joint_vel", fn=obs_joint_vel, noise_scale=0.05)
```

Each environment defines lists of these terms. Weights are applied separately in the env's `step()` method via a config dict — not in `RewardTerm` itself. This lets you retune weights without modifying reward functions.

## Training scripts

| Script | Use |
|--------|-----|
| `train_ppo_fast.py` | On-policy training (PPO) with `lax.scan` |
| `train_ppo.py` | On-policy training (PPO) with Python loop — slower, supports non-JIT envs |
| `train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py` | Off-policy training (per-algorithm scripts) |
| `train_flashsac.py` | FlashSAC training (standalone script) |
| `record_video.py` | Load a checkpoint and render a video |

## Next steps

- [Quickstart](quickstart.md) — Train your first policy
- [Train Locomotion](../tutorials/train-locomotion.md) — Train a Go2 quadruped to walk
- [API Reference](../api/algos.md) — Detailed algorithm documentation
