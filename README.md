# jax-learning

JAX-based reinforcement learning framework for robot learning research. Built for MuJoCo Playground environments, with a focus on locomotion and sim-to-real transfer.

## Quick Start

```bash
# Install dependencies
uv sync

# Train SAC on CheetahRun (simplest benchmark, ~8 min)
uv run python train_offpolicy.py --algo sac --env CheetahRun

# Train PPO on CartpoleBalance (fastest sanity check, ~2 min)
uv run python train_ppo_fast.py --env CartpoleBalance --total-timesteps 1000000

# Record a video of a trained policy
MUJOCO_GL=egl uv run python record_video.py --checkpoint checkpoints/<your_checkpoint>
```

## Algorithms

| Algorithm | Type | Script | Best Use Case |
|-----------|------|--------|---------------|
| **PPO** | On-policy | `train_ppo_fast.py` | Locomotion (Go2), high-dim obs |
| **SAC** | Off-policy | `train_offpolicy.py --algo sac` | General continuous control |
| **TD3** | Off-policy | `train_offpolicy.py --algo td3` | Low-dim action spaces |
| **FastTD3** | Off-policy | `train_offpolicy.py --algo fast_td3` | Large-scale (1024 envs), C51 distributional |
| **FastSAC** | Off-policy | `train_offpolicy.py --algo fast_sac` | Large-scale, high-dim actions (humanoid) |

`train_ppo_fast.py` uses `jax.lax.scan` for the collection phase and is ~3x faster than `train_ppo.py`. Use it for real training runs.

## Examples

### DM Control Suite benchmarks

```bash
# SAC on CheetahRun (expect ~771 eval @ 5M steps, ~8 min)
uv run python train_offpolicy.py --algo sac --env CheetahRun

# SAC on WalkerWalk (expect ~975 eval @ 5M steps)
uv run python train_offpolicy.py --algo sac --env WalkerWalk

# SAC on HumanoidRun (21-dim actions, expect ~426 eval @ 20M steps)
uv run python train_offpolicy.py --algo sac --env HumanoidRun --obs-norm

# FastSAC at scale (1024 envs, C51 distributional critic)
uv run python train_offpolicy.py --algo fast_sac --env HumanoidRun --obs-norm

# PPO on CheetahRun
uv run python train_ppo_fast.py --env CheetahRun --total-timesteps 20000000
```

### Go2 quadruped locomotion

```bash
# PPO on Go2 joystick walking (MJX backend, Menagerie MJCF)
uv run python train_ppo_fast.py --env Go2JoystickFlat --total-timesteps 50000000

# FastSAC on Go2 (Warp backend, unitree MJCF — best for sim2sim/sim2real)
uv run python train_offpolicy.py --algo fast_sac --env Go2WarpJoystickFlat --num-envs 1024

# FastSAC + domain randomization (recommended for transfer)
uv run python train_offpolicy.py --algo fast_sac --env Go2WarpJoystickFlat --num-envs 1024 --domain-rand

# Record a video of the trained walking policy
MUJOCO_GL=egl uv run python record_video.py --checkpoint checkpoints/<go2_checkpoint>

# Sim2sim validation on CPU MuJoCo (same unitree MJCF as Warp training)
MUJOCO_GL=egl uv run python deploy/sim2sim_direct.py \
    --checkpoint checkpoints/<warp_go2_checkpoint> \
    --vx 0.5 --duration 10 --record /tmp/sim2sim.mp4
```

**Two Go2 backends:**
- `Go2JoystickFlat` — MJX (JAX) backend, Menagerie go2_mjx.xml (simplified collision geometry). Fast, proven.
- `Go2WarpJoystickFlat` — MuJoCo Warp backend, unitree go2.xml (full cylinder collision geometry). Eliminates sim2sim gap. Use for deployment.

### Recording and visualizing policies

```bash
# Record video from checkpoint (headless — works on servers without a display)
MUJOCO_GL=egl uv run python record_video.py --checkpoint checkpoints/<checkpoint_dir>

# Record with custom settings
MUJOCO_GL=egl uv run python record_video.py \
    --checkpoint checkpoints/<checkpoint_dir> \
    --out my_video.mp4 \
    --max-steps 1000

# Live interactive viewer (requires display, Go2 joystick envs only)
# Arrow keys control velocity commands, space resets
uv run python live_viewer.py --checkpoint checkpoints/<go2_checkpoint>
```

### Common flags

```bash
# All training scripts support:
--env NAME              # Environment name (e.g., CheetahRun, Go2JoystickFlat)
--seed N                # Random seed (default: 0)
--total-timesteps N     # Total environment steps
--num-envs N            # Parallel environments (default: varies by env preset)
--eval-every N          # Evaluate every N episodes
--resume PATH           # Resume training from a checkpoint
--obs-norm              # Enable sample-time observation normalization

# Off-policy only (train_offpolicy.py):
--algo NAME             # Algorithm: sac, td3, fast_td3, fast_sac
--exploration-noise F   # TD3 exploration noise std (default: from config)
```

## Project Structure

```
├── train_ppo_fast.py          # PPO training (lax.scan collect, fastest)
├── train_ppo.py               # PPO training (Python loop, easier to read)
├── train_offpolicy.py         # Unified off-policy: SAC, TD3, FastTD3, FastSAC
├── record_video.py            # Record policy videos from checkpoints
├── live_viewer.py             # Interactive policy viewer (Go2)
│
├── jax_rl/
│   ├── algos/                 # Algorithm implementations
│   │   ├── ppo.py             #   Proximal Policy Optimization
│   │   ├── sac.py             #   Soft Actor-Critic
│   │   ├── td3.py             #   Twin Delayed DDPG
│   │   ├── fast_td3.py        #   TD3 + C51 distributional critic
│   │   └── fast_sac.py        #   SAC + C51 distributional critic
│   │
│   ├── networks/
│   │   ├── builders.py        #   Composed modules (encoder + head, swappable)
│   │   ├── activations.py     #   Activation function registry
│   │   ├── distributions.py   #   Gaussian sampling, log_prob, entropy
│   │   ├── encoders/
│   │   │   └── mlp.py         #   MLP encoder (obs → features)
│   │   └── heads/
│   │       ├── gaussian.py    #   Stochastic policy head (PPO, SAC)
│   │       ├── deterministic.py # Deterministic policy head (TD3)
│   │       ├── value.py       #   V(s) head (PPO critic)
│   │       ├── q_head.py      #   Scalar Q(s,a) head (SAC, TD3)
│   │       └── q_distributional.py # C51 Q(s,a) head (FastTD3, FastSAC)
│   │
│   ├── configs/               # Hyperparameter dataclasses + env presets
│   ├── training/              # Shared infrastructure (checkpointing, eval, logging)
│   ├── envs/                  # Custom environments (Go2 MJX + Warp backends)
│   ├── buffers/               # Replay buffer (off-policy) + rollout buffer (PPO)
│   └── utils/                 # Normalization, frame stacking, distributional math
│
├── checkpoints/               # Saved model checkpoints
├── tools/                     # Diagnostic scripts (kinematic sweep, Brax baselines)
├── tests/                     # Test suite (pytest)
└── .context/                  # Project docs, lessons, plans
```

## Benchmark Results

| Environment | PPO | SAC | TD3 | FastTD3 | FastSAC |
|-------------|-----|-----|-----|---------|---------|
| CheetahRun | 826 | **771** | 749 | **880** | 582 |
| WalkerWalk | 833 | **975** | 955 | — | — |
| HumanoidRun | ~10 | 426 | 4.3 | 665 | **892** |
| Go2 Joystick (MJX) | **244** | — | — | — | 226 |
| Go2 Joystick (Warp) | 132 | — | — | — | **276** |

SAC dominates on general continuous control. FastSAC excels on high-dim action spaces (HumanoidRun). PPO works well for locomotion with Go2 on MJX. **FastSAC on Warp achieves highest Go2 eval (276)** by training on unitree's exact MJCF.

## Key Design Decisions

- **JAX-native**: Everything runs on GPU via JAX/Flax. No PyTorch dependency.
- **MuJoCo Playground**: Uses MJX or Warp for GPU-parallelized physics (1024+ envs).
- **Encoder-swappable**: All algos use `builders.py` — swap MLP for CNN by changing the builder, not the algo.
- **Self-contained envs**: Each env handles its own obs, rewards, and action scaling. Training scripts are env-agnostic.
- **NaN/Inf safe**: MJX physics can crash stochastically. All training automatically guards against this.

## For New Contributors

Start by reading:
1. **This README** — overview and examples
2. **`.context/LESSONS.md`** — debugging lessons and gotchas (save yourself hours)
3. **`jax_rl/algos/sac.py`** — best-documented algo, explains the closure pattern all algos use
4. **`jax_rl/networks/builders.py`** — how networks are composed (encoder + head)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA
- [uv](https://docs.astral.sh/uv/) package manager

```bash
uv sync  # installs everything
```
