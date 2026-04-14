# jax-learning

JAX-based reinforcement learning framework for robot learning research. Built for MuJoCo Playground environments, with a focus on Unitree Go2 locomotion and sim-to-real transfer.

**[Documentation](https://stevenwman.github.io/jax-learning/)** | **[Quickstart](https://stevenwman.github.io/jax-learning/getting-started/quickstart/)** | **[Annotated Training Loop](https://stevenwman.github.io/jax-learning/reference/training-loop/)** | **[API Reference](https://stevenwman.github.io/jax-learning/api/algos/)**

## Quick Start

```bash
# Install dependencies
uv sync

# PPO on CartpoleBalance (fastest sanity check, ~2 min)
uv run python train_ppo_fast.py --env CartpoleBalance --total-timesteps 1000000

# SAC on CheetahRun (standard benchmark, ~8 min)
uv run python train_sac.py --env CheetahRun

# FastSAC on Go2 with domain randomization (best locomotion transfer)
uv run python train_fast_sac.py --env Go2WarpJoystickFlat --reset-mode per_step --wandb

# Record a video of a trained policy
MUJOCO_GL=egl uv run python record_video.py --checkpoint checkpoints/<run_dir>/best
```

## Algorithms

Each off-policy algo has its own entry script. The four below share a single loop implementation in `jax_rl/training/offpolicy_loop.py` (`run_offpolicy_loop`); FlashSAC stays standalone for BatchNorm + Zeta noise + adaptive reward scaling reasons (see [Architecture](https://stevenwman.github.io/jax-learning/reference/architecture/)).

| Algorithm | Type | Script | Notes |
|-----------|------|--------|-------|
| **PPO** | On-policy | `train_ppo_fast.py` | `lax.scan` rollout. `train_ppo.py` is the Python-loop variant (slower, easier to read). |
| **SAC** | Off-policy | `train_sac.py` | General continuous control baseline. |
| **TD3** | Off-policy | `train_td3.py` | Deterministic-policy twin-critic baseline. |
| **FastSAC** | Off-policy | `train_fast_sac.py` | SAC + C51 distributional critic + `policy_delay=4`. Large batches (8192), UTD 8. |
| **FastTD3** | Off-policy | `train_fast_td3.py` | TD3 + C51 distributional critic. Same scale as FastSAC. |
| **FlashSAC** | Off-policy | `train_flashsac.py` | Inverted residual blocks + BatchNorm + adaptive reward scaling. Standalone loop. |

Note: for FastSAC/FastTD3 the target-Q Polyak update is gated by `policy_delay`, so effective per-critic-step target decay is `tau / policy_delay`, not `tau`. See [API → configs](https://stevenwman.github.io/jax-learning/api/configs/).

## Examples

### DM Control benchmarks

```bash
# SAC on CheetahRun (~8 min, single seed)
uv run python train_sac.py --env CheetahRun

# SAC on WalkerWalk
uv run python train_sac.py --env WalkerWalk

# SAC on HumanoidRun (21-dim actions, needs obs normalization)
uv run python train_sac.py --env HumanoidRun --obs-norm

# FastSAC at scale
uv run python train_fast_sac.py --env HumanoidRun --obs-norm

# PPO on CheetahRun
uv run python train_ppo_fast.py --env CheetahRun --total-timesteps 20000000
```

### Go2 quadruped locomotion

```bash
# FastSAC on Go2 joystick (Warp backend, unitree MJCF, asymmetric critic)
uv run python train_fast_sac.py --env Go2WarpJoystickFlat --num-envs 1024

# FastSAC + per-step domain randomization (recommended for sim-to-real transfer)
uv run python train_fast_sac.py --env Go2WarpJoystickFlat --num-envs 1024 --reset-mode per_step

# FlashSAC on Go2 (standalone loop)
uv run python train_flashsac.py --env Go2WarpJoystickFlat --total-timesteps 10000000

# Sim2sim validation on CPU MuJoCo (same unitree MJCF)
MUJOCO_GL=egl uv run python deploy/sim2sim_direct.py \
    --checkpoint checkpoints/<warp_go2_checkpoint>/best \
    --vx 0.5 --duration 10 --record /tmp/sim2sim.mp4
```

**Go2 backend:** `Go2WarpJoystickFlat` uses MuJoCo Warp with full cylinder collision geometry (unitree MJCF). Eliminates the MJX → real sim2sim gap. Asymmetric critic: 48d actor obs (deployable sensors) + 122d privileged critic obs.

### Recording videos

```bash
# Record from checkpoint — headless EGL, works on servers without display
MUJOCO_GL=egl uv run python record_video.py --checkpoint checkpoints/<run_dir>/best
```

`record_video.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` by default so it coexists with concurrent training on the same GPU. Uses a Python loop over `jit(rollout_step)` rather than `lax.scan` to keep peak HBM low (~300 MB vs ~1 GB for scan).

### Common flags

```bash
# All training scripts support:
--env NAME                       # Environment name
--seed N                         # Random seed (default: 0)
--total-timesteps N              # Total environment steps
--num-envs N                     # Parallel envs (default: varies by preset)
--reset-mode MODE                # "legacy" (default) or "per_step" (enables DomainRandWrapper)
--obs-norm                       # Sample-time observation normalization
--frame-stack N                  # Stack N observation frames
--action-delay-ms N              # Simulate fixed N ms action latency
--action-delay-range-ms MIN MAX  # Randomized per-episode delay
--resume PATH                    # Resume from checkpoint
--wandb                          # Enable W&B experiment tracking
```

Full CLI reference: [docs.../reference/cli-flags](https://stevenwman.github.io/jax-learning/reference/cli-flags/).

## Project Structure

```
├── train_ppo_fast.py           # PPO (lax.scan rollout, fastest)
├── train_ppo.py                # PPO (Python loop, easier to read)
├── train_sac.py                # SAC          ─┐
├── train_td3.py                # TD3           │  thin wrappers around
├── train_fast_sac.py           # FastSAC       │  run_offpolicy_loop
├── train_fast_td3.py           # FastTD3      ─┘
├── train_flashsac.py           # FlashSAC (standalone)
├── record_video.py             # Render a rollout from a checkpoint
│
├── jax_rl/
│   ├── algos/                  # Pure-math algo implementations (no env knowledge)
│   │   ├── ppo.py              #   Proximal Policy Optimization
│   │   ├── sac.py, td3.py      #   Vanilla SAC / TD3
│   │   ├── fast_sac.py         #   SAC + C51 distributional critic
│   │   ├── fast_td3.py         #   TD3 + C51 distributional critic
│   │   └── flash_sac.py        #   Inverted residual + BatchNorm + weight norm
│   │
│   ├── training/               # Shared training plumbing
│   │   ├── offpolicy_loop.py   #   run_offpolicy_loop — shared SAC/TD3/FastSAC/FastTD3 loop
│   │   ├── env_setup.py        #   EnvBundle + make_env_bundle (env + step + obs dims)
│   │   ├── obs_pipeline.py     #   ObsPipeline (dict obs, running mean/std, frame stacking)
│   │   ├── checkpointing.py    #   Checkpoint save/load (orbax + meta.json)
│   │   └── eval_runner.py      #   Periodic eval + best-checkpoint tracking
│   │
│   ├── configs/                # Hyperparameter dataclasses + env presets
│   ├── networks/               # Encoder + head builders (MLP, Gaussian, C51, etc.)
│   ├── envs/
│   │   ├── locomotion/         #   Go2 (Warp + MJX archived), bongo handstand
│   │   └── wrappers/           #   Vmap, Episode, AutoReset, DomainRand, FrameStack, ActionDelay
│   ├── buffers/                # Off-policy replay + PPO rollout buffers
│   └── utils/                  # Normalization, distributional math
│
├── deploy/                     # Sim2sim + real-hardware deploy (separate Python 3.12 venv)
├── docs/                       # MkDocs site source
├── tests/                      # pytest suite (includes docs-drift tests)
├── tools/                      # Diagnostic scripts (kinematic sweep, Brax baselines)
└── .context/                   # Project journal, lessons, plans (internal voice)
```

## Benchmark Results

All numbers are **single-seed, post-truncation-fix (2026-04-12)**. Pre-fix numbers on long-horizon tasks (Go2, Humanoid) may not reproduce — the fix removed a systematic Q-underestimation bias. See [`lessons-learned.md`](https://stevenwman.github.io/jax-learning/reference/lessons-learned/) for context.

| Environment | PPO | SAC | FastSAC | FastTD3 | FlashSAC |
|-------------|----:|----:|--------:|--------:|---------:|
| CheetahRun (5M) | — | 771 | — | **515.9** | — |
| WalkerWalk (5M) | 833 | 975 | — | — | — |
| Go2WarpJoystickFlat (10–20M) | 132 | — | **283.8** (per_step DR) | 273.1 (per_step DR) | **284.5** |

FastTD3 CheetahRun 515.9 is at 5M steps (shorter than the ~880 number under 86M-step paper regime). Go2 scores within seed variance of each other — FlashSAC slightly ahead but A/B not yet characterized across seeds. See [AGENT_HANDOFF](.context/AGENT_HANDOFF.md) for full benchmark table.

## Key Design Decisions

- **Three-layer separation** — env / training / algo. Algos never import environments; they receive batches of `(obs, action, reward, next_obs, done, truncation)` and return updated parameters.
- **Shared off-policy loop** — `run_offpolicy_loop` handles the SAC/TD3/FastSAC/FastTD3 training script surface. Per-algo scripts just build the algo + optimizer + explore closure.
- **JAX-native** — everything on GPU via JAX/Flax. No PyTorch dependency.
- **MuJoCo Playground** — MJX or Warp for GPU-parallelized physics (1024+ envs). Warp is primary for Go2 (full cylinder collision).
- **Asymmetric critic** — off-policy algos support privileged critic obs (122d) with deployable actor obs (48d). Critic is discarded at deployment.
- **Closure-based networks** — algos build their networks once in `__init__` as closures; JIT-friendly, trace-stable.
- **Truncation-aware** — off-policy loss masks pure-timeout transitions; `done`-terminated transitions keep their bootstrap-zero target.

## For New Contributors

1. Read [Concepts](https://stevenwman.github.io/jax-learning/getting-started/concepts/) for the mental model
2. Read [Annotated Training Loop](https://stevenwman.github.io/jax-learning/reference/training-loop/) — the off-policy loop, end-to-end
3. Read `jax_rl/algos/sac.py` — simplest algo, shows the closure pattern all others use
4. Read `.context/LESSONS.md` — debugging gotchas, saves hours

## Requirements

- Python 3.13+
- NVIDIA GPU with CUDA (no CPU-only training — JAX compiles to GPU)
- Linux (CUDA JAX doesn't support macOS or Windows)
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync  # installs everything
```

No GPU? The CartpoleBalance quickstart runs on Google Colab's free T4 tier; Go2 needs a paid GPU or local RTX 3060+.
