# CLI Flags

All training scripts accept CLI flags that override the environment preset defaults. If a flag is not specified, the value comes from the preset (see [Environment Presets](env-presets.md)).

---

## train_ppo_fast.py

PPO training with `lax.scan`-based collection. Suitable for JAX-native envs (MJX, Playground).

```bash
uv run python train_ppo_fast.py --env CheetahRun --num-envs 2048 --wandb
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `CartpoleBalance` | Environment name (e.g., `CheetahRun`, `Go2JoystickFlat`) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | `None` | Resume from checkpoint directory path |
| `--num-envs` | int | from preset | Number of parallel environments |
| `--num-steps` | int | from preset | Rollout steps per env before each update |
| `--num-updates-per-batch` | int | from preset | Collect-update cycles per iteration |
| `--total-timesteps` | int | from preset | Total environment steps to train |
| `--lr` | float | from preset | Learning rate |
| `--policy-hidden-dim` | int+ | from preset | Actor network hidden layer sizes (e.g., `512 256 128`) |
| `--value-hidden-dim` | int+ | from preset | Critic network hidden layer sizes (e.g., `512 256 128`) |
| `--entropy-coef` | float | from preset | Entropy bonus coefficient (higher = more exploration) |
| `--eval-every` | int | from preset | Evaluate every N episodes |
| `--reward-scaling` | float | from preset | Multiply rewards by this factor |
| `--episode-length` | int | from preset | Max steps per episode |
| `--log-interval` | int | from preset | Print training stats every N iterations |
| `--domain-rand` | flag | `False` | Enable domain randomization (Go2 only) |
| `--wandb` | flag | `False` | Enable W&B experiment tracking |
| `--wandb-project` | str | `jax-rl` | W&B project name |
| `--frame-stack` | int | from preset | Number of stacked observation frames (1 = no stacking) |
| `--action-delay-ms` | int | from preset | Fixed action delay in ms (e.g., `120` for Go2 sim2real) |
| `--action-delay-range-ms` | int int | from preset | Randomized action delay range in ms (e.g., `40 120`) |

---

## train_offpolicy.py

Unified off-policy training for SAC, TD3, FastSAC, and FastTD3.

```bash
uv run python train_offpolicy.py --algo fast_sac --env HumanoidRun --obs-norm
uv run python train_offpolicy.py --algo fast_td3 --env CheetahRun --exploration-noise 0.15
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--algo` | str | **required** | RL algorithm: `sac`, `td3`, `fast_td3`, `fast_sac` |
| `--env` | str | `WalkerWalk` | Environment name (e.g., `CheetahRun`, `HumanoidRun`, `Go2JoystickFlat`) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | `None` | Resume from checkpoint directory path |
| `--num-envs` | int | from preset | Number of parallel environments |
| `--total-timesteps` | int | from preset | Total environment steps to train |
| `--lr` | float | from preset | Learning rate for actor and critic |
| `--batch-size` | int | from preset | Batch size for gradient updates |
| `--grad-updates-per-step` | int | from preset | Gradient updates per env step (UTD ratio) |
| `--buffer-size` | int | from preset | Replay buffer capacity |
| `--reward-scaling` | float | from preset | Multiply rewards by this factor |
| `--episode-length` | int | from preset | Max steps per episode |
| `--exploration-noise` | float | from preset | Exploration noise std (TD3-family only; SAC uses entropy) |
| `--target-entropy-scale` | float | from preset | `target_entropy = -scale * action_dim` (SAC-family only) |
| `--eval-every` | int | from preset | Evaluate every N episodes |
| `--obs-norm` | flag | `False` | Enable sample-time obs normalization (recommended for humanoid) |
| `--domain-rand` | flag | `False` | Enable domain randomization (Go2 only) |
| `--wandb` | flag | `False` | Enable W&B experiment tracking |
| `--wandb-project` | str | `jax-rl` | W&B project name |
| `--frame-stack` | int | from preset | Number of stacked observation frames (1 = no stacking) |
| `--action-delay-ms` | int | from preset | Fixed action delay in ms (e.g., `120` for Go2 sim2real) |
| `--action-delay-range-ms` | int int | from preset | Randomized action delay range in ms (e.g., `40 120`) |

---

## record_video.py

Record rollout videos from trained checkpoints or random policies.

```bash
MUJOCO_GL=egl uv run python record_video.py --checkpoint checkpoints/my_run --kicks
MUJOCO_GL=egl uv run python record_video.py --env CartpoleBalance  # random policy
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | from checkpoint | Environment name (auto-detected from checkpoint if not set) |
| `--checkpoint` | str | `None` | Checkpoint directory path (omit for random policy) |
| `--out` | str | `rollout.mp4` | Output video path (ignored if checkpoint is set; saves inside checkpoint dir) |
| `--max-steps` | int | `1000` | Maximum rollout steps |
| `--camera` | str | env default | Camera name for rendering (e.g., `side`, `track`, `fixed`) |
| `--seed` | int | `0` | Random seed for env reset |
| `--kicks` | flag | `False` | Zero velocity command + random velocity kicks every 1.5s (Go2) |

The script auto-detects the algorithm type from `meta.json` in the checkpoint and reconstructs the correct actor network. Works with all algo types (PPO, SAC, TD3, FastSAC, FastTD3).
