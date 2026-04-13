# CLI Flags

Auto-generated from argparse definitions. Regenerate with:

```bash
uv run python docs/scripts/gen_cli_reference.py
```

---

## `train_ppo_fast.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `CartpoleBalance` | Environment name |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Resume from checkpoint directory path |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--num-steps` | int | from preset | Rollout steps per environment per collect phase (default: from preset) |
| `--num-updates-per-batch` | int | from preset | Collect-update cycles per iteration (default: from preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Peak learning rate (default: from env preset) |
| `--policy-hidden-dim` | int+ | - | Policy network hidden layer sizes (e.g., 256 128) |
| `--value-hidden-dim` | int+ | - | Value network hidden layer sizes (e.g., 256 256 256) |
| `--entropy-coef` | float | from preset | Entropy bonus coefficient (default: from env preset) |
| `--eval-every` | int | - | Evaluate every N episodes (default: every 50000 episodes) |
| `--reward-scaling` | float | from preset | Multiply rewards by this factor (default: from preset) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--log-interval` | int | `10` | Print training stats every N iterations |
| `--wandb` | flag | off | Enable W&B experiment tracking |
| `--wandb-project` | str | `jax-rl` | W&B project name |
| `--frame-stack` | int | - | Number of stacked observation frames |
| `--action-delay-ms` | int | - | Fixed action delay in ms |
| `--action-delay-range-ms` | int int | - | Randomized action delay range in ms |

---

## `train_sac.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `WalkerWalk` | Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Resume from checkpoint directory path |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Learning rate for actor and critic (default: from algo config) |
| `--reward-scaling` | float | - | Multiply rewards by this factor (default: 1.0) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--target-entropy-scale` | float | from preset | target_entropy = -scale * action_dim (default: from algo config) |
| `--eval-every` | int | - | Evaluate every N episodes (default: every 512 episodes) |
| `--obs-norm` | flag | off | Enable sample-time obs normalization (recommended for humanoid tasks) |
| `--wandb` | flag | off | Enable W&B experiment tracking (requires wandb installed) |
| `--wandb-project` | str | `jax-rl` | W&B project name (default: jax-rl) |
| `--frame-stack` | int | - | Number of stacked observation frames (default: 1, use 3 for locomotion) |
| `--action-delay-ms` | int | - | Fixed action delay in ms (e.g., 120 for Go2 sim2real) |
| `--action-delay-range-ms` | int int | - | Randomized action delay range in ms (e.g., 40 120) |
| `--reset-mode` | str | - | Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper) |

---

## `train_td3.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `WalkerWalk` | Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Resume from checkpoint directory path |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Learning rate for actor and critic (default: from algo config) |
| `--reward-scaling` | float | - | Multiply rewards by this factor (default: 1.0) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--exploration-noise` | float | - | Exploration noise std for TD3-family |
| `--eval-every` | int | - | Evaluate every N episodes (default: every 512 episodes) |
| `--obs-norm` | flag | off | Enable sample-time obs normalization (recommended for humanoid tasks) |
| `--wandb` | flag | off | Enable W&B experiment tracking (requires wandb installed) |
| `--wandb-project` | str | `jax-rl` | W&B project name (default: jax-rl) |
| `--frame-stack` | int | - | Number of stacked observation frames (default: 1, use 3 for locomotion) |
| `--action-delay-ms` | int | - | Fixed action delay in ms (e.g., 120 for Go2 sim2real) |
| `--action-delay-range-ms` | int int | - | Randomized action delay range in ms (e.g., 40 120) |
| `--reset-mode` | str | - | Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper) |

---

## `train_fast_sac.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `WalkerWalk` | Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Resume from checkpoint directory path |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Learning rate for actor and critic (default: from algo config) |
| `--batch-size` | int | from preset | Batch size for gradient updates (default: from algo config) |
| `--grad-updates-per-step` | int | from preset | Gradient updates per env step (UTD ratio, default: from config) |
| `--buffer-size` | int | from preset | Replay buffer capacity (default: from algo config) |
| `--reward-scaling` | float | - | Multiply rewards by this factor (default: 1.0) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--target-entropy-scale` | float | from preset | target_entropy = -scale * action_dim (default: from algo config) |
| `--eval-every` | int | - | Evaluate every N episodes (default: every 512 episodes) |
| `--obs-norm` | flag | off | Enable sample-time obs normalization (recommended for humanoid tasks) |
| `--wandb` | flag | off | Enable W&B experiment tracking (requires wandb installed) |
| `--wandb-project` | str | `jax-rl` | W&B project name (default: jax-rl) |
| `--frame-stack` | int | - | Number of stacked observation frames (default: 1, use 3 for locomotion) |
| `--action-delay-ms` | int | - | Fixed action delay in ms (e.g., 120 for Go2 sim2real) |
| `--action-delay-range-ms` | int int | - | Randomized action delay range in ms (e.g., 40 120) |
| `--reset-mode` | str | - | Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper) |

---

## `train_fast_td3.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `WalkerWalk` | Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Resume from checkpoint directory path |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Learning rate for actor and critic (default: from algo config) |
| `--batch-size` | int | from preset | Batch size for gradient updates (default: from algo config) |
| `--grad-updates-per-step` | int | from preset | Gradient updates per env step (UTD ratio, default: from config) |
| `--buffer-size` | int | from preset | Replay buffer capacity (default: from algo config) |
| `--reward-scaling` | float | - | Multiply rewards by this factor (default: 1.0) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--exploration-noise` | float | - | Exploration noise std for TD3-family |
| `--eval-every` | int | - | Evaluate every N episodes (default: every 512 episodes) |
| `--obs-norm` | flag | off | Enable sample-time obs normalization (recommended for humanoid tasks) |
| `--wandb` | flag | off | Enable W&B experiment tracking (requires wandb installed) |
| `--wandb-project` | str | `jax-rl` | W&B project name (default: jax-rl) |
| `--frame-stack` | int | - | Number of stacked observation frames (default: 1, use 3 for locomotion) |
| `--action-delay-ms` | int | - | Fixed action delay in ms (e.g., 120 for Go2 sim2real) |
| `--action-delay-range-ms` | int int | - | Randomized action delay range in ms (e.g., 40 120) |
| `--reset-mode` | str | - | Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper) |

---

## `train_flashsac.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `CartpoleBalance` | Environment name (e.g., CartpoleBalance, CheetahRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Resume from checkpoint directory path |
| `--num-envs` | int | - | Number of parallel environments |
| `--total-timesteps` | int | - | Total environment steps to train |
| `--episode-length` | int | - | Max steps per episode |
| `--batch-size` | int | - | Batch size for gradient updates |
| `--gamma` | float | - | Discount factor |
| `--lr` | float | from preset | Peak learning rate (overrides lr_peak in FlashSACConfig) |
| `--lr-end` | float | - | End learning rate for cosine decay |
| `--buffer-size` | int | - | Replay buffer capacity |
| `--grad-updates-per-step` | int | - | Gradient updates per env step (UTD ratio) |
| `--no-reward-norm` | flag | off | Disable adaptive reward normalization |
| `--G-max` | float | - | Target max magnitude for discounted returns (reward norm) |
| `--no-weight-norm` | flag | off | Disable weight normalization after optimizer steps |
| `--eval-every` | int | - | Evaluate every N episodes |
| `--wandb` | flag | off | Enable W&B experiment tracking |
| `--wandb-project` | str | `jax-rl` | W&B project name |

---

## `record_video.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | - | Environment name (inferred from checkpoint if omitted) |
| `--checkpoint` | str | - | Checkpoint directory (random policy if omitted) |
| `--out` | str | `rollout.mp4` | Output video path |
| `--max-steps` | int | `1000` | Maximum rollout steps |
| `--camera` | str | - | Camera name override |
| `--seed` | int | - | Environment reset seed |
| `--kicks` | flag | off | Zero velocity command + random velocity kicks every 1.5s |
