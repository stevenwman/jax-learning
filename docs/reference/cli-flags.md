# CLI Flags

Auto-generated from argparse definitions. Regenerate with:

```bash
uv run python docs/scripts/gen_cli_reference.py
```

---

## `train_ppo.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `CartpoleBalance` | Environment name (e.g., CartpoleBalance, CheetahRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Warm-start from checkpoint directory: restores weights + opt state + norm state. Replay buffer is NOT persisted; refilled with loaded policy actions per --resume-warmup. Not exact training continuation. |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--num-steps` | int | from preset | Rollout steps per env before each update (default: from preset) |
| `--num-updates-per-batch` | int | from preset | SGD epochs over collected rollout data (default: from preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Learning rate (default: from preset) |
| `--policy-hidden-dim` | int+ | - | Actor network hidden layer sizes (e.g., 256 256) |
| `--value-hidden-dim` | int+ | - | Critic network hidden layer sizes (e.g., 256 256) |
| `--entropy-coef` | float | - | Entropy bonus coefficient (higher = more exploration) |
| `--reward-scaling` | float | - | Multiply rewards by this factor (default: 1.0) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--log-interval` | int | - | Print training stats every N iterations |
| `--eval-every` | int | - | Evaluate every N episodes (default: every 512 episodes) |
| `--wandb` | flag | off | Enable W&B experiment tracking |
| `--wandb-project` | str | `jax-rl` | W&B project name (default: jax-rl) |
| `--frame-stack` | int | - | Number of stacked observation frames (default: 1, use 3 for locomotion) |
| `--action-delay-ms` | int | - | Fixed action delay in ms (e.g., 120 for Go2 sim2real) |
| `--action-delay-range-ms` | int int | - | Randomized action delay range in ms (e.g., 40 120) |

---

## `train_ppo_fast.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `CartpoleBalance` | Environment name (e.g., CartpoleBalance, CheetahRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Warm-start from checkpoint: restores weights + opt state + norm state. Replay buffer is NOT persisted; refilled with loaded policy actions per --resume-warmup. Not exact training continuation. |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--num-steps` | int | from preset | Rollout steps per env before each update (default: from preset) |
| `--num-updates-per-batch` | int | from preset | SGD epochs over collected rollout data (default: from preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Learning rate (default: from preset) |
| `--policy-hidden-dim` | int+ | - | Actor network hidden layer sizes (e.g., 512 256 128) |
| `--value-hidden-dim` | int+ | - | Critic network hidden layer sizes (e.g., 512 256 128) |
| `--entropy-coef` | float | - | Entropy bonus coefficient (higher = more exploration) |
| `--eval-every` | int | - | Evaluate every N episodes (default: every 512 episodes) |
| `--reward-scaling` | float | - | Multiply rewards by this factor (default: 1.0) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--log-interval` | int | - | Print training stats every N iterations |
| `--wandb` | flag | off | Enable W&B experiment tracking |
| `--wandb-project` | str | `jax-rl` | W&B project name (default: jax-rl) |
| `--frame-stack` | int | - | Number of stacked observation frames (default: 1, use 3 for locomotion) |
| `--action-delay-ms` | int | - | Fixed action delay in ms (e.g., 120 for Go2 sim2real) |
| `--action-delay-range-ms` | int int | - | Randomized action delay range in ms (e.g., 40 120) |

---

## `train_ppo_contraction.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `Go2BongoHandstand` |  |
| `--seed` | int | `0` |  |
| `--resume` | str | - |  |
| `--num-envs` | int | - |  |
| `--total-timesteps` | int | - |  |
| `--lr` | float | - |  |
| `--reward-scaling` | float | - |  |
| `--episode-length` | int | - |  |
| `--log-interval` | int | - |  |
| `--frame-stack` | int | - |  |
| `--wandb` | flag | off |  |
| `--wandb-project` | str | `jax-rl` |  |
| `--alpha` | float | `0.1` | Contraction rate α |
| `--epsilon` | float | `0.001` | Strict-inequality slack ε |
| `--penalty-coef` | float | `1.0` | Reward-augment scale |
| `--metric-lr` | float | `0.001` |  |
| `--constraint-coef` | float | `1.0` |  |
| `--metric-hidden` | int+ | `[128, 128]` |  |

---

## `train_sac.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `WalkerWalk` | Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Warm-start from checkpoint: restores weights + opt state + norm state. Replay buffer is NOT persisted; refilled with loaded policy actions per --resume-warmup. Not exact training continuation. |
| `--resume-warmup` | str | `policy` | On resume, refill buffer using loaded policy actions (default, prevents eval drop) or legacy random uniform |
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
| `--buffer-size` | int | - | Replay buffer capacity (default: 4M) |
| `--batch-size` | int | - | Batch size (default: 512) |
| `--grad-updates-per-step` | int | - | Gradient updates per env step |

---

## `train_td3.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `WalkerWalk` | Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Warm-start from checkpoint: restores weights + opt state + norm state. Replay buffer is NOT persisted; refilled with loaded policy actions per --resume-warmup. Not exact training continuation. |
| `--resume-warmup` | str | `policy` | On resume, refill buffer using loaded policy actions (default, prevents eval drop) or legacy random uniform |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Learning rate for actor and critic (default: from algo config) |
| `--reward-scaling` | float | - | Multiply rewards by this factor (default: 1.0) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--exploration-noise` | float | from preset | Exploration noise std for TD3 (default: from algo config) |
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
| `--resume` | str | - | Warm-start from checkpoint: restores weights + opt state + norm state. Replay buffer is NOT persisted; refilled with loaded policy actions per --resume-warmup. Not exact training continuation. |
| `--resume-warmup` | str | `policy` | On resume, refill buffer using loaded policy actions (default, prevents eval drop) or legacy random uniform |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Learning rate for actor and critic (default: from algo config) |
| `--reward-scaling` | float | - | Multiply rewards by this factor (default: 1.0) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--target-entropy-scale` | float | from preset | target_entropy = -scale * action_dim (default: from algo config) |
| `--batch-size` | int | from preset | Batch size for gradient updates (default: from algo config) |
| `--grad-updates-per-step` | int | from preset | Gradient updates per environment step (default: from algo config) |
| `--buffer-size` | int | from preset | Replay buffer capacity (default: from algo config) |
| `--eval-every` | int | - | Evaluate every N episodes (default: every 512 episodes) |
| `--obs-norm` | flag | off | Enable sample-time obs normalization (recommended for humanoid tasks) |
| `--wandb` | flag | off | Enable W&B experiment tracking (requires wandb installed) |
| `--wandb-project` | str | `jax-rl` | W&B project name (default: jax-rl) |
| `--frame-stack` | int | - | Number of stacked observation frames (default: 1, use 3 for locomotion) |
| `--action-delay-ms` | int | - | Fixed action delay in ms (e.g., 120 for Go2 sim2real) |
| `--action-delay-range-ms` | int int | - | Randomized action delay range in ms (e.g., 40 120) |
| `--reset-mode` | str | - | Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper) |
| `--v-min` | float | from preset | C51 critic support lower bound (default: from algo config, -20 for FastSAC) |
| `--v-max` | float | from preset | C51 critic support upper bound (default: from algo config, +20 for FastSAC) |
| `--num-atoms` | int | from preset | C51 critic atom count (default: from algo config, 101) |
| `--tau` | float | - | Target network soft-update rate (default: 0.125) |
| `--gamma` | float | - | Discount factor (default: 0.99) |
| `--policy-delay` | int | - | Critic updates per actor update (default: 4) |

---

## `train_fast_td3.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `WalkerWalk` | Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Warm-start from checkpoint: restores weights + opt state + norm state. Replay buffer is NOT persisted; refilled with loaded policy actions per --resume-warmup. Not exact training continuation. |
| `--resume-warmup` | str | `policy` | On resume, refill buffer using loaded policy actions (default, prevents eval drop) or legacy random uniform |
| `--num-envs` | int | from preset | Number of parallel environments (default: from env preset) |
| `--total-timesteps` | int | from preset | Total environment steps to train (default: from env preset) |
| `--lr` | float | from preset | Learning rate for actor and critic (default: from algo config) |
| `--reward-scaling` | float | - | Multiply rewards by this factor (default: 1.0) |
| `--episode-length` | int | from preset | Max steps per episode (default: from env preset) |
| `--exploration-noise` | float | from preset | Exploration noise std for TD3 (default: from algo config) |
| `--eval-every` | int | - | Evaluate every N episodes (default: every 512 episodes) |
| `--obs-norm` | flag | off | Enable sample-time obs normalization (recommended for humanoid tasks) |
| `--wandb` | flag | off | Enable W&B experiment tracking (requires wandb installed) |
| `--wandb-project` | str | `jax-rl` | W&B project name (default: jax-rl) |
| `--frame-stack` | int | - | Number of stacked observation frames (default: 1, use 3 for locomotion) |
| `--action-delay-ms` | int | - | Fixed action delay in ms (e.g., 120 for Go2 sim2real) |
| `--action-delay-range-ms` | int int | - | Randomized action delay range in ms (e.g., 40 120) |
| `--reset-mode` | str | - | Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper) |
| `--batch-size` | int | from preset | Batch size for gradient updates (default: from algo config) |
| `--grad-updates-per-step` | int | from preset | Gradient updates per env step (default: from algo config) |
| `--buffer-size` | int | from preset | Replay buffer capacity (default: from algo config) |

---

## `train_flashsac.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | `CartpoleBalance` | Environment name (e.g., CartpoleBalance, CheetahRun, Go2WarpJoystickFlat) |
| `--seed` | int | `0` | Random seed |
| `--resume` | str | - | Warm-start from checkpoint: restores weights + opt state + norm state. Replay buffer is NOT persisted; refilled with loaded policy actions per --resume-warmup. Not exact training continuation. |
| `--resume-warmup` | str | `policy` | On resume, refill buffer using loaded policy actions (default, prevents eval drop) or legacy random uniform |
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
| `--reset-mode` | str | - | 'per_step' enables DomainRandWrapper / TerrainCurriculumDRWrapper |
| `--wandb` | flag | off | Enable W&B experiment tracking |
| `--wandb-project` | str | `jax-rl` | W&B project name |

---

## `train_tdmpc2.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | - | Env name (CheetahRun, HumanoidRun, AcrobotSwingup, ...) |
| `--total-timesteps` | int | `1000000` | Total environment steps to train (default: 1M) |
| `--seed` | int | `0` | Random seed (default: 0) |
| `--num-envs` | int | from preset | Override TDMPC2Config.num_envs (default: 8 from preset). |
| `--collect-mode` | str | from preset | Override TDMPC2Config.collect_mode (default: mppi). |
| `--eval-every` | int | from preset | Eval cadence in env steps (default: from preset) |
| `--ckpt-dir` | str | - | Checkpoint directory (default: no checkpointing) |
| `--wandb` | flag | off | Enable W&B experiment tracking |
| `--wandb-project` | str | `jax-rl-tdmpc2` | W&B project name (default: jax-rl-tdmpc2) |
| `--env-kwargs` | str | - | JSON dict forwarded to gym env factory. Example: --env-kwargs '{"obs_type":"keypoints","block_shape":"dr"}' |

---

## `train_pusht.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--reward-mode` | str | `dense` |  |
| `--total-timesteps` | int | `1000000` |  |
| `--num-envs` | int | `8` |  |
| `--buffer-size` | int | `500000` |  |
| `--batch-size` | int | `512` |  |
| `--grad-updates-per-step` | int | `1` |  |
| `--lr` | float | `0.0003` |  |
| `--gamma` | float | `0.99` |  |
| `--reward-scale` | float | `1.0` | Multiplier on env reward before replay. Use 0.1 for contact_gated. |
| `--grad-clip-norm` | float | - | Global grad norm clip (e.g. 1.0). None = off. |
| `--target-entropy-scale` | float | `1.0` | SAC target entropy = -scale * action_dim. Bigger = more explore. |
| `--obs-type` | str | `state` | state=5d; keypoints=25d (5d state + 10 dense arc-length KPs per shape); environment_state_agent_pos=18d (T-only keypoints + agent). |
| `--frame-stack` | int | `1` | Stack N consecutive obs. Implicit velocity; flattened to obs_dim × N. |
| `--action-repeat` | int | `1` | Repeat each action K env steps (frame skip). Commits policy to direction, classic RL trick for multi-contact manipulation (FiGAR / Atari frame skip). |
| `--coverage-shape` | str | `linear` | r_coverage shape. 'linear' = raw coverage. 'log_barrier' = -log(1 - cov + eps): unbounded near goal, amplifies final-mile precision. |
| `--coverage-eps` | float | `0.01` | Epsilon for log_barrier (sets max reward ceiling: ε=0.01 → r_max≈4.6). |
| `--success-threshold` | float | `0.95` | Coverage threshold for terminated=True. DP paper uses 0.95 (above human teleop peak 0.9489). Lower to 0.85 for tractable success events. |
| `--success-bonus` | float | `50.0` | Terminal reward on success (contact_gated mode only). Default 50. |
| `--block-shape` | str | `tee` | Block shape. 'dr' samples per episode from letter set {tee, l, k, s}. |
| `--seed` | int | `0` |  |
| `--eval-every-n-steps` | int | `50000` |  |
| `--wandb` | flag | off |  |

---

## `record_video.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | - |  |
| `--checkpoint` | str | - |  |
| `--out` | str | `rollout.mp4` |  |
| `--max-steps` | int | `1000` |  |
| `--camera` | str | - |  |
| `--seed` | int | `0` | Random seed for env reset |
| `--varied-cmds` | int | `0` | Resample uniform velocity command every N steps (75=1.5s at 50Hz). Mutually exclusive with --kicks. |
| `--cmd-max` | float | `[1.5, 0.8, 1.2]` | Symmetric ranges for --varied-cmds uniform sampler (default 1.5/0.8/1.2 — env's command_config.a). |
| `--cam-distance` | float | `6.0` | Free-camera tracking distance (default 6.0; use 3.0 for closer view). |
| `--kicks` | flag | off | Zero velocity command + random velocity kicks every 1.5s |
| `--force-zero-linvel` | flag | off | Curriculum Class A only: force cmd_vx=cmd_vy=0 for the whole episode (DR sanity check) |
| `--force-zero-yaw` | flag | off | Curriculum: force cmd_yaw_rate=0 for the whole episode (DR sanity check) |
| `--terrain-level` | int | - | Curriculum env only: force spawn at this level (0-9) |
| `--terrain-type` | str | - | Curriculum env only: force spawn at this terrain type |
| `--skill-index` | int | - | Fixed skill index for skill-discovery checkpoints (one-hot) |
| `--skill-vector` | str | - | Path to .csv/.npy with explicit skill vector |
| `--no-early-term` | flag | off | Don't break the rollout when env emits done=True. Keep rolling so the user can see the failure mode (post-fall dynamics, off-belt slide, etc.). |
| `--lock-cmd` | float | - | Pin cmd = [VX, VY, YAW] every step (overrides env's Markov-chain resample). Use to test linear-only (VY=YAW=0), rotation-only (VX=VY=0), or combined. Mutually exclusive with --kicks / --varied-cmds. |
| `--resolution` | int int | `[640, 480]` | Render resolution. Defaults to 640x480; use 1280 720 for HD, 1920 1080 for full HD. |
| `--video-quality` | int | `8` | imageio video quality (1-10, default 8). Higher = bigger file + sharper, lower = smaller. |

---

## `eval_tdmpc2.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | - | Env name (CheetahRun, HumanoidRun, ...) |
| `--load-ckpt` | str | - | Path to ckpt dir (containing actor_params.npz + world_model_params.npz). Pass `<dir>/best` to eval the peak ckpt. |
| `--num-evals` | int | `5` | Number of eval rounds. Each round runs cfg.num_eval_envs episodes. |
| `--seed` | int | `0` | Base seed; round k uses seed+k for reproducibility. |
| `--num-envs` | int | from preset | Override TDMPC2Config.num_envs (only affects state shape; eval uses num_eval_envs). |

---

## `check_tdmpc2_determinism.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--check` | str | - | Which determinism check: init / update / env |
| `--env` | str | `CheetahRun` | Env name (default: CheetahRun) |
| `--seed` | int | `0` | Seed (default: 0) |
| `--n` | int | `50` | Number of update / env steps to compare (default: 50) |

---

## `record_video_tdmpc2.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--env` | str | - | Env name (CheetahRun, HumanoidRun, ...) |
| `--load-ckpt` | str | - | Path to ckpt dir (actor_params.npz + world_model_params.npz) |
| `--mode` | str | `both` | Rollout mode: mppi planning, prior policy, or both (default) |
| `--num-steps` | int | `500` | Rollout length (default 500 = DMC episode length) |
| `--out-dir` | str | - | Defaults to <ckpt>/ |
| `--seed` | int | `0` | Seed (default 0) |
| `--width` | int | `480` | Render width (default 480) |
| `--height` | int | `480` | Render height (default 480) |
| `--num-envs` | int | from preset | Override TDMPC2Config.num_envs (state-shape only) |
