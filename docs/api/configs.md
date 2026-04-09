# Configs

Dataclass-based configuration system. `TrainConfig` is shared across all algorithms; each algorithm has its own config for algorithm-specific parameters.

| Config | Used by |
|--------|---------|
| [TrainConfig](#trainconfig) | All algorithms |
| [PPOConfig](#ppoconfig) | PPO |
| [SACConfig](#sacconfig) | SAC |
| [TD3Config](#td3config) | TD3 |
| [FastSACConfig](#fastsacconfig) | FastSAC |
| [FastTD3Config](#fasttd3config) | FastTD3 |
| [FlashSACConfig](#flashsacconfig) | FlashSAC |
| [EncoderConfig](#encoderconfig) | Network builders |
| [PolicyHeadConfig](#policyheadconfig) | Network builders |

---

## TrainConfig

```python
from jax_rl.configs.train_config import TrainConfig
```

Shared training config — environment, evaluation, and wrapper settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `env_name` | `str` | `"CartpoleBalance"` | Environment identifier |
| `episode_length` | `int` | `1000` | Max steps per episode |
| `num_envs` | `int` | `64` | Parallel environments |
| `total_timesteps` | `int` | `1_000_000` | Total training steps |
| `lr` | `float` | `3e-4` | Base learning rate |
| `gamma` | `float` | `0.99` | Discount factor |
| `reward_scaling` | `float` | `1.0` | Reward multiplier |
| `handle_truncation` | `bool` | `True` | Bootstrap at truncation |
| `domain_rand` | `bool` | `False` | Enable domain randomization |
| `n_frame_stack` | `int` | `1` | Frame stacking (1 = disabled) |
| `action_delay_ms` | `int` | `0` | Fixed action latency (0 = disabled) |
| `action_delay_range_ms` | `tuple | None` | `None` | Random delay range per-episode |
| `eval_every_n_episodes` | `int` | `5000` | Evaluation frequency |
| `num_eval_episodes` | `int` | `10` | Episodes per eval |
| `log_interval` | `int` | `1` | Logging frequency |
| `reset_mode` | `str` | `"legacy"` | `"legacy"`, `"per_step"`, or `"syncd"` |
| `ppo` | `PPOConfig | None` | `PPOConfig()` | PPO-specific config |

---

## PPOConfig

```python
from jax_rl.configs.ppo_config import PPOConfig
```

PPO algorithm parameters. Optimizer config (LR, schedule) is external — optimizers are passed to `PPO.__init__`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `clip_eps` | `float` | `0.3` | PPO clip coefficient |
| `entropy_coef` | `float` | `0.01` | Entropy regularization |
| `gae_lambda` | `float` | `0.95` | GAE lambda |
| `num_epochs` | `int` | `4` | Update epochs per rollout |
| `num_minibatches` | `int` | `32` | Minibatches per epoch |
| `num_steps` | `int` | `64` | Steps per rollout |
| `num_updates_per_batch` | `int` | `1` | Rollout+update cycles per iteration |
| `policy_hidden_dim` | `tuple` | `(32, 32, 32, 32)` | Actor MLP dims |
| `value_hidden_dim` | `tuple` | `(256, 256, 256, 256, 256)` | Critic MLP dims |
| `activation` | `str` | `"swish"` | Activation function |
| `squash` | `bool` | `True` | Tanh output squashing |
| `state_dependent_std` | `bool` | `False` | State-dependent policy std |
| `normalize_advantage` | `bool` | `True` | Normalize advantages |
| `max_grad_norm` | `float | None` | `None` | Gradient clipping |
| `anneal_lr` | `bool` | `True` | Learning rate annealing |

---

## SACConfig

```python
from jax_rl.configs.sac_config import SACConfig
```

SAC algorithm parameters. Vanilla SAC defaults — for high-UTD training, use [FastSACConfig](#fastsacconfig).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tau` | `float` | `0.005` | Polyak soft update coefficient |
| `target_entropy_scale` | `float` | `0.5` | `target_entropy = -scale * action_dim` |
| `alpha_lr` | `float` | `1e-3` | Temperature optimizer LR |
| `alpha_init` | `float` | `1.0` | Initial temperature |
| `max_std` | `float | None` | `None` | Cap on pre-tanh std |
| `policy_delay` | `int` | `1` | Actor update frequency |
| `grad_clip_norm` | `float | None` | `None` | Max gradient norm |
| `buffer_size` | `int` | `4_194_304` | Replay buffer size (4M) |
| `min_buffer_size` | `int` | `8_192` | Steps before first update |
| `batch_size` | `int` | `512` | Gradient batch size |
| `grad_updates_per_step` | `int` | `8` | UTD ratio |
| `hidden_dim` | `tuple` | `(256, 256)` | Actor MLP dims |
| `critic_hidden_dim` | `tuple | None` | `None` | Critic dims (None = same as actor) |
| `activation` | `str` | `"relu"` | Activation function |
| `q_layer_norm` | `bool` | `True` | Layer norm in Q-networks |
| `obs_normalization` | `bool` | `False` | Normalize observations |

---

## TD3Config

```python
from jax_rl.configs.td3_config import TD3Config
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tau` | `float` | `0.005` | Polyak soft update coefficient |
| `policy_delay` | `int` | `2` | Actor update frequency |
| `target_noise_std` | `float` | `0.2` | Target policy noise |
| `noise_clip` | `float` | `0.5` | Noise clipping range |
| `exploration_noise_std` | `float` | `0.1` | Exploration noise |
| `buffer_size` | `int` | `1_000_000` | Replay buffer size |
| `min_buffer_size` | `int` | `10_000` | Steps before first update |
| `batch_size` | `int` | `256` | Gradient batch size |
| `grad_updates_per_step` | `int` | `1` | UTD ratio |
| `hidden_dim` | `tuple` | `(256, 256)` | Actor MLP dims |
| `critic_hidden_dim` | `tuple | None` | `None` | Critic dims (None = same as actor) |
| `activation` | `str` | `"relu"` | Activation function |
| `q_layer_norm` | `bool` | `False` | Layer norm in Q-networks |
| `grad_clip_norm` | `float | None` | `1.0` | Max gradient norm |
| `obs_normalization` | `bool` | `False` | Normalize observations |

---

## FastSACConfig

```python
from jax_rl.configs.fast_sac_config import FastSACConfig
```

SAC + C51 distributional critic. Defaults from the paper (Seo et al. 2025) — diverges significantly from vanilla SAC.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tau` | `float` | `0.125` | 25x faster than SAC |
| `target_entropy_scale` | `float` | `0.0` | Prevents alpha collapse |
| `alpha_init` | `float` | `0.001` | Near-zero start |
| `max_std` | `float | None` | `1.0` | Caps pre-tanh std |
| `policy_delay` | `int` | `4` | Actor every 4th critic step |
| `batch_size` | `int` | `8_192` | 16x larger than SAC |
| `grad_updates_per_step` | `int` | `8` | UTD 8 |
| `hidden_dim` | `tuple` | `(512, 256, 128)` | Tapered actor |
| `critic_hidden_dim` | `tuple | None` | `(768, 384, 192)` | Wider critic |
| `activation` | `str` | `"swish"` | SiLU activation |
| `num_atoms` | `int` | `101` | C51 atoms |
| `v_min` / `v_max` | `float` | `-20.0` / `20.0` | Value distribution range |
| `alpha_lr` | `float` | `3e-4` | Temperature optimizer LR |
| `grad_clip_norm` | `float | None` | `None` | Max gradient norm |
| `buffer_size` | `int` | `4_194_304` | Replay buffer size (4M) |
| `min_buffer_size` | `int` | `8_192` | Steps before first update |
| `q_layer_norm` | `bool` | `True` | Layer norm in Q-networks |
| `q_aggregation` | `str` | `"avg"` | Avg (not min) of twin critics |
| `lr_end` | `float` | `3e-5` | Cosine decay target |
| `obs_normalization` | `bool` | `False` | Normalize observations |
| `obs_norm_eps` | `float` | `1e-2` | Obs normalization epsilon |

---

## FastTD3Config

```python
from jax_rl.configs.fast_td3_config import FastTD3Config
```

TD3 + C51 distributional critic. Same distributional approach as FastSAC with deterministic policy.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tau` | `float` | `0.125` | Fast target update |
| `policy_delay` | `int` | `2` | Actor update frequency |
| `exploration_noise_std` | `float` | `0.2` | Exploration noise |
| `noise_min` / `noise_max` | `float | None` | `None` | Mixed noise range (overrides exploration_noise_std) |
| `batch_size` | `int` | `8_192` | Gradient batch size |
| `grad_updates_per_step` | `int` | `8` | UTD 8 |
| `min_buffer_size` | `int` | `25_000` | Steps before first update |
| `hidden_dim` | `tuple` | `(512, 256, 128)` | Tapered actor |
| `critic_hidden_dim` | `tuple | None` | `(768, 384, 192)` | Wider critic |
| `activation` | `str` | `"swish"` | SiLU activation |
| `num_atoms` | `int` | `101` | C51 atoms |
| `v_min` / `v_max` | `float` | `-20.0` / `20.0` | Value distribution range |
| `q_aggregation` | `str` | `"avg"` | Avg of twin critics |
| `lr_end` | `float` | `3e-4` | Constant (no decay) |

---

## FlashSACConfig

```python
from jax_rl.configs.flash_sac_config import FlashSACConfig
```

Inverted residual blocks + BatchNorm + weight normalization + adaptive reward scaling. Defaults from Kim et al. 2026.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| **Architecture** | | | |
| `num_blocks` | `int` | `2` | Inverted residual blocks |
| `actor_hidden_dim` | `int` | `128` | Actor base dim |
| `critic_hidden_dim` | `int` | `256` | Critic base dim |
| `expansion` | `int` | `4` | Block expansion ratio |
| `weight_norm` | `bool` | `True` | Weight normalization |
| **Distributional** | | | |
| `num_atoms` | `int` | `101` | C51 atoms |
| `v_min` / `v_max` | `float` | `-5.0` / `5.0` | Tighter value range |
| **Training** | | | |
| `tau` | `float` | `0.01` | Polyak coefficient |
| `policy_delay` | `int` | `2` | Actor update frequency |
| `batch_size` | `int` | `2048` | Gradient batch size |
| `grad_updates_per_step` | `int` | `1` | UTD 1 |
| `alpha_init` | `float` | `0.01` | Initial temperature |
| `sigma_target` | `float` | `0.15` | Target entropy std |
| `normalize_reward` | `bool` | `True` | Adaptive reward scaling |
| `G_max` | `float` | `5.0` | Max scaled reward |
| **LR schedule** | | | |
| `lr_init` / `lr_peak` / `lr_end` | `float` | `3e-4` / `3e-4` / `1.5e-4` | Warmup → decay |
| **Noise** | | | |
| `noise_zeta_mu` | `float` | `2.0` | Zeta noise repetition mean |
| `noise_zeta_max` | `int` | `16` | Max repetitions |
| **Misc** | | | |
| `gamma` | `float` | `0.99` | Discount factor |
| `n_step` | `int` | `1` | n-step returns |
| `bc_alpha` | `float` | `0.0` | Behavior cloning regularization |
| `buffer_size` | `int` | `1_000_000` | Replay buffer size |
| `min_buffer_size` | `int` | `10_000` | Steps before first update |
| `lr_warmup_frac` | `float` | `1e-6` | LR warmup fraction |
| `lr_decay_frac` | `float` | `1.0` | LR decay fraction |

---

## EncoderConfig

```python
from jax_rl.configs.networks_config import EncoderConfig
```

Configuration for MLP encoders.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `obs_dim` | `int` | *required* | Observation dimension |
| `hidden_dim` | `tuple` | `(256, 256)` | Hidden layer dimensions |
| `activation` | `str` | `"relu"` | Activation function |
| `norm` | `str | None` | `None` | `"layer"` or `"spectral"` |
| `norm_placement` | `str` | `"pre"` | `"pre"` or `"post"` activation |
| `context_dim` | `int | None` | `None` | For goal-conditioned policies |
| `context_fusion` | `str` | `"concat"` | `"concat"`, `"film"`, or `"cross_attn"` |

---

## PolicyHeadConfig

```python
from jax_rl.configs.networks_config import PolicyHeadConfig
```

Configuration for Gaussian policy heads.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action_dim` | `int` | *required* | Action dimension |
| `log_std_min` / `log_std_max` | `float` | `-20.0` / `2.0` | Log-std bounds |
| `state_dependent_std` | `bool` | `False` | Dense-layer std vs learned param |
| `init_noise_std` | `float` | `1.0` | Initial std (state-independent) |
| `min_std` | `float` | `0.001` | Floor std (state-dependent) |
| `squash` | `bool` | `True` | Tanh output squashing |
