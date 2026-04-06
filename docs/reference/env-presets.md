# Environment Presets

Presets return fully-configured `(TrainConfig, AlgoConfig)` tuples with tuned hyperparameters per environment. CLI flags override individual fields via `dataclasses.replace()`.

If an environment is not listed, a default config is used with the environment name set.

---

## PPO Presets

Used by `train_ppo_fast.py`. Accessed via `get_preset(env_name)`.

| Environment | num_envs | total_timesteps | lr | gamma | reward_scaling | num_steps | num_epochs | entropy_coef | Notes |
|---|---|---|---|---|---|---|---|---|---|
| CartpoleBalance | 64 | 1M | 3e-4 | 0.99 | 1.0 | 64 | 4 | 0.01 | Baseline test env |
| CheetahRun | 2048 | 20M | 1e-3 | 0.995 | 10.0 | 30 | 16 | 1e-2 | Eval 826 |
| WalkerWalk | 2048 | 60M | 1e-3 | 0.995 | 10.0 | 30 | 16 | 1e-2 | |
| HumanoidRun | 2048 | 60M | 1e-3 | 0.995 | 10.0 | 480 | 16 | 1e-2 | state_dependent_std, anneal_lr=False, policy (128,128,128,128) |
| Go2JoystickFlat | 4096 | 100M | 3e-4 | 0.97 | 1.0 | 20 | 4 | 1e-2 | 32 minibatches, 4 updates/batch, policy/value (512,256,128) |
| Go2WarpJoystickFlat | 4096 | 100M | 3e-4 | 0.97 | 1.0 | 20 | 4 | 1e-2 | Same as Go2JoystickFlat; FastSAC preferred for Go2 Warp |

PPO defaults not shown: `clip_eps=0.3`, `gae_lambda=0.95`, `num_minibatches=32`, `num_updates_per_batch=1`, `max_grad_norm=None`, `anneal_lr=True`, `squash=True`, `state_dependent_std=False`, `policy_hidden_dim=(32,32,32,32)`, `value_hidden_dim=(256,256,256,256,256)`.

---

## SAC Presets

Used by `train_offpolicy.py --algo sac`. Accessed via `get_sac_preset(env_name)`.

Base config: `num_envs=128`, `lr=1e-3`, `gamma=0.99`, `episode_length=1000`, `handle_truncation=True`.

| Environment | total_timesteps | Notes |
|---|---|---|
| WalkerWalk | 5M | Eval 975 avg, 995 max |
| HumanoidRun | 5M | Eval 426 (vanilla SAC, 20M) |
| CheetahRun | 5M | |
| PandaPickCube | 10M | episode_length=150 |

SAC algo defaults: `tau=0.005`, `target_entropy_scale=0.5`, `alpha_lr=1e-3`, `buffer_size=4M`, `min_buffer_size=8192`, `batch_size=512`, `grad_updates_per_step=8`, `hidden_dim=(256,256)`, `activation=relu`, `q_layer_norm=True`.

---

## TD3 Presets

Used by `train_offpolicy.py --algo td3`. Accessed via `get_td3_preset(env_name)`.

Base config: `num_envs=128`, `lr=3e-4`, `gamma=0.99`, `episode_length=1000`, `handle_truncation=True`.

| Environment | total_timesteps | grad_updates_per_step | batch_size | q_layer_norm | Notes |
|---|---|---|---|---|---|
| CheetahRun | 5M | 4 | 256 | False | Eval 749 |
| WalkerWalk | 5M | 4 | 256 | False | |
| HumanoidRun | 5M | 4 | 256 | True | LayerNorm for stability on high-dim |

TD3 algo defaults: `tau=0.005`, `policy_delay=2`, `target_noise_std=0.2`, `noise_clip=0.5`, `exploration_noise_std=0.1`, `buffer_size=1M`, `min_buffer_size=10000`, `hidden_dim=(256,256)`, `activation=relu`, `grad_clip_norm=1.0`.

---

## FastTD3 Presets

Used by `train_offpolicy.py --algo fast_td3`. Accessed via `get_fast_td3_preset(env_name)`.

Based on Seo et al. 2025 (arXiv:2512.01996). Designed for large-scale training (1024+ envs, 50M+ steps).

Base config: `num_envs=1024`, `lr=3e-4`, `gamma=0.97`, `episode_length=1000`, `handle_truncation=True`.

| Environment | total_timesteps | Notes |
|---|---|---|
| CheetahRun | 100M | Eval 880 |
| WalkerWalk | 100M | |
| HumanoidRun | 100M | Eval 665 |

FastTD3 algo defaults: `tau=0.125`, `policy_delay=2`, `noise_min=0.01`, `noise_max=0.05` (mixed noise, overrides `exploration_noise_std=0.2`), `batch_size=8192`, `grad_updates_per_step=8`, `min_buffer_size=25000`, `num_atoms=101`, `v_min=-20`, `v_max=20`, `hidden_dim=(512,256,128)`, `critic_hidden_dim=(768,384,192)`, `activation=swish`, `q_layer_norm=True`, `lr_end=3e-4` (constant LR).

---

## FastSAC Presets

Used by `train_offpolicy.py --algo fast_sac`. Accessed via `get_fast_sac_preset(env_name)`.

Based on Seo et al. 2025 (arXiv:2512.01996). C51 distributional critic with SAC entropy. Preferred for Go2 locomotion.

Base config: `num_envs=1024`, `lr=3e-4`, `gamma=0.97`, `episode_length=1000`, `handle_truncation=True`.

| Environment | total_timesteps | Notes |
|---|---|---|
| CheetahRun | 100M | |
| WalkerWalk | 100M | |
| HumanoidRun | 100M | Eval 892 (SOTA for this framework) |
| Go2WarpJoystickFlat | 100M | Eval 276.5 @ 18M steps |

FastSAC algo defaults: `tau=0.125`, `target_entropy_scale=0.0`, `alpha_lr=3e-4`, `alpha_init=0.001`, `max_std=1.0`, `policy_delay=4`, `batch_size=8192`, `grad_updates_per_step=8`, `num_atoms=101`, `v_min=-20`, `v_max=20`, `hidden_dim=(512,256,128)`, `critic_hidden_dim=(768,384,192)`, `activation=swish`, `q_layer_norm=True`, `lr_end=3e-5` (cosine decay).
