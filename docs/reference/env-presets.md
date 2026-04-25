# Environment Presets

Auto-generated from `jax_rl/configs/env_presets.py`. Regenerate with:

```bash
uv run python docs/scripts/gen_env_presets.py
```

Presets return fully-configured `(TrainConfig, AlgoConfig)` tuples with tuned hyperparameters per environment. CLI flags override individual fields via `dataclasses.replace()`.

If an environment is not listed, a default config is used with the environment name set.

Select an algorithm tab below to see its presets. Defaults (gamma=0.99, reward_scaling=1) are omitted from rows and only appear in Notes when overridden.

=== "PPO"

    Used by `train_ppo_fast.py`. Accessed via `get_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | num_steps | epochs | entropy_coef | Notes |
    |---|---|---|---|---|---|---|---|
    | CartpoleBalance | 64 | 1M | 3e-04 | 64 | 4 | 0.01 | max_grad_norm=0.5 |
    | CartpoleSwingup | 256 | 3M | 3e-04 | 64 | 4 | 0.01 | clip_eps=0.2, policy_hidden_dim=(64, 64), value_hidden_dim=(64, 64), activation=tanh, squash=False, max_grad_norm=0.5, eval_every_n_episodes=500 |
    | CartpoleSwingupSparse | 256 | 5M | 3e-04 | 64 | 4 | 0.01 | clip_eps=0.2, policy_hidden_dim=(64, 64), value_hidden_dim=(64, 64), activation=tanh, squash=False, max_grad_norm=0.5 |
    | CheetahRun | 2,048 | 20M | 0.001 | 30 | 16 | 0.01 | gamma=0.995, reward_scaling=10 |
    | WalkerWalk | 2,048 | 60M | 0.001 | 30 | 16 | 0.01 | gamma=0.995, reward_scaling=10 |
    | HumanoidRun | 2,048 | 60M | 0.001 | 480 | 16 | 0.01 | gamma=0.995, reward_scaling=10, policy_hidden_dim=(128, 128, 128, 128), state_dependent_std=True, anneal_lr=False |
    | Go2WarpJoystickFlat | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1 |
    | Go2WarpJoystickFlatTorqueSpeed | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1 |
    | Go2WarpJoystickCurriculum | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, reset_mode=per_step |
    | Go2WarpJoystickCurriculumTorqueSpeed | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, reset_mode=per_step |
    | Go2BongoHandstand | 256 | 100M | 3e-04 | 64 | 4 | 0.01 | n_frame_stack=3 |
    | Go2BongoHandstandContraction | 256 | 100M | 3e-04 | 64 | 4 | 0.01 | n_frame_stack=3 |

    PPO algo defaults: `clip_eps=0.3`, `entropy_coef=0.01`, `gae_lambda=0.95`, `num_epochs=4`, `num_steps=64`, `num_updates_per_batch=1`, `policy_hidden_dim=(32, 32, 32, 32)`, `value_hidden_dim=(256, 256, 256, 256, 256)`, `activation=swish`, `squash=True`, `state_dependent_std=False`, `max_grad_norm=None`, `anneal_lr=True`, `critic_encoder=None`, `policy_head=None`, `normalize_advantage=True`, `contraction=None`.


=== "SAC"

    Used by `train_sac.py`. Accessed via `get_sac_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | WalkerWalk | 128 | 5M | 0.001 | 512 | 8 |  |
    | HumanoidRun | 128 | 5M | 0.001 | 512 | 8 |  |
    | CheetahRun | 128 | 5M | 0.001 | 512 | 8 |  |
    | PandaPickCube | 128 | 10M | 0.001 | 512 | 8 |  |

    SAC algo defaults: `tau=0.005`, `hidden_dim=(256, 256)`, `activation=relu`, `batch_size=512`, `grad_updates_per_step=8`, `buffer_size=4M`, `min_buffer_size=8,192`, `q_layer_norm=True`.


=== "TD3"

    Used by `train_td3.py`. Accessed via `get_td3_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | CheetahRun | 128 | 5M | 3e-04 | 256 | 4 |  |
    | WalkerWalk | 128 | 5M | 3e-04 | 256 | 4 |  |
    | HumanoidRun | 128 | 5M | 3e-04 | 256 | 4 | q_layer_norm=True |

    TD3 algo defaults: `tau=0.005`, `hidden_dim=(256, 256)`, `activation=relu`, `batch_size=256`, `grad_updates_per_step=1`, `buffer_size=1M`, `min_buffer_size=10,000`, `q_layer_norm=False`.


=== "FastTD3"

    Used by `train_fast_td3.py`. Accessed via `get_fast_td3_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | CheetahRun | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | WalkerWalk | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | HumanoidRun | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |

    FastTD3 algo defaults: `tau=0.125`, `hidden_dim=(512, 256, 128)`, `activation=swish`, `batch_size=8,192`, `grad_updates_per_step=8`, `buffer_size=1M`, `min_buffer_size=25,000`, `q_layer_norm=True`.


=== "FastSAC"

    Used by `train_fast_sac.py`. Accessed via `get_fast_sac_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | CheetahRun | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | WalkerWalk | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | HumanoidRun | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpJoystickFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpJoystickFlatTorqueSpeed | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpJoystickCurriculum | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpJoystickCurriculumTorqueSpeed | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |

    FastSAC algo defaults: `tau=0.125`, `hidden_dim=(512, 256, 128)`, `activation=swish`, `batch_size=8,192`, `grad_updates_per_step=8`, `buffer_size=4M`, `min_buffer_size=8,192`, `q_layer_norm=True`.


=== "FlashSAC"

    Used by `train_flashsac.py`. Accessed via `get_flash_sac_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | CheetahRun | 1,024 | 100M | 3e-04 | 2,048 | 8 |  |
    | WalkerWalk | 1,024 | 100M | 3e-04 | 2,048 | 8 |  |
    | HumanoidRun | 1,024 | 100M | 3e-04 | 2,048 | 8 |  |
    | Go2WarpJoystickFlat | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |
    | Go2WarpJoystickFlatTorqueSpeed | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |
    | Go2WarpJoystickCurriculum | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |
    | Go2WarpJoystickCurriculumTorqueSpeed | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |

    FlashSAC algo defaults: `tau=0.01`, `batch_size=2,048`, `grad_updates_per_step=1`, `buffer_size=1M`, `min_buffer_size=10,000`.

