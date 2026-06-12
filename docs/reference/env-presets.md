# Environment Presets

Auto-generated from `jax_rl/configs/env_presets.py`. Regenerate with:

```bash
uv run python docs/scripts/gen_env_presets.py
```

Presets return fully-configured `(TrainConfig, AlgoConfig)` tuples with tuned hyperparameters per environment. CLI flags override individual fields via `dataclasses.replace()`.

If an environment is not listed, a default config is used with the environment name set. `Go2Warp*` rows (except the splitbelt family) resolve from the variants table in `jax_rl/envs/locomotion/go2_warp_variants.py` rather than static preset entries.

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
    | FactoryPegInsert | 128 | 5M | 1e-04 | 64 | 4 | 0.01 | reward_scaling=0.1, clip_eps=0.2, policy_hidden_dim=(256, 256), value_hidden_dim=(256, 256), squash=False, max_grad_norm=0.5, reset_mode=per_step |
    | Go2BongoHandstand | 256 | 100M | 3e-04 | 64 | 4 | 0.01 | n_frame_stack=3 |
    | Go2BongoHandstandContraction | 256 | 100M | 3e-04 | 64 | 4 | 0.01 | n_frame_stack=3 |
    | Go2WarpSplitbelt | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, reset_mode=per_step |
    | Go2WarpFlatPosTrackProto | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1 |
    | Go2WarpJointRoughUni | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickCurriculum | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, reset_mode=per_step |
    | Go2WarpJoystickCurriculumTorqueSpeed | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, reset_mode=per_step |
    | Go2WarpJoystickFlat | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1 |
    | Go2WarpJoystickFlatHardKick | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1 |
    | Go2WarpJoystickFlatNoAccel | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1 |
    | Go2WarpJoystickFlatPhysical | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR4x | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatTorqueSpeed | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1 |
    | Go2WarpJoystickUnitree | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1 |
    | Go2WarpOscFlatSoftPhysical | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlat | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatJt | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp025 | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05 | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05HardKick | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp2 | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp4 | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscRoughUni | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisFlatPhysical | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisRoughUni | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysical | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud05 | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud10 | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22 | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22Heavy | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR4x | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisRoughUni | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingFlatPhysical | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarFlatPhysical | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisFlat | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisHardKickFlat | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceFlat | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceHardKickFlat | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarRoughUni | 4,096 | 100M | 3e-04 | 20 | 4 | 0.01 | gamma=0.97, num_updates_per_batch=4, policy_hidden_dim=(512, 256, 128), value_hidden_dim=(512, 256, 128), max_grad_norm=1, eval_every_n_episodes=500, reset_mode=per_step |

    PPO algo defaults: `clip_eps=0.3`, `entropy_coef=0.01`, `gae_lambda=0.95`, `num_epochs=4`, `num_steps=64`, `num_updates_per_batch=1`, `policy_hidden_dim=(32, 32, 32, 32)`, `value_hidden_dim=(256, 256, 256, 256, 256)`, `activation=swish`, `squash=True`, `state_dependent_std=False`, `max_grad_norm=None`, `anneal_lr=True`, `critic_encoder=None`, `policy_head=None`, `normalize_advantage=True`, `contraction=None`.


=== "SAC"

    Used by `train_sac.py`. Accessed via `get_sac_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | WalkerWalk | 128 | 5M | 0.001 | 512 | 8 |  |
    | HumanoidRun | 128 | 5M | 0.001 | 512 | 8 |  |
    | CheetahRun | 128 | 5M | 0.001 | 512 | 8 |  |
    | PandaPickCube | 128 | 10M | 0.001 | 512 | 8 |  |
    | FactoryPegInsert | 256 | 5M | 0.001 | 512 | 2 | target_entropy_scale=1, obs_normalization=True, reset_mode=per_step |
    | Go2WarpFlatPosTrackProto | 128 | 5M | 0.001 | 512 | 8 |  |
    | Go2WarpJointRoughUni | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickCurriculum | 128 | 5M | 0.001 | 512 | 8 | reset_mode=per_step |
    | Go2WarpJoystickCurriculumTorqueSpeed | 128 | 5M | 0.001 | 512 | 8 | reset_mode=per_step |
    | Go2WarpJoystickFlat | 128 | 5M | 0.001 | 512 | 8 |  |
    | Go2WarpJoystickFlatHardKick | 128 | 5M | 0.001 | 512 | 8 |  |
    | Go2WarpJoystickFlatNoAccel | 128 | 5M | 0.001 | 512 | 8 |  |
    | Go2WarpJoystickFlatPhysical | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR4x | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatTorqueSpeed | 128 | 5M | 0.001 | 512 | 8 |  |
    | Go2WarpJoystickUnitree | 128 | 5M | 0.001 | 512 | 8 |  |
    | Go2WarpOscFlatSoftPhysical | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlat | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatJt | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp025 | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05 | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05HardKick | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp2 | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp4 | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscRoughUni | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisFlatPhysical | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisRoughUni | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysical | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud05 | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud10 | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22 | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22Heavy | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR4x | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisRoughUni | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingFlatPhysical | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarFlatPhysical | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisFlat | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisHardKickFlat | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceFlat | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceHardKickFlat | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarRoughUni | 128 | 5M | 0.001 | 512 | 8 | eval_every_n_episodes=500, reset_mode=per_step |

    SAC algo defaults: `tau=0.005`, `hidden_dim=(256, 256)`, `activation=relu`, `batch_size=512`, `grad_updates_per_step=8`, `buffer_size=4M`, `min_buffer_size=8,192`, `q_layer_norm=True`.


=== "TD3"

    Used by `train_td3.py`. Accessed via `get_td3_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | CheetahRun | 128 | 5M | 3e-04 | 256 | 4 |  |
    | WalkerWalk | 128 | 5M | 3e-04 | 256 | 4 |  |
    | HumanoidRun | 128 | 5M | 3e-04 | 256 | 4 | q_layer_norm=True |
    | Go2WarpFlatPosTrackProto | 128 | 5M | 3e-04 | 256 | 4 |  |
    | Go2WarpJointRoughUni | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickCurriculum | 128 | 5M | 3e-04 | 256 | 4 | reset_mode=per_step |
    | Go2WarpJoystickCurriculumTorqueSpeed | 128 | 5M | 3e-04 | 256 | 4 | reset_mode=per_step |
    | Go2WarpJoystickFlat | 128 | 5M | 3e-04 | 256 | 4 |  |
    | Go2WarpJoystickFlatHardKick | 128 | 5M | 3e-04 | 256 | 4 |  |
    | Go2WarpJoystickFlatNoAccel | 128 | 5M | 3e-04 | 256 | 4 |  |
    | Go2WarpJoystickFlatPhysical | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR4x | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatTorqueSpeed | 128 | 5M | 3e-04 | 256 | 4 |  |
    | Go2WarpJoystickUnitree | 128 | 5M | 3e-04 | 256 | 4 |  |
    | Go2WarpOscFlatSoftPhysical | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlat | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatJt | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp025 | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05 | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05HardKick | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp2 | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp4 | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscRoughUni | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisFlatPhysical | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisRoughUni | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysical | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud05 | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud10 | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22 | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22Heavy | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR4x | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisRoughUni | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingFlatPhysical | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarFlatPhysical | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisFlat | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisHardKickFlat | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceFlat | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceHardKickFlat | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarRoughUni | 128 | 5M | 3e-04 | 256 | 4 | eval_every_n_episodes=500, reset_mode=per_step |

    TD3 algo defaults: `tau=0.005`, `hidden_dim=(256, 256)`, `activation=relu`, `batch_size=256`, `grad_updates_per_step=1`, `buffer_size=1M`, `min_buffer_size=10,000`, `q_layer_norm=False`.


=== "FastTD3"

    Used by `train_fast_td3.py`. Accessed via `get_fast_td3_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | CheetahRun | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | WalkerWalk | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | HumanoidRun | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | Go2WarpFlatPosTrackProto | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | Go2WarpJointRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickCurriculum | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, reset_mode=per_step |
    | Go2WarpJoystickCurriculumTorqueSpeed | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, reset_mode=per_step |
    | Go2WarpJoystickFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | Go2WarpJoystickFlatHardKick | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | Go2WarpJoystickFlatNoAccel | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | Go2WarpJoystickFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR4x | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatTorqueSpeed | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | Go2WarpJoystickUnitree | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05 |
    | Go2WarpOscFlatSoftPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatJt | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp025 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05HardKick | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp2 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp4 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud05 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud10 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22Heavy | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR4x | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisHardKickFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceHardKickFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, noise_min=0.01, noise_max=0.05, eval_every_n_episodes=500, reset_mode=per_step |

    FastTD3 algo defaults: `tau=0.125`, `hidden_dim=(512, 256, 128)`, `activation=swish`, `batch_size=8,192`, `grad_updates_per_step=8`, `buffer_size=1M`, `min_buffer_size=25,000`, `q_layer_norm=True`.


=== "FastSAC"

    Used by `train_fast_sac.py`. Accessed via `get_fast_sac_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | CheetahRun | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | WalkerWalk | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | HumanoidRun | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpSplitbelt | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, reset_mode=per_step |
    | Go2WarpSplitbeltDR | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, reset_mode=per_step |
    | Go2WarpSplitbeltPoseDR | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, reset_mode=per_step |
    | Go2WarpFlatPosTrackProto | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpJointRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickCurriculum | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, reset_mode=per_step |
    | Go2WarpJoystickCurriculumTorqueSpeed | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, reset_mode=per_step |
    | Go2WarpJoystickFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpJoystickFlatHardKick | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpJoystickFlatNoAccel | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpJoystickFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR4x | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatTorqueSpeed | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpJoystickUnitree | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97 |
    | Go2WarpOscFlatSoftPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatJt | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp025 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05HardKick | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp2 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp4 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud05 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud10 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22 | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22Heavy | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR4x | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarFlatPhysical | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisHardKickFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceHardKickFlat | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarRoughUni | 1,024 | 100M | 3e-04 | 8,192 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |

    FastSAC algo defaults: `tau=0.125`, `hidden_dim=(512, 256, 128)`, `activation=swish`, `batch_size=8,192`, `grad_updates_per_step=8`, `buffer_size=4M`, `min_buffer_size=8,192`, `q_layer_norm=True`.


=== "FlashSAC"

    Used by `train_flashsac.py`. Accessed via `get_flash_sac_preset(env_name)`.

    | Environment | num_envs | timesteps | lr | batch_size | UTD | Notes |
    |---|---|---|---|---|---|---|
    | CheetahRun | 1,024 | 100M | 3e-04 | 2,048 | 8 |  |
    | WalkerWalk | 1,024 | 100M | 3e-04 | 2,048 | 8 |  |
    | HumanoidRun | 1,024 | 100M | 3e-04 | 2,048 | 8 |  |
    | FactoryPegInsert | 128 | 2M | 3e-04 | 2,048 | 16 | alpha_init=0.1, sigma_target=0.3, eval_every_n_episodes=100, reset_mode=per_step |
    | FactoryGearMesh | 64 | 2M | 3e-04 | 2,048 | 16 | alpha_init=0.1, sigma_target=0.3, eval_every_n_episodes=1,500, reset_mode=per_step |
    | Go2WarpFlatPosTrackProto | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |
    | Go2WarpJointRoughUni | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickCurriculum | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, reset_mode=per_step |
    | Go2WarpJoystickCurriculumTorqueSpeed | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, reset_mode=per_step |
    | Go2WarpJoystickFlat | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |
    | Go2WarpJoystickFlatHardKick | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |
    | Go2WarpJoystickFlatNoAccel | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |
    | Go2WarpJoystickFlatPhysical | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatPhysicalMudDR4x | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpJoystickFlatTorqueSpeed | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |
    | Go2WarpJoystickUnitree | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97 |
    | Go2WarpOscFlatSoftPhysical | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlat | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatJt | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp025 | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05 | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp05HardKick | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp2 | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscJoystickFlatKp4 | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscRoughUni | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisFlatPhysical | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarAxisRoughUni | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysical | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud05 | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud10 | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22 | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMud22Heavy | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisFlatPhysicalMudDR4x | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingAxisRoughUni | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarDampingFlatPhysical | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarFlatPhysical | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisFlat | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceAxisHardKickFlat | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceFlat | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarImpedanceHardKickFlat | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |
    | Go2WarpOscVarRoughUni | 1,024 | 100M | 3e-04 | 2,048 | 8 | gamma=0.97, eval_every_n_episodes=500, reset_mode=per_step |

    FlashSAC algo defaults: `tau=0.01`, `batch_size=2,048`, `grad_updates_per_step=1`, `buffer_size=1M`, `min_buffer_size=10,000`.


=== "TDMPC2"

    Used by `train_tdmpc2.py`. Accessed via `get_tdmpc2_preset(env_name)` (raises `KeyError` for unlisted envs — `action_dim` has no safe default).

    | Environment | action_dim | total_steps | num_envs | horizon | batch_size | UTD | discount | Notes |
    |---|---|---|---|---|---|---|---|---|
    | CheetahRun | 6 | 1M | 8 | 3 | 256 | 1 | 0.99 |  |
    | HumanoidRun | 21 | 1M | 8 | 3 | 256 | 1 | 0.99 |  |
    | HopperHop | 4 | 1M | 8 | 3 | 256 | 1 | 0.99 |  |
    | AcrobotSwingup | 1 | 1M | 8 | 3 | 256 | 1 | 0.99 |  |
    | CartpoleSwingup | 1 | 1M | 8 | 3 | 256 | 1 | 0.99 |  |
    | PushT | 2 | 1M | 8 | 3 | 256 | 1 | 0.983333 |  |

    TDMPC2 algo defaults: `latent_dim=512`, `mlp_dim=512`, `num_q=5`, `num_bins=101`, `num_samples=512`, `num_elites=64`, `mppi_iterations=6`, `tau=0.01`, `lr=3e-04`, `seed_steps=2,500`.

