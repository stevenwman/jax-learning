"""Per-environment training presets (MuJoCo Playground reference configs)."""

import dataclasses

from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.train_config import TrainConfig

PRESETS: dict[str, TrainConfig] = {
    "CartpoleBalance": TrainConfig(
        env_name="CartpoleBalance",
        total_timesteps=1_000_000,
        num_envs=64,
        num_steps=64,
        gamma=0.99,
        lr=3e-4,
        reward_scaling=1.0,
        max_grad_norm=0.5,
        ppo=PPOConfig(entropy_coef=0.01, num_epochs=4),
    ),
    "CheetahRun": TrainConfig(
        env_name="CheetahRun",
        total_timesteps=20_000_000,
        num_envs=2048,
        num_steps=30,
        gamma=0.995,
        lr=1e-3,
        reward_scaling=10.0,
        ppo=PPOConfig(entropy_coef=1e-2, num_epochs=16),
    ),
    "WalkerWalk": TrainConfig(
        env_name="WalkerWalk",
        total_timesteps=60_000_000,
        num_envs=2048,
        num_steps=30,
        gamma=0.995,
        lr=1e-3,
        reward_scaling=10.0,
        ppo=PPOConfig(entropy_coef=1e-2, num_epochs=16),
    ),
    "HumanoidRun": TrainConfig(
        env_name="HumanoidRun",
        total_timesteps=60_000_000,
        num_envs=2048,
        num_steps=480,
        gamma=0.995,
        lr=1e-3,
        anneal_lr=False,
        reward_scaling=10.0,
        policy_hidden_dim=(128, 128, 128, 128),
        state_dependent_std=True,
        ppo=PPOConfig(entropy_coef=1e-2, num_epochs=16),
    ),
}


# SAC presets — matching MuJoCo Playground dm_control_suite_params.brax_sac_config()
# Reference HPs: lr=1e-3, batch_size=512, grad_updates_per_step=8, q_layer_norm=True
# num_envs=128, min_buffer=8192, max_buffer=4M, num_timesteps=5M
_SAC_BASE_CFG = TrainConfig(
    total_timesteps=5_000_000,
    num_envs=128,
    episode_length=1000,
    lr=1e-3,
    anneal_lr=False,
    reward_scaling=1.0,
    gamma=0.99,
    handle_truncation=True,
    ppo=None,
)

_SAC_BASE_ALGO = SACConfig(
    tau=0.005,
    target_entropy_scale=0.5,
    alpha_lr=1e-3,
    buffer_size=4_194_304,
    min_buffer_size=8_192,
    batch_size=512,
    grad_updates_per_step=8,
    hidden_dim=(256, 256),
    activation="relu",
    q_layer_norm=True,
)

SAC_PRESETS: dict[str, tuple[TrainConfig, SACConfig]] = {
    "WalkerWalk": (
        dataclasses.replace(_SAC_BASE_CFG, env_name="WalkerWalk"),
        _SAC_BASE_ALGO,
    ),
    "HumanoidRun": (
        dataclasses.replace(_SAC_BASE_CFG, env_name="HumanoidRun"),
        _SAC_BASE_ALGO,
    ),
    "CheetahRun": (
        dataclasses.replace(_SAC_BASE_CFG, env_name="CheetahRun"),
        _SAC_BASE_ALGO,
    ),
}


def get_sac_preset(env_name: str) -> tuple[TrainConfig, SACConfig]:
    """Return SAC preset (TrainConfig, SACConfig) for env, or a default."""
    if env_name in SAC_PRESETS:
        return SAC_PRESETS[env_name]
    return dataclasses.replace(_SAC_BASE_CFG, env_name=env_name), _SAC_BASE_ALGO


# TD3 presets — vanilla TD3 with 1:1 gradient ratio
_TD3_BASE_CFG = TrainConfig(
    total_timesteps=5_000_000,
    num_envs=128,
    episode_length=1000,
    lr=3e-4,
    anneal_lr=False,
    reward_scaling=1.0,
    gamma=0.99,
    handle_truncation=True,
    ppo=None,
)

_TD3_BASE_ALGO = TD3Config(
    grad_updates_per_step=4,  # 128 envs need higher replay ratio than vanilla TD3's 1:1
    batch_size=256,
)

TD3_PRESETS: dict[str, tuple[TrainConfig, TD3Config]] = {
    "CheetahRun": (
        dataclasses.replace(_TD3_BASE_CFG, env_name="CheetahRun"),
        _TD3_BASE_ALGO,
    ),
    "WalkerWalk": (
        dataclasses.replace(_TD3_BASE_CFG, env_name="WalkerWalk"),
        _TD3_BASE_ALGO,
    ),
    "HumanoidRun": (
        dataclasses.replace(_TD3_BASE_CFG, env_name="HumanoidRun"),
        dataclasses.replace(_TD3_BASE_ALGO, q_layer_norm=True),  # stability for high-dim
    ),
}


def get_td3_preset(env_name: str) -> tuple[TrainConfig, TD3Config]:
    """Return TD3 preset (TrainConfig, TD3Config) for env, or a default."""
    if env_name in TD3_PRESETS:
        return TD3_PRESETS[env_name]
    return dataclasses.replace(_TD3_BASE_CFG, env_name=env_name), _TD3_BASE_ALGO


def get_preset(env_name: str) -> TrainConfig:
    """Return preset config for env, or a default with env_name set."""
    if env_name in PRESETS:
        return PRESETS[env_name]
    return TrainConfig(env_name=env_name, total_timesteps=3_000_000)
