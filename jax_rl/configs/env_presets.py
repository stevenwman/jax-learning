"""Per-environment training presets (MuJoCo Playground reference configs)."""

import dataclasses

from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.configs.fast_dsac_config import FastDSACConfig
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


# FastTD3 presets — C51 distributional + large batch + LR decay
_FAST_TD3_BASE_CFG = TrainConfig(
    total_timesteps=100_000_000,
    num_envs=1024,
    episode_length=1000,
    lr=3e-4,
    anneal_lr=False,
    reward_scaling=1.0,
    gamma=0.99,
    num_eval_episodes=5,  # fewer eval envs to avoid OOM with 1024 training envs
    handle_truncation=True,
    ppo=None,
)

_FAST_TD3_BASE_ALGO = FastTD3Config()

# v_min/v_max must cover the actual Q-value range for each env.
# Q ≈ avg_reward_per_step / (1 - gamma). With gamma=0.99:
#   CheetahRun: reward ~0.8 → Q ~80. WalkerWalk: reward ~0.97 → Q ~97.
#   HumanoidRun: reward ~0.2 → Q ~20.
FAST_TD3_PRESETS: dict[str, tuple[TrainConfig, FastTD3Config]] = {
    "CheetahRun": (
        dataclasses.replace(_FAST_TD3_BASE_CFG, env_name="CheetahRun"),
        dataclasses.replace(_FAST_TD3_BASE_ALGO, v_min=-10.0, v_max=150.0),
    ),
    "WalkerWalk": (
        dataclasses.replace(_FAST_TD3_BASE_CFG, env_name="WalkerWalk"),
        dataclasses.replace(_FAST_TD3_BASE_ALGO, v_min=-10.0, v_max=150.0),
    ),
    "HumanoidRun": (
        dataclasses.replace(_FAST_TD3_BASE_CFG, env_name="HumanoidRun"),
        dataclasses.replace(_FAST_TD3_BASE_ALGO, v_min=-10.0, v_max=50.0,
                            exploration_noise_std=0.3),
    ),
}


def get_fast_td3_preset(env_name: str) -> tuple[TrainConfig, FastTD3Config]:
    """Return FastTD3 preset (TrainConfig, FastTD3Config) for env, or a default."""
    if env_name in FAST_TD3_PRESETS:
        return FAST_TD3_PRESETS[env_name]
    return dataclasses.replace(_FAST_TD3_BASE_CFG, env_name=env_name), _FAST_TD3_BASE_ALGO


# FastSAC presets — SAC + C51 distributional critic at FastTD3 scale
# Matches FastTD3 network/training scale: (512, 512), batch=8192, 12 grad updates, 1024 envs
_FAST_SAC_BASE_CFG = TrainConfig(
    total_timesteps=100_000_000,
    num_envs=1024,
    episode_length=1000,
    lr=1e-3,
    anneal_lr=False,
    reward_scaling=1.0,
    gamma=0.99,
    num_eval_episodes=5,
    handle_truncation=True,
    ppo=None,
)

_FAST_SAC_BASE_ALGO = SACConfig(
    tau=0.005,
    target_entropy_scale=0.5,
    alpha_lr=1e-3,
    buffer_size=4_194_304,
    min_buffer_size=8_192,
    batch_size=8_192,
    grad_updates_per_step=12,
    hidden_dim=(512, 512),
    activation="relu",
    q_layer_norm=True,
)

FAST_SAC_PRESETS: dict[str, tuple[TrainConfig, SACConfig]] = {
    "CheetahRun": (
        dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="CheetahRun"),
        _FAST_SAC_BASE_ALGO,
    ),
    "WalkerWalk": (
        dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="WalkerWalk"),
        _FAST_SAC_BASE_ALGO,
    ),
    "HumanoidRun": (
        dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="HumanoidRun"),
        _FAST_SAC_BASE_ALGO,
    ),
}


def get_fast_sac_preset(env_name: str) -> tuple[TrainConfig, SACConfig]:
    """Return FastSAC preset (TrainConfig, SACConfig) for env, or a default."""
    if env_name in FAST_SAC_PRESETS:
        return FAST_SAC_PRESETS[env_name]
    return dataclasses.replace(_FAST_SAC_BASE_CFG, env_name=env_name), _FAST_SAC_BASE_ALGO


# FastDSAC presets — Gaussian distributional critic + DEM
_FAST_DSAC_BASE_CFG = TrainConfig(
    total_timesteps=100_000_000,
    num_envs=1024,
    episode_length=1000,
    lr=3e-4,
    anneal_lr=False,
    reward_scaling=1.0,
    gamma=0.99,
    num_eval_episodes=5,
    handle_truncation=True,
    ppo=None,
)

_FAST_DSAC_BASE_ALGO = FastDSACConfig()

FAST_DSAC_PRESETS: dict[str, tuple[TrainConfig, FastDSACConfig]] = {
    "CheetahRun": (
        dataclasses.replace(_FAST_DSAC_BASE_CFG, env_name="CheetahRun"),
        _FAST_DSAC_BASE_ALGO,
    ),
    "WalkerWalk": (
        dataclasses.replace(_FAST_DSAC_BASE_CFG, env_name="WalkerWalk"),
        _FAST_DSAC_BASE_ALGO,
    ),
    "HumanoidRun": (
        dataclasses.replace(_FAST_DSAC_BASE_CFG, env_name="HumanoidRun"),
        dataclasses.replace(_FAST_DSAC_BASE_ALGO, q_layer_norm=True),
    ),
}


def get_fast_dsac_preset(env_name: str) -> tuple[TrainConfig, FastDSACConfig]:
    """Return FastDSAC preset (TrainConfig, FastDSACConfig) for env, or a default."""
    if env_name in FAST_DSAC_PRESETS:
        return FAST_DSAC_PRESETS[env_name]
    return dataclasses.replace(_FAST_DSAC_BASE_CFG, env_name=env_name), _FAST_DSAC_BASE_ALGO


def get_preset(env_name: str) -> TrainConfig:
    """Return preset config for env, or a default with env_name set."""
    if env_name in PRESETS:
        return PRESETS[env_name]
    return TrainConfig(env_name=env_name, total_timesteps=3_000_000)
