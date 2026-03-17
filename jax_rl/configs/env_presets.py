"""Per-environment training presets (MuJoCo Playground reference configs)."""

from jax_rl.configs.ppo_config import PPOConfig
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
    "HumanoidRun": TrainConfig(
        env_name="HumanoidRun",
        total_timesteps=60_000_000,
        num_envs=4096,
        num_steps=480,
        gamma=0.97,
        lr=1e-3,
        reward_scaling=10.0,
        ppo=PPOConfig(entropy_coef=0.0, num_epochs=4),
    ),
}


def get_preset(env_name: str) -> TrainConfig:
    """Return preset config for env, or a default with env_name set."""
    if env_name in PRESETS:
        return PRESETS[env_name]
    return TrainConfig(env_name=env_name, total_timesteps=3_000_000)
