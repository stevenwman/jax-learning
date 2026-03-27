"""Training configuration — shared fields used by all algorithms."""

from dataclasses import dataclass, field
from jax_rl.configs.ppo_config import PPOConfig


@dataclass
class TrainConfig:
    """Shared training config for all algorithms.

    Contains only fields used across PPO, SAC, TD3, and their variants.
    Algorithm-specific params live in their own configs (SACConfig, TD3Config, etc.).
    PPO-specific fields (network dims, rollout settings) are here for backward
    compatibility but should be accessed via cfg.ppo where possible.
    """

    # Environment
    env_name: str = "CartpoleBalance"
    episode_length: int = 1000

    # Scale
    num_envs: int = 64
    total_timesteps: int = 1_000_000

    # Optimizer (base LR — algos may build their own schedule on top)
    lr: float = 3e-4

    # Environment behavior
    handle_truncation: bool = True

    # Shared RL
    gamma: float = 0.99
    reward_scaling: float = 1.0

    # Logging
    log_interval: int = 1

    # Evaluation + checkpointing
    eval_every_n_episodes: int = 5000  # ~5M steps at 1024 envs
    num_eval_episodes: int = 10

    # Domain randomization
    domain_rand: bool = False

    # PPO config (None for off-policy algos)
    ppo: PPOConfig | None = field(default_factory=PPOConfig)
