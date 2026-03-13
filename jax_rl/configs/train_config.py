"""Training configuration — everything train() needs in one place."""

from dataclasses import dataclass, field
from jax_rl.configs.ppo_config import PPOConfig


@dataclass
class TrainConfig:
    """Full training config: env, rollout, optimizer, and algo config.

    Algorithm-specific params live in nested config (e.g., cfg.ppo).
    Shared concerns (env, optimizer, reward) live here.
    """

    # Environment
    env_name: str = "CartpoleBalance"
    episode_length: int = 1000

    # Rollout
    num_envs: int = 64
    num_steps: int = 64  # Unroll length per iteration
    total_timesteps: int = 1_000_000

    # Network
    hidden_dim: tuple[int, ...] = (64, 64)

    # Optimizer
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    # Shared RL
    gamma: float = 0.99

    # Reward
    reward_scaling: float = 1.0

    # Logging
    log_interval: int = 1

    # Algorithm-specific (set one)
    ppo: PPOConfig | None = field(default_factory=PPOConfig)
