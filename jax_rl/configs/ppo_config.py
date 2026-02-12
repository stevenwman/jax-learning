"""PPO algorithm configuration."""

from dataclasses import dataclass, field
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig, ValueHeadConfig


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""

    # Actor
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    actor_lr: float = 3e-4

    # Critic
    critic_lr: float = 3e-4

    # Shared
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 4
    batch_size: int = 2048
    max_grad_norm: float = 0.5

    # Environment
    num_envs: int = 4096
    num_steps: int = 32  # Rollout length before update

    # Network configs (nested)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    policy_head: PolicyHeadConfig = field(default_factory=PolicyHeadConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)

    # Advanced (from RSL-RL / 37 PPO details)
    normalize_advantage: bool = True  # Normalize advantages per minibatch
    clip_value_loss: bool = False  # Apply clipping to value loss too
    adaptive_lr: bool = False  # Adaptive LR based on KL divergence
    target_kl: float | None = None  # Target KL for adaptive LR (e.g., 0.01)
