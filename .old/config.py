from dataclasses import dataclass, field

@dataclass
class EncoderConfig:
    obs_dim: int
    hidden_dim: tuple[int, ...] = (256, 256)
    activation: str = "relu"

@dataclass
class PPOConfig:
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
    num_steps: int = 32

    # Network (nested config)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)