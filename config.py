from dataclasses import dataclass, field

@dataclass
class EncoderConfig:
    obs_dim: int
    hidden_dim: tuple[int] = (256,256)
    activation: str = "relu"

@dataclass
class PPOConfig:
    clip_eps: float
    entropy_coef: float
    actor_lr: float
    critic_lr: float
    gamma: float
    gae_lambda: float
    num_epochs: int
    batch_size: int
    max_grad_norm: int
    num_envs: int
    num_steps: int