"""TD3 algorithm config."""

from dataclasses import dataclass


@dataclass
class TD3Config:
    tau: float = 0.005
    policy_delay: int = 2
    target_noise_std: float = 0.2
    noise_clip: float = 0.5
    exploration_noise_std: float = 0.1
    buffer_size: int = 1_000_000
    min_buffer_size: int = 10_000
    batch_size: int = 256
    grad_updates_per_step: int = 1
    hidden_dim: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    q_layer_norm: bool = False
