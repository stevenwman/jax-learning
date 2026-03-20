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
    hidden_dim: tuple[int, ...] = (256, 256)       # Actor network dims
    critic_hidden_dim: tuple[int, ...] | None = None  # Critic dims. None = same as hidden_dim.
    activation: str = "relu"
    q_layer_norm: bool = False
    grad_clip_norm: float | None = 1.0  # Max grad norm. None=no clipping.
    obs_normalization: bool = False     # Normalize obs at sample time (not pre-storage)
    obs_norm_eps: float = 1e-2
