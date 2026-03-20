"""FastTD3 algorithm config — TD3 + C51 distributional critic."""

from dataclasses import dataclass


@dataclass
class FastTD3Config:
    # TD3 core
    tau: float = 0.125  # paper: 0.125 (fast target update for high UTD ratio)
    policy_delay: int = 2
    target_noise_std: float = 0.2
    noise_clip: float = 0.5
    exploration_noise_std: float = 0.2
    # Mixed noise: sample σ ~ U[noise_min, noise_max] each step (paper recipe)
    # When noise_min is set, exploration_noise_std is ignored
    noise_min: float | None = None
    noise_max: float | None = None

    # C51 distributional
    num_atoms: int = 101           # paper: 101 (not 51)
    v_min: float = -20.0          # paper: [-20, 20]
    v_max: float = 20.0
    q_aggregation: str = "avg"

    # Training
    buffer_size: int = 1_000_000
    min_buffer_size: int = 25_000
    batch_size: int = 8_192
    grad_updates_per_step: int = 8  # paper: 8

    # Network — paper: actor 512→256→128 (tapered), critic 768→384→192
    hidden_dim: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dim: tuple[int, ...] | None = (768, 384, 192)
    activation: str = "swish"      # paper: SiLU (= swish)
    q_layer_norm: bool = True

    # LR decay — paper uses constant LR (lr_end=lr). Set lr_end < lr for cosine decay.
    lr_end: float = 3e-4  # same as default lr = constant
