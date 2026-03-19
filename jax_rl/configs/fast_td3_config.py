"""FastTD3 algorithm config — TD3 + C51 distributional critic."""

from dataclasses import dataclass


@dataclass
class FastTD3Config:
    # TD3 core
    tau: float = 0.005
    policy_delay: int = 2
    target_noise_std: float = 0.2
    noise_clip: float = 0.5
    exploration_noise_std: float = 0.2

    # C51 distributional
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    q_aggregation: str = "avg"  # "avg" (FastTD3) or "min" (vanilla)

    # Training (paper: 12 updates/iter, batch=32768, 1024 envs)
    buffer_size: int = 1_000_000
    min_buffer_size: int = 25_000
    batch_size: int = 8_192
    grad_updates_per_step: int = 12

    # Network (paper: critic=1024, actor=512; we use shared hidden)
    hidden_dim: tuple[int, ...] = (512, 512)
    activation: str = "relu"
    q_layer_norm: bool = True

    # LR decay (cosine from lr → lr_end)
    lr_end: float = 3e-5
