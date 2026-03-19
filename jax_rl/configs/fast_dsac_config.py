"""FastDSAC config — SAC + continuous Gaussian distributional critic + DEM.

Paper: FastDSAC (arXiv:2603.12612)
"""

from dataclasses import dataclass


@dataclass
class FastDSACConfig:
    # SAC core
    tau: float = 0.005
    target_entropy: float = 0.0  # paper uses 0, NOT -dim(A)
    alpha_lr: float = 3e-4

    # Gaussian distributional critic
    variance_eps: float = 1e-6

    # DEM (dimension-wise entropy modulation)
    dem_temperature: float = 1.0
    beta_min: float = 0.01
    beta_max: float = 2.0

    # Training
    buffer_size: int = 1_000_000
    min_buffer_size: int = 25_000
    batch_size: int = 8_192
    grad_updates_per_step: int = 12

    # Network
    hidden_dim: tuple[int, ...] = (512, 512)
    activation: str = "relu"
    q_layer_norm: bool = True

    # AdamW optimizer
    weight_decay: float = 1e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.95

    # LR decay (cosine)
    lr_end: float = 3e-5
