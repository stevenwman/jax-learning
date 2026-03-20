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
    alpha_init: float = 0.001    # paper: 0.001 (HumanoidBench) / 0.01 (MuJoCo PG)

    # Gaussian distributional critic
    variance_eps: float = 1e-6

    # DEM (dimension-wise entropy modulation)
    dem_temperature: float = 1.0  # paper: per-task 0.5-10.0
    beta_min: float = 0.01
    beta_max: float = 2.0

    # Training — paper uses much smaller buffers (5K-51K) and larger batch (32K)
    buffer_size: int = 51_200      # paper: 51,200 for HumanoidBench
    min_buffer_size: int = 1_000   # paper: 1,000
    batch_size: int = 32_768       # paper: 32,768
    grad_updates_per_step: int = 8

    # Network
    hidden_dim: tuple[int, ...] = (512, 512)
    critic_hidden_dim: tuple[int, ...] | None = None  # None = same as hidden_dim
    activation: str = "relu"
    q_layer_norm: bool = True

    # Optimizer
    grad_clip_norm: float | None = None  # Max grad norm. None=no clipping.
    weight_decay: float = 1e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.95

    # LR decay (cosine)
    lr_end: float = 3e-5

    # Observation normalization
    obs_normalization: bool = False
    obs_norm_eps: float = 1e-2
