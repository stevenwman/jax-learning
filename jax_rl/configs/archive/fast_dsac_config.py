"""FastDSAC config — SAC + Huber-based distributional critic + DEM.

Paper: FastDSAC (arXiv:2603.12612)
NOTE: Despite the name "Gaussian distributional", the paper's actual code uses
Huber loss, NOT Gaussian NLL. The critic outputs (mean, std) via softplus,
and the loss uses bounded Huber terms with per-sample ratio weighting.
"""

from dataclasses import dataclass


@dataclass
class FastDSACConfig:
    # SAC core
    tau: float = 0.1              # paper: 0.1 (NOT 0.005, much faster target tracking)
    target_entropy: float = 0.0   # paper uses 0, NOT -dim(A)
    alpha_lr: float = 3e-4
    alpha_init: float = 0.001     # paper: 0.001 (HumanoidBench) / 0.01 (MuJoCo PG)

    # Distributional critic (Huber-based, not Gaussian NLL)
    huber_delta: float = 50.0     # paper: delta=50 for Huber loss
    bias: float = 1e-6            # paper: 0.000001, division safety in ratio/variance terms

    # DEM (dimension-wise entropy modulation)
    dem_temperature: float = 1.0  # paper: per-task 0.5-10.0
    beta_min: float = 0.01
    beta_max: float = 2.0

    # Training — paper: 32K batch, 2 updates/step, small buffer
    buffer_size: int = 51_200     # paper: 51,200 for HumanoidBench
    min_buffer_size: int = 10_240  # paper: learning_starts=10 iters. 10*1024envs=10240 samples
    batch_size: int = 32_768       # paper: 32,768
    grad_updates_per_step: int = 16 # paper: 2 updates per 128 envs = UTD ~512. Match with 1024 envs: 16 updates
    policy_delay: int = 2         # paper: policy_frequency=2

    # Network — paper: actor 512→256→128, critic 1024→512→256, GELU
    hidden_dim: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dim: tuple[int, ...] | None = (1024, 512, 256)
    activation: str = "gelu"      # paper: GELU (not ReLU)
    q_layer_norm: bool = True

    # Optimizer — paper: weight_decay=0.1 (NOT 1e-4!)
    grad_clip_norm: float | None = None
    weight_decay: float = 0.1     # paper: 0.1
    adam_b1: float = 0.9
    adam_b2: float = 0.95

    # LR decay (cosine)
    lr_end: float = 3e-5

    # Reward scaling — paper: 0.2
    reward_scale: float = 0.2

    # Observation normalization
    obs_normalization: bool = False
    obs_norm_eps: float = 1e-2
