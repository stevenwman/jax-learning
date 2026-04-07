"""FlashSAC algorithm config — Kim et al. 2026.

Standalone config (not inheriting FastSACConfig) because FlashSAC diverges
significantly: inverted residual blocks, BatchNorm, weight normalization,
adaptive reward scaling, Zeta noise repetition, unified entropy target.
All defaults from the reference implementation (Holiday-Robot/FlashSAC).
"""

from dataclasses import dataclass


@dataclass
class FlashSACConfig:
    """FlashSAC = SAC + inverted residual blocks + norm bounding (Kim et al. 2026).

    All defaults from the reference source code (Holiday-Robot/FlashSAC).
    """

    # Architecture
    num_blocks: int = 2
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    expansion: int = 4
    num_atoms: int = 101
    v_min: float = -5.0
    v_max: float = 5.0

    # Training
    tau: float = 0.01
    policy_delay: int = 2
    batch_size: int = 2048
    buffer_size: int = 1_000_000
    min_buffer_size: int = 10_000
    grad_updates_per_step: int = 1
    gamma: float = 0.99
    n_step: int = 1

    # Temperature
    alpha_init: float = 0.01
    sigma_target: float = 0.15

    # Actor regularization
    bc_alpha: float = 0.0

    # Reward scaling
    normalize_reward: bool = True
    G_max: float = 5.0

    # Exploration
    noise_zeta_mu: float = 2.0
    noise_zeta_max: int = 16

    # LR schedule (warmup -> cosine decay) — shared by actor, critic, AND temperature
    lr_init: float = 3e-4
    lr_peak: float = 3e-4
    lr_end: float = 1.5e-4
    lr_warmup_frac: float = 1e-6
    lr_decay_frac: float = 1.0

    # Weight norm
    weight_norm: bool = True
