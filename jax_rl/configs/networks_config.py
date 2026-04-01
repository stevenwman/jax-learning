"""Network configuration dataclasses."""

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """Configuration for encoder networks (MLP, CNN, etc.)."""

    obs_dim: int
    hidden_dim: tuple[int, ...] = (256, 256)
    activation: str = "relu"

    # Normalization (for future FastTD3/SAC)
    norm: str | None = None  # None, "layer", "spectral"
    norm_placement: str = "pre"  # "pre" or "post" activation

    # Context handling (for future goal-conditioned, USD)
    context_dim: int | None = None
    context_fusion: str = "concat"  # "concat", "film", "cross_attn"


@dataclass
class PolicyHeadConfig:
    """Configuration for policy heads (Gaussian, Deterministic)."""

    action_dim: int
    # For Gaussian policies
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    # State-independent std: learned param vector (PPO default, more stable)
    # State-dependent std: Dense layer from features (SAC needs this)
    state_dependent_std: bool = False
    init_noise_std: float = 1.0  # Initial std for state-independent mode
    min_std: float = 0.001  # Minimum std for state-dependent mode (softplus floor)
    # For squashing (tanh transform)
    squash: bool = True  # Output in [-1, 1] for bounded action spaces
    # DEM (dimension-wise entropy modulation) for FastDSAC
    dem: bool = False  # When True, output includes dem_logits


