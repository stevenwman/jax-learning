"""Configuration dataclasses."""

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """Configuration for MLP encoder."""
    obs_dim: int
    hidden_dim: tuple[int, ...] = (256, 256)  # Tuple to avoid mutable default
    activation: str = "relu"
