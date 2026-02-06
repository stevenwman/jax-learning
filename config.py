from dataclasses import dataclass, field

@dataclass
class EncoderConfig:
    obs_dim: int
    hidden_dim: tuple[int] = (256,256)
    activation: str = "relu"