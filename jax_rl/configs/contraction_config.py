"""Configuration for ContractionPPO metric training + reward augmentation.

Ref: /tmp/cppo/contraction_cfg.py. See plan Task 2 for field origins.
"""

from dataclasses import dataclass


@dataclass
class ContractionConfig:
    alpha: float = 0.1
    epsilon_contraction: float = 1e-3
    penalty_coef: float = 1.0
    constraint_coef: float = 1.0
    metric_lr: float = 1e-3
    constraint_dim: int = 0
    hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "elu"
    min_diagonal_value: float = 0.1
    spectral_norm_bound: float = 3.0

    def validate(self) -> None:
        if self.constraint_dim <= 0:
            raise ValueError(
                f"ContractionConfig.constraint_dim must be populated at runtime "
                f"(got {self.constraint_dim}). See plan Task 8 Step 0."
            )
