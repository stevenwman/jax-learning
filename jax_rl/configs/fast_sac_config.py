"""FastSAC algorithm config — SAC + C51 distributional critic."""

from dataclasses import dataclass

from jax_rl.configs.sac_config import SACConfig


@dataclass
class FastSACConfig(SACConfig):
    """FastSAC = SAC + C51 distributional critic (Seo et al. 2025).

    Inherits all SAC fields (tau, alpha, buffer, network, etc.)
    and adds C51 distributional + LR decay parameters.
    """

    # C51 distributional
    num_atoms: int = 101           # paper: 101
    v_min: float = -20.0           # paper: [-20, 20]
    v_max: float = 20.0
    q_aggregation: str = "avg"     # "avg" (paper) or "min" (vanilla)

    # LR decay — paper uses cosine decay. Set lr_end < lr for decay, lr_end == lr for constant.
    lr_end: float = 3e-5           # paper: cosine to near-zero
