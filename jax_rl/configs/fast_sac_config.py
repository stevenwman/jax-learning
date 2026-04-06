"""FastSAC algorithm config — SAC + C51 distributional critic.

Standalone config (not inheriting SACConfig) because the paper defaults
diverge from vanilla SAC on 9/14 shared fields. Inheritance would make
bare FastSACConfig() silently produce SAC defaults, which are wrong for
FastSAC (e.g. tau=0.005 instead of 0.125 — the exact bug that caused
NaN divergence on HumanoidRun, see .context/archive/FAST_ALGOS_LIT_MISMATCH.md).
"""

from dataclasses import dataclass


@dataclass
class FastSACConfig:
    """FastSAC = SAC + C51 distributional critic (Seo et al. 2025).

    All defaults are from the paper source code (holosoma), NOT vanilla SAC.
    """

    # Core SAC (paper defaults, NOT vanilla SAC defaults)
    tau: float = 0.125                  # paper: 0.125 (25x faster than SAC's 0.005)
    target_entropy_scale: float = 0.0   # paper: 0 (prevents alpha collapse at scale)
    alpha_lr: float = 3e-4              # paper: same as policy LR
    alpha_init: float = 0.001           # paper: start near-zero (SAC uses 1.0)
    max_std: float | None = 1.0         # paper: cap pre-tanh std (SAC uses None)
    policy_delay: int = 4               # paper: actor updates every 4th critic update
    grad_clip_norm: float | None = None # paper: disabled

    # Replay buffer
    buffer_size: int = 4_194_304        # 4M
    min_buffer_size: int = 8_192
    batch_size: int = 8_192             # paper: 8192 (SAC uses 512)
    grad_updates_per_step: int = 8      # paper: UTD=8

    # Network — paper: tapered 3-layer (not flat 2-layer like SAC)
    hidden_dim: tuple[int, ...] = (512, 256, 128)        # actor
    critic_hidden_dim: tuple[int, ...] | None = (768, 384, 192)  # wider critic
    activation: str = "swish"           # paper: SiLU (SAC uses relu)
    q_layer_norm: bool = True

    # C51 distributional
    num_atoms: int = 101                # paper: 101 (not 51)
    v_min: float = -20.0                # paper: [-20, 20]
    v_max: float = 20.0
    q_aggregation: str = "avg"          # paper: avg (not min)

    # LR decay — paper uses cosine decay
    lr_end: float = 3e-5                # paper: cosine to near-zero

    # Observation normalization — paper uses True. Defaulting to False until
    # we A/B test on Go2 (paper benchmarks are DM Control, not locomotion).
    obs_normalization: bool = False
    obs_norm_eps: float = 1e-2
