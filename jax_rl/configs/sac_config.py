"""SAC algorithm configuration."""

from dataclasses import dataclass


@dataclass
class SACConfig:
    """Configuration for SAC algorithm.

    Optimizer config (LR, grad clipping) is in TrainConfig.
    Replay buffer sizing is here since it's SAC-specific.
    """

    # Core SAC
    tau: float = 0.005                  # Polyak soft update coefficient
    target_entropy_scale: float = 0.5   # target_entropy = -scale * action_dim
                                        # 0.5 = Brax default (softer/more exploitation)
                                        # 1.0 = common textbook (more exploration)
                                        # 0.0 = FastSAC paper (target_entropy=0, prevents alpha collapse at scale)
    alpha_lr: float = 1e-3              # Temperature optimizer LR (separate from policy/Q)
    alpha_init: float = 1.0             # Initial alpha. Vanilla SAC: 1.0, FastSAC paper: 0.001
    max_std: float | None = None        # Cap on pre-tanh std. None=no cap, FastSAC paper: 1.0
    policy_delay: int = 1              # Actor update frequency. 1=every step (vanilla SAC), 4=FastSAC paper
    grad_clip_norm: float | None = None # Max grad norm. None=no clipping. Paper: disabled (0.0)

    # Replay buffer
    buffer_size: int = 4_194_304        # 4M — Playground default
    min_buffer_size: int = 8_192        # Steps before first gradient update
    batch_size: int = 512
    grad_updates_per_step: int = 8      # Gradient steps per env step

    # Network
    hidden_dim: tuple[int, ...] = (256, 256)       # Actor network dims
    critic_hidden_dim: tuple[int, ...] | None = None  # Critic dims. None = same as hidden_dim.
                                                       # FastSAC paper: (768, 384, 192) for critic
    activation: str = "relu"            # SAC uses ReLU (not swish like PPO)
    q_layer_norm: bool = True           # Layer norm in Q-network (Playground default)

    # Observation normalization
    obs_normalization: bool = False     # Normalize obs at sample time (not pre-storage).
                                        # Paper (holosoma) uses True with EmpiricalNormalization.
                                        # When True: raw obs in buffer, normalize after sampling with eps=1e-2.
    obs_norm_eps: float = 1e-2          # Epsilon for obs normalization (paper: 1e-2, prevents near-zero var blowup)
