"""PPO algorithm configuration."""

from dataclasses import dataclass, field
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig, ValueHeadConfig


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm.

    Optimizer config (LR, schedule, grad clipping) is not here — optimizers
    are constructed externally and passed to PPO.__init__.
    """

    # PPO objective
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95

    # Update
    num_epochs: int = 4
    minibatch_size: int = 32
    num_envs: int = 4096

    # Network configs (populated at runtime by train.py once env dims are known)
    encoder: EncoderConfig | None = None
    policy_head: PolicyHeadConfig | None = None
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)

    # Advanced
    normalize_advantage: bool = True  # Normalize advantages per minibatch
