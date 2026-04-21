"""PPO algorithm configuration."""

from dataclasses import dataclass, field
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.configs.contraction_config import ContractionConfig


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm.

    Optimizer config (LR, schedule, grad clipping) is not here — optimizers
    are constructed externally and passed to PPO.__init__.
    """

    # PPO objective
    clip_eps: float = 0.3
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95

    # Update
    num_epochs: int = 4
    num_minibatches: int = 32

    # Rollout collection
    num_steps: int = 64                     # steps per rollout before update
    num_updates_per_batch: int = 1          # how many rollout+update cycles per iteration

    # Network architecture
    policy_hidden_dim: tuple[int, ...] = (32, 32, 32, 32)
    value_hidden_dim: tuple[int, ...] = (256, 256, 256, 256, 256)
    activation: str = "swish"
    squash: bool = True
    state_dependent_std: bool = False

    # Optimizer
    max_grad_norm: float | None = None
    anneal_lr: bool = True

    # Runtime fields (populated by train.py — do not set manually)
    minibatch_size: int = 0
    num_envs: int = 0
    gamma: float = 0.99  # Populated from TrainConfig.gamma at runtime

    # Network configs (populated at runtime by train.py once env dims are known)
    encoder: EncoderConfig | None = None
    critic_encoder: EncoderConfig | None = None
    policy_head: PolicyHeadConfig | None = None

    # Advanced
    normalize_advantage: bool = True

    # Contraction theory (None = disabled, baseline PPO)
    contraction: ContractionConfig | None = None
