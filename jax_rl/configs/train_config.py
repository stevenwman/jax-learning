"""Training configuration — everything train() needs in one place."""

from dataclasses import dataclass, field
from jax_rl.configs.ppo_config import PPOConfig


@dataclass
class TrainConfig:
    """Full training config: env, rollout, optimizer, and algo config.

    Algorithm-specific params live in nested config (e.g., cfg.ppo).
    Shared concerns (env, optimizer, reward) live here.
    """

    # Environment
    env_name: str = "CartpoleBalance"
    episode_length: int = 1000

    # Rollout
    num_envs: int = 64
    num_steps: int = 64  # Unroll length per collect→update cycle
    num_updates_per_batch: int = 1  # Collect→update cycles per iteration (Brax uses 16)
    total_timesteps: int = 1_000_000

    # Network
    policy_hidden_dim: tuple[int, ...] = (32, 32, 32, 32)
    value_hidden_dim: tuple[int, ...] = (256, 256, 256, 256, 256)
    activation: str = "swish"
    squash: bool = True  # Tanh-squash actions to [-1, 1] (Brax default)
    state_dependent_std: bool = False  # Network-predicted std (Brax tanh_normal default)

    # Optimizer
    lr: float = 3e-4
    max_grad_norm: float | None = None  # None = no clipping (Brax default)
    anneal_lr: bool = True

    # Environment behavior
    # True for auto-reset envs (MuJoCo Playground, Brax, IsaacGym) where next_obs after
    # timeout is the reset observation, not the true terminal obs. False for raw Gymnasium,
    # raw dm_control, or real robots where next_obs is trustworthy.
    handle_truncation: bool = True

    # Shared RL
    gamma: float = 0.99

    # Reward
    reward_scaling: float = 1.0

    # Logging
    log_interval: int = 1

    # Evaluation + checkpointing (triggered together)
    eval_every_n_episodes: int = 500  # run deterministic eval + save checkpoint
    num_eval_episodes: int = 10       # episodes per eval

    # Algorithm-specific (set one)
    ppo: PPOConfig | None = field(default_factory=PPOConfig)
