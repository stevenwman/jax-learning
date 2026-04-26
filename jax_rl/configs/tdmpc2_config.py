"""TD-MPC2 configuration.

All HPs sourced from /tmp/tdmpc2/tdmpc2/config.yaml. Do NOT change defaults without
citing source lines.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)  # frozen → hashable → safe as `jax.jit(static_argnames="cfg")` arg
class TDMPC2Config:
    # Architecture
    latent_dim: int = 512
    mlp_dim: int = 512
    enc_dim: int = 256
    num_enc_layers: int = 2
    simnorm_dim: int = 8
    dropout: float = 0.01  # Q heads only
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    # Loss
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 1e-4
    rho: float = 0.5
    grad_clip_norm: float = 20.0

    # Optimization
    lr: float = 3e-4
    enc_lr_scale: float = 0.3          # applied to encoder param group in world model optimizer
    pi_optim_eps: float = 1e-5
    tau: float = 0.01                   # shared by target EMA and Q-scale EMA
    batch_size: int = 256
    horizon: int = 3
    discount: float = 0.99              # derived from episode_lengths[0] at preset load

    # Policy prior bounds
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    # MPPI
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    mppi_iterations: int = 6
    mppi_temperature: float = 0.5
    mppi_min_std: float = 0.05
    mppi_max_std: float = 2.0

    # Training loop
    total_steps: int = 1_000_000      # DMC default
    seed_steps: int = 2500            # source: max(1000, 5*episode_length)
    utd: int = 1
    collect_mode: str = "mppi"        # "mppi" | "prior"
    num_envs: int = 8
    num_eval_envs: int = 8
    eval_every: int = 50_000
    eval_episodes: int = 10
    buffer_size: int = 1_000_000

    # Discount heuristic params
    discount_denom: int = 5
    discount_min: float = 0.95
    discount_max: float = 0.995

    # Env spec (set by make_tdmpc2_config from env — NOT a true default)
    action_dim: int = 0            # MUST be overridden at preset load
    # Source dmcontrol.py:54-60 hardcodes range(2) → action_repeat=2 for ALL DMC tasks.
    # Each agent action drives 2 control steps; rewards summed. With ctrl_dt=0.025 →
    # 20Hz agent decisions. MPPI horizon=3 covers 0.15s vs 0.075s if repeat=1.
    action_repeat: int = 2

    # Multi-task C-seams (B-mode defaults)
    num_tasks: int = 1
    task_names: tuple[str, ...] = ("single",)
    episode_lengths: tuple[int, ...] = (500,)


def compute_discount(episode_length: int, denom: int, dmin: float, dmax: float) -> float:
    """Source tdmpc2.py:58-71 discount heuristic: clamp((frac-1)/frac, dmin, dmax)."""
    frac = episode_length / denom
    d = max(0.0, (frac - 1) / frac) if frac > 0 else 0.0
    return float(max(dmin, min(d, dmax)))


def make_tdmpc2_config(
    action_dim: int,
    episode_length: int = 500,
    task_name: str = "single",
    **overrides,
) -> TDMPC2Config:
    """Factory: builds TDMPC2Config with `discount` derived from episode_length and
    `action_dim` from the env spec. Both required at preset-load time.

    Use this instead of calling TDMPC2Config() directly.
    """
    assert action_dim > 0, f"action_dim must be positive, got {action_dim}"
    base = TDMPC2Config(
        action_dim=action_dim,
        episode_lengths=(episode_length,),
        task_names=(task_name,),
    )
    d = compute_discount(episode_length, base.discount_denom, base.discount_min, base.discount_max)
    return replace(base, discount=d, **overrides)
