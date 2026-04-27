"""FastTD3 training script (C51 distributional + TD3).

Usage:
    uv run python train_fast_td3.py --env CheetahRun
    uv run python train_fast_td3.py --env WalkerWalk --exploration-noise 0.15
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
sys.stdout.reconfigure(line_buffering=True)

import argparse

import jax
import optax

from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_fast_td3_preset
from jax_rl.training import (
    make_env_bundle, apply_cli_overrides, run_offpolicy_loop,
)


def train(cfg: TrainConfig, algo_cfg, seed: int = 0, resume: str | None = None,
          resume_warmup: str = "policy",
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    env_bundle = make_env_bundle(cfg, seed)

    # FastTD3-specific: cosine LR schedule (warmup-aware) + adamw for actor AND critic
    warmup_steps = algo_cfg.min_buffer_size // cfg.num_envs
    train_iters = (cfg.total_timesteps // cfg.num_envs) - warmup_steps
    total_grad_est = train_iters * algo_cfg.grad_updates_per_step
    lr_schedule = (
        optax.cosine_decay_schedule(cfg.lr, total_grad_est,
                                    alpha=algo_cfg.lr_end / cfg.lr)
        if algo_cfg.lr_end < cfg.lr else cfg.lr
    )
    actor_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
    critic_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)

    algo = FastTD3(
        config=algo_cfg,
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        gamma=cfg.gamma,
        critic_obs_dim=env_bundle.critic_obs_dim,
    )

    # FastTD3-specific: explore with Gaussian noise (like TD3)
    exploration_noise_std = getattr(algo_cfg, "exploration_noise_std", 0.1)
    noise_min = getattr(algo_cfg, "noise_min", None)
    noise_max = getattr(algo_cfg, "noise_max", None)

    def explore(actor_params, obs, key):
        key, noise_key = jax.random.split(key)
        if noise_min is not None:
            noise_std = jax.random.uniform(noise_key, (), minval=noise_min, maxval=noise_max)
        else:
            noise_std = exploration_noise_std
        return algo.select_action(
            actor_params, obs, key, deterministic=False, exploration_noise=noise_std,
        )

    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="fast_td3",
        env_bundle=env_bundle, explore_fn=explore,
        log_extra_fields=[],
        log_extra_keys=[],
        seed=seed, resume=resume, resume_warmup=resume_warmup,
        use_wandb=use_wandb, wandb_project=wandb_project,
    )


# ── CLI ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Importable for docs/tooling without parse_args()."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="WalkerWalk",
                        help="Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--resume-warmup", type=str, default="policy",
                        choices=["policy", "random"],
                        help="On resume, refill buffer using loaded policy actions "
                             "(default, prevents eval drop) or legacy random uniform")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from env preset)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train (default: from env preset)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for actor and critic (default: from algo config)")
    parser.add_argument("--reward-scaling", type=float, default=None,
                        help="Multiply rewards by this factor (default: 1.0)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode (default: from env preset)")
    parser.add_argument("--exploration-noise", type=float, default=None,
                        help="Exploration noise std for TD3 (default: from algo config)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 512 episodes)")
    parser.add_argument("--obs-norm", action="store_true",
                        help="Enable sample-time obs normalization (recommended for humanoid tasks)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking (requires wandb installed)")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name (default: jax-rl)")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of stacked observation frames (default: 1, use 3 for locomotion)")
    parser.add_argument("--action-delay-ms", type=int, default=None,
                        help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
    parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Randomized action delay range in ms (e.g., 40 120)")
    parser.add_argument("--reset-mode", type=str, default=None,
                        choices=["legacy", "per_step"],
                        help="Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for gradient updates (default: from algo config)")
    parser.add_argument("--grad-updates-per-step", type=int, default=None,
                        help="Gradient updates per env step (default: from algo config)")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer capacity (default: from algo config)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Load preset
    cfg, algo_cfg = get_fast_td3_preset(args.env)

    # Apply overrides
    cfg, algo_cfg = apply_cli_overrides(args, cfg, algo_cfg)

    train(cfg, algo_cfg, seed=args.seed, resume=args.resume,
          resume_warmup=args.resume_warmup,
          use_wandb=args.wandb, wandb_project=args.wandb_project)
