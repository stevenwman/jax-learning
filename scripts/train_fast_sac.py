"""FastSAC training script (C51 distributional + SAC).

Usage:
    uv run python train_fast_sac.py --env Go2WarpJoystickFlat
    uv run python train_fast_sac.py --env WalkerWalk --obs-norm
    uv run python train_fast_sac.py --env Go2WarpJoystickFlat --reset-mode per_step --wandb
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
sys.stdout.reconfigure(line_buffering=True)

import argparse

import optax

from jax_rl.algos.fast_sac import FastSAC
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_fast_sac_preset
from jax_rl.training import (
    make_env_bundle, apply_cli_overrides, run_offpolicy_loop,
)


def train(cfg: TrainConfig, algo_cfg, seed: int = 0, resume: str | None = None,
          resume_warmup: str = "policy",
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    env_bundle = make_env_bundle(cfg, seed)

    # FastSAC-specific: cosine LR schedule (warmup-aware) + adamw
    warmup_steps = algo_cfg.min_buffer_size // cfg.num_envs
    train_iters = (cfg.total_timesteps // cfg.num_envs) - warmup_steps
    total_grad_est = train_iters * algo_cfg.grad_updates_per_step
    lr_schedule = (
        optax.cosine_decay_schedule(cfg.lr, total_grad_est,
                                    alpha=algo_cfg.lr_end / cfg.lr)
        if algo_cfg.lr_end < cfg.lr else cfg.lr
    )
    optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
    alpha_optimizer = optax.adam(algo_cfg.alpha_lr)

    algo = FastSAC(
        config=algo_cfg,
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        optimizer=optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma,
        critic_obs_dim=env_bundle.critic_obs_dim,
    )

    # FastSAC-specific: explore is pass-through (SAC-family stochastic policy)
    def explore(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key)

    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="fast_sac",
        env_bundle=env_bundle, explore_fn=explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
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
                        help="Warm-start from checkpoint: restores weights + opt state + norm state. Replay buffer is NOT persisted; refilled with loaded policy actions per --resume-warmup. Not exact training continuation.")
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
    parser.add_argument("--target-entropy-scale", type=float, default=None,
                        help="target_entropy = -scale * action_dim (default: from algo config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for gradient updates (default: from algo config)")
    parser.add_argument("--grad-updates-per-step", type=int, default=None,
                        help="Gradient updates per environment step (default: from algo config)")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer capacity (default: from algo config)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 5000 episodes; "
                             "Go2 OSC/physical presets set 500)")
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
    parser.add_argument("--v-min", type=float, default=None,
                        help="C51 critic support lower bound (default: from algo config, -20 for FastSAC)")
    parser.add_argument("--v-max", type=float, default=None,
                        help="C51 critic support upper bound (default: from algo config, +20 for FastSAC)")
    parser.add_argument("--num-atoms", type=int, default=None,
                        help="C51 critic atom count (default: from algo config, 101)")
    parser.add_argument("--tau", type=float, default=None,
                        help="Target network soft-update rate (default: 0.125)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--policy-delay", type=int, default=None,
                        help="Critic updates per actor update (default: 4)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Load preset
    cfg, algo_cfg = get_fast_sac_preset(args.env)

    # Apply overrides
    cfg, algo_cfg = apply_cli_overrides(args, cfg, algo_cfg)

    train(cfg, algo_cfg, seed=args.seed, resume=args.resume,
          resume_warmup=args.resume_warmup,
          use_wandb=args.wandb, wandb_project=args.wandb_project)
