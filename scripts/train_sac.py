"""SAC training script.

Usage:
    uv run python train_sac.py --env Go2WarpJoystickFlat
    uv run python train_sac.py --env WalkerWalk --obs-norm
    uv run python train_sac.py --env Go2WarpJoystickFlat --reset-mode per_step --wandb
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
sys.stdout.reconfigure(line_buffering=True)

import argparse

import optax

from jax_rl.algos.sac import SAC
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_sac_preset
from jax_rl.training import (
    make_env_bundle, apply_cli_overrides, run_offpolicy_loop,
    add_common_train_args, add_env_shaping_args, add_replay_args,
)


def train(cfg: TrainConfig, algo_cfg, seed: int = 0, resume: str | None = None,
          resume_warmup: str = "policy",
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    env_bundle = make_env_bundle(cfg, seed)

    # SAC-specific: optimizer with optional grad clipping
    if algo_cfg.grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(algo_cfg.grad_clip_norm),
            optax.adam(cfg.lr),
        )
    else:
        optimizer = optax.adam(cfg.lr)
    alpha_optimizer = optax.adam(algo_cfg.alpha_lr)

    # SAC-specific: algo
    algo = SAC(
        config=algo_cfg,
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        optimizer=optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma,
        critic_obs_dim=env_bundle.critic_obs_dim,
    )

    # SAC-specific: explore is pass-through (stochastic policy)
    def explore(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key)

    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="sac",
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
    add_common_train_args(parser)
    add_env_shaping_args(parser)
    add_replay_args(parser)
    parser.add_argument("--target-entropy-scale", type=float, default=None,
                        help="target_entropy = -scale * action_dim (default: from algo config)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Load preset
    cfg, algo_cfg = get_sac_preset(args.env)

    # Apply overrides
    cfg, algo_cfg = apply_cli_overrides(args, cfg, algo_cfg)

    train(cfg, algo_cfg, seed=args.seed, resume=args.resume,
          resume_warmup=args.resume_warmup,
          use_wandb=args.wandb, wandb_project=args.wandb_project)
