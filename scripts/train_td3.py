"""TD3 training script.

Usage:
    uv run python train_td3.py --env CheetahRun
    uv run python train_td3.py --env WalkerWalk --exploration-noise 0.15
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
sys.stdout.reconfigure(line_buffering=True)

import argparse

import jax
import optax

from jax_rl.algos.td3 import TD3
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_td3_preset
from jax_rl.training import (
    make_env_bundle, apply_cli_overrides, run_offpolicy_loop,
    add_common_train_args, add_env_shaping_args,
)


def train(cfg: TrainConfig, algo_cfg, seed: int = 0, resume: str | None = None,
          resume_warmup: str = "policy",
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    env_bundle = make_env_bundle(cfg, seed)

    # TD3-specific: plain adam for both actor and critic
    algo = TD3(
        config=algo_cfg,
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        actor_optimizer=optax.adam(cfg.lr),
        critic_optimizer=optax.adam(cfg.lr),
        gamma=cfg.gamma,
        critic_obs_dim=env_bundle.critic_obs_dim,
    )

    # TD3-specific: explore injects Gaussian noise. noise_min/max optional
    # (randomized noise range from the Fast papers).
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
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="td3",
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
    add_common_train_args(parser)
    add_env_shaping_args(parser)
    parser.add_argument("--exploration-noise", type=float, default=None,
                        help="Exploration noise std for TD3 (default: from algo config)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Load preset
    cfg, algo_cfg = get_td3_preset(args.env)

    # Apply overrides
    cfg, algo_cfg = apply_cli_overrides(args, cfg, algo_cfg)

    train(cfg, algo_cfg, seed=args.seed, resume=args.resume,
          resume_warmup=args.resume_warmup,
          use_wandb=args.wandb, wandb_project=args.wandb_project)
