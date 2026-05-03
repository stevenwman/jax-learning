"""DIAYN skill discovery training script (vanilla SAC + SkillManager).

Usage:
    uv run python scripts/train_skill_discovery.py --env CheetahRun --num-skills 8
    uv run python scripts/train_skill_discovery.py --env CheetahRun --num-skills 8 --total-timesteps 1000000 --wandb
"""
import argparse
import dataclasses
import optax

from jax_rl.algos.sac import SAC
from jax_rl.configs.env_presets import get_sac_preset
from jax_rl.training.env_setup import make_env_bundle
from jax_rl.training.skill_offpolicy_loop import run_skill_offpolicy_loop
from jax_rl.skill_discovery.config import (
    SkillDiscoveryConfig, FactorConfig, SkillDeployConfig,
)
from jax_rl.skill_discovery.factors import register_extractor
from jax_rl.skill_discovery.manager import SkillManager


# Register the SD-B default factor extractor: full actor obs.
# The built-in `actor_obs_full` from factors.py uses dim=-1 sentinel; SD-B
# specifies the env-specific dim explicitly via FactorConfig.dim at runtime.


def _build_skill_cfg(num_skills: int, obs_dim: int) -> SkillDiscoveryConfig:
    """SD-B default: single DIAYN factor over full actor obs."""
    return SkillDiscoveryConfig(
        mode="diayn",
        total_skill_dim=num_skills,
        prior="one_hot",
        resample="episode",
        intrinsic_weight=1.0,
        task_reward_weight=0.0,  # pure DIAYN for SD-B; SD-C adds task mix
        factors=(FactorConfig(
            name="full_state",
            method="diayn",
            skill_dim=num_skills,
            source="actor_obs",
            extractor="actor_obs_full",
            dim=obs_dim,
        ),),
        deploy=SkillDeployConfig(
            skill_input_mode="fixed",
            default_skill=None,  # set at deploy time
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="DIAYN skill discovery training")
    parser.add_argument("--env", type=str, default="CheetahRun")
    parser.add_argument("--num-skills", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-norm", action="store_true")
    parser.add_argument("--reset-mode", type=str, default=None,
                        choices=["legacy", "per_step"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="jax-rl")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    cfg, algo_cfg = get_sac_preset(args.env)
    if args.total_timesteps:
        cfg = dataclasses.replace(cfg, total_timesteps=args.total_timesteps)
    if args.obs_norm:
        # obs_normalization lives on algo_cfg (SACConfig), NOT cfg (TrainConfig)
        algo_cfg = dataclasses.replace(algo_cfg, obs_normalization=True)
    if args.reset_mode is not None:
        cfg = dataclasses.replace(cfg, reset_mode=args.reset_mode)

    # Build env bundle to learn obs/action dims
    env_bundle = make_env_bundle(cfg, args.seed)
    obs_dim = env_bundle.obs_dim
    action_dim = env_bundle.action_dim

    # Skill discovery setup
    skill_cfg = _build_skill_cfg(args.num_skills, obs_dim)
    skill_manager = SkillManager(skill_cfg)

    # SAC sees augmented obs (raw obs + skill_z)
    augmented_obs_dim = obs_dim + skill_cfg.total_skill_dim

    # Optimizer: vanilla SAC w/ optional grad clipping
    if algo_cfg.grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(algo_cfg.grad_clip_norm),
            optax.adam(cfg.lr),
        )
    else:
        optimizer = optax.adam(cfg.lr)

    alpha_optimizer = optax.adam(algo_cfg.alpha_lr)
    algo = SAC(
        config=algo_cfg,
        obs_dim=augmented_obs_dim,  # actor sees obs + skill_z
        action_dim=action_dim,
        optimizer=optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma,
        critic_obs_dim=env_bundle.critic_obs_dim,  # None for CheetahRun
    )

    def explore_fn(actor_params, obs, key):
        # SAC.select_action samples from the squashed-Gaussian policy.
        # Matches scripts/train_sac.py:53 pattern.
        return algo.select_action(actor_params, obs, key)

    # log_extra_fields: list of (label, key, format_str) tuples per metrics_logger contract
    # log_extra_keys: list of bare key strings
    run_skill_offpolicy_loop(
        cfg=cfg,
        algo_cfg=algo_cfg,
        algo=algo,
        algo_name="sac_skill",
        env_bundle=env_bundle,
        explore_fn=explore_fn,
        log_extra_fields=[
            ("IntR", "intrinsic_reward_mean", ".3f"),
            ("DiscL", "full_state_disc_loss", ".3e"),
            ("DiscA", "full_state_disc_accuracy", ".3f"),
        ],
        log_extra_keys=[
            "intrinsic_reward_mean",
            "full_state_disc_loss",
            "full_state_disc_accuracy",
        ],
        skill_cfg=skill_cfg,
        skill_manager=skill_manager,
        seed=args.seed,
        resume=args.resume,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
