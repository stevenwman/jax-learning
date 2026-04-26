"""Eval + checkpoint trigger — replaces the 20-line eval/checkpoint block across all train scripts."""

import jax

from jax_rl.training.checkpointing import save_checkpoint, CheckpointManager
from jax_rl.training.episode_tracker import EpisodeTracker
from jax_rl.training.metrics_logger import wandb_log
from jax_rl.training.train_context import TrainContext
from jax_rl.utils.eval import evaluate, evaluate_gym


def _eval_fn_for(ctx: TrainContext):
    """Pick the eval function for ctx.backend_kind."""
    if ctx.backend_kind == "gym":
        return evaluate_gym
    return evaluate


def maybe_eval_and_checkpoint(
    select_action_fn,
    actor_params,
    eval_env,
    tracker: EpisodeTracker,
    ctx: TrainContext,
    training_state,
    norm_state,
    last_eval_eps: int,
    key: jax.Array,
    obs_normalize_fn=None,
    q_fn=None,
    critic_norm_state=None,
) -> tuple[int, jax.Array]:
    """Run eval + save checkpoint if enough episodes completed since last eval.

    Returns:
        (updated_last_eval_eps, updated_key)
    """
    cfg = ctx.cfg
    n_eps = tracker.n_episodes
    if n_eps < last_eval_eps + cfg.eval_every_n_episodes:
        return last_eval_eps, key

    import os
    metrics_log = ctx.metrics_log
    total_steps = metrics_log[-1]["total_steps"] if metrics_log else 0
    eval_log = os.path.join(ctx.ckpt_dir, "eval_log.csv")

    key, eval_key = jax.random.split(key)
    eval_metrics = _eval_fn_for(ctx)(
        select_action_fn, actor_params,
        eval_env, num_episodes=cfg.num_eval_episodes,
        episode_length=cfg.episode_length, key=eval_key,
        num_envs=cfg.num_envs,
        obs_normalize_fn=obs_normalize_fn,
        q_fn=q_fn,
        gamma=cfg.gamma,
        eval_log_path=eval_log,
        total_steps=total_steps,
    )

    q_str = ""
    if "q_bias" in eval_metrics:
        q_str = (f" | Q bias={eval_metrics['q_bias']:.2f}"
                 f" RMSE={eval_metrics['q_rmse']:.2f}"
                 f" corr={eval_metrics['q_corr']:.3f}")
    print(
        f"  EVAL @ {n_eps} eps | "
        f"Return {eval_metrics['eval_mean']:.1f} ± {eval_metrics['eval_std']:.1f} "
        f"[{eval_metrics['eval_min']:.0f}, {eval_metrics['eval_max']:.0f}]"
        f"{q_str}"
    )

    if metrics_log:
        metrics_log[-1].update(eval_metrics)

    # Log eval metrics to W&B (no-op if wandb not initialized)
    wandb_log(eval_metrics, step=metrics_log[-1]["total_steps"] if metrics_log else 0)

    if ctx.ckpt_mgr is not None:
        is_best = ctx.ckpt_mgr.save(
            training_state, norm_state, cfg, ctx.algo_cfg,
            ctx.algo_name, ctx.obs_dim, ctx.action_dim, metrics_log, ctx.resume,
            eval_mean=eval_metrics['eval_mean'],
            critic_norm_state=critic_norm_state,
        )
        if is_best:
            print(f"  New best! eval={ctx.ckpt_mgr.best_eval:.1f}")
        else:
            print(f"  Checkpoint saved to {ctx.ckpt_dir}")
    else:
        save_checkpoint(ctx.ckpt_dir, training_state, norm_state, cfg, ctx.algo_cfg,
                        ctx.algo_name, ctx.obs_dim, ctx.action_dim, metrics_log, ctx.resume,
                        critic_norm_state=critic_norm_state)
        print(f"  Checkpoint saved to {ctx.ckpt_dir}")

    return n_eps, key


def final_eval_and_checkpoint(
    select_action_fn,
    actor_params,
    eval_env,
    tracker: EpisodeTracker,
    ctx: TrainContext,
    training_state,
    norm_state,
    key: jax.Array,
    total_gradient_steps: int,
    obs_normalize_fn=None,
    q_fn=None,
    critic_norm_state=None,
) -> dict:
    """Run final eval + save checkpoint after training completes. Returns eval_metrics."""
    cfg = ctx.cfg
    metrics_log = ctx.metrics_log

    key, eval_key = jax.random.split(key)
    eval_metrics = _eval_fn_for(ctx)(
        select_action_fn, actor_params,
        eval_env, num_episodes=cfg.num_eval_episodes,
        episode_length=cfg.episode_length, key=eval_key,
        num_envs=cfg.num_envs,
        obs_normalize_fn=obs_normalize_fn,
        q_fn=q_fn,
        gamma=cfg.gamma,
    )

    if ctx.ckpt_mgr is not None:
        is_best = ctx.ckpt_mgr.save(
            training_state, norm_state, cfg, ctx.algo_cfg,
            ctx.algo_name, ctx.obs_dim, ctx.action_dim, metrics_log, ctx.resume,
            eval_mean=eval_metrics['eval_mean'],
            critic_norm_state=critic_norm_state,
        )
        if is_best:
            print(f"  New best! eval={ctx.ckpt_mgr.best_eval:.1f}")
    else:
        save_checkpoint(ctx.ckpt_dir, training_state, norm_state, cfg, ctx.algo_cfg,
                        ctx.algo_name, ctx.obs_dim, ctx.action_dim, metrics_log, ctx.resume,
                        critic_norm_state=critic_norm_state)

    print("=" * 80)
    print("Training complete.")
    if tracker.completed_returns:
        import numpy as np
        final = tracker.completed_returns[-100:]
        print(f"  Total episodes: {tracker.n_episodes}")
        print(f"  Online avg return (last 100 eps): {np.mean(final):.1f}")
    print(f"  Eval return: {eval_metrics['eval_mean']:.1f} ± {eval_metrics['eval_std']:.1f} "
          f"[{eval_metrics['eval_min']:.0f}, {eval_metrics['eval_max']:.0f}]")
    if "q_bias" in eval_metrics:
        print(f"  Q diagnostics: bias={eval_metrics['q_bias']:.2f}, "
              f"RMSE={eval_metrics['q_rmse']:.2f}, corr={eval_metrics['q_corr']:.3f}")
        print(f"  Q mean={eval_metrics['q_mean']:.2f}, MC mean={eval_metrics['mc_mean']:.2f}")
    print(f"  Total gradient steps: {total_gradient_steps:,}")
    print(f"  Final checkpoint: {ctx.ckpt_dir}")

    return eval_metrics
