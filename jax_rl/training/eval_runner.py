"""Eval + checkpoint trigger — replaces the 20-line eval/checkpoint block across all train scripts."""

import jax

from jax_rl.training.checkpointing import save_checkpoint, CheckpointManager
from jax_rl.training.episode_tracker import EpisodeTracker
from jax_rl.training.metrics_logger import wandb_log
from jax_rl.utils.eval import evaluate


def maybe_eval_and_checkpoint(
    select_action_fn,
    actor_params,
    eval_env,
    tracker: EpisodeTracker,
    cfg,
    algo_cfg,
    algo_name: str,
    ckpt_dir: str,
    training_state,
    norm_state,
    obs_dim: int,
    action_dim: int,
    metrics_log: list[dict],
    last_eval_eps: int,
    key: jax.Array,
    resume: str | None,
    obs_normalize_fn=None,
    q_fn=None,
    ckpt_mgr: CheckpointManager | None = None,
) -> tuple[int, jax.Array]:
    """Run eval + save checkpoint if enough episodes completed since last eval.

    Returns:
        (updated_last_eval_eps, updated_key)
    """
    n_eps = tracker.n_episodes
    if n_eps < last_eval_eps + cfg.eval_every_n_episodes:
        return last_eval_eps, key

    key, eval_key = jax.random.split(key)
    eval_metrics = evaluate(
        select_action_fn, actor_params,
        eval_env, num_episodes=cfg.num_eval_episodes,
        episode_length=cfg.episode_length, key=eval_key,
        num_envs=cfg.num_envs,
        obs_normalize_fn=obs_normalize_fn,
        q_fn=q_fn,
        gamma=cfg.gamma,
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

    if ckpt_mgr is not None:
        is_best = ckpt_mgr.save(
            training_state, norm_state, cfg, algo_cfg,
            algo_name, obs_dim, action_dim, metrics_log, resume,
            eval_mean=eval_metrics['eval_mean'],
        )
        if is_best:
            print(f"  New best! eval={ckpt_mgr.best_eval:.1f}")
        else:
            print(f"  Checkpoint saved to {ckpt_dir}")
    else:
        save_checkpoint(ckpt_dir, training_state, norm_state, cfg, algo_cfg,
                        algo_name, obs_dim, action_dim, metrics_log, resume)
        print(f"  Checkpoint saved to {ckpt_dir}")

    return n_eps, key


def final_eval_and_checkpoint(
    select_action_fn,
    actor_params,
    eval_env,
    tracker: EpisodeTracker,
    cfg,
    algo_cfg,
    algo_name: str,
    ckpt_dir: str,
    training_state,
    norm_state,
    obs_dim: int,
    action_dim: int,
    metrics_log: list[dict],
    key: jax.Array,
    resume: str | None,
    total_gradient_steps: int,
    obs_normalize_fn=None,
    q_fn=None,
    ckpt_mgr: CheckpointManager | None = None,
) -> dict:
    """Run final eval + save checkpoint after training completes. Returns eval_metrics."""
    key, eval_key = jax.random.split(key)
    eval_metrics = evaluate(
        select_action_fn, actor_params,
        eval_env, num_episodes=cfg.num_eval_episodes,
        episode_length=cfg.episode_length, key=eval_key,
        num_envs=cfg.num_envs,
        obs_normalize_fn=obs_normalize_fn,
        q_fn=q_fn,
        gamma=cfg.gamma,
    )

    if ckpt_mgr is not None:
        ckpt_mgr.save(training_state, norm_state, cfg, algo_cfg,
                       algo_name, obs_dim, action_dim, metrics_log, resume,
                       eval_mean=eval_metrics['eval_mean'])
    else:
        save_checkpoint(ckpt_dir, training_state, norm_state, cfg, algo_cfg,
                        algo_name, obs_dim, action_dim, metrics_log, resume)

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
    print(f"  Final checkpoint: {ckpt_dir}")

    return eval_metrics
