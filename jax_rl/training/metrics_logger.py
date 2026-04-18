"""Training metrics logging — stdout + CSV + optional W&B.

W&B integration is opt-in via --wandb flag. When enabled, all metrics
logged to CSV are also sent to wandb.ai for real-time experiment tracking.
W&B is never imported unless the flag is set.
"""

from jax_rl.training.episode_tracker import EpisodeTracker

# Flat metric key -> W&B prefixed key for dashboard sections.
# Keys not in this mapping pass through unchanged.
_WANDB_PREFIX = {
    # perf — the "how's it doing" section
    "eval_mean": "perf/eval_mean",
    "eval_std": "perf/eval_std",
    "eval_min": "perf/eval_min",
    "eval_max": "perf/eval_max",
    "avg_return": "perf/avg_return",
    "min_return": "perf/min_return",
    "max_return": "perf/max_return",
    # critic — value estimation health
    "q1_mean": "critic/q1_mean",
    "q2_mean": "critic/q2_mean",
    "q1_loss": "critic/q1_loss",
    "q2_loss": "critic/q2_loss",
    "q_bias": "eval/q_bias",
    "q_rmse": "eval/q_rmse",
    "q_corr": "eval/q_corr",
    "q_mean": "eval/q_mean",
    "mc_mean": "eval/mc_mean",
    # actor — policy optimization (superset of PPO + off-policy)
    "actor_loss": "actor/actor_loss",
    "policy_loss": "actor/policy_loss",
    "value_loss": "actor/value_loss",
    "entropy": "actor/entropy",
    "alpha": "actor/alpha",
    "alpha_loss": "actor/alpha_loss",
    "approx_kl": "actor/approx_kl",
    "clip_fraction": "actor/clip_fraction",
    "log_std_mean": "actor/log_std_mean",
    "log_std_min": "actor/log_std_min",
    "log_std_max": "actor/log_std_max",
    # infra — throughput & progress
    "sps": "infra/sps",
    "elapsed": "infra/elapsed",
    "grad_steps": "infra/grad_steps",
    "episodes": "infra/episodes",
    "iteration": "infra/iteration",
    "iter_time": "infra/iter_time",
}


def log_training_step(
    total_steps: int,
    tracker: EpisodeTracker,
    last_metrics: dict,
    sps: int,
    is_training: bool = True,
    buffer_size: int | None = None,
    min_buffer: int | None = None,
    extra_fields: list[tuple[str, str, str]] | None = None,
    elapsed: float | None = None,
) -> None:
    """Print training step to stdout.

    Args:
        total_steps: total environment steps so far
        tracker: episode tracker for return stats
        last_metrics: dict from algo.update() (q1_mean, actor_loss, etc.)
        sps: steps per second
        is_training: True if past buffer warmup
        buffer_size: current buffer size (for warmup display)
        min_buffer: minimum buffer size before training starts
        extra_fields: list of (label, metric_key, format_str) for algo-specific metrics
                      e.g. [("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")]
    """
    stats = tracker.recent_stats()

    if not is_training:
        print(
            f"Step {total_steps:>9,} | "
            f"Buffer {buffer_size:>7,}/{min_buffer:,} | "
            f"Warming up... | "
            f"{sps:>6,} sps"
        )
        return

    parts = [
        f"Step {total_steps:>9,}",
        f"Eps {stats['n_eps']:>5}",
        f"Return {stats['avg']:9.3g} [{stats['min']:7.3g},{stats['max']:7.3g}]",
        f"Q1 {float(last_metrics.get('q1_mean', 0)):.3e}",
        f"ActLoss {float(last_metrics.get('actor_loss', 0)):.3e}",
    ]

    if extra_fields:
        for label, key, fmt in extra_fields:
            val = float(last_metrics.get(key, 0))
            parts.append(f"{label} {val:{fmt}}")

    parts.append(f"{sps:>6,} sps")
    if elapsed is not None:
        parts.append(f"{elapsed:.0f}s")
    print(" | ".join(parts))


def make_metrics_row(
    total_steps: int,
    tracker: EpisodeTracker,
    last_metrics: dict,
    grad_steps: int,
    sps: int,
    elapsed: float,
    extra_keys: list[str] | None = None,
) -> dict:
    """Build dict for CSV logging.

    Args:
        extra_keys: algo-specific metric keys to include from last_metrics
                    e.g. ["entropy", "alpha", "alpha_loss"] for SAC
    """
    stats = tracker.recent_stats()
    row = {
        "total_steps": total_steps,
        "episodes": stats["n_eps"],
        "avg_return": stats["avg"],
        "min_return": stats["min"],
        "max_return": stats["max"],
        "q1_mean": float(last_metrics.get("q1_mean", float("nan"))),
        "q2_mean": float(last_metrics.get("q2_mean", float("nan"))),
        "q1_loss": float(last_metrics.get("q1_loss", float("nan"))),
        "q2_loss": float(last_metrics.get("q2_loss", float("nan"))),
        "actor_loss": float(last_metrics.get("actor_loss", float("nan"))),
        "grad_steps": grad_steps,
        "sps": sps,
        "elapsed": elapsed,
    }
    if extra_keys:
        for key in extra_keys:
            row[key] = float(last_metrics.get(key, float("nan")))
    return row


def wandb_init(project: str, name: str, config: dict) -> bool:
    """Initialize W&B run. Returns True if successful, False if wandb not installed.

    Args:
        project: W&B project name (e.g., "jax-rl")
        name: Run name (e.g., "sac_CheetahRun_seed0")
        config: Dict of hyperparameters to log (typically the meta dict from checkpointing)
    """
    try:
        import wandb
        wandb.init(project=project, name=name, config=config)
        return True
    except ImportError:
        print("WARNING: wandb not installed. Install with: uv add wandb")
        return False


def wandb_setup_metrics() -> None:
    """Define W&B metric summary behavior. Called once after wandb_init().

    Sets summary types so the W&B runs table shows useful values
    (e.g., best eval return, final sps). Guarded for version compat.
    """
    try:
        import wandb
        if wandb.run is None:
            return
        summaries = {
            "perf/eval_mean": "max",
            "perf/avg_return": "max",
            "infra/sps": "last",
            "infra/episodes": "max",
        }
        for metric, summary in summaries.items():
            try:
                wandb.define_metric(metric, summary=summary)
            except TypeError:
                break  # older wandb without summary param — skip all
    except ImportError:
        pass


def wandb_log(metrics: dict, step: int) -> None:
    """Log metrics to W&B if a run is active. No-op if wandb not initialized.

    Remaps flat metric keys to prefixed keys (e.g., q1_mean -> critic/q1_mean)
    for dashboard section grouping. Keys not in _WANDB_PREFIX pass through.
    """
    try:
        import wandb
        if wandb.run is not None:
            remapped = {_WANDB_PREFIX.get(k, k): v for k, v in metrics.items()}
            wandb.log(remapped, step=step)
    except ImportError:
        pass


def wandb_finish() -> None:
    """Finish the W&B run. No-op if not initialized."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


TERRAIN_TYPE_NAMES = ["rough", "pyramid_up", "pyramid_down", "tilted"]


def log_terrain_metrics(info: dict, terrain_type_names: list[str] = None,
                        num_levels: int = 10) -> dict[str, float]:
    """Extract per-terrain-type metrics from state.info at a snapshot.

    Snapshot approach: each env's values reflect its last completed episode.
    Aggregates over ~num_envs/num_types envs per type (noisy per-sample, smooth
    over training time).

    Returns empty dict if terrain_level not in info (non-curriculum envs).

    Metrics per terrain type:
      - mean_level, max_level, std_level, num_envs
      - reach_rate, fall_rate, promote_rate, demote_rate
      - level_hist_L{0..num_levels-1} — env count at each curriculum level (bug-hunt: clumps / bimodal)

    Global: mean_level, reach_rate, fall_rate, std_level.

    Goal/yaw diagnostics (if present in info):
      - goal_dist_mean — current ||robot_xy - goal_xy||, per-type and global
      - min_distance_mean — how close each env got this episode (before reset), per-type
    """
    import numpy as np
    if "terrain_level" not in info or "terrain_type" not in info:
        return {}
    if terrain_type_names is None:
        terrain_type_names = TERRAIN_TYPE_NAMES

    levels = np.asarray(info["terrain_level"])
    types = np.asarray(info["terrain_type"])
    reached = np.asarray(info.get("episode_reached_goal", np.zeros_like(levels, dtype=bool)))
    fallen = np.asarray(info.get("episode_fallen", np.zeros_like(levels, dtype=bool)))
    promoted = np.asarray(info.get("episode_promoted", np.zeros_like(levels, dtype=bool)))
    demoted = np.asarray(info.get("episode_demoted", np.zeros_like(levels, dtype=bool)))
    min_dist = info.get("episode_min_distance", None)
    init_dist = info.get("initial_distance", None)

    result = {}
    for type_idx, name in enumerate(terrain_type_names):
        mask = types == type_idx
        if mask.any():
            sub_levels = levels[mask]
            result[f"terrain/{name}/mean_level"]   = float(sub_levels.mean())
            result[f"terrain/{name}/max_level"]    = int(sub_levels.max())
            result[f"terrain/{name}/std_level"]    = float(sub_levels.std())
            result[f"terrain/{name}/num_envs"]     = int(mask.sum())
            result[f"terrain/{name}/reach_rate"]   = float(reached[mask].mean())
            result[f"terrain/{name}/fall_rate"]    = float(fallen[mask].mean())
            result[f"terrain/{name}/promote_rate"] = float(promoted[mask].mean())
            result[f"terrain/{name}/demote_rate"]  = float(demoted[mask].mean())
            # Level histogram — counts of envs at each level.
            # Catches clumping (all at one level) and bimodal distributions.
            hist, _ = np.histogram(sub_levels, bins=np.arange(num_levels + 1))
            for lvl in range(num_levels):
                result[f"terrain/{name}/level_hist_L{lvl}"] = int(hist[lvl])
            # Progress fraction: how close did robot get to goal this episode.
            if min_dist is not None and init_dist is not None:
                md = np.asarray(min_dist)[mask]
                ind = np.asarray(init_dist)[mask]
                ind_safe = np.where(ind > 1e-6, ind, 1.0)
                progress = 1.0 - md / ind_safe  # 1.0 = reached, 0.0 = didn't move
                result[f"terrain/{name}/progress_frac"] = float(progress.mean())

    result["terrain/global/mean_level"] = float(levels.mean())
    result["terrain/global/std_level"]  = float(levels.std())
    result["terrain/global/reach_rate"] = float(reached.mean())
    result["terrain/global/fall_rate"]  = float(fallen.mean())
    return result


def print_curriculum_dump(info: dict, step: int, terrain_type_names: list[str] = None) -> None:
    """Console dump — curriculum state snapshot. Call every ~10k steps during training.

    Catches hidden bugs: stuck levels, type imbalance, silent failures. Zero-op if
    no terrain keys in info.
    """
    import numpy as np
    if "terrain_level" not in info or "terrain_type" not in info:
        return
    if terrain_type_names is None:
        terrain_type_names = TERRAIN_TYPE_NAMES

    levels = np.asarray(info["terrain_level"])
    types = np.asarray(info["terrain_type"])
    reached = np.asarray(info.get("episode_reached_goal", np.zeros_like(levels, dtype=bool)))
    fallen = np.asarray(info.get("episode_fallen", np.zeros_like(levels, dtype=bool)))

    print(f"[curriculum @ {step:,} steps]  global mean_level={float(levels.mean()):.2f}  "
          f"reach={float(reached.mean()):.2f}  fall={float(fallen.mean()):.2f}")
    for type_idx, name in enumerate(terrain_type_names):
        mask = types == type_idx
        n = int(mask.sum())
        if n == 0:
            print(f"    {name:<14} (no envs)")
            continue
        sub_levels = levels[mask]
        r = float(reached[mask].mean())
        f = float(fallen[mask].mean())
        # Level distribution as inline compact histogram
        num_levels = 10
        hist, _ = np.histogram(sub_levels, bins=np.arange(num_levels + 1))
        hist_str = " ".join(str(int(h)) for h in hist)
        print(f"    {name:<14} n={n:<3}  mean={float(sub_levels.mean()):.2f} "
              f"(±{float(sub_levels.std()):.2f})  "
              f"reach={r:.2f}  fall={f:.2f}  hist=[{hist_str}]")
