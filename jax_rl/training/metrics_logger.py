"""Training metrics logging — stdout + CSV + optional W&B.

W&B integration is opt-in via --wandb flag. When enabled, all metrics
logged to CSV are also sent to wandb.ai for real-time experiment tracking.
W&B is never imported unless the flag is set.
"""

from jax_rl.training.episode_tracker import EpisodeTracker


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
            parts.append(f"{label} {val:.3e}")

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


def wandb_log(metrics: dict, step: int) -> None:
    """Log metrics to W&B if a run is active. No-op if wandb not initialized.

    Safe to call even if wandb is not installed or not initialized —
    silently does nothing.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
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
