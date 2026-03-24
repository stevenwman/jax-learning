"""Training metrics logging — replaces the stdout print + CSV row blocks across all train scripts."""

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
        f"Return {stats['avg']:7.1f} [{stats['min']:4.0f},{stats['max']:4.0f}]",
        f"Q1 {float(last_metrics.get('q1_mean', 0)):.3e}",
        f"ActLoss {float(last_metrics.get('actor_loss', 0)):.3e}",
    ]

    if extra_fields:
        for label, key, fmt in extra_fields:
            val = float(last_metrics.get(key, 0))
            parts.append(f"{label} {val:.3e}")

    parts.append(f"{sps:>6,} sps")
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
