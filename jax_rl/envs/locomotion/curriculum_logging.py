"""Curriculum-specific logging helpers for Go2 terrain envs.

Extracted from `jax_rl/training/metrics_logger.py` (B5.6, 2026-04-25):
the generic logger should not know about Go2 terrain types. These helpers
read fields populated by `WarpJoystickCurriculum` + `TerrainCurriculumDRWrapper`
into `state.info`. Non-curriculum envs return empty dicts (no-op).
"""

TERRAIN_TYPE_NAMES = ["rough", "pyramid_up", "pyramid_down", "tilted", "flat"]


def log_terrain_metrics(info: dict, terrain_type_names: list[str] = None,
                        num_levels: int = 10) -> dict[str, float]:
    """Extract per-terrain-type scalar metrics from state.info at a snapshot.

    Snapshot approach: each env's values reflect its last completed episode.
    Aggregates over ~num_envs/num_types envs per type (noisy per-sample, smooth
    over training time).

    Returns empty dict if terrain_level not in info (non-curriculum envs).

    Per-type scalars (4 envs × 6 stats = 24):
      - mean_level, num_envs, reach_rate, fall_rate, promote_rate, demote_rate
    Global scalars (3): mean_level, reach_rate, fall_rate.
    Total: 27 scalars (down from 76). Drop std/max/level_hist/progress —
    encoded in image panel from `log_terrain_image()`.
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

    result = {}
    for type_idx, name in enumerate(terrain_type_names):
        mask = types == type_idx
        if mask.any():
            sub_levels = levels[mask]
            result[f"terrain/{name}/mean_level"]   = float(sub_levels.mean())
            result[f"terrain/{name}/num_envs"]     = int(mask.sum())
            result[f"terrain/{name}/reach_rate"]   = float(reached[mask].mean())
            result[f"terrain/{name}/fall_rate"]    = float(fallen[mask].mean())
            result[f"terrain/{name}/promote_rate"] = float(promoted[mask].mean())
            result[f"terrain/{name}/demote_rate"]  = float(demoted[mask].mean())

    result["terrain/global/mean_level"] = float(levels.mean())
    result["terrain/global/reach_rate"] = float(reached.mean())
    result["terrain/global/fall_rate"]  = float(fallen.mean())
    return result


def log_terrain_image(info: dict, terrain_type_names: list[str] = None,
                      num_levels: int = 10):
    """Composite image: per-type level distribution + reach/fall summary.

    Returns dict with single key 'curriculum/snapshot' → wandb.Image (or empty
    dict if not curriculum env / wandb not installed). Replaces the 40
    level_hist scalar lines with one info-dense panel.
    """
    if "terrain_level" not in info or "terrain_type" not in info:
        return {}
    try:
        import wandb
    except ImportError:
        return {}
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if terrain_type_names is None:
        terrain_type_names = TERRAIN_TYPE_NAMES

    levels = np.asarray(info["terrain_level"])
    types = np.asarray(info["terrain_type"])
    reached = np.asarray(info.get("episode_reached_goal", np.zeros_like(levels, dtype=bool)))
    fallen = np.asarray(info.get("episode_fallen", np.zeros_like(levels, dtype=bool)))

    n_types = len(terrain_type_names)
    fig, axes = plt.subplots(1, n_types, figsize=(3.0 * n_types, 3.0), sharey=True)
    for i, name in enumerate(terrain_type_names):
        mask = types == i
        if not mask.any():
            axes[i].set_title(f"{name}\n(no envs)")
            continue
        sub_levels = levels[mask]
        hist, _ = np.histogram(sub_levels, bins=np.arange(num_levels + 1))
        bars = axes[i].bar(np.arange(num_levels), hist, color="steelblue")
        axes[i].set_xticks(np.arange(num_levels))
        axes[i].set_xlabel("level")
        if i == 0:
            axes[i].set_ylabel("env count")
        r = float(reached[mask].mean())
        f = float(fallen[mask].mean())
        m = float(sub_levels.mean())
        axes[i].set_title(f"{name}\nmean={m:.2f}  reach={r:.2f}  fall={f:.2f}", fontsize=10)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return {"curriculum/snapshot": img}


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
