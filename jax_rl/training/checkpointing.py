"""Checkpoint save/load — replaces the 46-line _save_checkpoint duplicated across all train scripts."""

import csv
import dataclasses
import json
import os

import jax
import numpy as np
import orbax.checkpoint as ocp

from jax_rl.utils.normalization import NormalizationState


def save_checkpoint(
    ckpt_dir: str,
    training_state,
    norm_state: NormalizationState,
    cfg,
    algo_cfg,
    algo_name: str,
    obs_dim: int,
    action_dim: int,
    metrics_log: list[dict],
    resume: str | None,
) -> None:
    """Save meta.json + metrics.csv + actor_params.npy + orbax checkpoint.

    Args:
        ckpt_dir: directory to save into
        training_state: algorithm TrainingState (flax struct)
        norm_state: observation normalization state
        cfg: TrainConfig
        algo_cfg: algorithm-specific config (SACConfig, TD3Config, etc.)
        algo_name: string identifier ("sac", "td3", "fast_td3", "fast_sac", "fast_dsac", "ppo")
        obs_dim: observation dimensionality
        action_dim: action dimensionality
        metrics_log: list of metric dicts for CSV
        resume: previous checkpoint dir (for appending metrics CSV), or None
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # Meta JSON
    algo_cfg_key = f"{algo_name}_config"
    meta = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "train_config": dataclasses.asdict(cfg),
        algo_cfg_key: dataclasses.asdict(algo_cfg),
        "algo": algo_name,
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Metrics CSV
    if metrics_log:
        csv_path = os.path.join(ckpt_dir, "metrics.csv")
        prior_rows = []
        if resume is not None:
            prev_csv = os.path.join(resume, "metrics.csv")
            if os.path.exists(prev_csv):
                with open(prev_csv) as f:
                    prior_rows = list(csv.DictReader(f))
        all_keys = dict.fromkeys(k for row in metrics_log for k in row)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in prior_rows:
                writer.writerow(row)
            writer.writerows(metrics_log)

    # Inference artifact: actor_params + norm stats
    np.save(
        os.path.join(ckpt_dir, "actor_params.npy"),
        {
            "actor_params": jax.device_get(training_state.actor_params),
            "norm_mean": jax.device_get(norm_state.mean),
            "norm_mean_of_squares": jax.device_get(norm_state.mean_of_squares),
            "norm_count": int(norm_state.count),
        },
        allow_pickle=True,
    )

    # Full checkpoint for resume
    orbax_dir = os.path.join(ckpt_dir, "orbax")
    ckpt = {"training_state": training_state, "norm_state": norm_state}
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.abspath(orbax_dir), ckpt, force=True)
    checkpointer.wait_until_finished()


def load_checkpoint(
    ckpt_dir: str,
    training_state,
    norm_state: NormalizationState,
) -> tuple:
    """Load orbax checkpoint. Returns (training_state, norm_state, start_step)."""
    orbax_dir = os.path.join(ckpt_dir, "orbax")
    target = {"training_state": training_state, "norm_state": norm_state}
    ckpt = ocp.StandardCheckpointer().restore(os.path.abspath(orbax_dir), target=target)

    start_step = 0
    metrics_csv = os.path.join(ckpt_dir, "metrics.csv")
    if os.path.exists(metrics_csv):
        with open(metrics_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            start_step = int(rows[-1]["total_steps"])

    return ckpt["training_state"], ckpt["norm_state"], start_step


def load_actor_for_inference(ckpt_dir: str) -> tuple[dict, dict, NormalizationState]:
    """Load meta.json + actor_params.npy for recording/eval. Algo-agnostic.

    Returns:
        meta: dict from meta.json (includes algo name, configs, dims)
        actor_params: dict of actor parameters
        norm_state: NormalizationState for obs normalization
    """
    import jax.numpy as jnp

    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)

    saved = np.load(os.path.join(ckpt_dir, "actor_params.npy"), allow_pickle=True).item()

    norm_state = NormalizationState(
        mean=jnp.array(saved["norm_mean"]),
        mean_of_squares=jnp.array(saved["norm_mean_of_squares"]),
        count=int(saved["norm_count"]),
    )

    return meta, saved["actor_params"], norm_state
