"""Checkpoint save/load — replaces the 46-line _save_checkpoint duplicated across all train scripts."""

import csv
import dataclasses
import json
import os
import shutil

import jax
import numpy as np
import orbax.checkpoint as ocp

from jax_rl.utils.normalization import NormalizationState


class CheckpointManager:
    """Wraps save_checkpoint with best-policy tracking.

    Usage:
        mgr = CheckpointManager(ckpt_dir)
        mgr.save(training_state, norm_state, ..., eval_mean=15.2)
        # Automatically saves to ckpt_dir/ (latest) and ckpt_dir/best/ (if new high)
    """

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self.best_eval = -float('inf')
        self.best_dir = os.path.join(ckpt_dir, "best")

    def save(self, training_state, norm_state, cfg, algo_cfg,
             algo_name, obs_dim, action_dim, metrics_log, resume=None,
             eval_mean: float | None = None,
             critic_norm_state: NormalizationState | None = None) -> bool:
        """Save latest checkpoint. Returns True if new best."""
        save_checkpoint(self.ckpt_dir, training_state, norm_state, cfg, algo_cfg,
                        algo_name, obs_dim, action_dim, metrics_log, resume,
                        critic_norm_state=critic_norm_state)
        is_best = False
        if eval_mean is not None and eval_mean > self.best_eval:
            self.best_eval = eval_mean
            save_checkpoint(self.best_dir, training_state, norm_state, cfg, algo_cfg,
                            algo_name, obs_dim, action_dim, metrics_log, resume,
                            critic_norm_state=critic_norm_state)
            is_best = True
        return is_best


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
    critic_norm_state: NormalizationState | None = None,
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

    # Artifact contract (Phase A): declare what kind of checkpoint this is.
    # Consumers (record_video, deploy/PolicyRunner, ONNX export) check this
    # before loading to fail loudly on shape mismatches.
    from jax_rl.training.artifact_contract import stamp_meta, KIND_SHARED_ACTOR
    stamp_meta(meta, KIND_SHARED_ACTOR)

    # Reproducibility: seed, git hash, DR specs
    # Seed: extract from ckpt_dir name (format: timestamp_algo_env_seedN)
    import re
    seed_match = re.search(r"seed(\d+)", ckpt_dir)
    if seed_match:
        meta["seed"] = int(seed_match.group(1))

    # Git hash
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_hash"] = git_hash
    except Exception:
        pass

    # DR specs + obs schema + deploy control metadata (if env declares them).
    # Single env load for all three. For deployable Go2 envs, missing
    # obs_schema or control metadata is a hard failure — silent loss is the
    # exact bug that produced the 2026-04-10 → 2026-04-24 deploy drift.
    is_go2 = "go2" in cfg.env_name.lower()
    try:
        from mujoco_playground import registry as pg_registry
        env = pg_registry.load(cfg.env_name)
        if hasattr(env, 'get_domain_randomization_spec'):
            specs = env.get_domain_randomization_spec()
            meta["dr_specs"] = [
                {k: v for k, v in dataclasses.asdict(s).items() if v is not None}
                for s in specs
            ]
        if hasattr(env, '_obs_groups'):
            from jax_rl.envs.obs_spec import schema_from_obs_groups
            meta["obs_schema"] = schema_from_obs_groups(env._obs_groups)
        elif is_go2:
            raise RuntimeError(
                f"Go2 env '{cfg.env_name}' has no _obs_groups — cannot stamp "
                f"obs_schema. Deploy will silently drift; refusing to save."
            )
        # Phase D: deploy-critical control metadata (Kp/Kd/action_scale/dts/...).
        # Go2 Warp envs implement get_control_metadata; other envs don't yet.
        # deploy/sim2sim_direct.py + deploy/robot_interface.py read this block
        # before falling back to constants in deploy/go2_constants.py.
        if hasattr(env, 'get_control_metadata'):
            meta["control"] = env.get_control_metadata()
        elif is_go2:
            raise RuntimeError(
                f"Go2 env '{cfg.env_name}' has no get_control_metadata — "
                f"cannot stamp meta['control']. Refusing to save."
            )
    except Exception as e:
        if is_go2:
            raise RuntimeError(
                f"save_checkpoint: Go2 env metadata stamp failed for "
                f"'{cfg.env_name}': {type(e).__name__}: {e}"
            ) from e
        # Non-Go2 env: keep silent fallback (some envs lack pg_registry).

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

    # Inference artifact: actor_params + norm stats (+ batch_stats for FlashSAC)
    inference_dict = {
        "actor_params": jax.device_get(training_state.actor_params),
        "norm_mean": jax.device_get(norm_state.mean),
        "norm_mean_of_squares": jax.device_get(norm_state.mean_of_squares),
        "norm_count": int(norm_state.count),
    }
    if hasattr(training_state, "actor_batch_stats") and training_state.actor_batch_stats:
        inference_dict["actor_batch_stats"] = jax.device_get(training_state.actor_batch_stats)
    np.save(
        os.path.join(ckpt_dir, "actor_params.npy"),
        inference_dict,
        allow_pickle=True,
    )

    # Full checkpoint for resume
    orbax_dir = os.path.join(ckpt_dir, "orbax")
    ckpt = {"training_state": training_state, "norm_state": norm_state}
    if critic_norm_state is not None:
        ckpt["critic_norm_state"] = critic_norm_state
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.abspath(orbax_dir), ckpt, force=True)
    checkpointer.wait_until_finished()


def load_checkpoint(
    ckpt_dir: str,
    training_state,
    norm_state: NormalizationState,
    critic_norm_state: NormalizationState | None = None,
) -> tuple:
    """Load orbax checkpoint. Returns (training_state, norm_state, start_step[, critic_norm_state]).

    When critic_norm_state is provided but the saved checkpoint predates the
    critic-norm feature, falls back to a 3-tuple return and the caller's
    critic_norm_state is not updated (fresh stats). This preserves backwards
    compat with pre-2026-04-24 checkpoints.
    """
    orbax_dir = os.path.join(ckpt_dir, "orbax")
    target = {"training_state": training_state, "norm_state": norm_state}
    if critic_norm_state is not None:
        target["critic_norm_state"] = critic_norm_state
    try:
        ckpt = ocp.StandardCheckpointer().restore(os.path.abspath(orbax_dir), target=target)
    except Exception as e:
        # Retry without critic_norm_state — old checkpoint predating the key.
        if critic_norm_state is None:
            raise
        target_compat = {"training_state": training_state, "norm_state": norm_state}
        ckpt = ocp.StandardCheckpointer().restore(
            os.path.abspath(orbax_dir), target=target_compat
        )
        print(f"  (load_checkpoint: old format, no critic_norm_state in ckpt — {type(e).__name__})")

    start_step = 0
    metrics_csv = os.path.join(ckpt_dir, "metrics.csv")
    if os.path.exists(metrics_csv):
        with open(metrics_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            start_step = int(rows[-1]["total_steps"])

    if critic_norm_state is not None and "critic_norm_state" in ckpt:
        return ckpt["training_state"], ckpt["norm_state"], start_step, ckpt["critic_norm_state"]
    return ckpt["training_state"], ckpt["norm_state"], start_step


def load_actor_for_inference(
    ckpt_dir: str,
) -> tuple[dict, dict, NormalizationState, dict | None]:
    """Load meta.json + actor_params.npy for recording/eval.

    Validates `meta["artifact_kind"]` against the shared_actor allowlist
    (Phase B of the artifact contract); fails loudly with a redirect
    pointing at TDMPC2-specific tooling for `tdmpc2_v1` ckpts. Pre-Phase-A
    ckpts (no field) are treated as legacy shared-actor with a warning.

    Returns:
        meta: dict from meta.json
        actor_params: dict of actor parameters
        norm_state: NormalizationState for obs normalization
        actor_batch_stats: optional BN running stats (FlashSAC only); None otherwise
    """
    import jax.numpy as jnp
    from jax_rl.training.artifact_contract import (
        assert_artifact_kind, validate_shared_actor_files,
        KIND_SHARED_ACTOR, KIND_LEGACY_SHARED_ACTOR,
    )

    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)

    assert_artifact_kind(
        meta,
        allowed=[KIND_SHARED_ACTOR, KIND_LEGACY_SHARED_ACTOR],
        tool_name="load_actor_for_inference",
        ckpt_path=ckpt_dir,
        redirect=(
            "For TD-MPC2 checkpoints (artifact_kind='tdmpc2_v1'), use "
            "scripts/record_video_tdmpc2.py / scripts/eval_tdmpc2.py — "
            "they handle the actor_params.npz + world_model_params.npz "
            "shape directly."
        ),
    )
    validate_shared_actor_files(ckpt_dir)

    saved = np.load(os.path.join(ckpt_dir, "actor_params.npy"), allow_pickle=True).item()

    norm_state = NormalizationState(
        mean=jnp.array(saved["norm_mean"]),
        mean_of_squares=jnp.array(saved["norm_mean_of_squares"]),
        count=int(saved["norm_count"]),
    )

    actor_batch_stats = saved.get("actor_batch_stats", None)
    return meta, saved["actor_params"], norm_state, actor_batch_stats
