"""Skill-discovery off-policy training loop.

Mirrors the structure of `run_offpolicy_loop` (SAC/TD3/FastSAC/FastTD3) with
six convention overrides documented in the SD-B plan
(`.superpowers/plans/2026-05-02-skill-discovery-sd-b.md`, Task 3):

  1. Buffer-add stores the *unscaled* env reward; `cfg.reward_scaling` is
     applied **once** at sample time on the composed reward (intrinsic + task).
  2. Eval bypasses `maybe_eval_and_checkpoint` — this loop owns its own
     `eval_skills(...)` per-z fixed-skill evaluation.
  3. Buffer extras only declare `skill_z` — the buffer auto-allocates a
     companion `next_skill_z` slot per `JaxReplayBuffer:77-82` convention.
  4. After `algo.update(...)`, run `mgr.update(aux_state, raw_batch)` on the
     same sampled batch. Actor/critic gradients can't flow through aux
     networks because aux_state was fixed during the algo update.
  5. Checkpoints write the base state via `CheckpointManager.save(...)`, then
     dump aux state and JSON-patch `meta.json` to embed a `skill_discovery`
     block.
  6. Resume restores aux state from the checkpoint's `skill_aux/` directory
     and re-samples per-env `current_z` fresh (loop-state, not aux-state).

Spec gradient-step order:
  1. Sample replay batch.
  2. Normalize raw obs through ObsPipeline.
  3. Append `skill_z` to actor / critic obs.
  4. Compute intrinsic reward using current aux state.
  5. Compose final batch reward (sample-time replacement).
  6. Update actor / critic with the composed reward.
  7. Update aux params on the same batch.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from datetime import datetime
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from jax_rl.configs.train_config import TrainConfig
from jax_rl.skill_discovery.checkpointing import (
    save_skill_aux_state, load_skill_aux_state, write_skill_meta_block,
)
from jax_rl.skill_discovery.config import SkillDiscoveryConfig
from jax_rl.skill_discovery.manager import SkillManager
from jax_rl.training.checkpointing import CheckpointManager, load_checkpoint
from jax_rl.training.env_bundle import EnvBundle
from jax_rl.training.episode_tracker import EpisodeTracker
from jax_rl.training.metrics_logger import (
    log_training_step, make_metrics_row,
    wandb_init, wandb_setup_metrics, wandb_log, wandb_finish,
)
from jax_rl.training.obs_pipeline import ObsPipeline
from jax_rl.training.train_context import TrainContext


# ── Helpers (importable, unit-tested) ─────────────────────────────────────


def compose_skill_batch(raw_batch: dict, raw_obs_dim: int, skill_dim: int) -> dict:
    """Append `skill_z` to actor obs / next_obs (and critic obs if asymmetric).

    Returns a *new* dict referencing the same underlying arrays for unchanged
    fields (action, reward, done, etc.). The skill_dim arg is informational —
    actual concat width comes from `raw_batch["skill_z"].shape[-1]`.

    Asymmetric critic handling: if `raw_batch` carries `critic_obs` /
    `critic_next_obs` (non-aliased keys produced by ObsPipeline.normalize_batch
    when `has_privileged=True`), we concat skill_z onto them too. SD-B
    CheetahRun is symmetric so the asymmetric branch lands here for SD-C Go2
    use; symmetric envs see `critic_obs is obs` after `normalize_batch`, in
    which case the augmented obs is reused (no re-concat).
    """
    z = raw_batch["skill_z"]
    next_z = raw_batch["next_skill_z"]

    # Argument shape sanity — informative failure if caller passes a mismatched
    # raw_obs_dim. Cheap, runs once per sample.
    assert raw_batch["obs"].shape[-1] == raw_obs_dim, (
        f"compose_skill_batch: obs last dim {raw_batch['obs'].shape[-1]} "
        f"!= raw_obs_dim {raw_obs_dim}"
    )
    assert z.shape[-1] == skill_dim, (
        f"compose_skill_batch: skill_z last dim {z.shape[-1]} != skill_dim {skill_dim}"
    )

    composed = dict(raw_batch)  # shallow copy — keeps action/reward/done refs
    composed["obs"] = jnp.concatenate([raw_batch["obs"], z], axis=-1)
    composed["next_obs"] = jnp.concatenate([raw_batch["next_obs"], next_z], axis=-1)

    # Asymmetric branch: critic obs is a distinct array (privileged_state).
    # When symmetric, ObsPipeline.normalize_batch aliases critic_obs to obs;
    # detect via identity check rather than has_privileged so this helper
    # stays pipeline-agnostic and unit-testable.
    if "critic_obs" in raw_batch and raw_batch["critic_obs"] is not raw_batch["obs"]:
        composed["critic_obs"] = jnp.concatenate(
            [raw_batch["critic_obs"], z], axis=-1
        )
        composed["critic_next_obs"] = jnp.concatenate(
            [raw_batch["critic_next_obs"], next_z], axis=-1
        )
    elif "critic_obs" in raw_batch:
        # Symmetric: alias to the augmented obs so algos that read
        # batch["critic_obs"] see the skill-conditioned vector.
        composed["critic_obs"] = composed["obs"]
        composed["critic_next_obs"] = composed["next_obs"]

    return composed


def intrinsic_reward_then_update(
    composed_batch: dict,
    raw_batch: dict,
    mgr: SkillManager,
    aux_state: dict,
    skill_cfg: SkillDiscoveryConfig,
    reward_scaling: float,
) -> tuple[dict, jax.Array]:
    """Compute intrinsic reward and replace `composed_batch["reward"]`.

    final = reward_scaling * (intrinsic_weight * intrinsic
                              + task_reward_weight * env_reward)

    DIAYN reads current state s (`raw_batch["obs"]`), not s'. The discriminator
    sees the same (post-norm) obs the actor sees. Style/safety weights are
    zero in SD-B; SD-C onward.

    Returns:
        (composed_batch_with_overwritten_reward, intrinsic_reward) — the second
        return is for logging.
    """
    intrinsic = mgr.compute_intrinsic_reward(aux_state, raw_batch)  # (B,)
    intrinsic_col = intrinsic.reshape(-1, 1)
    env_reward = raw_batch["reward"].reshape(-1, 1)

    final = reward_scaling * (
        skill_cfg.intrinsic_weight * intrinsic_col
        + skill_cfg.task_reward_weight * env_reward
    )

    composed_batch = dict(composed_batch)  # don't mutate caller's dict
    composed_batch["reward"] = final
    return composed_batch, intrinsic


# ── Eval (skill-conditioned) ──────────────────────────────────────────────


def _eval_skills(
    select_action_fn,
    actor_params,
    eval_env,
    cfg: TrainConfig,
    skill_cfg: SkillDiscoveryConfig,
    obs_normalize_fn: Callable | None,
    key: jax.Array,
) -> tuple[dict, jax.Array]:
    """Run fixed-z evaluation for each skill in [0, total_skill_dim).

    For each skill index i, fixes `z = one_hot(i)` across all eval envs and
    runs `cfg.num_eval_episodes` rollouts with skill-conditioned obs. Returns
    a metrics dict with per-skill means + an overall mean for `is_best` /
    progress tracking.
    """
    num_envs = cfg.num_envs
    episode_length = cfg.episode_length
    num_episodes = cfg.num_eval_episodes
    total_skill_dim = skill_cfg.total_skill_dim

    per_skill_means: list[float] = []
    metrics: dict = {}

    for skill_idx in range(total_skill_dim):
        key, reset_key, scan_key = jax.random.split(key, 3)
        env_state = eval_env.reset(jax.random.split(reset_key, num_envs))

        # Fixed one-hot skill, broadcast over envs
        z_fixed = jax.nn.one_hot(
            jnp.full((num_envs,), skill_idx, dtype=jnp.int32),
            total_skill_dim,
        )

        @jax.jit
        def _scan_eval(env_state, scan_key, z_fixed):
            from jax_rl.training.env_setup import _make_nan_safe_step
            env_step = _make_nan_safe_step(eval_env.step)

            def scan_step(carry, step_key):
                env_state, episode_returns, episode_done = carry
                obs = env_state.obs
                if obs_normalize_fn is not None:
                    obs = obs_normalize_fn(obs)
                obs_aug = jnp.concatenate([obs, z_fixed], axis=-1)
                action = select_action_fn(
                    actor_params, obs_aug, step_key, deterministic=True,
                )
                env_state = env_step(env_state, action)
                reward = env_state.reward
                done = env_state.done.astype(jnp.bool_)
                episode_returns = (
                    episode_returns + reward * (~episode_done).astype(reward.dtype)
                )
                episode_done = episode_done | done
                carry = (env_state, episode_returns, episode_done)
                return carry, None

            init_returns = jnp.zeros(num_envs)
            init_done = jnp.zeros(num_envs, dtype=jnp.bool_)
            step_keys = jax.random.split(scan_key, episode_length)
            (env_state, episode_returns, _), _ = jax.lax.scan(
                scan_step,
                (env_state, init_returns, init_done),
                step_keys,
            )
            return episode_returns

        episode_returns = _scan_eval(env_state, scan_key, z_fixed)
        results = np.asarray(episode_returns)[:num_episodes]
        mean = float(np.mean(results))
        per_skill_means.append(mean)
        metrics[f"eval_skill{skill_idx}_mean"] = mean
        metrics[f"eval_skill{skill_idx}_std"] = float(np.std(results))

    metrics["eval_mean"] = float(np.mean(per_skill_means))
    metrics["eval_std"] = float(np.std(per_skill_means))
    metrics["eval_min"] = float(np.min(per_skill_means))
    metrics["eval_max"] = float(np.max(per_skill_means))
    return metrics, key


def _maybe_eval_and_checkpoint_skills(
    select_action_fn,
    actor_params,
    eval_env,
    tracker: EpisodeTracker,
    ctx: TrainContext,
    training_state,
    norm_state,
    aux_state: dict,
    skill_cfg: SkillDiscoveryConfig,
    last_eval_eps: int,
    key: jax.Array,
    obs_normalize_fn: Callable | None,
    critic_norm_state=None,
) -> tuple[int, jax.Array]:
    """Skill-aware eval+checkpoint trigger. Mirrors `maybe_eval_and_checkpoint`
    but uses `_eval_skills` and patches meta.json with the skill block.
    """
    cfg = ctx.cfg
    n_eps = tracker.n_episodes
    if n_eps < last_eval_eps + cfg.eval_every_n_episodes:
        return last_eval_eps, key

    metrics_log = ctx.metrics_log
    eval_metrics, key = _eval_skills(
        select_action_fn, actor_params, eval_env, cfg, skill_cfg,
        obs_normalize_fn, key,
    )

    # Compact per-skill print
    per_skill_str = " ".join(
        f"z{i}={eval_metrics[f'eval_skill{i}_mean']:.1f}"
        for i in range(skill_cfg.total_skill_dim)
    )
    print(
        f"  EVAL @ {n_eps} eps | mean={eval_metrics['eval_mean']:.1f} "
        f"± {eval_metrics['eval_std']:.1f} | {per_skill_str}"
    )

    if metrics_log:
        metrics_log[-1].update(eval_metrics)

    wandb_log(eval_metrics, step=metrics_log[-1]["total_steps"] if metrics_log else 0)

    if ctx.ckpt_mgr is not None:
        is_best = ctx.ckpt_mgr.save(
            training_state, norm_state, cfg, ctx.algo_cfg,
            ctx.algo_name, ctx.obs_dim, ctx.action_dim, metrics_log, ctx.resume,
            eval_mean=eval_metrics['eval_mean'],
            critic_norm_state=critic_norm_state,
            env=ctx.env,
        )
        # Convention #5: save aux state + patch meta.json AFTER the base save.
        _save_skill_state_alongside(ctx.ckpt_dir, aux_state, skill_cfg)
        if is_best:
            _save_skill_state_alongside(ctx.ckpt_mgr.best_dir, aux_state, skill_cfg)
            print(f"  New best! eval={ctx.ckpt_mgr.best_eval:.1f}")
        else:
            print(f"  Checkpoint saved to {ctx.ckpt_dir}")

    return n_eps, key


def _save_skill_state_alongside(
    ckpt_dir: str, aux_state: dict, skill_cfg: SkillDiscoveryConfig
) -> None:
    """Write `skill_aux/` and patch `meta.json` with a `skill_discovery` block.

    Synchronous on the loop thread — no race with eval-driven concurrent saves.
    """
    save_skill_aux_state(aux_state, ckpt_dir)
    meta_path = os.path.join(ckpt_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["skill_discovery"] = write_skill_meta_block(skill_cfg)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ── Main loop ─────────────────────────────────────────────────────────────


def run_skill_offpolicy_loop(
    cfg: TrainConfig,
    algo_cfg,
    algo,
    algo_name: str,
    env_bundle: EnvBundle,
    explore_fn: Callable,
    log_extra_fields: list,
    log_extra_keys: list,
    skill_cfg: SkillDiscoveryConfig,
    skill_manager: SkillManager,
    seed: int = 0,
    resume: str | None = None,
    resume_warmup: str = "policy",
    use_wandb: bool = False,
    wandb_project: str = "jax-rl",
) -> None:
    """Run the skill-discovery off-policy training loop.

    See `run_offpolicy_loop` for the base structure. The 6 conventions that
    diverge are listed at the top of this module's docstring.
    """
    # ── Unpack bundle ──────────────────────────────────────────────────────
    env = env_bundle.env
    env_step = env_bundle.env_step
    env_state = env_bundle.env_state
    eval_env = env_bundle.eval_env
    obs_dim = env_bundle.obs_dim
    action_dim = env_bundle.action_dim
    critic_obs_dim = env_bundle.critic_obs_dim
    has_privileged = env_bundle.has_privileged
    dict_obs = env_bundle.dict_obs
    key = env_bundle.key
    num_envs = env_bundle.num_envs

    total_env_steps = cfg.total_timesteps
    skill_dim = skill_cfg.total_skill_dim
    mgr = skill_manager  # alias

    # Algo sees skill-augmented obs. obs_dim from bundle is the *raw* obs dim;
    # actor/critic networks were already constructed against (obs_dim + skill_dim)
    # at the script level.
    augmented_obs_dim = obs_dim + skill_dim

    # ── Banner ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"{algo_name.upper()} [SKILL] — {cfg.env_name} (backend={env_bundle.backend_kind})")
    print("=" * 80)
    print(f"  obs_dim={obs_dim} (+ skill_dim={skill_dim} = augmented {augmented_obs_dim})")
    print(f"  action_dim={action_dim}")
    if num_envs != cfg.num_envs:
        print(f"  num_envs={num_envs} (cfg requested {cfg.num_envs}, capped by backend), "
              f"episode_length={cfg.episode_length}")
    else:
        print(f"  num_envs={num_envs}, episode_length={cfg.episode_length}")
    print(f"  total_timesteps={total_env_steps:,}")
    print(f"  buffer_size={algo_cfg.buffer_size:,}, min_buffer={algo_cfg.min_buffer_size:,}")
    print(f"  batch_size={algo_cfg.batch_size}, grad_updates_per_step={algo_cfg.grad_updates_per_step}")
    print(f"  tau={algo_cfg.tau}, lr={cfg.lr}, gamma={cfg.gamma}")
    print(f"  reward_scaling={cfg.reward_scaling} (applied at sample time)")
    print(f"  skill_mode={skill_cfg.mode}, prior={skill_cfg.prior}, "
          f"intrinsic_weight={skill_cfg.intrinsic_weight}, "
          f"task_reward_weight={skill_cfg.task_reward_weight}")

    # ── Timestamp (shared by checkpoint dir + W&B run name) ────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")

    # ── W&B (optional) ─────────────────────────────────────────────────────
    if use_wandb:
        wandb_init(
            project=wandb_project,
            name=f"{timestamp}_{algo_name}_skill_{env_short}_seed{seed}",
            config={
                "algo": algo_name,
                "env": cfg.env_name,
                "seed": seed,
                "timestamp": timestamp,
                "skill_mode": skill_cfg.mode,
                "skill_total_dim": skill_cfg.total_skill_dim,
                **{k: v for k, v in dataclasses.asdict(cfg).items() if k != "ppo"},
                **{f"algo_{k}": v for k, v in dataclasses.asdict(algo_cfg).items()},
            },
        )
        wandb_setup_metrics()

    # ── Algo init ──────────────────────────────────────────────────────────
    key, init_key = jax.random.split(key)
    training_state = algo.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    # ── Skill manager init ─────────────────────────────────────────────────
    key, aux_key, z_key = jax.random.split(key, 3)
    aux_state = mgr.init(aux_key)
    current_z = mgr.sample_skills(z_key, num_envs)

    # ── ObsPipeline + buffer + norm_state ──────────────────────────────────
    use_obs_norm = algo_cfg.obs_normalization
    obs_norm_eps = getattr(algo_cfg, "obs_norm_eps", 1e-8)
    n_frame_stack = cfg.n_frame_stack

    pipe = ObsPipeline(dict_obs, has_privileged, use_obs_norm, n_frame_stack, obs_norm_eps)
    # Convention #3: only `skill_z` as named extra. Buffer auto-allocates the
    # `next_skill_z` companion. Do NOT also pass `next_skill_z` here.
    buffer = pipe.make_buffer(
        obs_dim, action_dim, algo_cfg.buffer_size,
        critic_obs_dim=critic_obs_dim, num_envs=num_envs,
        extra_obs_dims={"skill_z": skill_dim},
    )
    norm_state = pipe.init_norm_state(obs_dim)
    critic_norm_state = pipe.init_critic_norm_state(critic_obs_dim) if has_privileged else None

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if resume is not None:
        print(f"\n  Resuming from {resume}")
        loaded = load_checkpoint(resume, training_state, norm_state, critic_norm_state)
        if has_privileged and len(loaded) == 4:
            training_state, norm_state, start_step, critic_norm_state = loaded
        else:
            training_state, norm_state, start_step = loaded[:3]
        # Convention #6: load aux state into the freshly-init aux pytree as
        # template. current_z stays freshly sampled (loop state, not aux state).
        aux_state = load_skill_aux_state(resume, template=aux_state)
        print(f"  Resuming from step {start_step:,} (aux state restored)")

    # ── Tracker + ctx + checkpoint manager ─────────────────────────────────
    tracker = EpisodeTracker(num_envs)
    metrics_log: list[dict] = []
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_{algo_name}_skill_{env_short}_seed{seed}")
    ckpt_mgr = CheckpointManager(ckpt_dir)
    ctx = TrainContext(
        cfg=cfg, algo_cfg=algo_cfg, algo_name=algo_name,
        ckpt_dir=ckpt_dir, obs_dim=augmented_obs_dim, action_dim=action_dim,
        metrics_log=metrics_log, ckpt_mgr=ckpt_mgr, resume=resume,
        backend_kind=env_bundle.backend_kind,
        env=env_bundle.env,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\nCollecting {algo_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // num_envs)
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    for outer_step in range(start_step // num_envs, total_env_steps // num_envs):
        raw_steps = (outer_step + 1) * num_envs
        total_steps = raw_steps
        raw_obs = pipe.get_obs(env_state.obs)
        critic_raw_obs = pipe.get_critic_obs(env_state.obs) if has_privileged else None

        # Obs normalization stats update
        norm_state = pipe.update_stats(raw_obs, norm_state)
        if has_privileged:
            critic_norm_state = pipe.update_critic_stats(critic_raw_obs, critic_norm_state)
        obs_for_action = pipe.normalize_for_action(raw_obs, norm_state)

        # Skill-augmented obs for action selection
        obs_aug = jnp.concatenate([obs_for_action, current_z], axis=-1)

        # Action selection — same warmup contract as run_offpolicy_loop.
        is_warmup = len(buffer) < algo_cfg.min_buffer_size
        use_random = is_warmup and (start_step == 0 or resume_warmup == "random")
        if use_random:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action = explore_fn(training_state.actor_params, obs_aug, ak)

        # Env step
        env_state = env_step(env_state, action)
        if cfg.handle_truncation:
            if "truncation" not in env_state.info:
                raise KeyError(
                    "cfg.handle_truncation=True but env_state.info['truncation'] is "
                    "missing. Either populate info['truncation'] in your environment "
                    "or disable truncation handling with cfg.handle_truncation=False."
                )
            truncation = env_state.info["truncation"]
        else:
            truncation = jnp.zeros_like(env_state.done)

        # Buffer add. Convention #1: store *unscaled* env reward — reward
        # scaling is applied at sample time in `intrinsic_reward_then_update`.
        # Convention #3: pass skill_z + next_skill_z as kwargs; lifecycle rule
        # under resample="episode" is next_skill_z = skill_z (per spec).
        next_raw_obs = pipe.get_obs(env_state.obs)
        extra_kwargs = {"skill_z": current_z, "next_skill_z": current_z}
        if has_privileged:
            extra_kwargs["critic_obs"] = critic_raw_obs
            extra_kwargs["critic_next_obs"] = pipe.get_critic_obs(env_state.obs)
        buffer.add_batch(
            obs=raw_obs, action=action,
            reward=env_state.reward,  # NOT scaled — applied at sample time
            next_obs=next_raw_obs, done=env_state.done,
            truncation=truncation, **extra_kwargs,
        )
        tracker.step(np.asarray(env_state.reward), np.asarray(env_state.done))

        # Resample skill on episode end (resample="episode" lifecycle).
        key, rk = jax.random.split(key)
        current_z = mgr.resample_on_done(current_z, env_state.done, rk)

        # Gradient updates
        if len(buffer) >= algo_cfg.min_buffer_size:
            for _ in range(algo_cfg.grad_updates_per_step):
                key, sample_key = jax.random.split(key)
                raw_batch = buffer.sample(algo_cfg.batch_size, key=sample_key)
                # Step 2: normalize obs through ObsPipeline.
                raw_batch = pipe.normalize_batch(
                    raw_batch, norm_state, critic_norm_state=critic_norm_state,
                )
                # Step 3: append skill_z to obs / next_obs (and critic obs).
                composed = compose_skill_batch(raw_batch, raw_obs_dim=obs_dim,
                                               skill_dim=skill_dim)
                # Steps 4–5: intrinsic reward + sample-time reward composition
                # (incl. cfg.reward_scaling applied once on the composed reward).
                composed, intrinsic = intrinsic_reward_then_update(
                    composed, raw_batch, mgr, aux_state,
                    skill_cfg, cfg.reward_scaling,
                )
                # Step 6: actor/critic update with composed batch.
                training_state, step_metrics = algo.update(training_state, composed)
                # Step 7: aux update on the SAME sampled raw batch. aux_state
                # was fixed during algo.update so actor/critic gradients can't
                # flow through aux nets.
                aux_state, aux_metrics = mgr.update(aux_state, raw_batch)

                # Merge metrics for logging.
                step_metrics = dict(step_metrics)
                step_metrics.update(aux_metrics)
                step_metrics["intrinsic_reward_mean"] = float(jnp.mean(intrinsic))
                step_metrics["intrinsic_reward_std"] = float(jnp.std(intrinsic))
                step_metrics["env_reward_mean"] = float(jnp.mean(raw_batch["reward"]))

                total_gradient_steps += 1
                last_metrics = step_metrics

        # Logging
        if outer_step % log_every == 0 or total_steps >= total_env_steps:
            elapsed = time.time() - t0
            sps = int(total_steps / elapsed) if elapsed > 0 else 0
            is_training = last_metrics and len(buffer) >= algo_cfg.min_buffer_size

            log_training_step(
                total_steps, tracker, last_metrics, sps,
                is_training=is_training,
                buffer_size=len(buffer), min_buffer=algo_cfg.min_buffer_size,
                extra_fields=log_extra_fields,
                elapsed=elapsed,
            )

            if is_training:
                row = make_metrics_row(
                    total_steps, tracker, last_metrics, total_gradient_steps, sps, elapsed,
                    extra_keys=log_extra_keys,
                )
                # Skill-specific metrics on top of the algo row.
                row["intrinsic_reward_mean"] = last_metrics.get("intrinsic_reward_mean", float("nan"))
                row["intrinsic_reward_std"] = last_metrics.get("intrinsic_reward_std", float("nan"))
                row["env_reward_mean"] = last_metrics.get("env_reward_mean", float("nan"))
                # Per-factor metrics: DIAYN exposes disc_*; METRA exposes
                # phi_*, dual_*, log_dual_lam. Pull whatever the manager wrote.
                for fac in skill_cfg.factors:
                    if fac.method == "diayn":
                        suffixes = ("disc_loss", "disc_accuracy")
                    else:  # metra
                        suffixes = (
                            "phi_loss", "phi_alignment", "phi_cst_penalty",
                            "phi_diff_norm_sq", "dual_lam", "log_dual_lam",
                            "dual_loss",
                        )
                    for suffix in suffixes:
                        k = f"{fac.name}_{suffix}"
                        if k in last_metrics:
                            row[k] = float(last_metrics[k])
                metrics_log.append(row)
                info = getattr(env_state, "info", None)
                if info is not None and env_bundle.extra_metrics_fn is not None:
                    row.update(env_bundle.extra_metrics_fn(info))
                img_dict = (env_bundle.extra_image_fn(info)
                            if info is not None and env_bundle.extra_image_fn is not None else {})
                wandb_log({**row, **img_dict}, step=raw_steps)

                if (info is not None and env_bundle.debug_dump_fn is not None
                        and total_steps // 50_000 != (total_steps - log_every * num_envs) // 50_000):
                    env_bundle.debug_dump_fn(info, total_steps)

        # Eval + checkpoint (Convention #2: skill loop owns its own eval).
        obs_norm_fn = pipe.make_obs_norm_fn(norm_state)
        last_eval_eps, key = _maybe_eval_and_checkpoint_skills(
            algo.select_action, training_state.actor_params, eval_env, tracker,
            ctx, training_state, norm_state, aux_state, skill_cfg,
            last_eval_eps, key,
            obs_normalize_fn=obs_norm_fn,
            critic_norm_state=critic_norm_state,
        )

    # ── Final eval ─────────────────────────────────────────────────────────
    obs_norm_fn = pipe.make_obs_norm_fn(norm_state)
    final_metrics, key = _eval_skills(
        algo.select_action, training_state.actor_params, eval_env, cfg, skill_cfg,
        obs_norm_fn, key,
    )

    if ctx.ckpt_mgr is not None:
        is_best = ctx.ckpt_mgr.save(
            training_state, norm_state, cfg, ctx.algo_cfg,
            ctx.algo_name, ctx.obs_dim, ctx.action_dim, metrics_log, ctx.resume,
            eval_mean=final_metrics['eval_mean'],
            critic_norm_state=critic_norm_state,
            env=ctx.env,
        )
        _save_skill_state_alongside(ctx.ckpt_dir, aux_state, skill_cfg)
        if is_best:
            _save_skill_state_alongside(ctx.ckpt_mgr.best_dir, aux_state, skill_cfg)

    print("=" * 80)
    print("Skill training complete.")
    if tracker.completed_returns:
        final = tracker.completed_returns[-100:]
        print(f"  Total episodes: {tracker.n_episodes}")
        print(f"  Online avg return (last 100 eps): {np.mean(final):.1f}")
    per_skill_str = " ".join(
        f"z{i}={final_metrics[f'eval_skill{i}_mean']:.1f}"
        for i in range(skill_cfg.total_skill_dim)
    )
    print(f"  Final per-skill eval: {per_skill_str}")
    print(f"  Final mean: {final_metrics['eval_mean']:.1f} ± {final_metrics['eval_std']:.1f}")
    print(f"  Total gradient steps: {total_gradient_steps:,}")
    print(f"  Final checkpoint: {ctx.ckpt_dir}")

    wandb_finish()
