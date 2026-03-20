"""FastDSAC — SAC with Gaussian distributional critic + DEM.

Paper: FastDSAC (arXiv:2603.12612)
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.7")

import argparse
import dataclasses
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.algos.fast_dsac import FastDSAC
from jax_rl.buffers.replay_buffer import ReplayBuffer
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.fast_dsac_config import FastDSACConfig
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_fast_dsac_preset
from jax_rl.training import (
    make_envs, make_identity_norm_state,
    EpisodeTracker, load_checkpoint,
    log_training_step, make_metrics_row,
    maybe_eval_and_checkpoint, final_eval_and_checkpoint,
)
from jax_rl.utils.normalization import (
    init as norm_init, update as norm_update, normalize as norm_normalize,
)


def train(cfg: TrainConfig, dsac_cfg: FastDSACConfig, seed: int = 0,
          resume: str | None = None, jax_buffer: bool = True):
    # ── Environment ──────────────────────────────────────────────────────
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)
    total_env_steps = cfg.total_timesteps

    # LR schedule
    warmup_steps = dsac_cfg.min_buffer_size // cfg.num_envs
    train_iters = (total_env_steps // cfg.num_envs) - warmup_steps
    total_grad_steps_est = train_iters * dsac_cfg.grad_updates_per_step

    print("=" * 80)
    print(f"FastDSAC — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, episode_length={cfg.episode_length}")
    print(f"  total_timesteps={total_env_steps:,}")
    print(f"  buffer_size={dsac_cfg.buffer_size:,}, min_buffer={dsac_cfg.min_buffer_size:,}")
    print(f"  batch_size={dsac_cfg.batch_size}, grad_updates_per_step={dsac_cfg.grad_updates_per_step}")
    print(f"  Gaussian critic (no C51), DEM: τ={dsac_cfg.dem_temperature}, "
          f"β=[{dsac_cfg.beta_min},{dsac_cfg.beta_max}]")
    print(f"  target_entropy={dsac_cfg.target_entropy}")
    print(f"  tau={dsac_cfg.tau}, q_layer_norm={dsac_cfg.q_layer_norm}, "
          f"hidden={dsac_cfg.hidden_dim}")
    print(f"  lr={cfg.lr} → {dsac_cfg.lr_end} (cosine), AdamW "
          f"β=({dsac_cfg.adam_b1},{dsac_cfg.adam_b2}), wd={dsac_cfg.weight_decay}")
    print(f"  alpha_lr={dsac_cfg.alpha_lr}, gamma={cfg.gamma}")

    # ── FastDSAC setup ───────────────────────────────────────────────────
    lr_schedule = optax.cosine_decay_schedule(
        cfg.lr, total_grad_steps_est, alpha=dsac_cfg.lr_end / cfg.lr
    )
    base_opt = optax.adamw(lr_schedule, b1=dsac_cfg.adam_b1, b2=dsac_cfg.adam_b2,
                           weight_decay=dsac_cfg.weight_decay)
    if dsac_cfg.grad_clip_norm is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(dsac_cfg.grad_clip_norm), base_opt)
    else:
        optimizer = base_opt
    alpha_optimizer = optax.adam(dsac_cfg.alpha_lr)

    dsac = FastDSAC(
        config=dsac_cfg, obs_dim=obs_dim, action_dim=action_dim,
        optimizer=optimizer, alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
    )

    key, init_key = jax.random.split(key)
    training_state = dsac.init(init_key)

    # Population diversity: fixed per-env beta for DEM exploration heterogeneity
    key, beta_key = jax.random.split(key)
    beta_per_env = jax.random.uniform(
        beta_key, (cfg.num_envs, 1),
        minval=dsac_cfg.beta_min, maxval=dsac_cfg.beta_max,
    )

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    use_obs_norm = dsac_cfg.obs_normalization
    norm_state = norm_init(obs_dim) if use_obs_norm else make_identity_norm_state(obs_dim)
    BufferCls = JaxReplayBuffer if jax_buffer else ReplayBuffer
    buffer = BufferCls(obs_dim, action_dim, max_size=dsac_cfg.buffer_size)

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if resume is not None:
        print(f"\n  Resuming from {resume}")
        training_state, norm_state, start_step = load_checkpoint(resume, training_state, norm_state)
        print(f"  Resuming from step {start_step:,}")

    # ── Tracking + infra ─────────────────────────────────────────────────
    tracker = EpisodeTracker(cfg.num_envs)
    metrics_log: list[dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_fast_dsac_{env_short}_seed{seed}")

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nCollecting {dsac_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // cfg.num_envs)
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    for outer_step in range(start_step // cfg.num_envs, total_env_steps // cfg.num_envs):
        total_steps = (outer_step + 1) * cfg.num_envs
        obs = env_state.obs

        # ── Obs normalization (update stats) ─────────────────────────────
        if use_obs_norm:
            norm_state = norm_update(norm_state, obs)

        # ── Action selection (with population diversity β) ────────────────
        obs_for_action = norm_normalize(norm_state, obs, eps=dsac_cfg.obs_norm_eps) if use_obs_norm else obs
        if len(buffer) < dsac_cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action = dsac.collect_action(training_state.actor_params, obs_for_action, ak, beta_per_env)

        # ── Env step ──────────────────────────────────────────────────────
        env_state = env_step(env_state, action)
        truncation = (env_state.info["truncation"] if cfg.handle_truncation
                      else jnp.zeros_like(env_state.done))

        # ── Buffer ────────────────────────────────────────────────────────
        if jax_buffer:
            buffer.add_batch(obs=obs, action=action,
                             reward=env_state.reward * cfg.reward_scaling,
                             next_obs=env_state.obs, done=env_state.done,
                             truncation=truncation)
        else:
            buffer.add_batch(obs=np.asarray(obs), action=np.asarray(action),
                             reward=np.asarray(env_state.reward * cfg.reward_scaling),
                             next_obs=np.asarray(env_state.obs),
                             done=np.asarray(env_state.done),
                             truncation=np.asarray(truncation))

        tracker.step(np.asarray(env_state.reward), np.asarray(env_state.done))

        # ── Gradient updates ──────────────────────────────────────────────
        if len(buffer) >= dsac_cfg.min_buffer_size:
            for _ in range(dsac_cfg.grad_updates_per_step):
                if jax_buffer:
                    key, sample_key = jax.random.split(key)
                    jax_batch = buffer.sample(dsac_cfg.batch_size, key=sample_key)
                else:
                    batch = buffer.sample(dsac_cfg.batch_size)
                    jax_batch = {k: jnp.array(v) for k, v in batch.items()}
                if use_obs_norm:
                    jax_batch["obs"] = norm_normalize(norm_state, jax_batch["obs"], eps=dsac_cfg.obs_norm_eps)
                    jax_batch["next_obs"] = norm_normalize(norm_state, jax_batch["next_obs"], eps=dsac_cfg.obs_norm_eps)
                training_state, last_metrics = dsac.update(training_state, jax_batch)
                total_gradient_steps += 1

        # ── Logging ───────────────────────────────────────────────────────
        if outer_step % log_every == 0 or total_steps >= total_env_steps:
            elapsed = time.time() - t0
            sps = int(total_steps / elapsed) if elapsed > 0 else 0
            is_training = last_metrics and len(buffer) >= dsac_cfg.min_buffer_size

            log_training_step(
                total_steps, tracker, last_metrics, sps,
                is_training=is_training,
                buffer_size=len(buffer), min_buffer=dsac_cfg.min_buffer_size,
                extra_fields=[
                    ("Q1σ²", "q1_var", ".2f"),
                    ("Ent", "entropy", ".3f"),
                    ("Alpha", "alpha", ".4f"),
                ],
            )

            if is_training:
                metrics_log.append(make_metrics_row(
                    total_steps, tracker, last_metrics, total_gradient_steps, sps, elapsed,
                    extra_keys=["q1_var", "q2_var", "entropy", "alpha", "alpha_loss"],
                ))

        # ── Eval + checkpoint ─────────────────────────────────────────────
        obs_norm_fn = (lambda o: norm_normalize(norm_state, o, eps=dsac_cfg.obs_norm_eps)) if use_obs_norm else None
        last_eval_eps, key = maybe_eval_and_checkpoint(
            dsac.select_action, training_state.actor_params, eval_env, tracker,
            cfg, dsac_cfg, "fast_dsac", ckpt_dir, training_state, norm_state,
            obs_dim, action_dim, metrics_log, last_eval_eps, key, resume,
            obs_normalize_fn=obs_norm_fn,
        )

    # ── Final eval ────────────────────────────────────────────────────────
    obs_norm_fn = (lambda o: norm_normalize(norm_state, o, eps=dsac_cfg.obs_norm_eps)) if use_obs_norm else None
    final_eval_and_checkpoint(
        dsac.select_action, training_state.actor_params, eval_env, tracker,
        cfg, dsac_cfg, "fast_dsac", ckpt_dir, training_state, norm_state,
        obs_dim, action_dim, metrics_log, key, resume, total_gradient_steps,
        obs_normalize_fn=obs_norm_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CheetahRun")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-updates-per-step", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--reward-scaling", type=float, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--jax-buffer", action=argparse.BooleanOptionalAction, default=True,
                        help="Use GPU-resident JAX replay buffer (default: True)")
    parser.add_argument("--obs-norm", action="store_true", help="Enable sample-time obs normalization")
    args = parser.parse_args()

    cfg, dsac_cfg = get_fast_dsac_preset(args.env)

    cfg_overrides = {}
    dsac_overrides = {}
    if args.num_envs is not None:
        cfg_overrides["num_envs"] = args.num_envs
    if args.total_timesteps is not None:
        cfg_overrides["total_timesteps"] = args.total_timesteps
    if args.lr is not None:
        cfg_overrides["lr"] = args.lr
    if args.reward_scaling is not None:
        cfg_overrides["reward_scaling"] = args.reward_scaling
    if args.episode_length is not None:
        cfg_overrides["episode_length"] = args.episode_length
    if args.batch_size is not None:
        dsac_overrides["batch_size"] = args.batch_size
    if args.grad_updates_per_step is not None:
        dsac_overrides["grad_updates_per_step"] = args.grad_updates_per_step
    if args.buffer_size is not None:
        dsac_overrides["buffer_size"] = args.buffer_size
    if args.obs_norm:
        dsac_overrides["obs_normalization"] = True

    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)
    if dsac_overrides:
        dsac_cfg = dataclasses.replace(dsac_cfg, **dsac_overrides)

    train(cfg, dsac_cfg, seed=args.seed, resume=args.resume, jax_buffer=args.jax_buffer)
