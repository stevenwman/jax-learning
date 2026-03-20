"""FastSAC — SAC with C51 distributional critic.

SAC + distributional critic + Q averaging + LR cosine decay.
Entropy regularization + auto-tuned alpha are unchanged from vanilla SAC.
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

from jax_rl.algos.fast_sac import FastSAC
from jax_rl.buffers.replay_buffer import ReplayBuffer
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_fast_sac_preset
from jax_rl.training import (
    make_envs, make_identity_norm_state,
    EpisodeTracker, load_checkpoint,
    log_training_step, make_metrics_row,
    maybe_eval_and_checkpoint, final_eval_and_checkpoint,
)


def train(cfg: TrainConfig, sac_cfg: SACConfig, seed: int = 0, resume: str | None = None,
          num_atoms: int = 51, v_min: float = -10.0, v_max: float = 150.0,
          q_aggregation: str = "avg", lr_end: float = 3e-5,
          jax_buffer: bool = True):
    # ── Environment ──────────────────────────────────────────────────────
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)
    total_env_steps = cfg.total_timesteps

    # LR schedule
    warmup_steps = sac_cfg.min_buffer_size // cfg.num_envs
    train_iters = (total_env_steps // cfg.num_envs) - warmup_steps
    total_grad_steps_est = train_iters * sac_cfg.grad_updates_per_step

    print("=" * 80)
    print(f"FastSAC — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, episode_length={cfg.episode_length}")
    print(f"  total_timesteps={total_env_steps:,}")
    print(f"  buffer_size={sac_cfg.buffer_size:,}, min_buffer={sac_cfg.min_buffer_size:,}")
    print(f"  batch_size={sac_cfg.batch_size}, grad_updates_per_step={sac_cfg.grad_updates_per_step}")
    print(f"  C51: atoms={num_atoms}, v=[{v_min},{v_max}], q_agg={q_aggregation}")
    print(f"  target_entropy={-sac_cfg.target_entropy_scale * action_dim:.2f}")
    print(f"  tau={sac_cfg.tau}, q_layer_norm={sac_cfg.q_layer_norm}, "
          f"hidden={sac_cfg.hidden_dim}")
    print(f"  lr={cfg.lr} → {lr_end} (cosine), AdamW β2=0.95, wd=0.001")
    print(f"  alpha_lr={sac_cfg.alpha_lr}, alpha_init={sac_cfg.alpha_init}, gamma={cfg.gamma}")

    # ── FastSAC setup ────────────────────────────────────────────────────
    lr_schedule = optax.cosine_decay_schedule(cfg.lr, total_grad_steps_est, alpha=lr_end / cfg.lr)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001),
    )
    alpha_optimizer = optax.adam(sac_cfg.alpha_lr)

    sac = FastSAC(
        config=sac_cfg, obs_dim=obs_dim, action_dim=action_dim,
        optimizer=optimizer, alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
        num_atoms=num_atoms, v_min=v_min, v_max=v_max, q_aggregation=q_aggregation,
    )

    key, init_key = jax.random.split(key)
    training_state = sac.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    norm_state = make_identity_norm_state(obs_dim)
    BufferCls = JaxReplayBuffer if jax_buffer else ReplayBuffer
    buffer = BufferCls(obs_dim, action_dim, max_size=sac_cfg.buffer_size)

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
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_fast_sac_{env_short}_seed{seed}")

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nCollecting {sac_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // cfg.num_envs)
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    for outer_step in range(start_step // cfg.num_envs, total_env_steps // cfg.num_envs):
        total_steps = (outer_step + 1) * cfg.num_envs
        obs = env_state.obs

        # ── Action selection ──────────────────────────────────────────────
        if len(buffer) < sac_cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action = sac.select_action(training_state.actor_params, obs, ak)

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
        if len(buffer) >= sac_cfg.min_buffer_size:
            for _ in range(sac_cfg.grad_updates_per_step):
                if jax_buffer:
                    key, sample_key = jax.random.split(key)
                    jax_batch = buffer.sample(sac_cfg.batch_size, key=sample_key)
                else:
                    batch = buffer.sample(sac_cfg.batch_size)
                    jax_batch = {k: jnp.array(v) for k, v in batch.items()}
                training_state, last_metrics = sac.update(training_state, jax_batch)
                total_gradient_steps += 1

        # ── Logging ───────────────────────────────────────────────────────
        if outer_step % log_every == 0 or total_steps >= total_env_steps:
            elapsed = time.time() - t0
            sps = int(total_steps / elapsed) if elapsed > 0 else 0
            is_training = last_metrics and len(buffer) >= sac_cfg.min_buffer_size

            log_training_step(
                total_steps, tracker, last_metrics, sps,
                is_training=is_training,
                buffer_size=len(buffer), min_buffer=sac_cfg.min_buffer_size,
                extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
            )

            if is_training:
                metrics_log.append(make_metrics_row(
                    total_steps, tracker, last_metrics, total_gradient_steps, sps, elapsed,
                    extra_keys=["entropy", "alpha", "alpha_loss"],
                ))

        # ── Eval + checkpoint ─────────────────────────────────────────────
        last_eval_eps, key = maybe_eval_and_checkpoint(
            sac.select_action, training_state.actor_params, eval_env, tracker,
            cfg, sac_cfg, "fast_sac", ckpt_dir, training_state, norm_state,
            obs_dim, action_dim, metrics_log, last_eval_eps, key, resume,
        )

    # ── Final eval ────────────────────────────────────────────────────────
    final_eval_and_checkpoint(
        sac.select_action, training_state.actor_params, eval_env, tracker,
        cfg, sac_cfg, "fast_sac", ckpt_dir, training_state, norm_state,
        obs_dim, action_dim, metrics_log, key, resume, total_gradient_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="WalkerWalk")
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
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--jax-buffer", action=argparse.BooleanOptionalAction, default=True,
                        help="Use GPU-resident JAX replay buffer (default: True)")
    args = parser.parse_args()

    cfg, sac_cfg = get_fast_sac_preset(args.env)

    cfg_overrides = {}
    sac_overrides = {}
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
    if args.log_interval is not None:
        cfg_overrides["log_interval"] = args.log_interval
    if args.batch_size is not None:
        sac_overrides["batch_size"] = args.batch_size
    if args.grad_updates_per_step is not None:
        sac_overrides["grad_updates_per_step"] = args.grad_updates_per_step
    if args.buffer_size is not None:
        sac_overrides["buffer_size"] = args.buffer_size

    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)
    if sac_overrides:
        sac_cfg = dataclasses.replace(sac_cfg, **sac_overrides)

    train(cfg, sac_cfg, seed=args.seed, resume=args.resume, jax_buffer=args.jax_buffer)
