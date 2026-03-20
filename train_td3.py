"""TD3 training on MuJoCo Playground environments.

Training loop: single Python env step → replay buffer → gradient updates.
Vanilla TD3: 1 gradient update per env step (unlike SAC's 8).

Key differences from SAC:
  - Deterministic policy with additive exploration noise
  - Delayed policy updates (every 2 critic updates)
  - Target policy smoothing (clipped noise on target actions)
  - No entropy / alpha — simpler loss landscape
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

from jax_rl.algos.td3 import TD3
from jax_rl.buffers.replay_buffer import ReplayBuffer
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_td3_preset
from jax_rl.training import (
    make_envs, make_identity_norm_state,
    EpisodeTracker, load_checkpoint,
    log_training_step, make_metrics_row,
    maybe_eval_and_checkpoint, final_eval_and_checkpoint,
)


def train(cfg: TrainConfig, td3_cfg: TD3Config, seed: int = 0, resume: str | None = None,
          jax_buffer: bool = False):
    # ── Environment ──────────────────────────────────────────────────────
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)
    total_env_steps = cfg.total_timesteps

    print("=" * 80)
    print(f"TD3 — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, episode_length={cfg.episode_length}")
    print(f"  total_timesteps={total_env_steps:,}")
    print(f"  buffer_size={td3_cfg.buffer_size:,}, min_buffer={td3_cfg.min_buffer_size:,}")
    print(f"  batch_size={td3_cfg.batch_size}, grad_updates_per_step={td3_cfg.grad_updates_per_step}")
    print(f"  policy_delay={td3_cfg.policy_delay}, target_noise={td3_cfg.target_noise_std}, "
          f"noise_clip={td3_cfg.noise_clip}")
    print(f"  exploration_noise={td3_cfg.exploration_noise_std}")
    print(f"  tau={td3_cfg.tau}, q_layer_norm={td3_cfg.q_layer_norm}, "
          f"hidden={td3_cfg.hidden_dim}, activation={td3_cfg.activation}")
    print(f"  lr={cfg.lr}, gamma={cfg.gamma}")
    print(f"  handle_truncation={cfg.handle_truncation}, reward_scaling={cfg.reward_scaling}")

    # ── TD3 setup ────────────────────────────────────────────────────────
    actor_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr))
    critic_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr))

    td3 = TD3(
        config=td3_cfg, obs_dim=obs_dim, action_dim=action_dim,
        actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
        gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
    )

    key, init_key = jax.random.split(key)
    training_state = td3.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    norm_state = make_identity_norm_state(obs_dim)
    BufferCls = JaxReplayBuffer if jax_buffer else ReplayBuffer
    buffer = BufferCls(obs_dim, action_dim, max_size=td3_cfg.buffer_size)

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
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_td3_{env_short}_seed{seed}")

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nCollecting {td3_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // cfg.num_envs)
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    for outer_step in range(start_step // cfg.num_envs, total_env_steps // cfg.num_envs):
        total_steps = (outer_step + 1) * cfg.num_envs
        obs = env_state.obs

        # ── Action selection (with exploration noise) ─────────────────────
        if len(buffer) < td3_cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action = td3.select_action(
                training_state.actor_params, obs, ak,
                deterministic=False,
                exploration_noise=td3_cfg.exploration_noise_std,
            )

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
        if len(buffer) >= td3_cfg.min_buffer_size:
            last_metrics = {}
            for _ in range(td3_cfg.grad_updates_per_step):
                if jax_buffer:
                    key, sample_key = jax.random.split(key)
                    jax_batch = buffer.sample(td3_cfg.batch_size, key=sample_key)
                else:
                    batch = buffer.sample(td3_cfg.batch_size)
                    jax_batch = {k: jnp.array(v) for k, v in batch.items()}
                training_state, step_metrics = td3.update(training_state, jax_batch)
                total_gradient_steps += 1
                # Keep last real actor loss (non-zero = actor actually updated, not delayed)
                if float(step_metrics.get("actor_loss", 0.0)) != 0.0:
                    last_metrics = step_metrics
                else:
                    last_metrics = {**step_metrics, "actor_loss": last_metrics.get("actor_loss", 0.0)}

        # ── Logging ───────────────────────────────────────────────────────
        if outer_step % log_every == 0 or total_steps >= total_env_steps:
            elapsed = time.time() - t0
            sps = int(total_steps / elapsed) if elapsed > 0 else 0
            is_training = last_metrics and len(buffer) >= td3_cfg.min_buffer_size

            log_training_step(
                total_steps, tracker, last_metrics, sps,
                is_training=is_training,
                buffer_size=len(buffer), min_buffer=td3_cfg.min_buffer_size,
            )

            if is_training:
                metrics_log.append(make_metrics_row(
                    total_steps, tracker, last_metrics, total_gradient_steps, sps, elapsed,
                ))

        # ── Eval + checkpoint ─────────────────────────────────────────────
        last_eval_eps, key = maybe_eval_and_checkpoint(
            td3.select_action, training_state.actor_params, eval_env, tracker,
            cfg, td3_cfg, "td3", ckpt_dir, training_state, norm_state,
            obs_dim, action_dim, metrics_log, last_eval_eps, key, resume,
        )

    # ── Final eval ────────────────────────────────────────────────────────
    final_eval_and_checkpoint(
        td3.select_action, training_state.actor_params, eval_env, tracker,
        cfg, td3_cfg, "td3", ckpt_dir, training_state, norm_state,
        obs_dim, action_dim, metrics_log, key, resume, total_gradient_steps,
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
    parser.add_argument("--exploration-noise", type=float, default=None)
    parser.add_argument("--jax-buffer", action="store_true", help="Use GPU-resident JAX replay buffer")
    args = parser.parse_args()

    cfg, td3_cfg = get_td3_preset(args.env)

    cfg_overrides = {}
    td3_overrides = {}
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
        td3_overrides["batch_size"] = args.batch_size
    if args.grad_updates_per_step is not None:
        td3_overrides["grad_updates_per_step"] = args.grad_updates_per_step
    if args.buffer_size is not None:
        td3_overrides["buffer_size"] = args.buffer_size
    if args.exploration_noise is not None:
        td3_overrides["exploration_noise_std"] = args.exploration_noise

    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)
    if td3_overrides:
        td3_cfg = dataclasses.replace(td3_cfg, **td3_overrides)

    train(cfg, td3_cfg, seed=args.seed, resume=args.resume, jax_buffer=args.jax_buffer)
