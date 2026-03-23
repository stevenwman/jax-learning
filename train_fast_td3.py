"""FastTD3 training on MuJoCo Playground environments.

TD3 + C51 distributional critic + Q averaging + LR cosine decay.
Paper: https://arxiv.org/abs/2505.22642
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

from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.buffers.replay_buffer import ReplayBuffer
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_fast_td3_preset
from jax_rl.training import (
    make_envs, make_identity_norm_state,
    EpisodeTracker, load_checkpoint,
    log_training_step, make_metrics_row,
    maybe_eval_and_checkpoint, final_eval_and_checkpoint,
)
from jax_rl.utils.distributional import logits_to_q, make_support
from jax_rl.utils.normalization import (
    init as norm_init, update as norm_update, normalize as norm_normalize,
)


def train(cfg: TrainConfig, td3_cfg: FastTD3Config, seed: int = 0, resume: str | None = None,
          jax_buffer: bool = True):
    # ── Environment ──────────────────────────────────────────────────────
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)
    total_env_steps = cfg.total_timesteps

    # LR schedule
    warmup_steps = td3_cfg.min_buffer_size // cfg.num_envs
    train_iters = (total_env_steps // cfg.num_envs) - warmup_steps
    total_grad_steps_est = train_iters * td3_cfg.grad_updates_per_step

    print("=" * 80)
    print(f"FastTD3 — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, episode_length={cfg.episode_length}")
    print(f"  total_timesteps={total_env_steps:,}")
    print(f"  buffer_size={td3_cfg.buffer_size:,}, min_buffer={td3_cfg.min_buffer_size:,}")
    print(f"  batch_size={td3_cfg.batch_size}, grad_updates_per_step={td3_cfg.grad_updates_per_step}")
    print(f"  C51: atoms={td3_cfg.num_atoms}, v=[{td3_cfg.v_min},{td3_cfg.v_max}], "
          f"q_agg={td3_cfg.q_aggregation}")
    print(f"  policy_delay={td3_cfg.policy_delay}, target_noise={td3_cfg.target_noise_std}")
    if td3_cfg.noise_min is not None:
        print(f"  exploration_noise=U[{td3_cfg.noise_min},{td3_cfg.noise_max}] (mixed)")
    else:
        print(f"  exploration_noise={td3_cfg.exploration_noise_std}")
    print(f"  tau={td3_cfg.tau}, q_layer_norm={td3_cfg.q_layer_norm}, "
          f"hidden={td3_cfg.hidden_dim}")
    print(f"  lr={cfg.lr}, AdamW β2=0.95 wd=0.001, gamma={cfg.gamma}")

    # ── FastTD3 setup ────────────────────────────────────────────────────
    # Paper uses constant LR + no grad clipping. We keep cosine decay as an option
    # via lr_end (set lr_end=lr for constant). No grad clipping per paper.
    lr_schedule = optax.cosine_decay_schedule(
        cfg.lr, total_grad_steps_est, alpha=td3_cfg.lr_end / cfg.lr
    ) if td3_cfg.lr_end < cfg.lr else cfg.lr
    actor_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
    critic_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)

    td3 = FastTD3(
        config=td3_cfg, obs_dim=obs_dim, action_dim=action_dim,
        actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
        gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
    )

    key, init_key = jax.random.split(key)
    training_state = td3.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    use_obs_norm = td3_cfg.obs_normalization
    norm_state = norm_init(obs_dim) if use_obs_norm else make_identity_norm_state(obs_dim)
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
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_fast_td3_{env_short}_seed{seed}")

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

        # ── Obs normalization (update stats) ─────────────────────────────
        if use_obs_norm:
            norm_state = norm_update(norm_state, obs)

        # ── Action selection ──────────────────────────────────────────────
        obs_for_action = norm_normalize(norm_state, obs, eps=td3_cfg.obs_norm_eps) if use_obs_norm else obs
        if len(buffer) < td3_cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak, noise_key = jax.random.split(key, 3)
            # Mixed noise schedule: sample σ ~ U[noise_min, noise_max] each step
            if td3_cfg.noise_min is not None:
                noise_std = jax.random.uniform(
                    noise_key, (), minval=td3_cfg.noise_min, maxval=td3_cfg.noise_max)
            else:
                noise_std = td3_cfg.exploration_noise_std
            action = td3.select_action(
                training_state.actor_params, obs_for_action, ak,
                deterministic=False,
                exploration_noise=noise_std,
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
                if use_obs_norm:
                    jax_batch["obs"] = norm_normalize(norm_state, jax_batch["obs"], eps=td3_cfg.obs_norm_eps)
                    jax_batch["next_obs"] = norm_normalize(norm_state, jax_batch["next_obs"], eps=td3_cfg.obs_norm_eps)
                training_state, step_metrics = td3.update(training_state, jax_batch)
                total_gradient_steps += 1
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
        obs_norm_fn = (lambda o: norm_normalize(norm_state, o, eps=td3_cfg.obs_norm_eps)) if use_obs_norm else None
        _ts = training_state
        _support = make_support(td3_cfg.v_min, td3_cfg.v_max, td3_cfg.num_atoms)
        def _q_fn(obs, action):
            return logits_to_q(td3.q1.apply(_ts.q1_params, obs, action), _support)
        last_eval_eps, key = maybe_eval_and_checkpoint(
            td3.select_action, training_state.actor_params, eval_env, tracker,
            cfg, td3_cfg, "fast_td3", ckpt_dir, training_state, norm_state,
            obs_dim, action_dim, metrics_log, last_eval_eps, key, resume,
            obs_normalize_fn=obs_norm_fn, q_fn=_q_fn,
        )

    # ── Final eval ────────────────────────────────────────────────────────
    obs_norm_fn = (lambda o: norm_normalize(norm_state, o, eps=td3_cfg.obs_norm_eps)) if use_obs_norm else None
    _support = make_support(td3_cfg.v_min, td3_cfg.v_max, td3_cfg.num_atoms)
    def _q_fn_final(obs, action):
        return logits_to_q(td3.q1.apply(training_state.q1_params, obs, action), _support)
    final_eval_and_checkpoint(
        td3.select_action, training_state.actor_params, eval_env, tracker,
        cfg, td3_cfg, "fast_td3", ckpt_dir, training_state, norm_state,
        obs_dim, action_dim, metrics_log, key, resume, total_gradient_steps,
        obs_normalize_fn=obs_norm_fn, q_fn=_q_fn_final,
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
    parser.add_argument("--jax-buffer", action=argparse.BooleanOptionalAction, default=True,
                        help="Use GPU-resident JAX replay buffer (default: True)")
    parser.add_argument("--obs-norm", action="store_true", help="Enable sample-time obs normalization")
    args = parser.parse_args()

    cfg, td3_cfg = get_fast_td3_preset(args.env)

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
    if args.obs_norm:
        td3_overrides["obs_normalization"] = True

    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)
    if td3_overrides:
        td3_cfg = dataclasses.replace(td3_cfg, **td3_overrides)

    train(cfg, td3_cfg, seed=args.seed, resume=args.resume, jax_buffer=args.jax_buffer)
