"""FastTD3 training script (C51 distributional + TD3).

Usage:
    uv run python train_fast_td3.py --env CheetahRun
    uv run python train_fast_td3.py --env WalkerWalk --exploration-noise 0.15
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.7")
sys.stdout.reconfigure(line_buffering=True)

import argparse
import dataclasses
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_fast_td3_preset
from jax_rl.training import (
    make_envs,
    EpisodeTracker, load_checkpoint,
    log_training_step, make_metrics_row,
    maybe_eval_and_checkpoint, final_eval_and_checkpoint,
    ObsPipeline, TrainContext, apply_cli_overrides,
)
from jax_rl.training.checkpointing import CheckpointManager
from jax_rl.training.metrics_logger import wandb_init, wandb_setup_metrics, wandb_log, wandb_finish


# ── Training ───────────────────────────────────────────────────────────────

def train(cfg: TrainConfig, algo_cfg, seed: int = 0, resume: str | None = None,
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    algo_name = "fast_td3"

    # ── Environment ────────────────────────────────────────────────────────
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)

    # Dict obs support (with optional privileged obs for asymmetric critic)
    dict_obs = isinstance(env_state.obs, dict)
    has_privileged = False
    critic_obs_dim = None
    if dict_obs:
        obs_dim = env_state.obs["state"].shape[-1]
        has_privileged = "privileged_state" in env_state.obs
        if has_privileged:
            critic_obs_dim = env_state.obs["privileged_state"].shape[-1]
            print(f"  Dict obs detected: actor={obs_dim}d, critic={critic_obs_dim}d (asymmetric)")
        else:
            print(f"  Dict obs detected: using 'state' key ({obs_dim}d) for off-policy")

    total_env_steps = cfg.total_timesteps

    print("=" * 80)
    print(f"{algo_name.upper()} — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, episode_length={cfg.episode_length}")
    print(f"  total_timesteps={total_env_steps:,}")
    print(f"  buffer_size={algo_cfg.buffer_size:,}, min_buffer={algo_cfg.min_buffer_size:,}")
    print(f"  batch_size={algo_cfg.batch_size}, grad_updates_per_step={algo_cfg.grad_updates_per_step}")
    print(f"  tau={algo_cfg.tau}, lr={cfg.lr}, gamma={cfg.gamma}")
    print(f"  reward_scaling={cfg.reward_scaling}")

    # ── Timestamp (shared by checkpoint dir + W&B run name) ─────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")

    # ── W&B (optional) ─────────────────────────────────────────────────────
    if use_wandb:
        wandb_init(
            project=wandb_project,
            name=f"{timestamp}_{algo_name}_{env_short}_seed{seed}",
            config={
                "algo": algo_name,
                "env": cfg.env_name,
                "seed": seed,
                "timestamp": timestamp,
                **{k: v for k, v in dataclasses.asdict(cfg).items() if k != "ppo"},
                **{f"algo_{k}": v for k, v in dataclasses.asdict(algo_cfg).items()},
            },
        )
        wandb_setup_metrics()

    # ── Algo setup ─────────────────────────────────────────────────────────
    warmup_steps = algo_cfg.min_buffer_size // cfg.num_envs
    train_iters = (cfg.total_timesteps // cfg.num_envs) - warmup_steps
    total_grad_est = train_iters * algo_cfg.grad_updates_per_step
    lr_schedule = optax.cosine_decay_schedule(cfg.lr, total_grad_est, alpha=algo_cfg.lr_end / cfg.lr) if algo_cfg.lr_end < cfg.lr else cfg.lr
    actor_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
    critic_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
    algo = FastTD3(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
                   actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                   gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
                   critic_obs_dim=critic_obs_dim)

    key, init_key = jax.random.split(key)
    training_state = algo.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    use_obs_norm = algo_cfg.obs_normalization
    obs_norm_eps = getattr(algo_cfg, 'obs_norm_eps', 1e-8)
    n_frame_stack = cfg.n_frame_stack

    pipe = ObsPipeline(dict_obs, has_privileged, use_obs_norm, n_frame_stack, obs_norm_eps)
    if has_privileged:
        buffer = pipe.make_buffer_with_critic(obs_dim, action_dim, algo_cfg.buffer_size,
                                              critic_obs_dim, num_envs=cfg.num_envs)
    else:
        buffer = pipe.make_buffer(obs_dim, action_dim, algo_cfg.buffer_size, num_envs=cfg.num_envs)
    norm_state = pipe.init_norm_state(obs_dim)

    # ── Exploration ────────────────────────────────────────────────────────
    exploration_noise_std = getattr(algo_cfg, 'exploration_noise_std', 0.1)
    noise_min = getattr(algo_cfg, 'noise_min', None)
    noise_max = getattr(algo_cfg, 'noise_max', None)

    def explore(actor_params, obs, key):
        key, noise_key = jax.random.split(key)
        if noise_min is not None:
            noise_std = jax.random.uniform(noise_key, (), minval=noise_min, maxval=noise_max)
        else:
            noise_std = exploration_noise_std
        return algo.select_action(actor_params, obs, key, deterministic=False, exploration_noise=noise_std)
    log_extra_fields = []
    log_extra_keys = []

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if resume is not None:
        print(f"\n  Resuming from {resume}")
        training_state, norm_state, start_step = load_checkpoint(resume, training_state, norm_state)
        print(f"  Resuming from step {start_step:,}")

    # ── Tracking + infra ───────────────────────────────────────────────────
    tracker = EpisodeTracker(cfg.num_envs)
    metrics_log: list[dict] = []
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_{algo_name}_{env_short}_seed{seed}")
    ckpt_mgr = CheckpointManager(ckpt_dir)
    ctx = TrainContext(
        cfg=cfg, algo_cfg=algo_cfg, algo_name=algo_name,
        ckpt_dir=ckpt_dir, obs_dim=obs_dim, action_dim=action_dim,
        metrics_log=metrics_log, ckpt_mgr=ckpt_mgr, resume=resume,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\nCollecting {algo_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // cfg.num_envs)
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    for outer_step in range(start_step // cfg.num_envs, total_env_steps // cfg.num_envs):
        raw_steps = (outer_step + 1) * cfg.num_envs
        total_steps = raw_steps
        raw_obs = pipe.get_obs(env_state.obs)
        critic_raw_obs = pipe.get_critic_obs(env_state.obs) if has_privileged else None

        # ── Obs normalization ──────────────────────────────────────────
        norm_state = pipe.update_stats(raw_obs, norm_state)
        obs_for_action = pipe.normalize_for_action(raw_obs, norm_state)

        # ── Action selection ───────────────────────────────────────────
        if len(buffer) < algo_cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action = explore(training_state.actor_params, obs_for_action, ak)

        # ── Env step ───────────────────────────────────────────────────
        env_state = env_step(env_state, action)

        truncation = (env_state.info.get("truncation", jnp.zeros_like(env_state.done))
                      if cfg.handle_truncation else jnp.zeros_like(env_state.done))

        # ── Buffer ─────────────────────────────────────────────────────
        next_raw_obs = pipe.get_obs(env_state.obs)
        extra_kwargs = {}
        if has_privileged:
            extra_kwargs["critic_obs"] = critic_raw_obs
            extra_kwargs["critic_next_obs"] = pipe.get_critic_obs(env_state.obs)
        buffer.add_batch(obs=raw_obs, action=action,
                         reward=env_state.reward * cfg.reward_scaling,
                         next_obs=next_raw_obs, done=env_state.done,
                         truncation=truncation, **extra_kwargs)

        tracker.step(np.asarray(env_state.reward), np.asarray(env_state.done))

        # ── Gradient updates ───────────────────────────────────────────
        if len(buffer) >= algo_cfg.min_buffer_size:
            for _ in range(algo_cfg.grad_updates_per_step):
                key, sample_key = jax.random.split(key)
                jax_batch = buffer.sample(algo_cfg.batch_size, key=sample_key)
                jax_batch = pipe.normalize_batch(jax_batch, norm_state)
                training_state, step_metrics = algo.update(training_state, jax_batch)
                total_gradient_steps += 1
                last_metrics = step_metrics

        # ── Logging ────────────────────────────────────────────────────
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
                metrics_log.append(row)
                wandb_log(row, step=raw_steps)

        # ── Eval + checkpoint ──────────────────────────────────────────
        obs_norm_fn = pipe.make_obs_norm_fn(norm_state)
        _ts = training_state
        last_eval_eps, key = maybe_eval_and_checkpoint(
            algo.select_action, training_state.actor_params, eval_env, tracker,
            ctx, training_state, norm_state, last_eval_eps, key,
            obs_normalize_fn=obs_norm_fn,
            q_fn=lambda obs, action: algo.get_q_value(
                _ts, pipe.get_obs(obs), action,
                critic_obs=obs["privileged_state"] if isinstance(obs, dict) and "privileged_state" in obs else None),
        )

    # ── Final eval ─────────────────────────────────────────────────────────
    obs_norm_fn = pipe.make_obs_norm_fn(norm_state)
    final_eval_and_checkpoint(
        algo.select_action, training_state.actor_params, eval_env, tracker,
        ctx, training_state, norm_state, key, total_gradient_steps,
        obs_normalize_fn=obs_norm_fn,
        q_fn=lambda obs, action: algo.get_q_value(
            training_state, pipe.get_obs(obs), action,
            critic_obs=obs["privileged_state"] if isinstance(obs, dict) and "privileged_state" in obs else None),
    )

    wandb_finish()


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="WalkerWalk",
                        help="Environment name (e.g., CheetahRun, HumanoidRun, Go2WarpJoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from env preset)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train (default: from env preset)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for actor and critic (default: from algo config)")
    parser.add_argument("--reward-scaling", type=float, default=None,
                        help="Multiply rewards by this factor (default: 1.0)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode (default: from env preset)")
    parser.add_argument("--exploration-noise", type=float, default=None,
                        help="Exploration noise std for TD3 (default: from algo config)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 512 episodes)")
    parser.add_argument("--obs-norm", action="store_true",
                        help="Enable sample-time obs normalization (recommended for humanoid tasks)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking (requires wandb installed)")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name (default: jax-rl)")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of stacked observation frames (default: 1, use 3 for locomotion)")
    parser.add_argument("--action-delay-ms", type=int, default=None,
                        help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
    parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Randomized action delay range in ms (e.g., 40 120)")
    parser.add_argument("--reset-mode", type=str, default=None,
                        choices=["legacy", "per_step"],
                        help="Reset mode: legacy (AutoReset) or per_step (DomainRandWrapper)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for gradient updates (default: from algo config)")
    parser.add_argument("--grad-updates-per-step", type=int, default=None,
                        help="Gradient updates per env step (default: from algo config)")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer capacity (default: from algo config)")
    args = parser.parse_args()

    # Load preset
    cfg, algo_cfg = get_fast_td3_preset(args.env)

    # Apply overrides
    cfg, algo_cfg = apply_cli_overrides(args, cfg, algo_cfg)

    train(cfg, algo_cfg, seed=args.seed, resume=args.resume,
          use_wandb=args.wandb, wandb_project=args.wandb_project)
