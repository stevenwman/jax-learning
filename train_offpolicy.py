"""Unified off-policy training: SAC, TD3, FastSAC, FastTD3.

Usage:
    uv run python train_offpolicy.py --algo sac --env Go2JoystickFlat
    uv run python train_offpolicy.py --algo fast_td3 --env CheetahRun --exploration-noise 0.15
    uv run python train_offpolicy.py --algo fast_sac --env HumanoidRun --obs-norm

Supports dict obs (Go2 asymmetric actor-critic), CheckpointManager with
best-policy tracking, and all QoL from train_ppo_fast.py.
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

from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import (
    get_sac_preset, get_td3_preset, get_fast_sac_preset, get_fast_td3_preset,
)
from jax_rl.training import (
    make_envs, make_identity_norm_state,
    EpisodeTracker, load_checkpoint,
    log_training_step, make_metrics_row,
    maybe_eval_and_checkpoint, final_eval_and_checkpoint,
)
from jax_rl.training.checkpointing import CheckpointManager
from jax_rl.utils.normalization import (
    init as norm_init, update as norm_update, normalize as norm_normalize,
)

# ── Algo registry ──────────────────────────────────────────────────────────

ALGO_REGISTRY = {}


def _register(name, preset_fn, family):
    ALGO_REGISTRY[name] = {"preset_fn": preset_fn, "family": family}


_register("sac", get_sac_preset, "sac")
_register("td3", get_td3_preset, "td3")
_register("fast_sac", get_fast_sac_preset, "sac")
_register("fast_td3", get_fast_td3_preset, "td3")


def _make_algo(algo_name, algo_cfg, obs_dim, action_dim, cfg):
    """Instantiate algo + optimizer. Returns (algo, training_state, key)."""
    if algo_name == "sac":
        from jax_rl.algos.sac import SAC
        if algo_cfg.grad_clip_norm is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(algo_cfg.grad_clip_norm), optax.adam(cfg.lr))
        else:
            optimizer = optax.adam(cfg.lr)
        alpha_optimizer = optax.adam(algo_cfg.alpha_lr)
        return SAC(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
                    optimizer=optimizer, alpha_optimizer=alpha_optimizer,
                    gamma=cfg.gamma, handle_truncation=cfg.handle_truncation)

    elif algo_name == "td3":
        from jax_rl.algos.td3 import TD3
        optimizer = optax.adam(cfg.lr)
        return TD3(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
                    actor_optimizer=optimizer, critic_optimizer=optax.adam(cfg.lr),
                    gamma=cfg.gamma, handle_truncation=cfg.handle_truncation)

    elif algo_name == "fast_sac":
        from jax_rl.algos.fast_sac import FastSAC
        warmup_steps = algo_cfg.min_buffer_size // cfg.num_envs
        train_iters = (cfg.total_timesteps // cfg.num_envs) - warmup_steps
        total_grad_est = train_iters * algo_cfg.grad_updates_per_step
        lr_schedule = optax.cosine_decay_schedule(cfg.lr, total_grad_est, alpha=algo_cfg.lr_end / cfg.lr) if algo_cfg.lr_end < cfg.lr else cfg.lr
        optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
        alpha_optimizer = optax.adam(algo_cfg.alpha_lr)
        return FastSAC(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
                        optimizer=optimizer, alpha_optimizer=alpha_optimizer,
                        gamma=cfg.gamma, handle_truncation=cfg.handle_truncation)

    elif algo_name == "fast_td3":
        from jax_rl.algos.fast_td3 import FastTD3
        warmup_steps = algo_cfg.min_buffer_size // cfg.num_envs
        train_iters = (cfg.total_timesteps // cfg.num_envs) - warmup_steps
        total_grad_est = train_iters * algo_cfg.grad_updates_per_step
        lr_schedule = optax.cosine_decay_schedule(cfg.lr, total_grad_est, alpha=algo_cfg.lr_end / cfg.lr) if algo_cfg.lr_end < cfg.lr else cfg.lr
        actor_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
        critic_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
        return FastTD3(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
                        actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                        gamma=cfg.gamma, handle_truncation=cfg.handle_truncation)

    raise ValueError(f"Unknown algo: {algo_name}")


# ── Training ───────────────────────────────────────────────────────────────

def train(cfg: TrainConfig, algo_cfg, algo_name: str, seed: int = 0, resume: str | None = None):
    family = ALGO_REGISTRY[algo_name]["family"]

    # ── Environment ────────────────────────────────────────────────────────
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)

    # Dict obs support
    dict_obs = isinstance(env_state.obs, dict)
    if dict_obs:
        # Off-policy: actor and critic both see "state" (no asymmetric for Q-networks)
        obs_dim = env_state.obs["state"].shape[-1]
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

    # ── Algo setup ─────────────────────────────────────────────────────────
    algo = _make_algo(algo_name, algo_cfg, obs_dim, action_dim, cfg)

    key, init_key = jax.random.split(key)
    training_state = algo.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    use_obs_norm = algo_cfg.obs_normalization
    obs_norm_eps = getattr(algo_cfg, 'obs_norm_eps', 1e-8)
    norm_state = norm_init(obs_dim) if use_obs_norm else make_identity_norm_state(obs_dim)
    buffer = JaxReplayBuffer(obs_dim, action_dim, max_size=algo_cfg.buffer_size)

    # ── Exploration closures (family-specific) ─────────────────────────────
    if family == "sac":
        def explore(actor_params, obs, key):
            return algo.select_action(actor_params, obs, key)
        log_extra_fields = [("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")]
        log_extra_keys = ["entropy", "alpha", "alpha_loss"]
    else:  # td3
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_{algo_name}_{env_short}_seed{seed}")
    ckpt_mgr = CheckpointManager(ckpt_dir)

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\nCollecting {algo_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // cfg.num_envs)
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    def _get_obs(obs):
        """Extract flat obs from dict or flat."""
        return obs["state"] if dict_obs else obs

    for outer_step in range(start_step // cfg.num_envs, total_env_steps // cfg.num_envs):
        total_steps = (outer_step + 1) * cfg.num_envs
        raw_obs = _get_obs(env_state.obs)

        # ── Obs normalization ──────────────────────────────────────────
        if use_obs_norm:
            norm_state = norm_update(norm_state, raw_obs)
        obs_for_action = norm_normalize(norm_state, raw_obs, eps=obs_norm_eps) if use_obs_norm else raw_obs

        # ── Action selection ───────────────────────────────────────────
        if len(buffer) < algo_cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action = explore(training_state.actor_params, obs_for_action, ak)

        # ── Env step ───────────────────────────────────────────────────
        env_state = env_step(env_state, action)
        truncation = (env_state.info["truncation"] if cfg.handle_truncation
                      else jnp.zeros_like(env_state.done))

        # ── Buffer ─────────────────────────────────────────────────────
        next_raw_obs = _get_obs(env_state.obs)
        buffer.add_batch(obs=raw_obs, action=action,
                         reward=env_state.reward * cfg.reward_scaling,
                         next_obs=next_raw_obs, done=env_state.done,
                         truncation=truncation)

        tracker.step(np.asarray(env_state.reward), np.asarray(env_state.done))

        # ── Gradient updates ───────────────────────────────────────────
        if len(buffer) >= algo_cfg.min_buffer_size:
            for _ in range(algo_cfg.grad_updates_per_step):
                key, sample_key = jax.random.split(key)
                jax_batch = buffer.sample(algo_cfg.batch_size, key=sample_key)
                if use_obs_norm:
                    jax_batch["obs"] = norm_normalize(norm_state, jax_batch["obs"], eps=obs_norm_eps)
                    jax_batch["next_obs"] = norm_normalize(norm_state, jax_batch["next_obs"], eps=obs_norm_eps)
                training_state, step_metrics = algo.update(training_state, jax_batch)
                total_gradient_steps += 1

                # TD3 policy_delay: keep last non-zero actor_loss
                if family == "td3":
                    if float(step_metrics.get("actor_loss", 0.0)) != 0.0:
                        last_metrics = step_metrics
                    else:
                        last_metrics = {**step_metrics, "actor_loss": last_metrics.get("actor_loss", 0.0)}
                else:
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
            )

            if is_training:
                metrics_log.append(make_metrics_row(
                    total_steps, tracker, last_metrics, total_gradient_steps, sps, elapsed,
                    extra_keys=log_extra_keys,
                ))

        # ── Eval + checkpoint ──────────────────────────────────────────
        obs_norm_fn = (lambda o: norm_normalize(norm_state, _get_obs(o), eps=obs_norm_eps)) if use_obs_norm else (lambda o: _get_obs(o)) if dict_obs else None
        _ts = training_state
        last_eval_eps, key = maybe_eval_and_checkpoint(
            algo.select_action, training_state.actor_params, eval_env, tracker,
            cfg, algo_cfg, algo_name, ckpt_dir, training_state, norm_state,
            obs_dim, action_dim, metrics_log, last_eval_eps, key, resume,
            obs_normalize_fn=obs_norm_fn,
            q_fn=lambda obs, action: algo.get_q_value(_ts, obs, action),
        )

    # ── Final eval ─────────────────────────────────────────────────────────
    obs_norm_fn = (lambda o: norm_normalize(norm_state, _get_obs(o), eps=obs_norm_eps)) if use_obs_norm else (lambda o: _get_obs(o)) if dict_obs else None
    final_eval_and_checkpoint(
        algo.select_action, training_state.actor_params, eval_env, tracker,
        cfg, algo_cfg, algo_name, ckpt_dir, training_state, norm_state,
        obs_dim, action_dim, metrics_log, key, resume, total_gradient_steps,
        obs_normalize_fn=obs_norm_fn,
        q_fn=lambda obs, action: algo.get_q_value(training_state, obs, action),
    )


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=list(ALGO_REGISTRY.keys()))
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
    parser.add_argument("--exploration-noise", type=float, default=None, help="TD3-family only")
    parser.add_argument("--eval-every", type=int, default=None, help="Eval every N episodes")
    parser.add_argument("--obs-norm", action="store_true", help="Enable sample-time obs normalization")
    args = parser.parse_args()

    # Load preset
    preset_fn = ALGO_REGISTRY[args.algo]["preset_fn"]
    cfg, algo_cfg = preset_fn(args.env)

    # Apply overrides
    cfg_overrides = {}
    algo_overrides = {}
    if args.num_envs is not None: cfg_overrides["num_envs"] = args.num_envs
    if args.total_timesteps is not None: cfg_overrides["total_timesteps"] = args.total_timesteps
    if args.lr is not None: cfg_overrides["lr"] = args.lr
    if args.reward_scaling is not None: cfg_overrides["reward_scaling"] = args.reward_scaling
    if args.episode_length is not None: cfg_overrides["episode_length"] = args.episode_length
    if args.eval_every is not None: cfg_overrides["eval_every_n_episodes"] = args.eval_every
    if args.batch_size is not None: algo_overrides["batch_size"] = args.batch_size
    if args.grad_updates_per_step is not None: algo_overrides["grad_updates_per_step"] = args.grad_updates_per_step
    if args.buffer_size is not None: algo_overrides["buffer_size"] = args.buffer_size
    if args.exploration_noise is not None: algo_overrides["exploration_noise_std"] = args.exploration_noise
    if args.obs_norm: algo_overrides["obs_normalization"] = True

    if cfg_overrides: cfg = dataclasses.replace(cfg, **cfg_overrides)
    if algo_overrides: algo_cfg = dataclasses.replace(algo_cfg, **algo_overrides)

    train(cfg, algo_cfg, algo_name=args.algo, seed=args.seed, resume=args.resume)
