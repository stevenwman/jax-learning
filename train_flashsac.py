"""FlashSAC standalone training script.

Usage:
    uv run python train_flashsac.py --env CartpoleBalance
    uv run python train_flashsac.py --env CheetahRun --num-envs 128 --total-timesteps 5000000
    uv run python train_flashsac.py --env Go2WarpJoystickFlat --wandb

Trains FlashSAC (Kim et al. 2026) with:
  - Zeta-distributed noise repetition during collection
  - Adaptive reward scaling (outside JIT, updated per env step)
  - Warmup + cosine decay LR schedule (shared across actor, critic, temperature)
  - Cross-batch BatchNorm + weight normalization (handled inside FlashSAC.update)
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

from jax_rl.algos.flash_sac import FlashSAC, NoiseState
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.configs.train_config import TrainConfig
from jax_rl.training import (
    make_envs, make_identity_norm_state,
    EpisodeTracker, load_checkpoint,
    log_training_step, make_metrics_row,
    maybe_eval_and_checkpoint, final_eval_and_checkpoint,
)
from jax_rl.training.checkpointing import CheckpointManager
from jax_rl.training.metrics_logger import wandb_init, wandb_setup_metrics, wandb_log, wandb_finish
from jax_rl.utils.reward_scaling import init_reward_norm, update_reward_stats, scale_reward


# ── Zeta noise repetition ──────────────────────────────────────────────────

def _make_zeta_cdf(mu: float, max_n: int) -> jnp.ndarray:
    """Precompute CDF for Zeta distribution P(k) ∝ k^(-mu), k=1..max_n."""
    ks = jnp.arange(1, max_n + 1, dtype=jnp.float32)
    pmf = ks ** (-mu)
    pmf = pmf / pmf.sum()
    return jnp.cumsum(pmf)


# ── Training ───────────────────────────────────────────────────────────────

def train(cfg: TrainConfig, algo_cfg: FlashSACConfig, seed: int = 0,
          resume: str | None = None, use_wandb: bool = False,
          wandb_project: str = "jax-rl"):

    # ── Environment ────────────────────────────────────────────────────────
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)

    dict_obs = isinstance(env_state.obs, dict)
    has_privileged = False
    critic_obs_dim = None
    if dict_obs:
        obs_dim = env_state.obs["state"].shape[-1]
        has_privileged = "privileged_state" in env_state.obs
        if has_privileged:
            critic_obs_dim = env_state.obs["privileged_state"].shape[-1]
            print(f"  Dict obs: actor={obs_dim}d, critic={critic_obs_dim}d (asymmetric)")
        else:
            print(f"  Dict obs: using 'state' key ({obs_dim}d)")

    total_env_steps = cfg.total_timesteps

    print("=" * 80)
    print(f"FlashSAC — {cfg.env_name}")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, episode_length={cfg.episode_length}")
    print(f"  total_timesteps={total_env_steps:,}")
    print(f"  buffer_size={algo_cfg.buffer_size:,}, min_buffer={algo_cfg.min_buffer_size:,}")
    print(f"  batch_size={algo_cfg.batch_size}, grad_updates_per_step={algo_cfg.grad_updates_per_step}")
    print(f"  tau={algo_cfg.tau}, gamma={cfg.gamma}")
    print(f"  lr_peak={algo_cfg.lr_peak}, lr_end={algo_cfg.lr_end}")
    print(f"  normalize_reward={algo_cfg.normalize_reward}, G_max={algo_cfg.G_max}")
    print(f"  noise_zeta_mu={algo_cfg.noise_zeta_mu}, noise_zeta_max={algo_cfg.noise_zeta_max}")

    # ── Timestamp ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")

    # ── W&B ───────────────────────────────────────────────────────────────
    if use_wandb:
        wandb_init(
            project=wandb_project,
            name=f"{timestamp}_flash_sac_{env_short}_seed{seed}",
            config={
                "algo": "flash_sac",
                "env": cfg.env_name,
                "seed": seed,
                "timestamp": timestamp,
                **{k: v for k, v in dataclasses.asdict(cfg).items() if k != "ppo"},
                **{f"algo_{k}": v for k, v in dataclasses.asdict(algo_cfg).items()},
            },
        )
        wandb_setup_metrics()

    # ── LR schedule ────────────────────────────────────────────────────────
    # Estimate total gradient steps for schedule shape.
    warmup_env_steps = algo_cfg.min_buffer_size
    train_env_steps = total_env_steps - warmup_env_steps
    total_gradient_steps_est = int(
        (train_env_steps / cfg.num_envs) * algo_cfg.grad_updates_per_step
    )
    warmup_steps = max(1, int(algo_cfg.lr_warmup_frac * total_gradient_steps_est))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=algo_cfg.lr_init,
        peak_value=algo_cfg.lr_peak,
        end_value=algo_cfg.lr_end,
        warmup_steps=warmup_steps,
        decay_steps=total_gradient_steps_est,
    )
    optimizer = optax.adamw(learning_rate=schedule, b2=0.95, weight_decay=0.001)
    alpha_optimizer = optax.adam(learning_rate=schedule)

    # ── Algo setup ─────────────────────────────────────────────────────────
    algo = FlashSAC(
        config=algo_cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        optimizer=optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma,
        handle_truncation=cfg.handle_truncation,
        critic_obs_dim=critic_obs_dim,
        num_envs=cfg.num_envs,
    )

    key, init_key = jax.random.split(key)
    training_state = algo.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    # ── Replay buffer ──────────────────────────────────────────────────────
    extra_obs_dims = {"critic_obs": critic_obs_dim} if has_privileged else None
    buffer = JaxReplayBuffer(
        obs_dim, action_dim,
        max_size=algo_cfg.buffer_size,
        extra_obs_dims=extra_obs_dims,
    )

    # ── Reward norm state (lives OUTSIDE TrainingState) ────────────────────
    reward_norm_state = init_reward_norm(cfg.num_envs)

    # ── Zeta CDF (precomputed once) ────────────────────────────────────────
    zeta_cdf = _make_zeta_cdf(algo_cfg.noise_zeta_mu, algo_cfg.noise_zeta_max)

    # ── Noise-aware action selection (closure over algo.actor) ─────────────
    actor_net = algo.actor

    @jax.jit
    def _select_action_with_noise(actor_params, actor_batch_stats, obs, noise_state, key):
        """Sample action with Zeta-distributed noise repetition."""
        mean, log_std = actor_net.apply(
            {'params': actor_params, 'batch_stats': actor_batch_stats},
            obs, train=False,
        )
        std = jnp.exp(log_std)

        # Reinitialize noise where count==0 or count>=repeat_n
        reinit = (noise_state.count == 0) | (noise_state.count >= noise_state.repeat_n)

        k1, k2 = jax.random.split(key)
        new_noise = jax.random.normal(k1, mean.shape)
        u = jax.random.uniform(k2, (mean.shape[0],))
        new_n = (jnp.searchsorted(zeta_cdf, u) + 1).astype(jnp.int32)

        noise = jnp.where(reinit[:, None], new_noise, noise_state.noise)
        repeat_n = jnp.where(reinit, new_n, noise_state.repeat_n)
        count = jnp.where(
            reinit,
            jnp.ones_like(noise_state.count),
            noise_state.count + 1,
        )

        action = jnp.tanh(mean + std * noise)
        new_noise_state = noise_state.replace(noise=noise, count=count, repeat_n=repeat_n)
        return action, new_noise_state

    # ── Obs helpers ────────────────────────────────────────────────────────
    def _get_obs(obs):
        return obs["state"] if dict_obs else obs

    def _get_critic_obs(obs):
        return obs["privileged_state"] if has_privileged else _get_obs(obs)

    # ── Resume ─────────────────────────────────────────────────────────────
    # norm_state is a dummy (FlashSAC uses reward_norm not obs_norm), needed
    # only so load_checkpoint has a compatible third return value.
    dummy_norm_state = make_identity_norm_state(obs_dim)
    start_step = 0
    if resume is not None:
        print(f"\n  Resuming from {resume}")
        training_state, dummy_norm_state, start_step = load_checkpoint(
            resume, training_state, dummy_norm_state
        )
        print(f"  Resuming from step {start_step:,}")

    # ── Tracking + infra ───────────────────────────────────────────────────
    tracker = EpisodeTracker(cfg.num_envs)
    metrics_log: list[dict] = []
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_flash_sac_{env_short}_seed{seed}")
    ckpt_mgr = CheckpointManager(ckpt_dir)

    log_extra_fields = [("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f"),
                        ("RewScale", "reward_scale_denom", ".3f")]
    log_extra_keys = ["entropy", "alpha", "alpha_loss",
                      "reward_G_var", "reward_G_r_max", "reward_scale_denom"]

    print(f"\nCollecting {algo_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // cfg.num_envs)
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    for outer_step in range(start_step // cfg.num_envs, total_env_steps // cfg.num_envs):
        total_steps = (outer_step + 1) * cfg.num_envs
        raw_obs = _get_obs(env_state.obs)
        critic_raw_obs = _get_critic_obs(env_state.obs) if has_privileged else None

        # ── Action selection ───────────────────────────────────────────
        if len(buffer) < algo_cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action, new_noise_state = _select_action_with_noise(
                training_state.actor_params,
                training_state.actor_batch_stats,
                raw_obs,
                training_state.noise_state,
                ak,
            )
            training_state = training_state.replace(noise_state=new_noise_state)

        # ── Env step ───────────────────────────────────────────────────
        env_state = env_step(env_state, action)
        truncation = (env_state.info["truncation"] if cfg.handle_truncation
                      else jnp.zeros_like(env_state.done))

        # ── Reward normalization (outside JIT) ─────────────────────────
        if algo_cfg.normalize_reward:
            reward_norm_state = update_reward_stats(
                reward_norm_state,
                env_state.reward,
                terminated=env_state.done,
                truncated=truncation,
                gamma=cfg.gamma,
            )

        # ── Buffer ─────────────────────────────────────────────────────
        next_raw_obs = _get_obs(env_state.obs)
        extra_kwargs = {}
        if has_privileged:
            extra_kwargs["critic_obs"] = critic_raw_obs
            extra_kwargs["critic_next_obs"] = _get_critic_obs(env_state.obs)

        buffer.add_batch(
            obs=raw_obs,
            action=action,
            reward=env_state.reward,   # raw reward; scaling happens at sample time
            next_obs=next_raw_obs,
            done=env_state.done,
            truncation=truncation,
            **extra_kwargs,
        )

        tracker.step(np.asarray(env_state.reward), np.asarray(env_state.done))

        # ── Gradient updates ───────────────────────────────────────────
        if len(buffer) >= algo_cfg.min_buffer_size:
            for _ in range(algo_cfg.grad_updates_per_step):
                key, sample_key = jax.random.split(key)
                jax_batch = buffer.sample(algo_cfg.batch_size, key=sample_key)

                if algo_cfg.normalize_reward:
                    jax_batch["reward"] = scale_reward(
                        reward_norm_state, jax_batch["reward"], G_max=algo_cfg.G_max
                    )

                # Non-privileged envs: critic sees actor obs
                if not has_privileged:
                    jax_batch["critic_obs"] = jax_batch["obs"]
                    jax_batch["critic_next_obs"] = jax_batch["next_obs"]

                training_state, step_metrics = algo.update(training_state, jax_batch)
                total_gradient_steps += 1

                # Inject reward scaling diagnostics
                if algo_cfg.normalize_reward:
                    G_var = float(reward_norm_state.G_var)
                    G_r_max = float(reward_norm_state.G_r_max)
                    denom = max(G_var ** 0.5, G_r_max / algo_cfg.G_max)
                    step_metrics = {
                        **step_metrics,
                        "reward_G_var": G_var,
                        "reward_G_r_max": G_r_max,
                        "reward_scale_denom": denom,
                    }

                # Carry forward actor metrics on skip steps (policy_delay pattern)
                if float(step_metrics.get("actor_loss", 0.0)) != 0.0:
                    last_metrics = step_metrics
                else:
                    last_metrics = {
                        **step_metrics,
                        "actor_loss": last_metrics.get("actor_loss", 0.0),
                        "entropy": last_metrics.get("entropy", 0.0),
                    }

        # ── Logging ────────────────────────────────────────────────────
        if outer_step % log_every == 0 or total_steps >= total_env_steps:
            elapsed = time.time() - t0
            sps = int(total_steps / elapsed) if elapsed > 0 else 0
            is_training = bool(last_metrics) and len(buffer) >= algo_cfg.min_buffer_size

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
                wandb_log(row, step=total_steps)

        # ── Eval + checkpoint ──────────────────────────────────────────
        # Update actor batch_stats so select_action uses current BN running stats
        algo._default_actor_bs = training_state.actor_batch_stats
        _ts = training_state

        last_eval_eps, key = maybe_eval_and_checkpoint(
            algo.select_action,
            training_state.actor_params,
            eval_env, tracker,
            cfg, algo_cfg, "flash_sac", ckpt_dir, training_state, dummy_norm_state,
            obs_dim, action_dim, metrics_log, last_eval_eps, key, resume,
            obs_normalize_fn=(lambda o: _get_obs(o)) if dict_obs else None,
            q_fn=lambda obs, action: algo.get_q_value(
                _ts, _get_obs(obs), action,
                critic_obs=(obs["privileged_state"]
                            if isinstance(obs, dict) and "privileged_state" in obs
                            else None),
            ),
            ckpt_mgr=ckpt_mgr,
        )

    # ── Final eval ─────────────────────────────────────────────────────────
    algo._default_actor_bs = training_state.actor_batch_stats
    _ts = training_state
    final_eval_and_checkpoint(
        algo.select_action,
        training_state.actor_params,
        eval_env, tracker,
        cfg, algo_cfg, "flash_sac", ckpt_dir, training_state, dummy_norm_state,
        obs_dim, action_dim, metrics_log, key, resume, total_gradient_steps,
        obs_normalize_fn=(lambda o: _get_obs(o)) if dict_obs else None,
        q_fn=lambda obs, action: algo.get_q_value(
            _ts, _get_obs(obs), action,
            critic_obs=(obs["privileged_state"]
                        if isinstance(obs, dict) and "privileged_state" in obs
                        else None),
        ),
        ckpt_mgr=ckpt_mgr,
    )

    wandb_finish()


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FlashSAC on any supported environment.")
    parser.add_argument("--env", type=str, default="CartpoleBalance",
                        help="Environment name (e.g., CartpoleBalance, CheetahRun, Go2WarpJoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for gradient updates")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate (overrides lr_peak in FlashSACConfig)")
    parser.add_argument("--lr-end", type=float, default=None,
                        help="End learning rate for cosine decay")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer capacity")
    parser.add_argument("--grad-updates-per-step", type=int, default=None,
                        help="Gradient updates per env step (UTD ratio)")
    parser.add_argument("--no-reward-norm", action="store_true",
                        help="Disable adaptive reward normalization")
    parser.add_argument("--G-max", type=float, default=None,
                        help="Target max magnitude for discounted returns (reward norm)")
    parser.add_argument("--no-weight-norm", action="store_true",
                        help="Disable weight normalization after optimizer steps")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name")
    args = parser.parse_args()

    # ── Base configs ───────────────────────────────────────────────────────
    # Use sensible defaults; no separate preset registry for FlashSAC yet.
    cfg = TrainConfig(
        env_name=args.env,
        total_timesteps=2_000_000,
        num_envs=64,
        gamma=0.99,
        reward_scaling=1.0,  # raw rewards; FlashSAC normalizes adaptively
        handle_truncation=True,
        eval_every_n_episodes=2000,
        num_eval_episodes=10,
    )
    algo_cfg = FlashSACConfig()

    # ── Apply CLI overrides ────────────────────────────────────────────────
    cfg_overrides = {}
    algo_overrides = {}
    if args.num_envs is not None:          cfg_overrides["num_envs"] = args.num_envs
    if args.total_timesteps is not None:   cfg_overrides["total_timesteps"] = args.total_timesteps
    if args.episode_length is not None:    cfg_overrides["episode_length"] = args.episode_length
    if args.gamma is not None:             cfg_overrides["gamma"] = args.gamma
    if args.eval_every is not None:        cfg_overrides["eval_every_n_episodes"] = args.eval_every
    if args.batch_size is not None:        algo_overrides["batch_size"] = args.batch_size
    if args.buffer_size is not None:       algo_overrides["buffer_size"] = args.buffer_size
    if args.grad_updates_per_step is not None:
        algo_overrides["grad_updates_per_step"] = args.grad_updates_per_step
    if args.lr is not None:                algo_overrides["lr_peak"] = args.lr
    if args.lr_end is not None:            algo_overrides["lr_end"] = args.lr_end
    if args.no_reward_norm:                algo_overrides["normalize_reward"] = False
    if args.G_max is not None:             algo_overrides["G_max"] = args.G_max
    if args.no_weight_norm:                algo_overrides["weight_norm"] = False

    if cfg_overrides:     cfg = dataclasses.replace(cfg, **cfg_overrides)
    if algo_overrides: algo_cfg = dataclasses.replace(algo_cfg, **algo_overrides)

    train(cfg, algo_cfg, seed=args.seed, resume=args.resume,
          use_wandb=args.wandb, wandb_project=args.wandb_project)
