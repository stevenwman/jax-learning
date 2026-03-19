"""FastSAC — SAC with C51 distributional critic.

SAC + distributional critic + Q averaging + LR cosine decay.
Entropy regularization + auto-tuned alpha are unchanged from vanilla SAC.
"""

import argparse
import csv
import dataclasses
import json
import os
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from mujoco_playground import dm_control_suite
from mujoco_playground._src.wrapper import wrap_for_brax_training

from jax_rl.algos.fast_sac import FastSAC
from jax_rl.buffers.replay_buffer import ReplayBuffer
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_sac_preset
from jax_rl.utils.eval import evaluate
from jax_rl.utils.normalization import NormalizationState


def _save_checkpoint(ckpt_dir, training_state, norm_state, cfg, sac_cfg,
                     obs_dim, action_dim, metrics_log, resume):
    os.makedirs(ckpt_dir, exist_ok=True)

    meta = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "train_config": dataclasses.asdict(cfg),
        "sac_config": dataclasses.asdict(sac_cfg),
        "algo": "fast_sac",
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

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

    # Inference artifact: actor_params + norm stats (no orbax needed for video/rollout)
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


def train(cfg: TrainConfig, sac_cfg: SACConfig, seed: int = 0, resume: str | None = None,
          num_atoms: int = 51, v_min: float = -10.0, v_max: float = 150.0,
          q_aggregation: str = "avg", lr_end: float = 3e-5,
          jax_buffer: bool = True):
    env = dm_control_suite.load(cfg.env_name)
    env = wrap_for_brax_training(env, episode_length=cfg.episode_length)
    env_step = jax.jit(env.step)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, cfg.num_envs))

    obs_dim = env_state.obs.shape[-1]
    action_dim = env.action_size
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
    print(f"  lr={cfg.lr} → {lr_end} (cosine), alpha_lr={sac_cfg.alpha_lr}, gamma={cfg.gamma}")

    # ── FastSAC setup ─────────────────────────────────────────────────────
    lr_schedule = optax.cosine_decay_schedule(cfg.lr, total_grad_steps_est, alpha=lr_end / cfg.lr)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
    alpha_optimizer = optax.adam(sac_cfg.alpha_lr)

    sac = FastSAC(
        config=sac_cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        optimizer=optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma,
        handle_truncation=cfg.handle_truncation,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        q_aggregation=q_aggregation,
    )

    key, init_key = jax.random.split(key)
    training_state = sac.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    norm_state = NormalizationState(
        mean=jnp.zeros(obs_dim),
        mean_of_squares=jnp.ones(obs_dim),
        count=1,
    )

    BufferCls = JaxReplayBuffer if jax_buffer else ReplayBuffer
    buffer = BufferCls(obs_dim, action_dim, max_size=sac_cfg.buffer_size)

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if resume is not None:
        print(f"\n  Resuming from {resume}")
        target = {"training_state": training_state, "norm_state": norm_state}
        orbax_dir = os.path.join(resume, "orbax")
        ckpt = ocp.StandardCheckpointer().restore(os.path.abspath(orbax_dir), target=target)
        training_state = ckpt["training_state"]
        norm_state = ckpt["norm_state"]
        metrics_csv = os.path.join(resume, "metrics.csv")
        if os.path.exists(metrics_csv):
            with open(metrics_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                start_step = int(rows[-1]["total_steps"])
                print(f"  Resuming from step {start_step:,}")

    # ── Episode return tracking ───────────────────────────────────────────
    episode_rewards = np.zeros(cfg.num_envs)
    completed_returns: list[float] = []
    metrics_log: list[dict] = []

    # ── Eval env (separate instance, not disturbing training) ──────────────
    eval_env = dm_control_suite.load(cfg.env_name)
    eval_env = wrap_for_brax_training(eval_env, episode_length=cfg.episode_length)

    # ── Checkpoint dir ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_fast_sac_{env_short}_seed{seed}")

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nCollecting {sac_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // cfg.num_envs)  # log roughly every 10k env steps
    last_log_step = start_step
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    for outer_step in range(start_step // cfg.num_envs,
                            total_env_steps // cfg.num_envs):
        total_steps = (outer_step + 1) * cfg.num_envs

        obs = env_state.obs

        # Select action (no obs normalization — Q-network LayerNorm handles scaling,
        # and online normalization is unstable with off-policy replay buffers)
        if len(buffer) < sac_cfg.min_buffer_size:
            # Random exploration until buffer is warm
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action = sac.select_action(training_state.actor_params, obs, ak)

        # Step env
        env_state = env_step(env_state, action)

        truncation = (
            env_state.info["truncation"] if cfg.handle_truncation
            else jnp.zeros_like(env_state.done)
        )

        # Store raw obs (no normalization for off-policy)
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

        # Episode return tracking
        step_rewards = np.asarray(env_state.reward)
        step_dones = np.asarray(env_state.done)
        episode_rewards += step_rewards
        done_mask = step_dones.astype(bool)
        if done_mask.any():
            completed_returns.extend(episode_rewards[done_mask].tolist())
            episode_rewards[done_mask] = 0.0

        # ── Gradient updates ─────────────────────────────────────────────
        if len(buffer) >= sac_cfg.min_buffer_size:
            last_metrics = {}
            for _ in range(sac_cfg.grad_updates_per_step):
                if jax_buffer:
                    key, sample_key = jax.random.split(key)
                    jax_batch = buffer.sample(sac_cfg.batch_size, key=sample_key)
                else:
                    batch = buffer.sample(sac_cfg.batch_size)
                    jax_batch = {k: jnp.array(v) for k, v in batch.items()}
                training_state, last_metrics = sac.update(training_state, jax_batch)
                total_gradient_steps += 1

        # ── Logging ──────────────────────────────────────────────────────
        if outer_step % log_every == 0 or total_steps >= total_env_steps:
            elapsed = time.time() - t0
            sps = int(total_steps / elapsed) if elapsed > 0 else 0

            if completed_returns:
                recent = completed_returns[-100:]
                avg_ret = np.mean(recent)
                min_ret = np.min(recent)
                max_ret = np.max(recent)
                n_eps = len(completed_returns)
            else:
                avg_ret = min_ret = max_ret = float("nan")
                n_eps = 0

            if last_metrics and len(buffer) >= sac_cfg.min_buffer_size:
                print(
                    f"Step {total_steps:>9,} | "
                    f"Eps {n_eps:>5} | "
                    f"Return {avg_ret:7.1f} [{min_ret:4.0f},{max_ret:4.0f}] | "
                    f"Q1 {float(last_metrics['q1_mean']):7.2f} | "
                    f"ActLoss {float(last_metrics['actor_loss']):7.3f} | "
                    f"Ent {float(last_metrics['entropy']):.3f} | "
                    f"Alpha {float(last_metrics['alpha']):.4f} | "
                    f"{sps:>6,} sps"
                )
            else:
                print(
                    f"Step {total_steps:>9,} | "
                    f"Buffer {len(buffer):>7,}/{sac_cfg.min_buffer_size:,} | "
                    f"Warming up... | "
                    f"{sps:>6,} sps"
                )

            if last_metrics and len(buffer) >= sac_cfg.min_buffer_size:
                metrics_log.append({
                    "total_steps": total_steps,
                    "episodes": n_eps,
                    "avg_return": float(avg_ret),
                    "min_return": float(min_ret),
                    "max_return": float(max_ret),
                    "q1_mean": float(last_metrics.get("q1_mean", float("nan"))),
                    "q2_mean": float(last_metrics.get("q2_mean", float("nan"))),
                    "actor_loss": float(last_metrics.get("actor_loss", float("nan"))),
                    "entropy": float(last_metrics.get("entropy", float("nan"))),
                    "alpha": float(last_metrics.get("alpha", float("nan"))),
                    "alpha_loss": float(last_metrics.get("alpha_loss", float("nan"))),
                    "grad_steps": total_gradient_steps,
                    "sps": sps,
                    "elapsed": elapsed,
                })
            last_log_step = total_steps

        # ── Eval + checkpoint (triggered by episode count) ─────────────────
        n_eps = len(completed_returns)
        if n_eps >= last_eval_eps + cfg.eval_every_n_episodes:
            key, eval_key = jax.random.split(key)
            eval_metrics = evaluate(
                sac.select_action, training_state.actor_params,
                eval_env, num_episodes=cfg.num_eval_episodes,
                episode_length=cfg.episode_length, key=eval_key,
            )
            print(
                f"  EVAL @ {n_eps} eps ({total_steps:,} steps) | "
                f"Return {eval_metrics['eval_mean']:.1f} ± {eval_metrics['eval_std']:.1f} "
                f"[{eval_metrics['eval_min']:.0f}, {eval_metrics['eval_max']:.0f}]"
            )
            if metrics_log:
                metrics_log[-1].update(eval_metrics)
            _save_checkpoint(ckpt_dir, training_state, norm_state, cfg, sac_cfg,
                             obs_dim, action_dim, metrics_log, resume)
            print(f"  Checkpoint saved to {ckpt_dir}")
            last_eval_eps = n_eps

    # ── Final eval + checkpoint ───────────────────────────────────────────
    key, eval_key = jax.random.split(key)
    eval_metrics = evaluate(
        sac.select_action, training_state.actor_params,
        eval_env, num_episodes=cfg.num_eval_episodes,
        episode_length=cfg.episode_length, key=eval_key,
    )
    _save_checkpoint(ckpt_dir, training_state, norm_state, cfg, sac_cfg,
                     obs_dim, action_dim, metrics_log, resume)
    print("=" * 80)
    print(f"Training complete.")
    if completed_returns:
        final = completed_returns[-100:]
        print(f"  Total episodes: {len(completed_returns)}")
        print(f"  Online avg return (last 100 eps): {np.mean(final):.1f}")
    print(f"  Eval return: {eval_metrics['eval_mean']:.1f} ± {eval_metrics['eval_std']:.1f} "
          f"[{eval_metrics['eval_min']:.0f}, {eval_metrics['eval_max']:.0f}]")
    print(f"  Total gradient steps: {total_gradient_steps:,}")
    print(f"  Final checkpoint: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="WalkerWalk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    # Common overrides
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

    cfg, sac_cfg = get_sac_preset(args.env)

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
