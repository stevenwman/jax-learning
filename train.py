import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.7")

"""PPO training on MuJoCo Playground environments.

Wires together: env, collect loop, GAE, PPO update, obs normalization.
Uses wrap_for_brax_training for vectorized auto-reset + truncation tracking.

Truncation handling (matches Brax):
  Playground envs signal timeout via info['truncation']. The env auto-resets,
  so obs after truncation is the RESET state, not the terminal state.
  In GAE, we zero out the entire TD error at truncation steps:
    - Advantage at truncation = 0 (no gradient for policy)
    - Value target = V(s_t) (no gradient for critic)
    - Propagation stops (next episode doesn't leak in)
"""

import argparse
import csv
import dataclasses
import json
import os
import time
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from mujoco_playground import dm_control_suite
from mujoco_playground._src.wrapper import wrap_for_brax_training

from jax_rl.algos.ppo import PPO
from jax_rl.buffers import RolloutBuffer
from jax_rl.configs import EncoderConfig, PolicyHeadConfig, TrainConfig, get_preset
from jax_rl.utils.eval import evaluate
from jax_rl.utils.normalization import (
    init as norm_init,
    update as norm_update,
    normalize as norm_normalize,
    NormalizationState,
)


def _save_checkpoint(ckpt_dir, training_state, norm_state, cfg, obs_dim, action_dim,
                     metrics_log, resume):
    """Save checkpoint, meta, and metrics CSV to ckpt_dir."""
    os.makedirs(ckpt_dir, exist_ok=True)

    # Write meta and metrics first (fast, survives interrupted orbax saves)
    meta = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "train_config": dataclasses.asdict(cfg),
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

    # Save inference-only params (actor_params + norm_state) as numpy — no orbax needed for rollout/video
    np.save(os.path.join(ckpt_dir, "actor_params.npy"),
            {"actor_params": jax.device_get(training_state.actor_params),
             "norm_mean": jax.device_get(norm_state.mean),
             "norm_mean_of_squares": jax.device_get(norm_state.mean_of_squares),
             "norm_count": jax.device_get(norm_state.count)},
            allow_pickle=True)

    # Orbax save last (slow, async — if killed mid-save, we still have meta+CSV+actor_params)
    # Saved to orbax/ subdir so the run folder stays clean alongside meta.json/metrics.csv
    orbax_dir = os.path.join(ckpt_dir, "orbax")
    ckpt = {"training_state": training_state, "norm_state": norm_state}
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.abspath(orbax_dir), ckpt, force=True)
    checkpointer.wait_until_finished()


def train(cfg: TrainConfig, seed: int = 0, resume: str | None = None):
    # ── Environment ──────────────────────────────────────────────────────
    env = dm_control_suite.load(cfg.env_name)
    env = wrap_for_brax_training(env, episode_length=cfg.episode_length)
    env_step = jax.jit(env.step)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, cfg.num_envs))

    obs_dim = env_state.obs.shape[-1]
    action_dim = env.action_size

    samples_per_update = cfg.num_envs * cfg.num_steps
    samples_per_iter = samples_per_update * cfg.num_updates_per_batch
    num_iterations = cfg.total_timesteps // samples_per_iter

    print("=" * 80)
    print(f"PPO — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, num_steps={cfg.num_steps}, "
          f"num_updates_per_batch={cfg.num_updates_per_batch}, episode_length={cfg.episode_length}")
    print(f"  samples/update={samples_per_update:,}, samples/iter={samples_per_iter:,}, "
          f"iterations={num_iterations}, total_steps={cfg.total_timesteps:,}")

    # ── PPO setup ────────────────────────────────────────────────────────
    ppo_cfg = cfg.ppo
    num_minibatches = ppo_cfg.num_minibatches
    minibatch_size = samples_per_update // num_minibatches
    if samples_per_update % num_minibatches != 0:
        usable = minibatch_size * num_minibatches
        print(f"  WARNING: samples_per_update ({samples_per_update}) not divisible by "
              f"num_minibatches ({num_minibatches}). {samples_per_update - usable} "
              f"samples will be dropped per update.")
    total_gradient_steps = num_iterations * cfg.num_updates_per_batch * ppo_cfg.num_epochs * num_minibatches

    # Populate network/env-derived fields into the algo config
    ppo_config = dataclasses.replace(
        ppo_cfg,
        encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=cfg.policy_hidden_dim, activation=cfg.activation),
        critic_encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=cfg.value_hidden_dim, activation=cfg.activation),
        policy_head=PolicyHeadConfig(action_dim=action_dim, squash=cfg.squash,
                                     state_dependent_std=cfg.state_dependent_std),
        num_envs=cfg.num_envs,
        minibatch_size=minibatch_size,
        gamma=cfg.gamma,
    )

    # ── Optimizer (decoupled from PPO) ──────────────────────────────────
    if cfg.anneal_lr:
        lr_schedule = optax.linear_schedule(cfg.lr, 0.0, total_gradient_steps)
    else:
        lr_schedule = cfg.lr

    if cfg.max_grad_norm is not None:
        actor_optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr_schedule))
        critic_optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr_schedule))
    else:
        actor_optimizer = optax.adam(lr_schedule)
        critic_optimizer = optax.adam(lr_schedule)

    print(f"  policy_net={cfg.policy_hidden_dim}, value_net={cfg.value_hidden_dim}, "
          f"activation={cfg.activation}, minibatch_size={minibatch_size}, "
          f"num_epochs={ppo_cfg.num_epochs}, updates_per_batch={cfg.num_updates_per_batch}, "
          f"grad_updates/iter={num_minibatches * ppo_cfg.num_epochs * cfg.num_updates_per_batch}")
    lr_desc = f"{cfg.lr} (linear anneal → 0 over {total_gradient_steps:,} grad steps)" if cfg.anneal_lr else f"{cfg.lr}"
    print(f"  lr={lr_desc}, max_grad_norm={cfg.max_grad_norm}")
    print(f"  clip_eps={ppo_cfg.clip_eps}, entropy_coef={ppo_cfg.entropy_coef}, reward_scaling={cfg.reward_scaling}")
    print(f"  gamma={cfg.gamma}, gae_lambda={ppo_cfg.gae_lambda}")

    ppo = PPO(ppo_config, obs_dim, action_dim, actor_optimizer, critic_optimizer)
    key, init_key = jax.random.split(key)
    training_state = ppo.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    critic_param_count = sum(x.size for x in jax.tree.leaves(training_state.critic_params))
    print(f"  actor_params={actor_param_count:,}, critic_params={critic_param_count:,}")

    # ── Observation normalization (Welford running stats) ────────────────
    norm_state = norm_init(obs_dim)

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_iteration = 0
    if resume is not None:
        print(f"\n  Resuming from {resume}")
        target = {"training_state": training_state, "norm_state": norm_state}
        orbax_dir = os.path.join(resume, "orbax")
        ckpt = ocp.StandardCheckpointer().restore(os.path.abspath(orbax_dir), target=target)
        training_state = ckpt["training_state"]
        norm_state = ckpt["norm_state"]

        # Infer iteration from metrics.csv if available
        metrics_csv = os.path.join(resume, "metrics.csv")
        if os.path.exists(metrics_csv):
            with open(metrics_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last_iter = int(rows[-1]["iteration"])
                start_iteration = last_iter + 1
                print(f"  Resuming from iteration {start_iteration} "
                      f"(step {start_iteration * samples_per_iter:,})")
        if start_iteration == 0:
            print("  WARNING: no metrics.csv found, starting from iteration 0 with loaded weights")

    # ── Episode return tracking ──────────────────────────────────────────
    episode_rewards = np.zeros(cfg.num_envs)
    completed_returns: list[float] = []
    metrics_log: list[dict] = []

    # ── Eval env (separate instance) ────────────────────────────────────
    eval_env = dm_control_suite.load(cfg.env_name)
    eval_env = wrap_for_brax_training(eval_env, episode_length=cfg.episode_length)


    # ── Checkpoint dir (created once, reused for periodic saves) ─────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_{env_short}_seed{seed}")
    last_eval_eps = 0

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nJIT-compiling first iteration (expect a delay)...")
    print("-" * 80)

    for iteration in range(start_iteration, num_iterations):
        t0 = time.time()

        # ── Inner loop: multiple collect→update cycles per iteration ─────
        for _update_cycle in range(cfg.num_updates_per_batch):
            buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, obs_dim, action_dim)

            # ── Collect rollout ──────────────────────────────────────────
            for step in range(cfg.num_steps):
                obs = env_state.obs

                # Update running stats and normalize
                norm_state = norm_update(norm_state, obs)
                normed_obs = norm_normalize(norm_state, obs)

                # Select action
                key, action_key = jax.random.split(key)
                action, log_prob, value = ppo.select_action(
                    training_state, normed_obs, action_key
                )

                # Clip actions to env bounds (squash=False → unbounded Gaussian)
                clipped_action = jnp.clip(action, -1.0, 1.0)

                # Step environment
                env_state = env_step(env_state, clipped_action)

                truncation = (env_state.info["truncation"] if cfg.handle_truncation
                              else jnp.zeros_like(env_state.done))

                # Store transition (unclipped action matches log_prob)
                buffer.add(
                    obs=normed_obs,
                    action=action,
                    reward=env_state.reward * cfg.reward_scaling,
                    done=env_state.done,
                    truncation=truncation,
                    log_prob=log_prob,
                    value=value,
                )

                # Track episode returns (Python-side bookkeeping, unscaled)
                step_rewards = np.asarray(env_state.reward)
                step_dones = np.asarray(env_state.done)
                episode_rewards += step_rewards
                done_mask = step_dones.astype(bool)
                if done_mask.any():
                    completed_returns.extend(episode_rewards[done_mask].tolist())
                    episode_rewards[done_mask] = 0.0

            # ── Bootstrap value for last obs ─────────────────────────────
            normed_next_obs = norm_normalize(norm_state, env_state.obs)
            key, bootstrap_key = jax.random.split(key)
            _, _, next_value = ppo.select_action(
                training_state, normed_next_obs, bootstrap_key, deterministic=True
            )

            # ── Compute advantages + returns via GAE ─────────────────────
            batch = buffer.get(next_value, gamma=cfg.gamma, gae_lambda=ppo_cfg.gae_lambda)

            # ── PPO update ───────────────────────────────────────────────
            key, update_key = jax.random.split(key)
            training_state, metrics = ppo.update(training_state, batch, update_key,
                                                  next_obs=normed_next_obs)

        # ── Logging ──────────────────────────────────────────────────────
        iter_time = time.time() - t0
        total_steps = (iteration + 1) * samples_per_iter

        if iteration % cfg.log_interval == 0 or iteration == num_iterations - 1:
            if completed_returns:
                recent = completed_returns[-100:]
                avg_ret = np.mean(recent)
                min_ret = np.min(recent)
                max_ret = np.max(recent)
                n_eps = len(completed_returns)
            else:
                avg_ret = min_ret = max_ret = float("nan")
                n_eps = 0

            sps = int(samples_per_iter / iter_time) if iter_time > 0 else 0
            print(
                f"Iter {iteration:4d}/{num_iterations} | "
                f"Steps {total_steps:>9,} | "
                f"Eps {n_eps:>5} | "
                f"Return {avg_ret:7.1f} [{min_ret:4.0f},{max_ret:4.0f}] | "
                f"PLoss {metrics['policy_loss']:7.4f} | "
                f"VLoss {metrics['value_loss']:8.2f} | "
                f"Ent {metrics['entropy']:.3f} | "
                f"KL {metrics['approx_kl']:.4f} | "
                f"Clip {metrics['clip_fraction']:.3f} | "
                f"logσ {metrics['log_std_mean']:.2f} [{metrics['log_std_min']:.2f},{metrics['log_std_max']:.2f}] | "
                f"{sps:>6,} sps | "
                f"{iter_time:.1f}s"
            )
            if iteration == 0:
                print(f"  ^ first iteration includes JIT compilation time")

            metrics_log.append({
                "iteration": iteration,
                "total_steps": total_steps,
                "episodes": n_eps,
                "avg_return": float(avg_ret),
                "min_return": float(min_ret),
                "max_return": float(max_ret),
                "policy_loss": float(metrics["policy_loss"]),
                "value_loss": float(metrics["value_loss"]),
                "entropy": float(metrics["entropy"]),
                "approx_kl": float(metrics["approx_kl"]),
                "clip_fraction": float(metrics["clip_fraction"]),
                "log_std_mean": float(metrics["log_std_mean"]),
                "log_std_min": float(metrics["log_std_min"]),
                "log_std_max": float(metrics["log_std_max"]),
                "sps": sps,
                "iter_time": iter_time,
            })

        # ── Eval + checkpoint (triggered by episode count) ─────────────
        n_eps_total = len(completed_returns)
        if n_eps_total >= last_eval_eps + cfg.eval_every_n_episodes:
            # PPO wrapper: normalize obs, then call select_action with full state
            frozen_norm = norm_state
            frozen_state = training_state
            def _ppo_eval_action(actor_params, obs, key, deterministic=True):
                normed = norm_normalize(frozen_norm, obs)
                action, _, _ = ppo.select_action(frozen_state, normed, key, deterministic=True)
                return jnp.clip(action, -1.0, 1.0)

            key, eval_key = jax.random.split(key)
            eval_metrics = evaluate(
                _ppo_eval_action, training_state.actor_params,
                eval_env, num_episodes=cfg.num_eval_episodes,
                episode_length=cfg.episode_length, key=eval_key,
                num_envs=cfg.num_envs,
            )
            print(
                f"  EVAL @ {n_eps_total} eps ({total_steps:,} steps) | "
                f"Return {eval_metrics['eval_mean']:.1f} ± {eval_metrics['eval_std']:.1f} "
                f"[{eval_metrics['eval_min']:.0f}, {eval_metrics['eval_max']:.0f}]"
            )
            if metrics_log:
                metrics_log[-1].update(eval_metrics)
            _save_checkpoint(ckpt_dir, training_state, norm_state, cfg,
                             obs_dim, action_dim, metrics_log, resume)
            print(f"  Checkpoint saved to {ckpt_dir}")
            last_eval_eps = n_eps_total

    # ── Final eval + checkpoint ───────────────────────────────────────────
    frozen_norm = norm_state
    frozen_state = training_state
    def _ppo_eval_action(actor_params, obs, key, deterministic=True):
        normed = norm_normalize(frozen_norm, obs)
        action, _, _ = ppo.select_action(frozen_state, normed, key, deterministic=True)
        return jnp.clip(action, -1.0, 1.0)

    key, eval_key = jax.random.split(key)
    eval_metrics = evaluate(
        _ppo_eval_action, training_state.actor_params,
        eval_env, num_episodes=cfg.num_eval_episodes,
        episode_length=cfg.episode_length, key=eval_key,
        num_envs=cfg.num_envs,
    )
    _save_checkpoint(ckpt_dir, training_state, norm_state, cfg,
                     obs_dim, action_dim, metrics_log, resume)
    print("=" * 80)
    print(f"Training complete.")
    if completed_returns:
        final = completed_returns[-100:]
        print(f"  Total episodes: {len(completed_returns)}")
        print(f"  Online avg return (last 100 eps): {np.mean(final):.1f}")
    print(f"  Eval return: {eval_metrics['eval_mean']:.1f} ± {eval_metrics['eval_std']:.1f} "
          f"[{eval_metrics['eval_min']:.0f}, {eval_metrics['eval_max']:.0f}]")
    print(f"  Final checkpoint: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartpoleBalance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")
    # CLI overrides — applied on top of env preset
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-updates-per-batch", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--policy-hidden-dim", type=int, nargs="+", default=None)
    parser.add_argument("--value-hidden-dim", type=int, nargs="+", default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--reward-scaling", type=float, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    args = parser.parse_args()

    # Start from env preset, then apply CLI overrides
    cfg = get_preset(args.env)
    overrides = {}
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    if args.num_steps is not None:
        overrides["num_steps"] = args.num_steps
    if args.num_updates_per_batch is not None:
        overrides["num_updates_per_batch"] = args.num_updates_per_batch
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.policy_hidden_dim is not None:
        overrides["policy_hidden_dim"] = tuple(args.policy_hidden_dim)
    if args.value_hidden_dim is not None:
        overrides["value_hidden_dim"] = tuple(args.value_hidden_dim)
    if args.reward_scaling is not None:
        overrides["reward_scaling"] = args.reward_scaling
    if args.episode_length is not None:
        overrides["episode_length"] = args.episode_length
    if args.log_interval is not None:
        overrides["log_interval"] = args.log_interval
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    # PPO-specific CLI overrides
    if args.entropy_coef is not None:
        cfg = dataclasses.replace(cfg, ppo=dataclasses.replace(cfg.ppo, entropy_coef=args.entropy_coef))

    train(cfg, seed=args.seed, resume=args.resume)
