"""Fast PPO training using lax.scan for the collect phase.

Same algorithm as train_ppo.py but ~5x faster wall-clock on JIT-able envs
(MjxEnv / MuJoCo Playground). The entire collect loop (env.step + select_action
+ obs normalization) runs inside a single jax.lax.scan — zero Python-GPU sync
during collection.

NOT suitable for envs that require Python-level operations per step (e.g.,
MJWarp rendering). Use train_ppo.py for those.
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.7")
sys.stdout.reconfigure(line_buffering=True)  # Unbuffered output without PYTHONUNBUFFERED

import argparse
import dataclasses
import time
from datetime import datetime
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import optax

from jax_rl.algos.ppo import PPO
from jax_rl.buffers import RolloutBatch
from jax_rl.configs import EncoderConfig, PolicyHeadConfig, TrainConfig, get_preset
from jax_rl.training import make_envs, EpisodeTracker, load_checkpoint
from jax_rl.training.metrics_logger import wandb_init, wandb_setup_metrics, wandb_log, wandb_finish
from jax_rl.training.checkpointing import save_checkpoint, CheckpointManager
from jax_rl.utils.eval import evaluate
from jax_rl.utils.normalization import (
    NormalizationState,
    init as norm_init,
    update as norm_update,
    normalize as norm_normalize,
)


class StepData(NamedTuple):
    """Per-step data accumulated by lax.scan. Stacked to (T, E, ...) arrays."""
    obs: jax.Array            # (E, obs_dim) — normed policy obs
    critic_obs: jax.Array     # (E, critic_obs_dim)
    action: jax.Array         # (E, action_dim)
    log_prob: jax.Array       # (E,)
    value: jax.Array          # (E,)
    reward: jax.Array         # (E,)
    done: jax.Array           # (E,)
    truncation: jax.Array     # (E,)


def _make_eval_action(ppo, get_policy_obs):
    """Build a JIT'd eval action function. Norm state passed as kwarg (no recompilation)."""
    @jax.jit
    def _ppo_eval_action(actor_params, obs, key=None, deterministic=True, *, norm_state):
        policy_obs = get_policy_obs(obs)
        normed = norm_normalize(norm_state, policy_obs)
        mean, _log_std = ppo.actor.apply(actor_params, normed)
        return jnp.clip(mean, -1.0, 1.0)
    return _ppo_eval_action


def train(cfg: TrainConfig, seed: int = 0, resume: str | None = None,
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    # ── Environment ──────────────────────────────────────────────────────
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)

    dict_obs = isinstance(env_state.obs, dict)
    if dict_obs:
        critic_obs_dim = env_state.obs["privileged_state"].shape[-1]
        print(f"  Asymmetric actor-critic: policy obs={obs_dim}, critic obs={critic_obs_dim}")
    else:
        critic_obs_dim = obs_dim

    ppo_cfg = cfg.ppo
    samples_per_update = cfg.num_envs * ppo_cfg.num_steps
    samples_per_iter = samples_per_update * ppo_cfg.num_updates_per_batch
    num_iterations = cfg.total_timesteps // samples_per_iter

    print("=" * 80)
    print(f"PPO (fast/scan) — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, critic_obs_dim={critic_obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, num_steps={ppo_cfg.num_steps}, "
          f"num_updates_per_batch={ppo_cfg.num_updates_per_batch}, episode_length={cfg.episode_length}")
    print(f"  samples/update={samples_per_update:,}, samples/iter={samples_per_iter:,}, "
          f"iterations={num_iterations}, total_steps={cfg.total_timesteps:,}")

    # ── Timestamp (shared by checkpoint dir + W&B run name) ─────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")

    # ── W&B (optional) ─────────────────────────────────────────────────────
    if use_wandb:
        wandb_init(
            project=wandb_project,
            name=f"{timestamp}_ppo_{env_short}_seed{seed}",
            config={**dataclasses.asdict(cfg)},
        )
        wandb_setup_metrics()

    # ── PPO setup ────────────────────────────────────────────────────────
    num_minibatches = ppo_cfg.num_minibatches
    minibatch_size = samples_per_update // num_minibatches
    if samples_per_update % num_minibatches != 0:
        usable = minibatch_size * num_minibatches
        print(f"  WARNING: samples_per_update ({samples_per_update}) not divisible by "
              f"num_minibatches ({num_minibatches}). {samples_per_update - usable} "
              f"samples will be dropped per update.")
    total_gradient_steps = num_iterations * ppo_cfg.num_updates_per_batch * ppo_cfg.num_epochs * num_minibatches

    ppo_config = dataclasses.replace(
        ppo_cfg,
        encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=ppo_cfg.policy_hidden_dim, activation=ppo_cfg.activation),
        critic_encoder=EncoderConfig(obs_dim=critic_obs_dim, hidden_dim=ppo_cfg.value_hidden_dim, activation=ppo_cfg.activation),
        policy_head=PolicyHeadConfig(action_dim=action_dim, squash=ppo_cfg.squash,
                                     state_dependent_std=ppo_cfg.state_dependent_std),
        num_envs=cfg.num_envs,
        minibatch_size=minibatch_size,
        gamma=cfg.gamma,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    if ppo_cfg.anneal_lr:
        lr_schedule = optax.linear_schedule(cfg.lr, 0.0, total_gradient_steps)
    else:
        lr_schedule = cfg.lr

    if ppo_cfg.max_grad_norm is not None:
        actor_optimizer = optax.chain(optax.clip_by_global_norm(ppo_cfg.max_grad_norm), optax.adam(lr_schedule))
        critic_optimizer = optax.chain(optax.clip_by_global_norm(ppo_cfg.max_grad_norm), optax.adam(lr_schedule))
    else:
        actor_optimizer = optax.adam(lr_schedule)
        critic_optimizer = optax.adam(lr_schedule)

    print(f"  policy_net={ppo_cfg.policy_hidden_dim}, value_net={ppo_cfg.value_hidden_dim}, "
          f"activation={ppo_cfg.activation}, minibatch_size={minibatch_size}, "
          f"num_epochs={ppo_cfg.num_epochs}, updates_per_batch={ppo_cfg.num_updates_per_batch}, "
          f"grad_updates/iter={num_minibatches * ppo_cfg.num_epochs * ppo_cfg.num_updates_per_batch}")
    lr_desc = f"{cfg.lr} (linear anneal → 0 over {total_gradient_steps:,} grad steps)" if ppo_cfg.anneal_lr else f"{cfg.lr}"
    print(f"  lr={lr_desc}, max_grad_norm={ppo_cfg.max_grad_norm}")
    print(f"  clip_eps={ppo_cfg.clip_eps}, entropy_coef={ppo_cfg.entropy_coef}, reward_scaling={cfg.reward_scaling}")
    print(f"  gamma={cfg.gamma}, gae_lambda={ppo_cfg.gae_lambda}")

    ppo = PPO(ppo_config, obs_dim, action_dim, actor_optimizer, critic_optimizer,
              critic_obs_dim=critic_obs_dim)
    key, init_key = jax.random.split(key)
    training_state = ppo.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    critic_param_count = sum(x.size for x in jax.tree.leaves(training_state.critic_params))
    print(f"  actor_params={actor_param_count:,}, critic_params={critic_param_count:,}")

    # ── Observation normalization ─────────────────────────────────────────
    norm_state = norm_init(obs_dim)
    critic_norm_state = norm_init(critic_obs_dim)

    # ── Resume ────────────────────────────────────────────────────────────
    start_iteration = 0
    if resume is not None:
        print(f"\n  Resuming from {resume}")
        training_state, norm_state, start_step = load_checkpoint(resume, training_state, norm_state)
        start_iteration = start_step // samples_per_iter if start_step > 0 else 0
        print(f"  Resuming from iteration {start_iteration} (step {start_iteration * samples_per_iter:,})")

    # ── Build scan-based collect function ─────────────────────────────────
    # Obs extraction (baked at trace time based on dict_obs)
    if dict_obs:
        def _get_policy_obs(obs): return obs["state"]
        def _get_critic_obs(obs): return obs["privileged_state"]
    else:
        def _get_policy_obs(obs): return obs
        def _get_critic_obs(obs): return obs

    reward_scaling = cfg.reward_scaling
    handle_truncation = cfg.handle_truncation
    num_steps = ppo_cfg.num_steps

    # Access the raw JIT'd functions from PPO (no Python-level dispatch)
    _select_stochastic = ppo._select_stochastic
    _select_deterministic = ppo._select_deterministic

    @jax.jit
    def _collect(training_state, env_state, norm_state, critic_norm_state, key, running_ep_return):
        """Collect num_steps of experience using lax.scan. Zero Python overhead."""

        def collect_step(carry, _unused):
            env_state, ns, cns, key, ep_return, ep_count, ep_return_sum = carry

            policy_obs = _get_policy_obs(env_state.obs)
            critic_obs = _get_critic_obs(env_state.obs)

            # Normalize with FROZEN stats (updated after full rollout, like Brax)
            normed_obs = norm_normalize(ns, policy_obs)
            normed_critic_obs = norm_normalize(cns, critic_obs)

            # Select action
            key, action_key = jax.random.split(key)
            action, log_prob, value = _select_stochastic(
                training_state.actor_params, training_state.critic_params,
                normed_obs, normed_critic_obs, action_key,
            )
            clipped_action = jnp.clip(action, -1.0, 1.0)

            # Step environment
            env_state = env_step(env_state, clipped_action)

            truncation = jnp.where(
                handle_truncation,
                env_state.info["truncation"],
                jnp.zeros_like(env_state.done),
            )

            # Track episode returns inside scan
            ep_return = ep_return + env_state.reward
            done_mask = env_state.done
            ep_return_sum = ep_return_sum + jnp.sum(ep_return * done_mask)
            ep_count = ep_count + jnp.sum(done_mask)
            ep_return = ep_return * (1.0 - done_mask)  # Reset on done

            step_data = StepData(
                obs=normed_obs,
                critic_obs=normed_critic_obs,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=env_state.reward * reward_scaling,
                done=env_state.done,
                truncation=truncation,
            )

            return (env_state, ns, cns, key, ep_return, ep_count, ep_return_sum), (step_data, policy_obs, critic_obs)

        init_ep_count = jnp.zeros(())
        init_ep_return_sum = jnp.zeros(())

        (env_state, norm_state, critic_norm_state, key, running_ep_return, ep_count, ep_return_sum), (rollout, raw_policy_obs, raw_critic_obs) = jax.lax.scan(
            collect_step,
            (env_state, norm_state, critic_norm_state, key, running_ep_return, init_ep_count, init_ep_return_sum),
            None,
            length=num_steps,
        )

        # Update norm stats AFTER collection (Brax-style: frozen during rollout)
        # raw_policy_obs: (num_steps, num_envs, obs_dim) → reshape to (N, obs_dim)
        flat_policy_obs = raw_policy_obs.reshape(-1, raw_policy_obs.shape[-1])
        flat_critic_obs = raw_critic_obs.reshape(-1, raw_critic_obs.shape[-1])
        norm_state = norm_update(norm_state, flat_policy_obs)
        critic_norm_state = norm_update(critic_norm_state, flat_critic_obs)

        # Bootstrap value for the state after the last step
        next_policy_obs = _get_policy_obs(env_state.obs)
        next_critic_obs = _get_critic_obs(env_state.obs)
        normed_next = norm_normalize(norm_state, next_policy_obs)
        normed_next_critic = norm_normalize(critic_norm_state, next_critic_obs)
        _, _, next_value = _select_deterministic(
            training_state.actor_params, training_state.critic_params,
            normed_next, normed_next_critic,
        )

        return (env_state, norm_state, critic_norm_state, key,
                rollout, normed_next, normed_next_critic, next_value,
                ep_count, ep_return_sum, running_ep_return)

    # ── Tracking + infra ─────────────────────────────────────────────────
    tracker = EpisodeTracker(cfg.num_envs)
    metrics_log: list[dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_ppo_{env_short}_seed{seed}")
    last_eval_eps = 0

    # Create eval action fn ONCE (no recompilation per eval call).
    _ppo_eval_action = _make_eval_action(ppo, _get_policy_obs)

    # ── Training loop ────────────────────────────────────────────────────
    running_ep_return = jnp.zeros(cfg.num_envs)
    ckpt_mgr = CheckpointManager(ckpt_dir)
    import time as _time
    _t_start = _time.time()
    print(f"\nJIT-compiling first iteration (expect a delay)...")
    print("-" * 80)
    import sys; sys.stdout.reconfigure(line_buffering=True)

    for iteration in range(start_iteration, num_iterations):
        t0 = time.time()

        for _update_cycle in range(ppo_cfg.num_updates_per_batch):
            # ── Scan-based collect ──────────────────────────────────────
            (env_state, norm_state, critic_norm_state, key,
             rollout, normed_next, normed_next_critic, next_value,
             scan_ep_count, scan_ep_return_sum, running_ep_return) = _collect(
                training_state, env_state, norm_state, critic_norm_state, key, running_ep_return,
            )

            # ── Episode tracking (from scan-accumulated stats) ─────────
            scan_n = int(scan_ep_count)
            if scan_n > 0:
                scan_avg = float(scan_ep_return_sum / scan_ep_count)
                tracker.completed_returns.extend([scan_avg] * scan_n)

            # ── PPO update ──────────────────────────────────────────────
            batch = RolloutBatch(
                obs=rollout.obs,
                actions=rollout.action,
                rewards=rollout.reward,
                dones=rollout.done,
                truncations=rollout.truncation,
                log_probs=rollout.log_prob,
                values=rollout.value,
            )

            key, update_key = jax.random.split(key)
            training_state, metrics = ppo.update(
                training_state, batch, update_key,
                next_obs=normed_next,
                critic_obs=rollout.critic_obs if dict_obs else None,
                critic_next_obs=normed_next_critic if dict_obs else None,
            )

        # ── Logging ───────────────────────────────────────────────────────
        iter_time = time.time() - t0
        total_steps = (iteration + 1) * samples_per_iter

        if iteration % cfg.log_interval == 0 or iteration == num_iterations - 1:
            stats = tracker.recent_stats()
            sps = int(samples_per_iter / iter_time) if iter_time > 0 else 0
            print(
                f"Iter {iteration:4d}/{num_iterations} | "
                f"Steps {total_steps:>9,} | "
                f"Eps {stats['n_eps']:>5} | "
                f"Return {stats['avg']:9.3g} [{stats['min']:7.3g},{stats['max']:7.3g}] | "
                f"PLoss {metrics['policy_loss']:.3e} | "
                f"VLoss {metrics['value_loss']:.3e} | "
                f"Ent {metrics['entropy']:.3e} | "
                f"KL {metrics['approx_kl']:.3e} | "
                f"Clip {metrics['clip_fraction']:.3e} | "
                f"logσ {metrics['log_std_mean']:.2f} [{metrics['log_std_min']:.2f},{metrics['log_std_max']:.2f}] | "
                f"{sps:>6,} sps | "
                f"{iter_time:.1f}s | "
                f"{_time.time() - _t_start:.0f}s elapsed"
            )
            if iteration == 0:
                print(f"  ^ first iteration includes JIT compilation time")

            metrics_log.append({
                "iteration": iteration,
                "total_steps": total_steps,
                "episodes": stats["n_eps"],
                "avg_return": stats["avg"],
                "min_return": stats["min"],
                "max_return": stats["max"],
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
            if iteration > 0:  # skip iter 0 — VLoss is garbage before first real update
                wandb_log(metrics_log[-1], step=total_steps)

        # ── Eval + checkpoint ─────────────────────────────────────────────
        n_eps_total = tracker.n_episodes
        if n_eps_total >= last_eval_eps + cfg.eval_every_n_episodes:
            frozen_norm = norm_state
            frozen_actor = training_state.actor_params

            key, eval_key = jax.random.split(key)
            eval_metrics = evaluate(
                _ppo_eval_action, frozen_actor,
                eval_env, num_episodes=cfg.num_eval_episodes,
                episode_length=cfg.episode_length, key=eval_key,
                num_envs=cfg.num_envs,
                action_fn_kwargs={"norm_state": frozen_norm},
            )
            print(
                f"  EVAL @ {n_eps_total} eps ({total_steps:,} steps) | "
                f"Return {eval_metrics['eval_mean']:.1f} ± {eval_metrics['eval_std']:.1f} "
                f"[{eval_metrics['eval_min']:.0f}, {eval_metrics['eval_max']:.0f}]"
            )
            if metrics_log:
                metrics_log[-1].update(eval_metrics)
            wandb_log(eval_metrics, step=total_steps)
            is_best = ckpt_mgr.save(
                training_state, norm_state, cfg, cfg.ppo,
                "ppo", obs_dim, action_dim, metrics_log, resume,
                eval_mean=eval_metrics['eval_mean'],
            )
            if is_best:
                print(f"  New best! eval={ckpt_mgr.best_eval:.1f}")
            else:
                print(f"  Checkpoint saved to {ckpt_dir}")
            last_eval_eps = n_eps_total

    # ── Final eval ────────────────────────────────────────────────────────
    frozen_norm = norm_state
    frozen_actor = training_state.actor_params

    key, eval_key = jax.random.split(key)
    eval_metrics = evaluate(
        _ppo_eval_action, frozen_actor,
        eval_env, num_episodes=cfg.num_eval_episodes,
        episode_length=cfg.episode_length, key=eval_key,
        num_envs=cfg.num_envs,
        action_fn_kwargs={"norm_state": frozen_norm},
    )
    ckpt_mgr.save(training_state, norm_state, cfg, cfg.ppo,
                   "ppo", obs_dim, action_dim, metrics_log, resume,
                   eval_mean=eval_metrics['eval_mean'])
    print("=" * 80)
    print(f"Training complete.")
    if tracker.completed_returns:
        final = tracker.completed_returns[-100:]
        print(f"  Total episodes: {tracker.n_episodes}")
        print(f"  Online avg return (last 100 eps): {np.mean(final):.1f}")
    print(f"  Eval return: {eval_metrics['eval_mean']:.1f} ± {eval_metrics['eval_std']:.1f} "
          f"[{eval_metrics['eval_min']:.0f}, {eval_metrics['eval_max']:.0f}]")
    print(f"  Final checkpoint: {ckpt_dir}")

    wandb_finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartpoleBalance",
                        help="Environment name (e.g., CartpoleBalance, CheetahRun, Go2JoystickFlat)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory path")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from env preset)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Rollout steps per env before each update (default: from preset)")
    parser.add_argument("--num-updates-per-batch", type=int, default=None,
                        help="SGD epochs over collected rollout data (default: from preset)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment steps to train (default: from env preset)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: from preset)")
    parser.add_argument("--policy-hidden-dim", type=int, nargs="+", default=None,
                        help="Actor network hidden layer sizes (e.g., 512 256 128)")
    parser.add_argument("--value-hidden-dim", type=int, nargs="+", default=None,
                        help="Critic network hidden layer sizes (e.g., 512 256 128)")
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="Entropy bonus coefficient (higher = more exploration)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes (default: every 512 episodes)")
    parser.add_argument("--reward-scaling", type=float, default=None,
                        help="Multiply rewards by this factor (default: 1.0)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Max steps per episode (default: from env preset)")
    parser.add_argument("--log-interval", type=int, default=None,
                        help="Print training stats every N iterations")
    parser.add_argument("--domain-rand", action="store_true",
                        help="Enable domain randomization (Go2 only)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="jax-rl",
                        help="W&B project name (default: jax-rl)")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of stacked observation frames (default: 1, use 3 for locomotion)")
    parser.add_argument("--action-delay-ms", type=int, default=None,
                        help="Fixed action delay in ms (e.g., 120 for Go2 sim2real)")
    parser.add_argument("--action-delay-range-ms", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Randomized action delay range in ms (e.g., 40 120)")
    args = parser.parse_args()

    cfg = get_preset(args.env)
    cfg_overrides = {}
    ppo_overrides = {}
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
    if args.num_steps is not None:
        ppo_overrides["num_steps"] = args.num_steps
    if args.num_updates_per_batch is not None:
        ppo_overrides["num_updates_per_batch"] = args.num_updates_per_batch
    if args.policy_hidden_dim is not None:
        ppo_overrides["policy_hidden_dim"] = tuple(args.policy_hidden_dim)
    if args.value_hidden_dim is not None:
        ppo_overrides["value_hidden_dim"] = tuple(args.value_hidden_dim)
    if args.entropy_coef is not None:
        ppo_overrides["entropy_coef"] = args.entropy_coef
    if args.eval_every is not None:
        cfg_overrides["eval_every_n_episodes"] = args.eval_every
    if args.domain_rand:
        cfg_overrides["domain_rand"] = True
    if args.frame_stack is not None:
        cfg_overrides["n_frame_stack"] = args.frame_stack
    if args.action_delay_ms is not None:
        cfg_overrides["action_delay_ms"] = args.action_delay_ms
    if args.action_delay_range_ms is not None:
        cfg_overrides["action_delay_range_ms"] = tuple(args.action_delay_range_ms)
    if ppo_overrides:
        cfg_overrides["ppo"] = dataclasses.replace(cfg.ppo, **ppo_overrides)
    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)

    train(cfg, seed=args.seed, resume=args.resume,
          use_wandb=args.wandb, wandb_project=args.wandb_project)
