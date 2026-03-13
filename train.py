"""PPO training on MuJoCo Playground environments.

Wires together: env, collect loop, GAE, PPO update, obs normalization.
Uses wrap_for_brax_training for vectorized auto-reset + truncation tracking.

Truncation handling:
  Playground envs signal timeout via info['truncation']. For GAE we need to
  distinguish true terminals (done=1, trunc=0) from timeouts (done=1, trunc=1).
  We pass effective_done = done * (1 - truncation) to the buffer so GAE
  bootstraps through timeouts instead of zeroing the value estimate.
"""

import argparse
import csv
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

from jax_rl.algos.ppo_scan import PPO
from jax_rl.buffers import RolloutBuffer
from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.utils.normalization import (
    init as norm_init,
    update as norm_update,
    normalize as norm_normalize,
)


# Per-env defaults: (hidden_dim, total_timesteps, num_envs, num_steps, entropy_coef)
ENV_DEFAULTS = {
    "CartpoleBalance": ((64, 64), 1_000_000, 64, 64, 0.01),
    "CheetahRun":      ((256, 256), 20_000_000, 256, 64, 0.0),
}


def train(
    env_name: str = "CartpoleBalance",
    seed: int = 0,
    num_envs: int | None = None,
    num_steps: int | None = None,
    total_timesteps: int | None = None,
    hidden_dim: tuple[int, ...] | None = None,
    entropy_coef: float | None = None,
    episode_length: int = 1000,
    log_interval: int = 1,
):
    # ── Resolve defaults per env ──────────────────────────────────────────
    defaults = ENV_DEFAULTS.get(env_name, ((256, 256), 3_000_000, 128, 64, 0.001))
    hidden_dim = hidden_dim or defaults[0]
    total_timesteps = total_timesteps or defaults[1]
    num_envs = num_envs or defaults[2]
    num_steps = num_steps or defaults[3]
    entropy_coef = entropy_coef if entropy_coef is not None else defaults[4]

    # ── Environment ──────────────────────────────────────────────────────
    env = dm_control_suite.load(env_name)
    env = wrap_for_brax_training(env, episode_length=episode_length)
    env_step = jax.jit(env.step)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, num_envs))

    obs_dim = env_state.obs.shape[-1]
    action_dim = env.action_size

    samples_per_iter = num_envs * num_steps
    num_iterations = total_timesteps // samples_per_iter

    print("=" * 80)
    print(f"PPO — {env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={num_envs}, num_steps={num_steps}, episode_length={episode_length}")
    print(f"  samples/iter={samples_per_iter:,}, iterations={num_iterations}, total_steps={total_timesteps:,}")

    # ── PPO setup ────────────────────────────────────────────────────────
    minibatch_size = min(2048, num_envs * num_steps)
    num_minibatches = samples_per_iter // minibatch_size
    num_epochs = 4
    total_gradient_steps = num_iterations * num_epochs * num_minibatches

    config = PPOConfig(
        encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=hidden_dim),
        policy_head=PolicyHeadConfig(action_dim=action_dim, squash=False),
        num_envs=num_envs,
        num_steps=num_steps,
        minibatch_size=minibatch_size,
        num_epochs=num_epochs,
        entropy_coef=entropy_coef,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # ── Optimizer (decoupled from PPO) ──────────────────────────────────
    lr = 3e-4
    lr_schedule = optax.linear_schedule(lr, 0.0, total_gradient_steps)
    max_grad_norm = 0.5

    actor_optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_schedule))
    critic_optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_schedule))

    print(f"  hidden_dim={hidden_dim}, minibatch_size={config.minibatch_size}, "
          f"num_epochs={config.num_epochs}, grad_updates/iter={num_minibatches * config.num_epochs}")
    print(f"  lr={lr} (linear anneal → 0 over {total_gradient_steps:,} grad steps)")
    print(f"  clip_eps={config.clip_eps}, entropy_coef={config.entropy_coef}")
    print(f"  gamma={config.gamma}, gae_lambda={config.gae_lambda}")

    ppo = PPO(config, obs_dim, action_dim, actor_optimizer, critic_optimizer)
    key, init_key = jax.random.split(key)
    training_state = ppo.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    critic_param_count = sum(x.size for x in jax.tree.leaves(training_state.critic_params))
    print(f"  actor_params={actor_param_count:,}, critic_params={critic_param_count:,}")

    # ── Observation normalization (Welford running stats) ────────────────
    norm_state = norm_init(obs_dim)

    # ── Episode return tracking ──────────────────────────────────────────
    episode_rewards = np.zeros(num_envs)
    completed_returns: list[float] = []
    metrics_log: list[dict] = []

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nJIT-compiling first iteration (expect a delay)...")
    print("-" * 80)

    for iteration in range(num_iterations):
        t0 = time.time()
        buffer = RolloutBuffer(num_steps, num_envs, obs_dim, action_dim)

        # ── Collect rollout ──────────────────────────────────────────────
        for step in range(num_steps):
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

            # Truncation handling for GAE:
            #   done=1, trunc=0  → true terminal  → effective_done=1 (don't bootstrap)
            #   done=1, trunc=1  → timeout         → effective_done=0 (bootstrap through)
            truncation = env_state.info["truncation"]
            effective_done = env_state.done * (1.0 - truncation)

            # Store transition (unclipped action matches log_prob)
            buffer.add(
                obs=normed_obs,
                action=action,
                reward=env_state.reward,
                done=effective_done,
                log_prob=log_prob,
                value=value,
            )

            # Track episode returns (Python-side bookkeeping)
            step_rewards = np.asarray(env_state.reward)
            step_dones = np.asarray(env_state.done)
            episode_rewards += step_rewards
            for i in range(num_envs):
                if step_dones[i]:
                    completed_returns.append(episode_rewards[i])
                    episode_rewards[i] = 0.0

        # ── Bootstrap value for last obs ─────────────────────────────────
        normed_next_obs = norm_normalize(norm_state, env_state.obs)
        key, bootstrap_key = jax.random.split(key)
        _, _, next_value = ppo.select_action(
            training_state, normed_next_obs, bootstrap_key, deterministic=True
        )

        # ── Compute advantages + returns via GAE ─────────────────────────
        batch = buffer.get(next_value, gamma=config.gamma, gae_lambda=config.gae_lambda)

        # ── PPO update ───────────────────────────────────────────────────
        key, update_key = jax.random.split(key)
        training_state, metrics = ppo.update(training_state, batch, update_key)

        # ── Logging ──────────────────────────────────────────────────────
        iter_time = time.time() - t0
        total_steps = (iteration + 1) * num_envs * num_steps

        if iteration % log_interval == 0 or iteration == num_iterations - 1:
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
                "sps": sps,
                "iter_time": iter_time,
            })

    print("=" * 80)
    if completed_returns:
        final = completed_returns[-100:]
        print(f"Training complete.")
        print(f"  Total episodes: {len(completed_returns)}")
        print(f"  Final avg return (last 100 eps): {np.mean(final):.1f}")
        print(f"  Final max return (last 100 eps): {np.max(final):.1f}")
    else:
        print("Done. No episodes completed.")

    # ── Save checkpoint ───────────────────────────────────────────────────
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = env_name.lower().replace(" ", "_")
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_{env_short}_seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt = {"training_state": training_state, "norm_state": norm_state}
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.abspath(ckpt_dir), ckpt, force=True)
    checkpointer.wait_until_finished()

    # Save config metadata so record_video.py can reconstruct the network
    meta = {"env_name": env_name, "hidden_dim": list(hidden_dim),
            "obs_dim": obs_dim, "action_dim": action_dim}
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # Save metrics CSV
    if metrics_log:
        csv_path = os.path.join(ckpt_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_log[0].keys())
            writer.writeheader()
            writer.writerows(metrics_log)

    print(f"  Checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartpoleBalance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1)
    args = parser.parse_args()

    train(
        env_name=args.env,
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        hidden_dim=tuple(args.hidden_dim) if args.hidden_dim else None,
        entropy_coef=args.entropy_coef,
        episode_length=args.episode_length,
        log_interval=args.log_interval,
    )
