"""PPO training on MuJoCo Playground CartpoleBalance.

Wires together: env, collect loop, GAE, PPO update, obs normalization.
Uses wrap_for_brax_training for vectorized auto-reset + truncation tracking.

Truncation handling:
  Playground envs signal timeout via info['truncation']. For GAE we need to
  distinguish true terminals (done=1, trunc=0) from timeouts (done=1, trunc=1).
  We pass effective_done = done * (1 - truncation) to the buffer so GAE
  bootstraps through timeouts instead of zeroing the value estimate.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

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


def train(
    seed: int = 0,
    num_envs: int = 64,
    num_steps: int = 64,
    total_timesteps: int = 1_000_000,
    episode_length: int = 1000,
    log_interval: int = 1,
):
    """Run PPO training on CartpoleBalance.

    Args:
        seed: Random seed
        num_envs: Number of parallel environments
        num_steps: Rollout length per iteration
        total_timesteps: Total environment steps
        episode_length: Max steps per episode before truncation
        log_interval: Print metrics every N iterations
    """
    # ── Environment ──────────────────────────────────────────────────────
    env = dm_control_suite.load("CartpoleBalance")
    env = wrap_for_brax_training(env, episode_length=episode_length)
    env_step = jax.jit(env.step)  # Fuse MJX physics into one GPU kernel

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, num_envs))

    obs_dim = env_state.obs.shape[-1]  # 5 for CartpoleBalance
    action_dim = env.action_size       # 1 for CartpoleBalance

    samples_per_iter = num_envs * num_steps
    num_iterations = total_timesteps // samples_per_iter

    print("=" * 80)
    print("PPO — CartpoleBalance (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={num_envs}, num_steps={num_steps}, episode_length={episode_length}")
    print(f"  samples/iter={samples_per_iter:,}, iterations={num_iterations}, total_steps={total_timesteps:,}")

    # ── PPO setup ────────────────────────────────────────────────────────
    config = PPOConfig(
        encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=(64, 64)),
        policy_head=PolicyHeadConfig(action_dim=action_dim, squash=False),
        num_envs=num_envs,
        num_steps=num_steps,
        minibatch_size=min(256, num_envs * num_steps),
        num_epochs=4,
        actor_lr=3e-4,
        critic_lr=3e-4,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
    )

    num_minibatches = samples_per_iter // config.minibatch_size
    print(f"  minibatch_size={config.minibatch_size}, num_epochs={config.num_epochs}, "
          f"grad_updates/iter={num_minibatches * config.num_epochs}")
    print(f"  actor_lr={config.actor_lr}, critic_lr={config.critic_lr}, "
          f"clip_eps={config.clip_eps}, entropy_coef={config.entropy_coef}")
    print(f"  gamma={config.gamma}, gae_lambda={config.gae_lambda}")

    ppo = PPO(config, obs_dim, action_dim)
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

    print("=" * 80)
    if completed_returns:
        final = completed_returns[-100:]
        print(f"Training complete.")
        print(f"  Total episodes: {len(completed_returns)}")
        print(f"  Final avg return (last 100 eps): {np.mean(final):.1f}")
        print(f"  Final max return (last 100 eps): {np.max(final):.1f}")
        print(f"  Target: >= 950")
    else:
        print("Done. No episodes completed.")


if __name__ == "__main__":
    train()
