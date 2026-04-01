"""Test determinism of env and full training pipeline.

Test 1: Env determinism — same seed, same actions → same trajectory?
Test 2: Full train.py determinism — same seed → same metrics at every iteration?
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mujoco_playground import dm_control_suite
from jax_rl.envs.wrappers import wrap_for_training

from jax_rl.algos.ppo import PPO
from jax_rl.buffers import RolloutBuffer
from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.utils.normalization import (
    init as norm_init,
    update as norm_update,
    normalize as norm_normalize,
)


def test_env_determinism():
    """Check if env.reset and env.step are deterministic."""
    print("=" * 70)
    print("TEST 1: Environment determinism")
    print("=" * 70)

    num_envs = 64
    num_steps = 100

    results = []
    for run in range(3):
        env = dm_control_suite.load("CartpoleBalance")
        env = wrap_for_training(env, episode_length=1000)
        env_step = jax.jit(env.step)

        key = jax.random.PRNGKey(0)
        key, reset_key = jax.random.split(key)
        state = env.reset(jax.random.split(reset_key, num_envs))

        obs_trace = [np.array(jax.device_get(state.obs))]
        reward_trace = []

        for step in range(num_steps):
            key, action_key = jax.random.split(key)
            action = jax.random.uniform(action_key, (num_envs, env.action_size), minval=-1, maxval=1)
            state = env_step(state, action)
            obs_trace.append(np.array(jax.device_get(state.obs)))
            reward_trace.append(np.array(jax.device_get(state.reward)))

        obs_trace = np.stack(obs_trace)
        reward_trace = np.stack(reward_trace)
        results.append((obs_trace, reward_trace))
        print(f"  Run {run + 1}: obs hash={hash(obs_trace.tobytes())}, "
              f"reward sum={reward_trace.sum():.6f}")

    # Compare
    all_match = True
    for i in range(1, len(results)):
        obs_match = np.array_equal(results[0][0], results[i][0])
        rew_match = np.array_equal(results[0][1], results[i][1])
        if not obs_match or not rew_match:
            all_match = False
            # Find first divergence
            for step in range(num_steps + 1):
                if not np.array_equal(results[0][0][step], results[i][0][step]):
                    print(f"  DIVERGE: Run 1 vs Run {i+1} at step {step} (obs)")
                    print(f"    Run 1: {results[0][0][step][0][:5]}")
                    print(f"    Run {i+1}: {results[i][0][step][0][:5]}")
                    break

    if all_match:
        print("  RESULT: Environment is BIT-IDENTICAL across runs ✓")
    else:
        print("  RESULT: Environment is NON-DETERMINISTIC ✗")
    return all_match


def run_full_training(seed=0, num_iters=50):
    """Exact copy of train.py logic — not a simplified version."""
    env = dm_control_suite.load("CartpoleBalance")
    env = wrap_for_training(env, episode_length=1000)
    env_step = jax.jit(env.step)

    num_envs = 64
    num_steps = 64

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, num_envs))

    obs_dim = env_state.obs.shape[-1]
    action_dim = env.action_size

    config = PPOConfig(
        encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=(64, 64)),
        critic_encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=(64, 64)),
        policy_head=PolicyHeadConfig(action_dim=action_dim, squash=False),
        num_envs=num_envs,
        minibatch_size=min(256, num_envs * num_steps),
        num_epochs=4,
        entropy_coef=0.01,
    )
    gamma = 0.99
    gae_lambda = 0.95

    actor_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    critic_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    ppo = PPO(config, obs_dim, action_dim, actor_opt, critic_opt)
    key, init_key = jax.random.split(key)
    training_state = ppo.init(init_key)
    norm_state = norm_init(obs_dim)

    episode_rewards = np.zeros(num_envs)
    completed_returns = []

    # Per-iteration tracking
    iter_policy_losses = []
    iter_value_losses = []
    iter_returns = []

    for iteration in range(num_iters):
        buffer = RolloutBuffer(num_steps, num_envs, obs_dim, action_dim)

        for step in range(num_steps):
            obs = env_state.obs
            norm_state = norm_update(norm_state, obs)
            normed_obs = norm_normalize(norm_state, obs)

            key, action_key = jax.random.split(key)
            action, log_prob, value = ppo.select_action(
                training_state, normed_obs, action_key
            )
            clipped_action = jnp.clip(action, -1.0, 1.0)
            env_state = env_step(env_state, clipped_action)

            truncation = env_state.info["truncation"]
            effective_done = env_state.done * (1.0 - truncation)

            buffer.add(
                obs=normed_obs, action=action, reward=env_state.reward,
                done=effective_done, truncation=truncation,
                log_prob=log_prob, value=value,
            )

            step_rewards = np.asarray(env_state.reward)
            step_dones = np.asarray(env_state.done)
            episode_rewards += step_rewards
            for i in range(num_envs):
                if step_dones[i]:
                    completed_returns.append(episode_rewards[i])
                    episode_rewards[i] = 0.0

        normed_next_obs = norm_normalize(norm_state, env_state.obs)
        key, bootstrap_key = jax.random.split(key)
        _, _, next_value = ppo.select_action(
            training_state, normed_next_obs, bootstrap_key, deterministic=True
        )

        batch = buffer.get(next_value, gamma=gamma, gae_lambda=gae_lambda)
        key, update_key = jax.random.split(key)
        training_state, metrics = ppo.update(training_state, batch, update_key, next_obs=normed_next_obs)

        pl = float(jax.device_get(metrics["policy_loss"]))
        vl = float(jax.device_get(metrics["value_loss"]))
        avg_ret = float(np.mean(completed_returns[-100:])) if completed_returns else 0.0

        iter_policy_losses.append(pl)
        iter_value_losses.append(vl)
        iter_returns.append(avg_ret)

    return (np.array(iter_policy_losses), np.array(iter_value_losses),
            np.array(iter_returns))


def test_training_determinism():
    """Run full training 3 times, compare everything."""
    print("\n" + "=" * 70)
    print("TEST 2: Full training determinism (mirrors train.py exactly)")
    print("=" * 70)

    num_runs = 3
    num_iters = 50

    all_pl = []
    all_vl = []
    all_ret = []

    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)
        pl, vl, ret = run_full_training(seed=0, num_iters=num_iters)
        all_pl.append(pl)
        all_vl.append(vl)
        all_ret.append(ret)
        print(f"done (ploss={pl[-1]:.6f}, vloss={vl[-1]:.4f}, ret={ret[-1]:.1f})")

    all_pl = np.array(all_pl)
    all_vl = np.array(all_vl)
    all_ret = np.array(all_ret)

    print(f"\n{'Iter':>5} | {'PL Run1':>12} | {'PL Run2':>12} | {'PL Run3':>12} | {'Max Diff':>12}")
    print("-" * 70)

    first_diverge = None
    for i in range(num_iters):
        vals = all_pl[:, i]
        max_diff = vals.max() - vals.min()
        if max_diff > 0 and first_diverge is None:
            first_diverge = i
        if i < 5 or i == first_diverge or i >= num_iters - 3:
            match = "✓" if max_diff == 0.0 else f"{max_diff:.2e}"
            print(f"{i:5d} | {vals[0]:12.8f} | {vals[1]:12.8f} | {vals[2]:12.8f} | {match}")

    if first_diverge is not None:
        print(f"\n  RESULT: DIVERGENCE at iteration {first_diverge} ✗")
    else:
        print(f"\n  RESULT: BIT-IDENTICAL across all {num_iters} iterations ✓")

    print(f"\n  Final returns: {all_ret[:, -1]}")


if __name__ == "__main__":
    env_ok = test_env_determinism()
    test_training_determinism()
