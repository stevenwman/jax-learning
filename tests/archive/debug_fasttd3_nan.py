"""Debug FastTD3 NaN on HumanoidRun.

Runs a short training with NaN checks at every critical point:
- After env.step: check obs, reward, done
- After select_action: check action
- After buffer.sample: check batch
- After algo.update: check all params, Q values, gradients
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.7")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.configs.train_config import TrainConfig
from jax_rl.configs.env_presets import get_fast_td3_preset
from jax_rl.training import make_envs, make_identity_norm_state
from jax_rl.utils.normalization import init as norm_init, update as norm_update, normalize as norm_normalize


def has_nan(x, name=""):
    """Check if any NaN in a pytree."""
    leaves = jax.tree.leaves(x)
    for i, leaf in enumerate(leaves):
        if jnp.any(jnp.isnan(leaf)):
            print(f"  NaN found in {name} leaf[{i}] shape={leaf.shape} dtype={leaf.dtype}")
            nan_count = jnp.sum(jnp.isnan(leaf))
            print(f"    NaN count: {nan_count}/{leaf.size}")
            print(f"    Non-NaN range: [{jnp.nanmin(leaf):.6f}, {jnp.nanmax(leaf):.6f}]")
            return True
    return False


def check_batch(batch, step):
    """Check a batch dict for NaN."""
    for k, v in batch.items():
        if jnp.any(jnp.isnan(v)):
            print(f"  Step {step}: NaN in batch['{k}'] shape={v.shape}")
            print(f"    NaN count: {jnp.sum(jnp.isnan(v))}/{v.size}")
            print(f"    Range: [{jnp.nanmin(v):.4f}, {jnp.nanmax(v):.4f}]")
            return True
    return False


def main():
    cfg, td3_cfg = get_fast_td3_preset("HumanoidRun")
    cfg = cfg.__class__(**{**cfg.__dict__, "total_timesteps": 2_000_000})

    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, 0)

    print(f"FastTD3 NaN Debug — HumanoidRun")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, num_envs={cfg.num_envs}")
    print(f"  tau={td3_cfg.tau}, hidden={td3_cfg.hidden_dim}, critic={td3_cfg.critic_hidden_dim}")
    print(f"  C51: atoms={td3_cfg.num_atoms}, v=[{td3_cfg.v_min}, {td3_cfg.v_max}]")
    print(f"  activation={td3_cfg.activation}, q_layer_norm={td3_cfg.q_layer_norm}")
    print()

    # Optimizer (paper: AdamW, no grad clip)
    lr_schedule = cfg.lr
    optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)

    td3 = FastTD3(
        config=td3_cfg, obs_dim=obs_dim, action_dim=action_dim,
        actor_optimizer=optimizer, critic_optimizer=optimizer,
        gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
    )

    key, init_key = jax.random.split(key)
    training_state = td3.init(init_key)

    # Obs normalization
    use_obs_norm = td3_cfg.obs_normalization
    norm_state = norm_init(obs_dim) if use_obs_norm else make_identity_norm_state(obs_dim)
    print(f"  obs_normalization={use_obs_norm}")

    buffer = JaxReplayBuffer(obs_dim, action_dim, max_size=td3_cfg.buffer_size)

    # Check initial params for NaN
    print("\n--- Initial param check ---")
    has_nan(training_state.actor_params, "actor_params")
    has_nan(training_state.q1_params, "q1_params")
    has_nan(training_state.q2_params, "q2_params")
    print("Initial params OK" if not has_nan(training_state, "training_state") else "INITIAL PARAMS HAVE NaN!")

    # Check initial obs
    print(f"\nInitial obs range: [{env_state.obs.min():.4f}, {env_state.obs.max():.4f}]")
    print(f"Initial obs mean: {env_state.obs.mean():.4f}, std: {env_state.obs.std():.4f}")

    print("\n--- Training loop ---")
    total_steps = 0
    nan_found = False

    for outer_step in range(cfg.total_timesteps // cfg.num_envs):
        total_steps = (outer_step + 1) * cfg.num_envs
        obs = env_state.obs

        # Check obs
        if jnp.any(jnp.isnan(obs)):
            print(f"\nStep {total_steps}: NaN in env obs!")
            has_nan(obs, "env_state.obs")
            nan_found = True
            break

        # Obs normalization
        if use_obs_norm:
            norm_state = norm_update(norm_state, obs)
            obs_for_action = norm_normalize(norm_state, obs, eps=td3_cfg.obs_norm_eps)
            if jnp.any(jnp.isnan(obs_for_action)):
                print(f"\nStep {total_steps}: NaN after obs normalization!")
                print(f"  Raw obs range: [{obs.min():.4f}, {obs.max():.4f}]")
                print(f"  Normalized range: [{jnp.nanmin(obs_for_action):.4f}, {jnp.nanmax(obs_for_action):.4f}]")
                print(f"  Norm state mean range: [{norm_state.mean.min():.6f}, {norm_state.mean.max():.6f}]")
                var = norm_state.mean_of_squares - norm_state.mean**2
                print(f"  Norm state var range: [{var.min():.6f}, {var.max():.6f}]")
                print(f"  Norm state count: {norm_state.count}")
                nan_found = True
                break
        else:
            obs_for_action = obs

        # Action selection
        if len(buffer) < td3_cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak, noise_key = jax.random.split(key, 3)
            if td3_cfg.noise_min is not None:
                noise_std = jax.random.uniform(noise_key, (), minval=td3_cfg.noise_min, maxval=td3_cfg.noise_max)
            else:
                noise_std = td3_cfg.exploration_noise_std
            action = td3.select_action(training_state.actor_params, obs_for_action, ak,
                                       deterministic=False, exploration_noise=noise_std)

        if jnp.any(jnp.isnan(action)):
            print(f"\nStep {total_steps}: NaN in action!")
            print(f"  Action range: [{jnp.nanmin(action):.4f}, {jnp.nanmax(action):.4f}]")
            print(f"  NaN count: {jnp.sum(jnp.isnan(action))}/{action.size}")
            has_nan(training_state.actor_params, "actor_params at NaN action")
            nan_found = True
            break

        # Env step
        env_state = env_step(env_state, action)
        truncation = env_state.info["truncation"] if cfg.handle_truncation else jnp.zeros_like(env_state.done)

        # Buffer add
        buffer.add_batch(obs=obs, action=action,
                         reward=env_state.reward * cfg.reward_scaling,
                         next_obs=env_state.obs, done=env_state.done,
                         truncation=truncation)

        # Gradient updates
        if len(buffer) >= td3_cfg.min_buffer_size:
            for grad_step in range(td3_cfg.grad_updates_per_step):
                key, sample_key = jax.random.split(key)
                jax_batch = buffer.sample(td3_cfg.batch_size, key=sample_key)

                if use_obs_norm:
                    jax_batch["obs"] = norm_normalize(norm_state, jax_batch["obs"], eps=td3_cfg.obs_norm_eps)
                    jax_batch["next_obs"] = norm_normalize(norm_state, jax_batch["next_obs"], eps=td3_cfg.obs_norm_eps)

                if check_batch(jax_batch, total_steps):
                    print(f"  (grad_step {grad_step})")
                    nan_found = True
                    break

                training_state, metrics = td3.update(training_state, jax_batch)

                # Check metrics
                q1 = float(metrics.get("q1_mean", 0))
                actor_loss = float(metrics.get("actor_loss", 0))
                if np.isnan(q1) or np.isnan(actor_loss):
                    print(f"\nStep {total_steps}, grad_step {grad_step}: NaN in metrics!")
                    print(f"  Q1={q1}, ActorLoss={actor_loss}")
                    print(f"  Checking params...")
                    has_nan(training_state.actor_params, "actor_params")
                    has_nan(training_state.q1_params, "q1_params")
                    has_nan(training_state.q2_params, "q2_params")
                    has_nan(training_state.target_q1_params, "target_q1_params")

                    # Check Q logits range
                    print(f"\n  Checking Q network outputs on batch...")
                    nan_found = True
                    break

            if nan_found:
                break

        # Periodic logging
        if outer_step % 10 == 0 and len(buffer) >= td3_cfg.min_buffer_size:
            q1 = float(metrics.get("q1_mean", 0))
            al = float(metrics.get("actor_loss", 0))
            obs_range = f"[{obs.min():.2f}, {obs.max():.2f}]"
            if use_obs_norm:
                nobs = norm_normalize(norm_state, obs, eps=td3_cfg.obs_norm_eps)
                nobs_range = f"[{nobs.min():.2f}, {nobs.max():.2f}]"
            else:
                nobs_range = obs_range

            # Check param magnitudes
            actor_max = max(float(jnp.max(jnp.abs(l))) for l in jax.tree.leaves(training_state.actor_params))
            q1_max = max(float(jnp.max(jnp.abs(l))) for l in jax.tree.leaves(training_state.q1_params))

            print(f"Step {total_steps:>10,} | Q1 {q1:>8.3f} | ActLoss {al:>8.3f} | "
                  f"obs {obs_range} | nobs {nobs_range} | "
                  f"actor_max {actor_max:.3f} | q1_max {q1_max:.3f}")

    if not nan_found:
        print(f"\nCompleted {total_steps:,} steps without NaN!")
    else:
        print(f"\nNaN detected at step {total_steps:,}")
        print("Dumping final state for analysis...")
        print(f"  norm_state.count: {norm_state.count}")
        print(f"  norm_state.mean range: [{norm_state.mean.min():.6f}, {norm_state.mean.max():.6f}]")
        var = norm_state.mean_of_squares - norm_state.mean**2
        print(f"  norm_state.var range: [{var.min():.6f}, {var.max():.6f}]")


if __name__ == "__main__":
    main()
