"""Debug FastDSAC NaN — isolate exact source of divergence.

Tests batch sizes 8K, 16K, 32K with real env data at 1024 envs.
For the failing batch size, instruments the loss function to find
the exact computation that produces NaN first.
"""

import os
import sys
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.7"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial

from jax_rl.algos.fast_dsac import FastDSAC
from jax_rl.configs.fast_dsac_config import FastDSACConfig
from jax_rl.configs.env_presets import get_fast_dsac_preset
from jax_rl.training import make_envs
from jax_rl.utils.normalization import (
    init as norm_init, update as norm_update, normalize as norm_normalize,
)
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer


def check_nan(x, name):
    """Check array for NaN, return True if found."""
    if jnp.any(jnp.isnan(x)):
        print(f"  NaN in {name}: shape={x.shape}, "
              f"nan_count={jnp.sum(jnp.isnan(x))}/{x.size}, "
              f"non-NaN range=[{jnp.nanmin(x):.4f}, {jnp.nanmax(x):.4f}]")
        return True
    return False


def main():
    cfg, dsac_cfg = get_fast_dsac_preset("HumanoidRun")
    obs_dim, action_dim = 67, 21

    # ── Step 1: Collect real data at 1024 envs ──
    print("=" * 70)
    print("STEP 1: Collecting real env data at 1024 envs")
    print("=" * 70)

    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, 0)
    norm_state = norm_init(obs_dim)
    buffer = JaxReplayBuffer(obs_dim, action_dim, max_size=dsac_cfg.buffer_size)

    for step in range(dsac_cfg.min_buffer_size // cfg.num_envs + 2):
        obs = env_state.obs
        norm_state = norm_update(norm_state, obs)
        key, ak = jax.random.split(key)
        action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1, maxval=1)
        env_state = env_step(env_state, action)
        buffer.add_batch(obs=obs, action=action, reward=env_state.reward,
                         next_obs=env_state.obs, done=env_state.done,
                         truncation=env_state.info.get("truncation", jnp.zeros_like(env_state.done)))

    print(f"Buffer: {len(buffer)} samples")
    var = norm_state.mean_of_squares - norm_state.mean**2
    print(f"Norm variance: [{var.min():.6f}, {var.max():.6f}]")
    print(f"Zero-var dims: {(var < 1e-6).sum()}/{obs_dim}")

    # ── Step 2: Test batch sizes ──
    print("\n" + "=" * 70)
    print("STEP 2: Testing batch sizes [8192, 16384, 32768]")
    print("=" * 70)

    failing_batch_size = None

    for bs in [8192, 16384, 32768]:
        print(f"\n--- batch_size={bs} ---")

        test_cfg = FastDSACConfig(**{**dsac_cfg.__dict__, "batch_size": bs})

        optimizer = optax.adamw(cfg.lr, b1=test_cfg.adam_b1, b2=test_cfg.adam_b2,
                                weight_decay=test_cfg.weight_decay)
        alpha_optimizer = optax.adamw(test_cfg.alpha_lr, b1=test_cfg.adam_b1,
                                      b2=test_cfg.adam_b2, weight_decay=test_cfg.weight_decay)

        dsac = FastDSAC(
            config=test_cfg, obs_dim=obs_dim, action_dim=action_dim,
            optimizer=optimizer, alpha_optimizer=alpha_optimizer,
            gamma=cfg.gamma, handle_truncation=True,
        )

        key, ik = jax.random.split(key)
        state = dsac.init(ik)

        nan_step = None
        for i in range(30):
            key, sk = jax.random.split(key)
            batch = buffer.sample(bs, key=sk)
            batch["obs"] = norm_normalize(norm_state, batch["obs"], eps=test_cfg.obs_norm_eps)
            batch["next_obs"] = norm_normalize(norm_state, batch["next_obs"], eps=test_cfg.obs_norm_eps)

            state, metrics = dsac.update(state, batch)
            q1 = float(metrics["q1_mean"])

            if np.isnan(q1):
                nan_step = i
                print(f"  NaN at step {i}!")
                break

            if i % 10 == 0:
                print(f"  Step {i}: Q1={q1:.4f}, Q1σ²={float(metrics['q1_var']):.4f}, "
                      f"ent={float(metrics['entropy']):.1f}, alpha={float(metrics['alpha']):.6f}")

        if nan_step is not None:
            failing_batch_size = bs
            print(f"  FAILED at batch_size={bs}")
            break
        else:
            print(f"  PASSED 30 steps at batch_size={bs}")

    if failing_batch_size is None:
        print("\nAll batch sizes passed! Cannot reproduce NaN.")
        return

    # ── Step 3: Instrument the failing batch size ──
    print("\n" + "=" * 70)
    print(f"STEP 3: Instrumenting NaN source at batch_size={failing_batch_size}")
    print("=" * 70)

    # Re-init fresh
    test_cfg = FastDSACConfig(**{**dsac_cfg.__dict__, "batch_size": failing_batch_size})
    optimizer = optax.adamw(cfg.lr, b1=test_cfg.adam_b1, b2=test_cfg.adam_b2,
                            weight_decay=test_cfg.weight_decay)
    alpha_optimizer = optax.adamw(test_cfg.alpha_lr, b1=test_cfg.adam_b1,
                                  b2=test_cfg.adam_b2, weight_decay=test_cfg.weight_decay)

    dsac = FastDSAC(
        config=test_cfg, obs_dim=obs_dim, action_dim=action_dim,
        optimizer=optimizer, alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma, handle_truncation=True,
    )

    key, ik = jax.random.split(key)
    state = dsac.init(ik)

    # Run until NaN, then on the NaN step, manually trace through the computation
    last_good_state = state
    last_good_batch = None

    for i in range(30):
        key, sk = jax.random.split(key)
        batch = buffer.sample(failing_batch_size, key=sk)
        batch["obs"] = norm_normalize(norm_state, batch["obs"], eps=test_cfg.obs_norm_eps)
        batch["next_obs"] = norm_normalize(norm_state, batch["next_obs"], eps=test_cfg.obs_norm_eps)

        prev_state = state
        state, metrics = dsac.update(state, batch)
        q1 = float(metrics["q1_mean"])

        if np.isnan(q1):
            print(f"\nNaN at step {i}. Tracing through the computation...")

            # Check the batch
            print("\n--- Batch check ---")
            for k, v in batch.items():
                check_nan(v, f"batch['{k}']")
                print(f"  batch['{k}']: range=[{v.min():.4f}, {v.max():.4f}], "
                      f"mean={v.mean():.4f}, std={v.std():.4f}")

            # Check prev state params
            print("\n--- Pre-update state ---")
            for name in ["actor_params", "q1_params", "q2_params",
                          "target_q1_params", "target_q2_params"]:
                params = getattr(prev_state, name)
                has = any(jnp.any(jnp.isnan(l)) for l in jax.tree.leaves(params))
                maxv = max(float(jnp.nanmax(jnp.abs(l))) for l in jax.tree.leaves(params))
                print(f"  {name}: NaN={has}, max_abs={maxv:.4f}")
            print(f"  log_alpha: {float(prev_state.log_alpha):.6f}")

            # Manually run forward passes to find where NaN appears
            print("\n--- Manual forward pass trace ---")

            # Actor forward on batch obs
            obs = batch["obs"]
            actor_params = prev_state.actor_params

            # Actor encoder + head
            enc_params, head_params = actor_params
            features = dsac.actor_enc.apply(enc_params, obs)
            check_nan(features, "actor_encoder(obs)")

            # Q network forward
            action = batch["action"]
            q1_out = dsac.q1.apply(prev_state.q1_params, obs, action)
            if isinstance(q1_out, tuple):
                q1_mean, q1_var = q1_out
                check_nan(q1_mean, "q1_mean")
                check_nan(q1_var, "q1_var")
                print(f"  q1_mean: [{q1_mean.min():.4f}, {q1_mean.max():.4f}]")
                print(f"  q1_var: [{q1_var.min():.6f}, {q1_var.max():.6f}]")

            # Target Q on next_obs
            next_obs = batch["next_obs"]

            # Actor forward for next action (with DEM)
            next_features = dsac.actor_enc.apply(enc_params, next_obs)
            check_nan(next_features, "actor_encoder(next_obs)")

            # Target Q
            tq1_out = dsac.q1.apply(prev_state.target_q1_params, next_obs, action)
            if isinstance(tq1_out, tuple):
                tq1_mean, tq1_var = tq1_out
                check_nan(tq1_mean, "target_q1_mean")
                check_nan(tq1_var, "target_q1_var")

            # Check post-update state
            print("\n--- Post-update state ---")
            for name in ["actor_params", "q1_params", "q2_params"]:
                params = getattr(state, name)
                has = any(jnp.any(jnp.isnan(l)) for l in jax.tree.leaves(params))
                maxv = max(float(jnp.nanmax(jnp.abs(l))) for l in jax.tree.leaves(params))
                print(f"  {name}: NaN={has}, max_abs={maxv:.4f}")
            print(f"  log_alpha: {float(state.log_alpha):.6f}")

            # Check metrics
            print("\n--- All metrics ---")
            for k, v in sorted(metrics.items()):
                print(f"  {k}: {float(v)}")

            break
    else:
        print(f"\nCompleted 30 steps without NaN! Cannot reproduce.")


if __name__ == "__main__":
    main()
