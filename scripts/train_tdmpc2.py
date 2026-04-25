"""TD-MPC2 training script (standalone, not using offpolicy_loop).

Usage:
    uv run python train_tdmpc2.py --env CheetahRun
    uv run python train_tdmpc2.py --env HumanoidRun --seed 42
    uv run python train_tdmpc2.py --env CheetahRun --total-timesteps 0  # init-only smoke
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
sys.stdout.reconfigure(line_buffering=True)

import argparse
import csv
import dataclasses
import json
import shutil
from dataclasses import asdict

import numpy as np
import jax
import jax.numpy as jnp

from jax_rl.algos.tdmpc2 import (
    TDMPC2State,
    make_plan_batched, make_update_step,
)
from jax_rl.algos.tdmpc2_runtime import (
    build_modules, init_train_state, run_eval,
    build_train_config_from_tdmpc2, _pipe_obs,
)
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.tdmpc2_config import TDMPC2Config
from jax_rl.configs.env_presets import get_tdmpc2_preset
from jax_rl.training.env_setup import make_env_bundle


def run_warmup(
    state: "TDMPC2State",
    env_bundle,
    buffer,
    update_step,
    cfg: "TDMPC2Config",
    key: jax.Array,
) -> tuple:
    """Run seed_steps random-action collect, then seed_steps gradient updates.

    Source: /tmp/tdmpc2/tdmpc2/trainer/online_trainer.py:107-122.

    Returns: (state, env_state, key, episode_ids_per_env, prev_done_or_trunc)
             — the post-warmup handoff tuple.
    """
    env_step = env_bundle.env_step
    env_state = env_bundle.env_state
    dict_obs = env_bundle.dict_obs
    num_envs = cfg.num_envs
    action_dim = cfg.action_dim

    print(f"[tdmpc2] warmup: {cfg.seed_steps} random-action collect steps")
    episode_ids_per_env = jnp.zeros(num_envs, dtype=jnp.int32)
    prev_done_or_trunc = jnp.zeros(num_envs, dtype=jnp.bool_)

    for step in range(cfg.seed_steps):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(
            subkey, (num_envs, action_dim), minval=-1.0, maxval=1.0
        )
        obs = _pipe_obs(env_state.obs, dict_obs)
        env_state = env_step(env_state, action)
        next_obs = _pipe_obs(env_state.obs, dict_obs)
        reward = env_state.reward
        done = env_state.done
        if hasattr(env_state, "info") and isinstance(env_state.info, dict):
            truncation = env_state.info.get("truncation", jnp.zeros_like(done))
        else:
            truncation = jnp.zeros_like(done)

        # Increment episode_id on PREVIOUS done-or-trunc
        episode_ids_now = episode_ids_per_env + prev_done_or_trunc.astype(jnp.int32)

        buffer.add_batch(
            obs=np.asarray(obs),
            action=np.asarray(action),
            reward=np.asarray(reward),
            next_obs=np.asarray(next_obs),
            done=np.asarray(done),
            truncation=np.asarray(truncation),
            episode_ids=np.asarray(episode_ids_now),
        )
        prev_done_or_trunc = done.astype(jnp.bool_) | truncation.astype(jnp.bool_)
        episode_ids_per_env = episode_ids_now

    print(f"[tdmpc2] warmup collect done; buffer size={buffer.size}")

    # Gradient burst: only if buffer has enough for a valid sequence window
    min_needed = cfg.horizon + 1
    if buffer.size >= min_needed:
        # Source-faithful: gradient burst count = number of env transitions collected
        # = cfg.seed_steps * num_envs (1:1 grad-to-env-step ratio matching source's
        # warmup of seed_steps env-steps + seed_steps grad updates).
        burst_count = cfg.seed_steps * cfg.num_envs
        print(f"[tdmpc2] warmup: {burst_count} gradient updates "
              f"({cfg.seed_steps} seed_steps × {cfg.num_envs} num_envs)")
        burst_log_every = max(1, burst_count // 20)
        for step in range(burst_count):
            key, batch_key = jax.random.split(key)
            batch = buffer.sample_sequence(
                cfg.batch_size, cfg.horizon, batch_key, stride=cfg.num_envs,
            )
            state, metrics = update_step(state, batch)
            if step % burst_log_every == 0:
                print(
                    f"[tdmpc2]   burst step {step}/{burst_count} "
                    f"L_world={float(metrics['L_world_total']):.4f} "
                    f"L_policy={float(metrics['L_policy']):.4f}"
                )
    else:
        print(
            f"[tdmpc2] warmup: skipping gradient burst — buffer size {buffer.size} "
            f"< horizon+1 ({min_needed})"
        )

    print(f"[tdmpc2] warmup complete; state.step={int(state.step)}")
    return state, env_state, key, episode_ids_per_env, prev_done_or_trunc


def run_main_loop(
    state: "TDMPC2State",
    env_bundle,
    env_state,  # post-warmup env state (not env_bundle.env_state, which is stale)
    buffer,
    update_step,
    plan_fn,
    modules: tuple,
    cfg: "TDMPC2Config",
    key: jax.Array,
    start_env_step: int,
    episode_ids: jax.Array,
    prev_done_or_trunc: jax.Array,
    total_timesteps: int,
    seed: int,                              # for eval RNG isolation (see eval branch)
    env_name: str = "",
    ckpt_dir: str | None = None,
    use_wandb: bool = False,
    wandb_project: str = "jax-rl-tdmpc2",
) -> "TDMPC2State":
    """Main TD-MPC2 training loop: MPPI/prior collect + UTD=1 updates.

    Source: /tmp/tdmpc2/tdmpc2/trainer/online_trainer.py:113-135.

    Args:
        start_env_step: env steps already consumed (i.e. seed_steps). Main loop
                        starts here and increments by num_envs per outer iteration.
    Returns:
        final TDMPC2State after all training steps.
    """
    encoder, dynamics, reward_net, q_ensemble, policy = modules
    env_step = env_bundle.env_step
    dict_obs = env_bundle.dict_obs
    num_envs = cfg.num_envs
    action_dim = cfg.action_dim

    # How often to print: ~100 lines total across the full run
    outer_steps_total = max(1, (total_timesteps - start_env_step) // num_envs)
    log_every = max(1, outer_steps_total // 100)

    # Dedicated eval RNG (NOT consumed from training key). Each eval folds in its
    # eval_index so the eval cadence (cfg.eval_every) cannot perturb the training
    # PRNG stream — runs with different eval_every become byte-identical in training.
    eval_base_key = jax.random.PRNGKey(seed + 9000)

    step_counter = start_env_step  # counts total env steps consumed so far
    outer_idx = 0
    metrics = {}  # last update_step metrics (needed even if no update ran yet)

    # Checkpointing + metrics tracking
    best_eval_tracker = [-float("inf")]  # list for mutability across iterations
    metrics_csv_path = None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        metrics_csv_path = _init_metrics_csv(ckpt_dir, METRICS_HEADER)

    print(
        f"[tdmpc2] main loop: {start_env_step:_} → {total_timesteps:_} env steps, "
        f"collect_mode={cfg.collect_mode}, utd={cfg.utd}"
    )

    while step_counter < total_timesteps:
        # ------------------------------------------------------------------ #
        # 1. Action selection
        # ------------------------------------------------------------------ #
        obs = _pipe_obs(env_state.obs, dict_obs)  # (num_envs, obs_dim)
        z_0 = encoder.apply(state.encoder_params, obs)  # (num_envs, latent_dim)

        if cfg.collect_mode == "mppi":
            plan_params = {
                "encoder": state.encoder_params,
                "dynamics": state.dynamics_params,
                "reward": state.reward_params,
                "q_ensemble": state.q_ensemble_params,
                "policy": state.policy_params,
            }
            t0_b = prev_done_or_trunc  # (num_envs,) bool
            key, plan_key = jax.random.split(key)
            plan_keys = jax.random.split(plan_key, num_envs)
            action, new_prev_mean = plan_fn(
                plan_params, z_0, state.prev_mean, t0_b, cfg, plan_keys, False
            )
            state = state.replace(prev_mean=new_prev_mean)
        else:  # "prior"
            key, pi_key = jax.random.split(key)
            pi_keys = jax.random.split(pi_key, num_envs)

            def _sample_prior(z, k):
                a, _ = policy.apply(state.policy_params, z[None, :], k)
                return a[0]

            action = jax.vmap(_sample_prior, in_axes=(0, 0))(z_0, pi_keys)

        # ------------------------------------------------------------------ #
        # 2. Env step
        # ------------------------------------------------------------------ #
        env_state = env_step(env_state, action)
        next_obs = _pipe_obs(env_state.obs, dict_obs)
        reward = env_state.reward
        done = env_state.done
        if hasattr(env_state, "info") and isinstance(env_state.info, dict):
            truncation = env_state.info.get("truncation", jnp.zeros_like(done))
        else:
            truncation = jnp.zeros_like(done)

        # ------------------------------------------------------------------ #
        # 3. Buffer add
        # ------------------------------------------------------------------ #
        episode_ids_now = episode_ids + prev_done_or_trunc.astype(jnp.int32)
        buffer.add_batch(
            obs=np.asarray(obs),
            action=np.asarray(action),
            reward=np.asarray(reward),
            next_obs=np.asarray(next_obs),
            done=np.asarray(done),
            truncation=np.asarray(truncation),
            episode_ids=np.asarray(episode_ids_now),
        )

        # ------------------------------------------------------------------ #
        # 4. Gradient updates: cfg.utd is per ENV-STEP (source semantics).
        # With num_envs > 1, we run cfg.utd * num_envs updates per outer step
        # to maintain source's 1 update/env-step ratio. Otherwise effective
        # UTD scales as 1/num_envs and training starves (8× deficit at num_envs=8).
        # ------------------------------------------------------------------ #
        for _ in range(cfg.utd * num_envs):
            key, batch_key = jax.random.split(key)
            batch = buffer.sample_sequence(
                cfg.batch_size, cfg.horizon, batch_key, stride=num_envs,
            )
            state, metrics = update_step(state, batch)

        # ------------------------------------------------------------------ #
        # 5. Update trackers for next iteration
        # ------------------------------------------------------------------ #
        prev_done_or_trunc = done.astype(jnp.bool_) | truncation.astype(jnp.bool_)
        episode_ids = episode_ids_now
        step_counter += num_envs
        outer_idx += 1

        # ------------------------------------------------------------------ #
        # 6. Periodic eval (MPPI + prior modes)
        # ------------------------------------------------------------------ #
        # Window check (not exact %): step_counter increments by num_envs, so it
        # rarely lands exactly on cfg.eval_every. Fire once per eval_every window.
        prev_step = step_counter - num_envs
        crossed_eval = (prev_step // cfg.eval_every) < (step_counter // cfg.eval_every)
        if crossed_eval or step_counter >= total_timesteps:
            eval_index = (step_counter - 1) // cfg.eval_every
            eval_key = jax.random.fold_in(eval_base_key, eval_index)
            eval_metrics = run_eval(state, env_bundle, plan_fn, modules, cfg, eval_key)

            # Pull per-h consistency tensor (length H) and unpack first 3 indices into named cols
            consistency_per_h = metrics.get("consistency_per_h", None)
            latent_err = {}
            if consistency_per_h is not None:
                arr = np.asarray(consistency_per_h)
                for h in range(min(3, arr.shape[0])):
                    latent_err[f"latent_err_h{h}"] = float(arr[h])

            combined_metrics = {
                "step": step_counter,
                **eval_metrics,
                "L_world_total": float(metrics.get("L_world_total", 0.0)),
                "L_policy": float(metrics.get("L_policy", 0.0)),
                "L_consistency_raw": float(metrics.get("L_consistency_raw", 0.0)),
                "L_reward_raw": float(metrics.get("L_reward_raw", 0.0)),
                "L_value_raw": float(metrics.get("L_value_raw", 0.0)),
                "q_scale_range_ema": float(metrics.get("q_scale_range_ema", 1.0)),
                # Tier B
                "wm_grad_norm": float(metrics.get("wm_grad_norm", 0.0)),
                "pi_grad_norm": float(metrics.get("pi_grad_norm", 0.0)),
                "pi_entropy": float(metrics.get("pi_entropy", 0.0)),
                "scaled_entropy_mean": float(metrics.get("scaled_entropy_mean", 0.0)),
                "max_reward_observed": float(metrics.get("max_reward_observed", 0.0)),
                "q_p5_batch": float(metrics.get("q_p5_batch", 0.0)),
                "q_p95_batch": float(metrics.get("q_p95_batch", 0.0)),
                **latent_err,
            }

            is_best = eval_metrics["mppi_return"] > best_eval_tracker[0]
            if is_best:
                best_eval_tracker[0] = eval_metrics["mppi_return"]

            if ckpt_dir:
                _append_metrics_row(metrics_csv_path, combined_metrics, METRICS_HEADER)
                _save_checkpoint(
                    ckpt_dir, state, cfg, env_name, step_counter,
                    best_eval_tracker[0], is_best=is_best,
                )

            if use_wandb:
                import wandb
                wandb.log(combined_metrics, step=step_counter)

            print(
                f"[tdmpc2] step={step_counter:_} EVAL "
                f"mppi={eval_metrics['mppi_return']:.2f} "
                f"prior={eval_metrics['prior_return']:.2f} "
                f"gap={eval_metrics['mppi_prior_gap']:+.2f}"
                + (" [best]" if is_best else "")
            )

        # ------------------------------------------------------------------ #
        # 7. Periodic logging
        # ------------------------------------------------------------------ #
        if outer_idx % log_every == 0 and metrics:
            print(
                f"[tdmpc2] step={step_counter:_} "
                f"L_world={float(metrics['L_world_total']):.4f} "
                f"L_policy={float(metrics['L_policy']):.4f}"
            )

    print(f"[tdmpc2] main loop complete; total_env_steps={step_counter:_}")
    return state


METRICS_HEADER = [
    # Step
    "step",
    # Eval (paper-comparable + diagnostic)
    "mppi_return", "prior_return", "mppi_prior_gap",
    # Loss components (raw, pre-coefficient)
    "L_world_total", "L_policy", "L_consistency_raw", "L_reward_raw", "L_value_raw",
    # Q scale
    "q_scale_range_ema", "q_p5_batch", "q_p95_batch",
    # Tier B: gradient norms (pre-clip)
    "wm_grad_norm", "pi_grad_norm",
    # Tier B: policy entropy + saturation watch
    "pi_entropy", "scaled_entropy_mean", "max_reward_observed",
    # Tier B: per-h consistency (cfg.horizon=3 → 3 cols; H=H reads first 3 of consistency_per_h)
    "latent_err_h0", "latent_err_h1", "latent_err_h2",
]


def _flatten_params_for_save(params):
    """Convert Flax params tree to a dict of numpy arrays keyed by dotted path."""
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        key = ".".join(
            str(p.key) if hasattr(p, "key") else str(p)
            for p in path
        )
        out[key] = np.asarray(leaf)
    return out


def _save_checkpoint(
    ckpt_dir: str,
    state: "TDMPC2State",
    cfg: "TDMPC2Config",
    env_name: str,
    step: int,
    best_eval: float,
    is_best: bool = False,
):
    """Save actor_params.npz + world_model_params.npz + meta.json to ckpt_dir.
    If is_best, also mirror into ckpt_dir/best/.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # Actor (policy) — deployable
    actor_flat = _flatten_params_for_save(state.policy_params)
    np.savez(os.path.join(ckpt_dir, "actor_params.npz"), **actor_flat)

    # World model — for MPPI at inference
    wm_params = {
        "encoder": state.encoder_params,
        "dynamics": state.dynamics_params,
        "reward": state.reward_params,
        "q_ensemble": state.q_ensemble_params,
    }
    wm_flat = _flatten_params_for_save(wm_params)
    np.savez(os.path.join(ckpt_dir, "world_model_params.npz"), **wm_flat)

    # Meta
    meta = {
        "env_name": env_name,
        "step": int(step),
        "best_eval": float(best_eval),
        "cfg": asdict(cfg),
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if is_best:
        best_dir = os.path.join(ckpt_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        for fname in ("actor_params.npz", "world_model_params.npz", "meta.json"):
            shutil.copy(
                os.path.join(ckpt_dir, fname),
                os.path.join(best_dir, fname),
            )


def _init_metrics_csv(ckpt_dir: str, header_cols: list[str]) -> str:
    """Create metrics.csv with header if it doesn't exist. Returns file path."""
    path = os.path.join(ckpt_dir, "metrics.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header_cols)
    return path


def _append_metrics_row(path: str, row: dict, header_cols: list[str]):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([row.get(k, "") for k in header_cols])


def train(
    cfg: TDMPC2Config,
    env_name: str,
    total_timesteps: int,
    seed: int = 0,
    ckpt_dir: str | None = None,
    use_wandb: bool = False,
    wandb_project: str = "jax-rl-tdmpc2",
):
    print(f"[tdmpc2] env={env_name} total_timesteps={total_timesteps:_} seed={seed}")
    print(f"[tdmpc2] cfg: latent_dim={cfg.latent_dim} horizon={cfg.horizon} "
          f"num_envs={cfg.num_envs} batch_size={cfg.batch_size} "
          f"discount={cfg.discount:.4f} action_dim={cfg.action_dim}")

    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=f"tdmpc2-{env_name}-s{seed}",
            config={**asdict(cfg), "env_name": env_name, "seed": seed},
        )

    # Build TrainConfig adapter for env bundle
    train_cfg = build_train_config_from_tdmpc2(cfg, env_name, total_timesteps, seed)
    env_bundle = make_env_bundle(train_cfg, seed)
    print(f"[tdmpc2] obs_dim={env_bundle.obs_dim} action_dim={env_bundle.action_dim}")

    if env_bundle.action_dim != cfg.action_dim:
        raise ValueError(
            f"Env action_dim={env_bundle.action_dim} but cfg.action_dim={cfg.action_dim}. "
            f"Update the preset or pass --action-dim."
        )

    # Replay buffer (episode_ids supported from Phase B)
    buffer = JaxReplayBuffer(
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        max_size=cfg.buffer_size,
    )
    print(f"[tdmpc2] buffer size={cfg.buffer_size:_}")

    # Train state + optimizers
    state, wm_opt, pol_opt = init_train_state(cfg, env_bundle.obs_dim, seed)
    print(f"[tdmpc2] state initialized; step={int(state.step)}")

    # Build update_step and plan_batched (for H2-H4 use; not called here)
    encoder, dynamics, reward_net, q_ensemble, policy = build_modules(cfg)
    update_step = make_update_step(
        cfg, wm_opt, pol_opt,
        encoder=encoder, dynamics=dynamics, reward_net=reward_net,
        q_ensemble_net=q_ensemble, policy_net=policy,
    )
    plan_fn = make_plan_batched(
        dynamics=dynamics, reward_net=reward_net,
        q_ensemble_net=q_ensemble, policy_net=policy,
    )
    print(f"[tdmpc2] update_step and plan_fn built")

    if total_timesteps == 0:
        print(f"[tdmpc2] --total-timesteps=0 → init-only smoke, exiting cleanly")
        return state

    # H2: warmup (random collect + gradient burst)
    key = jax.random.PRNGKey(seed + 42)  # separate from init key
    state, env_state, key, episode_ids, prev_done_or_trunc = run_warmup(
        state, env_bundle, buffer, update_step, cfg, key,
    )

    # H3: main loop
    modules = (encoder, dynamics, reward_net, q_ensemble, policy)
    state = run_main_loop(
        state, env_bundle, env_state, buffer, update_step, plan_fn, modules, cfg,
        key, start_env_step=cfg.seed_steps,
        episode_ids=episode_ids, prev_done_or_trunc=prev_done_or_trunc,
        total_timesteps=total_timesteps,
        seed=seed,
        env_name=env_name,
        ckpt_dir=ckpt_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
    )
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        help="Env name (CheetahRun, HumanoidRun, AcrobotSwingup, ...)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override TDMPC2Config.num_envs (default: 8 from preset).")
    parser.add_argument("--collect-mode", type=str, default=None,
                        choices=["mppi", "prior"],
                        help="Override TDMPC2Config.collect_mode (default: mppi).")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="jax-rl-tdmpc2")
    args = parser.parse_args()

    cfg = get_tdmpc2_preset(args.env)
    # Apply CLI overrides
    overrides = {}
    if args.num_envs is not None: overrides["num_envs"] = args.num_envs
    if args.collect_mode is not None: overrides["collect_mode"] = args.collect_mode
    if args.eval_every is not None: overrides["eval_every"] = args.eval_every
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    train(
        cfg, args.env, args.total_timesteps,
        seed=args.seed, ckpt_dir=args.ckpt_dir,
        use_wandb=args.wandb, wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
