"""Single off-policy training loop — shared across SAC, TD3, FastSAC, FastTD3.

Each per-algo script builds its optimizer + algo + explore closure, then calls
this function. FlashSAC does NOT use this helper (BN state, Zeta noise, and
reward normalization don't fit the shared shape).

Variation points (the 4 things each script passes in):
    - algo: already-constructed SAC/TD3/FastSAC/FastTD3 with optimizer bound
    - explore_fn: (actor_params, obs, key) -> action — handles any noise injection
    - log_extra_fields: algo-specific metrics for stdout logging
    - log_extra_keys: algo-specific metrics for W&B CSV rows
"""
import dataclasses
import os
import time
from datetime import datetime
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from jax_rl.configs.train_config import TrainConfig
from jax_rl.training.checkpointing import CheckpointManager, load_checkpoint
from jax_rl.training.env_setup import EnvBundle
from jax_rl.training.episode_tracker import EpisodeTracker
from jax_rl.training.eval_runner import maybe_eval_and_checkpoint, final_eval_and_checkpoint
from jax_rl.training.metrics_logger import (
    log_training_step, make_metrics_row,
    wandb_init, wandb_setup_metrics, wandb_log, wandb_finish,
)
from jax_rl.envs.locomotion.curriculum_logging import (
    log_terrain_metrics, log_terrain_image, print_curriculum_dump,
)
from jax_rl.training.obs_pipeline import ObsPipeline
from jax_rl.training.train_context import TrainContext


def run_offpolicy_loop(
    cfg: TrainConfig,
    algo_cfg,
    algo,
    algo_name: str,
    env_bundle: EnvBundle,
    explore_fn: Callable,
    log_extra_fields: list,
    log_extra_keys: list,
    seed: int = 0,
    resume: str | None = None,
    resume_warmup: str = "policy",
    use_wandb: bool = False,
    wandb_project: str = "jax-rl",
) -> None:
    """Run the off-policy training loop.

    See train_sac.py, train_td3.py, train_fast_sac.py, train_fast_td3.py for
    concrete usage. FlashSAC does NOT use this helper — it has algo-specific
    loop state (BN stats, Zeta noise, adaptive reward scaling) that don't fit.
    """
    # ── Unpack bundle ──────────────────────────────────────────────────────
    env = env_bundle.env
    env_step = env_bundle.env_step
    env_state = env_bundle.env_state
    eval_env = env_bundle.eval_env
    obs_dim = env_bundle.obs_dim
    action_dim = env_bundle.action_dim
    critic_obs_dim = env_bundle.critic_obs_dim
    has_privileged = env_bundle.has_privileged
    dict_obs = env_bundle.dict_obs
    key = env_bundle.key

    total_env_steps = cfg.total_timesteps

    # ── Banner ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"{algo_name.upper()} — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, episode_length={cfg.episode_length}")
    print(f"  total_timesteps={total_env_steps:,}")
    print(f"  buffer_size={algo_cfg.buffer_size:,}, min_buffer={algo_cfg.min_buffer_size:,}")
    print(f"  batch_size={algo_cfg.batch_size}, grad_updates_per_step={algo_cfg.grad_updates_per_step}")
    print(f"  tau={algo_cfg.tau}, lr={cfg.lr}, gamma={cfg.gamma}")
    print(f"  reward_scaling={cfg.reward_scaling}")

    # ── Timestamp (shared by checkpoint dir + W&B run name) ────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")

    # ── W&B (optional) ─────────────────────────────────────────────────────
    if use_wandb:
        wandb_init(
            project=wandb_project,
            name=f"{timestamp}_{algo_name}_{env_short}_seed{seed}",
            config={
                "algo": algo_name,
                "env": cfg.env_name,
                "seed": seed,
                "timestamp": timestamp,
                **{k: v for k, v in dataclasses.asdict(cfg).items() if k != "ppo"},
                **{f"algo_{k}": v for k, v in dataclasses.asdict(algo_cfg).items()},
            },
        )
        wandb_setup_metrics()

    # ── Algo init ──────────────────────────────────────────────────────────
    key, init_key = jax.random.split(key)
    training_state = algo.init(init_key)

    actor_param_count = sum(x.size for x in jax.tree.leaves(training_state.actor_params))
    q_param_count = sum(x.size for x in jax.tree.leaves(training_state.q1_params))
    print(f"  actor_params={actor_param_count:,}, Q_params (each)={q_param_count:,}")

    # ── ObsPipeline + buffer + norm_state ──────────────────────────────────
    use_obs_norm = algo_cfg.obs_normalization
    obs_norm_eps = getattr(algo_cfg, "obs_norm_eps", 1e-8)
    n_frame_stack = cfg.n_frame_stack

    pipe = ObsPipeline(dict_obs, has_privileged, use_obs_norm, n_frame_stack, obs_norm_eps)
    buffer = pipe.make_buffer(
        obs_dim, action_dim, algo_cfg.buffer_size,
        critic_obs_dim=critic_obs_dim, num_envs=cfg.num_envs,
    )
    norm_state = pipe.init_norm_state(obs_dim)
    critic_norm_state = pipe.init_critic_norm_state(critic_obs_dim) if has_privileged else None

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if resume is not None:
        print(f"\n  Resuming from {resume}")
        loaded = load_checkpoint(resume, training_state, norm_state, critic_norm_state)
        if has_privileged and len(loaded) == 4:
            training_state, norm_state, start_step, critic_norm_state = loaded
        else:
            training_state, norm_state, start_step = loaded[:3]
        print(f"  Resuming from step {start_step:,}")

    # ── Tracker + ctx + checkpoint manager ─────────────────────────────────
    tracker = EpisodeTracker(cfg.num_envs)
    metrics_log: list[dict] = []
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_{algo_name}_{env_short}_seed{seed}")
    ckpt_mgr = CheckpointManager(ckpt_dir)
    ctx = TrainContext(
        cfg=cfg, algo_cfg=algo_cfg, algo_name=algo_name,
        ckpt_dir=ckpt_dir, obs_dim=obs_dim, action_dim=action_dim,
        metrics_log=metrics_log, ckpt_mgr=ckpt_mgr, resume=resume,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\nCollecting {algo_cfg.min_buffer_size:,} samples before first gradient update...")
    print("-" * 80)

    t0 = time.time()
    log_every = max(1, 10_000 // cfg.num_envs)
    last_eval_eps = 0
    last_metrics: dict = {}
    total_gradient_steps = 0

    for outer_step in range(start_step // cfg.num_envs, total_env_steps // cfg.num_envs):
        raw_steps = (outer_step + 1) * cfg.num_envs
        total_steps = raw_steps
        raw_obs = pipe.get_obs(env_state.obs)
        critic_raw_obs = pipe.get_critic_obs(env_state.obs) if has_privileged else None

        # Obs normalization
        norm_state = pipe.update_stats(raw_obs, norm_state)
        if has_privileged:
            critic_norm_state = pipe.update_critic_stats(critic_raw_obs, critic_norm_state)
        obs_for_action = pipe.normalize_for_action(raw_obs, norm_state)

        # Action selection
        # Cold-start: random uniform until buffer fills, for exploration.
        # Resume default ("policy"): use loaded policy from step 0 — random refill
        # would corrupt the converged policy's data distribution and tank first eval.
        # Resume "random": legacy behavior, restored via --resume-warmup random.
        is_warmup = len(buffer) < algo_cfg.min_buffer_size
        use_random = is_warmup and (start_step == 0 or resume_warmup == "random")
        if use_random:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action = explore_fn(training_state.actor_params, obs_for_action, ak)

        # Env step
        env_state = env_step(env_state, action)
        if cfg.handle_truncation:
            if "truncation" not in env_state.info:
                raise KeyError(
                    "cfg.handle_truncation=True but env_state.info['truncation'] is "
                    "missing. Either populate info['truncation'] in your environment "
                    "or disable truncation handling with cfg.handle_truncation=False."
                )
            truncation = env_state.info["truncation"]
        else:
            truncation = jnp.zeros_like(env_state.done)

        # Buffer
        next_raw_obs = pipe.get_obs(env_state.obs)
        extra_kwargs = {}
        if has_privileged:
            extra_kwargs["critic_obs"] = critic_raw_obs
            extra_kwargs["critic_next_obs"] = pipe.get_critic_obs(env_state.obs)
        buffer.add_batch(
            obs=raw_obs, action=action,
            reward=env_state.reward * cfg.reward_scaling,
            next_obs=next_raw_obs, done=env_state.done,
            truncation=truncation, **extra_kwargs,
        )
        tracker.step(np.asarray(env_state.reward), np.asarray(env_state.done))

        # Gradient updates
        if len(buffer) >= algo_cfg.min_buffer_size:
            for _ in range(algo_cfg.grad_updates_per_step):
                key, sample_key = jax.random.split(key)
                jax_batch = buffer.sample(algo_cfg.batch_size, key=sample_key)
                jax_batch = pipe.normalize_batch(jax_batch, norm_state, critic_norm_state=critic_norm_state)
                training_state, step_metrics = algo.update(training_state, jax_batch)
                total_gradient_steps += 1
                last_metrics = step_metrics

        # Logging
        if outer_step % log_every == 0 or total_steps >= total_env_steps:
            elapsed = time.time() - t0
            sps = int(total_steps / elapsed) if elapsed > 0 else 0
            is_training = last_metrics and len(buffer) >= algo_cfg.min_buffer_size

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
                terrain_metrics = log_terrain_metrics(env_state.info) if hasattr(env_state, "info") else {}
                row.update(terrain_metrics)
                # Composite snapshot image (replaces 40+ scalar level_hist lines)
                img_dict = log_terrain_image(env_state.info) if hasattr(env_state, "info") else {}
                wandb_log({**row, **img_dict}, step=raw_steps)

                # Console curriculum dump every ~50k env steps (bug-hunt diagnostic).
                # Zero-op for non-curriculum envs.
                if total_steps // 50_000 != (total_steps - log_every * cfg.num_envs) // 50_000:
                    print_curriculum_dump(env_state.info, step=total_steps)

        # Eval + checkpoint. q_fn closes over training_state directly: the
        # lambda is called synchronously inside maybe_eval_and_checkpoint
        # → evaluate() → q_fn(obs, action), so there is no cross-iteration
        # capture risk. Critic obs is normalized via critic_norm_state so
        # Q-value logs are comparable to training-time Q.
        obs_norm_fn = pipe.make_obs_norm_fn(norm_state)

        def _q_fn(obs, action):
            policy_obs = pipe.get_obs(obs)
            raw_critic = (obs["privileged_state"]
                          if isinstance(obs, dict) and "privileged_state" in obs
                          else None)
            normed_critic = (pipe.normalize_critic(raw_critic, critic_norm_state)
                             if raw_critic is not None and has_privileged else None)
            return algo.get_q_value(training_state, policy_obs, action,
                                    critic_obs=normed_critic)

        last_eval_eps, key = maybe_eval_and_checkpoint(
            algo.select_action, training_state.actor_params, eval_env, tracker,
            ctx, training_state, norm_state, last_eval_eps, key,
            obs_normalize_fn=obs_norm_fn,
            q_fn=_q_fn,
            critic_norm_state=critic_norm_state,
        )

    # ── Final eval ─────────────────────────────────────────────────────────
    obs_norm_fn = pipe.make_obs_norm_fn(norm_state)

    def _q_fn_final(obs, action):
        policy_obs = pipe.get_obs(obs)
        raw_critic = (obs["privileged_state"]
                      if isinstance(obs, dict) and "privileged_state" in obs
                      else None)
        normed_critic = (pipe.normalize_critic(raw_critic, critic_norm_state)
                         if raw_critic is not None and has_privileged else None)
        return algo.get_q_value(training_state, policy_obs, action,
                                critic_obs=normed_critic)

    final_eval_and_checkpoint(
        algo.select_action, training_state.actor_params, eval_env, tracker,
        ctx, training_state, norm_state, key, total_gradient_steps,
        obs_normalize_fn=obs_norm_fn,
        q_fn=_q_fn_final,
        critic_norm_state=critic_norm_state,
    )

    wandb_finish()
