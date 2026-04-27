"""PPO + ContractionPPO training (fast/scan path).

Fork of train_ppo_fast.py minimally modified to:
1. Instantiate PPOContraction (not PPO).
2. Populate ContractionConfig.constraint_dim from env obs.
3. Use make_collect() with extras + reward_augment hooks.
4. Forward contraction_c / contraction_c_dot into RolloutBatch.

Baseline train_ppo_fast.py is untouched — contraction-enabled runs go through
this script, contraction=None would use train_ppo_fast.py.

Usage:
    uv run python train_ppo_contraction.py --env Go2BongoHandstand --total-timesteps 1_000_000
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
sys.stdout.reconfigure(line_buffering=True)

import argparse
import dataclasses
import time
from datetime import datetime
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import optax

from jax_rl.algos import PPOContraction
from jax_rl.buffers import RolloutBatch
from jax_rl.configs import (
    ContractionConfig,
    EncoderConfig,
    PolicyHeadConfig,
    TrainConfig,
    get_preset,
)
from jax_rl.training import (
    make_env_bundle, EpisodeTracker, load_checkpoint,
    make_collect,
)
from jax_rl.training.metrics_logger import (
    wandb_init, wandb_setup_metrics, wandb_log, wandb_finish,
)
from jax_rl.training.checkpointing import save_checkpoint, CheckpointManager
from jax_rl.utils.eval import evaluate
from jax_rl.utils.normalization import init as norm_init


class ContractionExtras(NamedTuple):
    """Per-step extras emitted by extra_rollout_fn, stacked over scan."""
    c: jax.Array       # (num_envs, constraint_dim)
    c_dot: jax.Array   # (num_envs, constraint_dim)


def _make_eval_action(ppo, get_policy_obs, n_frame_stack=1):
    from jax_rl.utils.normalization import (
        normalize as norm_normalize,
        normalize_stacked as norm_normalize_stacked,
    )

    @jax.jit
    def _ppo_eval_action(actor_params, obs, key=None, deterministic=True, *, norm_state):
        policy_obs = get_policy_obs(obs)
        normed = (norm_normalize_stacked(norm_state, policy_obs, n_frame_stack)
                  if n_frame_stack > 1 else norm_normalize(norm_state, policy_obs))
        mean, _log_std = ppo.actor.apply(actor_params, normed)
        return jnp.clip(mean, -1.0, 1.0)
    return _ppo_eval_action


def train(cfg: TrainConfig, seed: int = 0, resume: str | None = None,
          use_wandb: bool = False, wandb_project: str = "jax-rl"):
    # ── Environment ──────────────────────────────────────────────────────
    bundle = make_env_bundle(cfg, seed)
    if bundle.backend_kind != "mjx":
        raise ValueError(
            f"train_ppo_contraction requires an MJX env bundle, but env "
            f"{cfg.env_name!r} routes to backend_kind={bundle.backend_kind!r}.\n"
            f"\n"
            f"PPOContraction uses lax.scan rollout collection + a "
            f"contraction-state feature in env obs that's only emitted by "
            f"MJX envs (Playground/Warp). Gym envs don't have it.\n"
            f"\n"
            f"For PPO on gym envs today, no equivalent script exists yet."
        )
    env, env_step, env_state, eval_env = bundle.env, bundle.env_step, bundle.env_state, bundle.eval_env
    obs_dim, action_dim, key = bundle.obs_dim, bundle.action_dim, bundle.key
    dict_obs = bundle.dict_obs
    if dict_obs:
        critic_obs_dim = env_state.obs["privileged_state"].shape[-1]
        if "contraction_state" not in env_state.obs:
            raise ValueError(
                "env_state.obs missing 'contraction_state'. Enable the env's "
                "observe_contraction config flag (e.g. cfg.observe_contraction=True)."
            )
        c_state = env_state.obs["contraction_state"]
        constraint_dim = c_state.shape[-1] // 2
        print(f"  Contraction: constraint_dim={constraint_dim} (packed={c_state.shape[-1]}) [dict obs]")
    else:
        # Flat-obs path: symmetric actor/critic, per-env contraction feature fn.
        critic_obs_dim = obs_dim
        if cfg.env_name in ("CartpoleSwingup", "CartpoleSwingupSparse"):
            # obs = [cart_pos, cos θ, sin θ, cart_vel, θ̇]. Equilibrium = upright (θ=0).
            constraint_dim = 3
        else:
            raise ValueError(
                f"Flat-obs contraction not wired for env '{cfg.env_name}'. "
                "Add a feature fn case or use a dict-obs env."
            )
        print(f"  Contraction: constraint_dim={constraint_dim} [flat obs, env={cfg.env_name}]")

    # ── Config plumbing ───────────────────────────────────────────────────
    ppo_cfg = cfg.ppo
    if ppo_cfg.contraction is None:
        raise ValueError("cfg.ppo.contraction must be set for this script.")
    ppo_cfg.contraction.constraint_dim = constraint_dim
    ppo_cfg.contraction.validate()

    samples_per_update = cfg.num_envs * ppo_cfg.num_steps
    samples_per_iter = samples_per_update * ppo_cfg.num_updates_per_batch
    num_iterations = cfg.total_timesteps // samples_per_iter

    print("=" * 80)
    print(f"PPOContraction — {cfg.env_name} (MuJoCo Playground)")
    print("=" * 80)
    print(f"  obs_dim={obs_dim}, critic_obs_dim={critic_obs_dim}, action_dim={action_dim}")
    print(f"  num_envs={cfg.num_envs}, num_steps={ppo_cfg.num_steps}, episode_length={cfg.episode_length}")
    print(f"  alpha={ppo_cfg.contraction.alpha}, eps={ppo_cfg.contraction.epsilon_contraction}, "
          f"penalty_coef={ppo_cfg.contraction.penalty_coef}, metric_lr={ppo_cfg.contraction.metric_lr}")
    print(f"  samples/iter={samples_per_iter:,}, iterations={num_iterations}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_short = cfg.env_name.lower().replace(" ", "_")

    if use_wandb:
        wandb_init(
            project=wandb_project,
            name=f"{timestamp}_ppocontr_{env_short}_seed{seed}",
            config={**dataclasses.asdict(cfg)},
        )
        wandb_setup_metrics()

    num_minibatches = ppo_cfg.num_minibatches
    minibatch_size = samples_per_update // num_minibatches
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

    # ── Optimizers ────────────────────────────────────────────────────────
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

    ppo = PPOContraction(ppo_config, obs_dim, action_dim, actor_optimizer, critic_optimizer,
                         critic_obs_dim=critic_obs_dim)
    key, init_key = jax.random.split(key)
    training_state = ppo.init(init_key)

    # ── Normalization ────────────────────────────────────────────────────
    n_frame_stack = cfg.n_frame_stack
    policy_raw_dim = obs_dim // n_frame_stack if n_frame_stack > 1 else obs_dim
    # FrameStackWrapper only stacks policy obs, not privileged_state → critic
    # sees single-frame privileged obs (critic_raw_dim = critic_obs_dim).
    critic_raw_dim = critic_obs_dim
    norm_state = norm_init(policy_raw_dim)
    critic_norm_state = norm_init(critic_raw_dim)

    start_iteration = 0
    if resume is not None:
        loaded = load_checkpoint(resume, training_state, norm_state, critic_norm_state)
        if len(loaded) == 4:
            training_state, norm_state, start_step, critic_norm_state = loaded
        else:
            training_state, norm_state, start_step = loaded[:3]
            print("  WARNING: no critic_norm_state in checkpoint — resuming with fresh "
                  "critic stats. Results pre- and post-resume may diverge.")
        start_iteration = start_step // samples_per_iter if start_step > 0 else 0
        print(f"  Resuming from iteration {start_iteration}")

    # ── Obs extractors + hooks ────────────────────────────────────────────
    if dict_obs:
        def _get_policy_obs(obs): return obs["state"]
        def _get_critic_obs(obs): return obs["privileged_state"]
        def _get_contraction_obs(obs):
            cs = obs["contraction_state"]
            return cs[..., :constraint_dim], cs[..., constraint_dim:]
    else:
        def _get_policy_obs(obs): return obs
        def _get_critic_obs(obs): return obs
        if cfg.env_name in ("CartpoleSwingup", "CartpoleSwingupSparse"):
            def _get_contraction_obs(obs):
                cart_pos = obs[..., 0:1]
                cos_t = obs[..., 1:2]
                sin_t = obs[..., 2:3]
                cart_vel = obs[..., 3:4]
                th_dot = obs[..., 4:5]
                c = jnp.concatenate([cart_pos, cos_t - 1.0, sin_t], axis=-1)
                c_dot = jnp.concatenate([cart_vel, -sin_t * th_dot, cos_t * th_dot], axis=-1)
                return c, c_dot
        else:
            raise ValueError(f"Flat-obs contraction fn not wired for '{cfg.env_name}'.")

    def extra_rollout_fn(ts, env_state, key):
        c, c_dot = _get_contraction_obs(env_state.obs)
        return ContractionExtras(c=c, c_dot=c_dot)

    def reward_augment_fn(ts, env_state, extras):
        return ppo.compute_contraction_reward(ts.metric_params, extras.c, extras.c_dot)

    # ── Build collect fn ─────────────────────────────────────────────────
    _collect = make_collect(
        env_step=env_step,
        select_stochastic=ppo._select_stochastic,
        select_deterministic=ppo._select_deterministic,
        num_steps=ppo_cfg.num_steps,
        num_envs=cfg.num_envs,
        reward_scaling=cfg.reward_scaling,
        handle_truncation=cfg.handle_truncation,
        get_policy_obs=_get_policy_obs,
        get_critic_obs=_get_critic_obs,
        n_frame_stack=n_frame_stack,
        policy_raw_dim=policy_raw_dim,
        critic_raw_dim=critic_raw_dim,
        extra_rollout_fn=extra_rollout_fn,
        reward_augment_fn=reward_augment_fn,
    )

    # ── Outer loop ───────────────────────────────────────────────────────
    tracker = EpisodeTracker(cfg.num_envs)
    metrics_log: list[dict] = []
    ckpt_dir = os.path.join("checkpoints", f"{timestamp}_ppocontr_{env_short}_seed{seed}")
    last_eval_eps = 0

    _ppo_eval_action = _make_eval_action(ppo, _get_policy_obs, n_frame_stack)

    running_ep_return = jnp.zeros(cfg.num_envs)
    ckpt_mgr = CheckpointManager(ckpt_dir)
    print(f"\nJIT-compiling first iteration...")
    print("-" * 80)

    _t_start = time.time()
    for iteration in range(start_iteration, num_iterations):
        t0 = time.time()

        for _ in range(ppo_cfg.num_updates_per_batch):
            (env_state, norm_state, critic_norm_state, key,
             rollout, normed_next, normed_next_critic, next_value,
             scan_ep_count, scan_ep_return_sum, running_ep_return) = _collect(
                training_state, env_state, norm_state, critic_norm_state, key, running_ep_return,
            )

            scan_n = int(scan_ep_count)
            if scan_n > 0:
                scan_avg = float(scan_ep_return_sum / scan_ep_count)
                tracker.completed_returns.extend([scan_avg] * scan_n)

            batch = RolloutBatch(
                obs=rollout.obs,
                actions=rollout.action,
                rewards=rollout.reward,
                dones=rollout.done,
                truncations=rollout.truncation,
                log_probs=rollout.log_prob,
                values=rollout.value,
                contraction_c=rollout.extras.c,
                contraction_c_dot=rollout.extras.c_dot,
            )

            key, update_key = jax.random.split(key)
            training_state, metrics = ppo.update(
                training_state, batch, update_key,
                next_obs=normed_next,
                critic_obs=rollout.critic_obs,
                critic_next_obs=normed_next_critic,
            )

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
                f"PLoss {metrics['policy_loss']:.2e} | "
                f"VLoss {metrics['value_loss']:.2e} | "
                f"CPen {metrics['contraction_penalty']:.2e} | "
                f"V_mean {metrics['V_mean']:.2e} | "
                f"{sps:>6,} sps | {iter_time:.1f}s"
            )
            metrics_log.append({
                "iteration": iteration,
                "total_steps": total_steps,
                "episodes": stats["n_eps"],
                "avg_return": stats["avg"],
                "policy_loss": float(metrics["policy_loss"]),
                "value_loss": float(metrics["value_loss"]),
                "entropy": float(metrics["entropy"]),
                "approx_kl": float(metrics["approx_kl"]),
                "clip_fraction": float(metrics["clip_fraction"]),
                "contraction_penalty": float(metrics["contraction_penalty"]),
                "V_mean": float(metrics["V_mean"]),
                "V_dot_mean": float(metrics["V_dot_mean"]),
                "sps": sps,
            })
            if iteration > 0:
                wandb_log(metrics_log[-1], step=total_steps)

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
            print(f"  EVAL @ {n_eps_total} eps | Return {eval_metrics['eval_mean']:.1f}")
            if metrics_log:
                metrics_log[-1].update(eval_metrics)
            wandb_log(eval_metrics, step=total_steps)
            ckpt_mgr.save(training_state, norm_state, cfg, cfg.ppo,
                          "ppocontr", obs_dim, action_dim, metrics_log, resume,
                          eval_mean=eval_metrics['eval_mean'],
                          critic_norm_state=critic_norm_state)
            last_eval_eps = n_eps_total

    print("=" * 80)
    print(f"Training complete. Total elapsed: {time.time() - _t_start:.0f}s")
    wandb_finish()


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Importable for docs/tooling without parse_args()."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Go2BongoHandstand")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--reward-scaling", type=float, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="jax-rl")
    # Contraction knobs
    parser.add_argument("--alpha", type=float, default=0.1, help="Contraction rate α")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Strict-inequality slack ε")
    parser.add_argument("--penalty-coef", type=float, default=1.0, help="Reward-augment scale")
    parser.add_argument("--metric-lr", type=float, default=1e-3)
    parser.add_argument("--constraint-coef", type=float, default=1.0)
    parser.add_argument("--metric-hidden", type=int, nargs="+", default=[128, 128])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    cfg = get_preset(args.env)
    cfg_overrides = {}
    if args.num_envs is not None: cfg_overrides["num_envs"] = args.num_envs
    if args.total_timesteps is not None: cfg_overrides["total_timesteps"] = args.total_timesteps
    if args.lr is not None: cfg_overrides["lr"] = args.lr
    if args.reward_scaling is not None: cfg_overrides["reward_scaling"] = args.reward_scaling
    if args.episode_length is not None: cfg_overrides["episode_length"] = args.episode_length
    if args.log_interval is not None: cfg_overrides["log_interval"] = args.log_interval
    if args.frame_stack is not None: cfg_overrides["n_frame_stack"] = args.frame_stack

    # Inject ContractionConfig into ppo
    contraction = ContractionConfig(
        alpha=args.alpha,
        epsilon_contraction=args.epsilon,
        penalty_coef=args.penalty_coef,
        constraint_coef=args.constraint_coef,
        metric_lr=args.metric_lr,
        hidden_dims=tuple(args.metric_hidden),
    )
    ppo_cfg_with_contraction = dataclasses.replace(cfg.ppo, contraction=contraction)
    cfg_overrides["ppo"] = ppo_cfg_with_contraction

    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)

    train(cfg, seed=args.seed, resume=args.resume,
          use_wandb=args.wandb, wandb_project=args.wandb_project)
