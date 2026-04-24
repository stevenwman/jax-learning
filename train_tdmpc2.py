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
import dataclasses

import jax
import jax.numpy as jnp

from jax_rl.algos.tdmpc2 import (
    Encoder, Dynamics, Reward, QEnsemble, PolicyPrior,
    TDMPC2State,
    build_world_model_optimizer, build_policy_optimizer,
    make_plan_batched, make_update_step,
)
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.tdmpc2_config import TDMPC2Config
from jax_rl.configs.env_presets import get_tdmpc2_preset
from jax_rl.configs.train_config import TrainConfig
from jax_rl.training.env_setup import make_env_bundle
from jax_rl.utils.qscale import qscale_init


def build_modules(cfg: TDMPC2Config):
    """Instantiate the five network modules from config."""
    encoder = Encoder(
        enc_dim=cfg.enc_dim, num_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim, simnorm_dim=cfg.simnorm_dim,
    )
    dynamics = Dynamics(
        mlp_dim=cfg.mlp_dim, latent_dim=cfg.latent_dim, simnorm_dim=cfg.simnorm_dim,
    )
    reward_net = Reward(mlp_dim=cfg.mlp_dim, num_bins=cfg.num_bins)
    q_ensemble = QEnsemble(
        mlp_dim=cfg.mlp_dim, num_bins=cfg.num_bins,
        num_q=cfg.num_q, dropout=cfg.dropout,
    )
    policy = PolicyPrior(
        mlp_dim=cfg.mlp_dim, action_dim=cfg.action_dim,
        log_std_min=cfg.log_std_min, log_std_max=cfg.log_std_max,
    )
    return encoder, dynamics, reward_net, q_ensemble, policy


def init_train_state(
    cfg: TDMPC2Config,
    obs_dim: int,
    seed: int,
) -> tuple[TDMPC2State, object, object]:
    """Build modules + init all params + init optimizers + build TDMPC2State.

    Returns (state, wm_optimizer, policy_optimizer) for downstream use.
    """
    encoder, dynamics, reward_net, q_ensemble, policy = build_modules(cfg)

    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 6)

    # Init params with dummy input
    enc_params = encoder.init(ks[0], jnp.zeros((1, obs_dim)))
    dyn_params = dynamics.init(ks[1], jnp.zeros((1, cfg.latent_dim)),
                                jnp.zeros((1, cfg.action_dim)))
    rwd_params = reward_net.init(ks[2], jnp.zeros((1, cfg.latent_dim)),
                                  jnp.zeros((1, cfg.action_dim)))
    q_params = q_ensemble.init(
        {"params": ks[3]},
        jnp.zeros((1, cfg.latent_dim)), jnp.zeros((1, cfg.action_dim)),
        deterministic=True,
    )
    pol_params = policy.init(ks[4], jnp.zeros((1, cfg.latent_dim)), ks[5])

    # Target = online at init
    target_enc = jax.tree_util.tree_map(lambda x: x, enc_params)
    target_dyn = jax.tree_util.tree_map(lambda x: x, dyn_params)
    target_rwd = jax.tree_util.tree_map(lambda x: x, rwd_params)
    target_q = jax.tree_util.tree_map(lambda x: x, q_params)

    # Optimizers
    wm_opt = build_world_model_optimizer(cfg)
    pol_opt = build_policy_optimizer(cfg)
    wm_params = {"encoder": enc_params, "dynamics": dyn_params,
                 "reward": rwd_params, "q_ensemble": q_params}
    wm_opt_state = wm_opt.init(wm_params)
    pol_opt_state = pol_opt.init(
        {"policy": pol_params, "q_ensemble": q_params},
    )

    state = TDMPC2State(
        encoder_params=enc_params,
        dynamics_params=dyn_params,
        reward_params=rwd_params,
        q_ensemble_params=q_params,
        policy_params=pol_params,
        encoder_target_params=target_enc,
        dynamics_target_params=target_dyn,
        reward_target_params=target_rwd,
        q_ensemble_target_params=target_q,
        world_model_opt_state=wm_opt_state,
        policy_opt_state=pol_opt_state,
        qscale=qscale_init(),
        prev_mean=jnp.zeros((cfg.num_envs, cfg.horizon, cfg.action_dim)),
        key=jax.random.PRNGKey(seed + 1000),  # separate from init key
        step=jnp.array(0, dtype=jnp.int32),
    )
    return state, wm_opt, pol_opt


def build_train_config_from_tdmpc2(
    tdmpc2_cfg: TDMPC2Config,
    env_name: str,
    total_timesteps: int,
    seed: int,
) -> TrainConfig:
    """Adapter: build a minimal TrainConfig from TDMPC2Config + env info.

    TrainConfig carries env-framework state (env_name, num_envs, episode_length).
    TDMPC2Config carries algorithm state. These are separate; keep them so until
    we hit a reason to merge.
    """
    # episode_length default 1000 for DMC; for Go2/PushT get from TDMPC2Config.
    ep_len = tdmpc2_cfg.episode_lengths[0] if tdmpc2_cfg.episode_lengths else 1000
    return TrainConfig(
        env_name=env_name,
        total_timesteps=total_timesteps,
        num_envs=tdmpc2_cfg.num_envs,
        episode_length=ep_len,
        gamma=tdmpc2_cfg.discount,
        lr=tdmpc2_cfg.lr,
        reward_scaling=1.0,
        handle_truncation=True,
    )


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

    # H2-H5 will fill in warmup + main loop + eval + checkpointing here
    raise NotImplementedError("Training loop comes in Task H2-H5")


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
