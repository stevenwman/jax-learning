"""Eval-only mode for TD-MPC2: load a checkpoint and run N rounds of eval to
get a mean ± std on MPPI and prior returns. Useful for confirming whether a
peak/dip in training-time eval was real or just single-eval variance.

Usage:
    uv run python scripts/eval_tdmpc2.py --env CheetahRun \\
        --load-ckpt .temp/tdmpc2_j3_smoke_v4 --num-evals 5

Each round runs cfg.num_eval_envs episodes (default 8). 5 rounds → 40 episodes.
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
sys.stdout.reconfigure(line_buffering=True)

import argparse
import dataclasses

import numpy as np
import jax

from jax_rl.algos.tdmpc2 import make_plan_batched
from jax_rl.algos.tdmpc2.runtime import (
    build_modules, init_train_state, run_eval,
    build_train_config_from_tdmpc2, load_params_into_state,
)
from jax_rl.configs.env_presets import get_tdmpc2_preset
from jax_rl.training.env_setup import make_env_bundle


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Importable for docs/tooling without parse_args()."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        help="Env name (CheetahRun, HumanoidRun, ...)")
    parser.add_argument("--load-ckpt", type=str, required=True,
                        help="Path to ckpt dir (containing actor_params.npz + world_model_params.npz). "
                             "Pass `<dir>/best` to eval the peak ckpt.")
    parser.add_argument("--num-evals", type=int, default=5,
                        help="Number of eval rounds. Each round runs cfg.num_eval_envs episodes.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; round k uses seed+k for reproducibility.")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override TDMPC2Config.num_envs (only affects state shape; eval uses num_eval_envs).")
    return parser


def main():
    args = build_parser().parse_args()

    cfg = get_tdmpc2_preset(args.env)
    if args.num_envs is not None:
        cfg = dataclasses.replace(cfg, num_envs=args.num_envs)

    print(f"[eval] env={args.env} ckpt={args.load_ckpt} num_evals={args.num_evals}")

    train_cfg = build_train_config_from_tdmpc2(cfg, args.env, total_timesteps=0, seed=args.seed)
    env_bundle = make_env_bundle(train_cfg, args.seed)
    print(f"[eval] obs_dim={env_bundle.obs_dim} action_dim={env_bundle.action_dim}")

    state, _wm_opt, _pol_opt = init_train_state(cfg, env_bundle.obs_dim, args.seed)
    state = load_params_into_state(state, args.load_ckpt)
    print(f"[eval] params loaded from {args.load_ckpt}")

    modules = build_modules(cfg)
    _, dynamics, reward_net, q_ensemble, policy = modules
    plan_fn = make_plan_batched(
        dynamics=dynamics, reward_net=reward_net,
        q_ensemble_net=q_ensemble, policy_net=policy,
    )

    mppi_returns, prior_returns = [], []
    for k in range(args.num_evals):
        key = jax.random.PRNGKey(args.seed + k)
        result = run_eval(state, env_bundle, plan_fn, modules, cfg, key)
        mppi_returns.append(result["mppi_return"])
        prior_returns.append(result["prior_return"])
        print(f"[eval] round {k}: mppi={result['mppi_return']:.2f} "
              f"prior={result['prior_return']:.2f} "
              f"gap={result['mppi_prior_gap']:+.2f}")

    mppi = np.array(mppi_returns)
    prior = np.array(prior_returns)
    print(f"\n[eval] Summary over {args.num_evals} rounds (each = {cfg.num_eval_envs} episodes):")
    print(f"[eval]   mppi:  mean={mppi.mean():.2f} std={mppi.std():.2f} "
          f"min={mppi.min():.2f} max={mppi.max():.2f}")
    print(f"[eval]   prior: mean={prior.mean():.2f} std={prior.std():.2f} "
          f"min={prior.min():.2f} max={prior.max():.2f}")


if __name__ == "__main__":
    main()
