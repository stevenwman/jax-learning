"""Determinism diagnostics for TD-MPC2. Run twice in separate processes and
diff the printed hashes — if all hashes match, that subsystem is bit-identical
across processes.

Subsystems tested:
  --check init     : init_train_state(seed=0) → hash all params
  --check update   : init + run N update_steps on a fixed fake batch → hash params
  --check env      : eval_env.reset + N env_steps with fixed actions → hash qpos/qvel

Use:
  uv run python scripts/check_tdmpc2_determinism.py --check init
  uv run python scripts/check_tdmpc2_determinism.py --check update --n 50
  uv run python scripts/check_tdmpc2_determinism.py --check env --n 50

Then run the same command in a second process and diff the outputs.
"""

import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
sys.stdout.reconfigure(line_buffering=True)

import argparse
import hashlib

import numpy as np
import jax
import jax.numpy as jnp

from jax_rl.algos.tdmpc2 import make_update_step
from jax_rl.algos.tdmpc2.runtime import (
    build_modules, init_train_state, build_train_config_from_tdmpc2,
)
from jax_rl.configs.env_presets import get_tdmpc2_preset
from jax_rl.training.env_setup import make_env_bundle


def _hash_pytree(tree) -> str:
    """SHA-256 over all leaves' raw bytes. Stable across processes."""
    h = hashlib.sha256()
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        arr = np.asarray(leaf)
        h.update(arr.tobytes())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
    return h.hexdigest()[:16]


def check_init(env: str, seed: int):
    cfg = get_tdmpc2_preset(env)
    train_cfg = build_train_config_from_tdmpc2(cfg, env, total_timesteps=0, seed=seed)
    env_bundle = make_env_bundle(train_cfg, seed)
    state, _, _ = init_train_state(cfg, env_bundle.obs_dim, seed)
    print(f"[det] init   encoder={_hash_pytree(state.encoder_params)}")
    print(f"[det] init   dynamics={_hash_pytree(state.dynamics_params)}")
    print(f"[det] init   reward={_hash_pytree(state.reward_params)}")
    print(f"[det] init   q_ens={_hash_pytree(state.q_ensemble_params)}")
    print(f"[det] init   policy={_hash_pytree(state.policy_params)}")


def check_update(env: str, seed: int, n_steps: int):
    cfg = get_tdmpc2_preset(env)
    train_cfg = build_train_config_from_tdmpc2(cfg, env, total_timesteps=0, seed=seed)
    env_bundle = make_env_bundle(train_cfg, seed)
    state, wm_opt, pol_opt = init_train_state(cfg, env_bundle.obs_dim, seed)
    encoder, dynamics, reward_net, q_ensemble, policy = build_modules(cfg)
    update_step = make_update_step(
        cfg, wm_opt, pol_opt,
        encoder=encoder, dynamics=dynamics, reward_net=reward_net,
        q_ensemble_net=q_ensemble, policy_net=policy,
    )

    # Fixed fake batch — deterministic from seed.
    batch_key = jax.random.PRNGKey(seed + 7777)
    H, B = cfg.horizon, cfg.batch_size
    obs = jax.random.normal(batch_key, (H + 1, B, env_bundle.obs_dim))
    act = jax.random.uniform(batch_key, (H, B, cfg.action_dim), minval=-1, maxval=1)
    rew = jax.random.uniform(batch_key, (H, B, 1), minval=0, maxval=1)
    done = jnp.zeros((H, B, 1))
    trunc = jnp.zeros((H, B, 1))
    batch = {"obs": obs, "actions": act, "rewards": rew, "dones": done, "truncations": trunc}

    print(f"[det] update before  encoder={_hash_pytree(state.encoder_params)}")
    for i in range(n_steps):
        state, _ = update_step(state, batch)
    print(f"[det] update after{n_steps:>4}  encoder={_hash_pytree(state.encoder_params)}")
    print(f"[det] update after{n_steps:>4}  dynamics={_hash_pytree(state.dynamics_params)}")
    print(f"[det] update after{n_steps:>4}  q_ens={_hash_pytree(state.q_ensemble_params)}")
    print(f"[det] update after{n_steps:>4}  policy={_hash_pytree(state.policy_params)}")
    print(f"[det] update after{n_steps:>4}  state.key={int(jnp.sum(state.key))}")


def check_env(env: str, seed: int, n_steps: int):
    cfg = get_tdmpc2_preset(env)
    train_cfg = build_train_config_from_tdmpc2(cfg, env, total_timesteps=0, seed=seed)
    env_bundle = make_env_bundle(train_cfg, seed)
    eval_env = env_bundle.eval_env
    env_step = env_bundle.env_step
    num_envs = cfg.num_eval_envs

    reset_key = jax.random.PRNGKey(seed)
    env_state = eval_env.reset(jax.random.split(reset_key, num_envs))

    # Fixed deterministic actions.
    act_key = jax.random.PRNGKey(seed + 1234)
    actions = jax.random.uniform(act_key, (n_steps, num_envs, cfg.action_dim),
                                  minval=-1, maxval=1)

    print(f"[det] env reset  obs_hash={_hash_pytree(env_state.obs)}")
    for i in range(n_steps):
        env_state = env_step(env_state, actions[i])
    print(f"[det] env after{n_steps:>4}  obs_hash={_hash_pytree(env_state.obs)}")
    print(f"[det] env after{n_steps:>4}  reward_sum={float(jnp.sum(env_state.reward)):.6f}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Importable for docs/tooling without parse_args()."""
    p = argparse.ArgumentParser()
    p.add_argument("--check", choices=["init", "update", "env"], required=True,
                   help="Which determinism check: init / update / env")
    p.add_argument("--env", default="CheetahRun",
                   help="Env name (default: CheetahRun)")
    p.add_argument("--seed", type=int, default=0, help="Seed (default: 0)")
    p.add_argument("--n", type=int, default=50,
                   help="Number of update / env steps to compare (default: 50)")
    return p


def main():
    args = build_parser().parse_args()

    if args.check == "init":
        check_init(args.env, args.seed)
    elif args.check == "update":
        check_update(args.env, args.seed, args.n)
    elif args.check == "env":
        check_env(args.env, args.seed, args.n)


if __name__ == "__main__":
    main()
