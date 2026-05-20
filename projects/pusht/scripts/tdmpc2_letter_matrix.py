"""5×4 TDMPC2 letter transfer matrix on PushT.

Loads 5 TDMPC2 ckpts (T/L/K/S/DR), evaluates each on 4 test shapes (T/L/K/S)
via MPPI planning. Reads PushT info["coverage"] at episode end. Reports
mean (min-max) over n_episodes per cell.
"""
import argparse
import dataclasses
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from jax_rl.algos.tdmpc2 import make_plan_batched
from jax_rl.algos.tdmpc2.runtime import (
    build_modules,
    build_train_config_from_tdmpc2,
    init_train_state,
    load_params_into_state,
)
from jax_rl.configs.env_presets import get_tdmpc2_preset
from jax_rl.training import make_env_bundle


TRAIN_SHAPES = ("tee", "l", "k", "s", "dr")
TEST_SHAPES = ("tee", "l", "k", "s")


def _make_bundle_for_shape(cfg, block_shape, seed=2000):
    """Build a single-env gym bundle for PushT with the given block_shape."""
    train_cfg = build_train_config_from_tdmpc2(cfg, "PushT", total_timesteps=0, seed=seed)
    train_cfg = dataclasses.replace(
        train_cfg,
        num_envs=1,
        env_kwargs={
            "obs_type": "keypoints",
            "block_shape": block_shape,
            "reward_mode": "contact_gated",
            "coverage_shape": "log_barrier",
            "coverage_eps": 0.01,
            "render_mode": "rgb_array",
        },
    )
    return make_env_bundle(train_cfg, seed)


def eval_cell(state, plan_fn, modules, cfg, block_shape, n_episodes=5, base_seed=2000, max_steps=150):
    """Run n_episodes MPPI rollouts on PushT with the given block_shape. Return covs."""
    encoder, dynamics, reward_net, q_ensemble, policy = modules
    plan_params = {
        "encoder": state.encoder_params,
        "dynamics": state.dynamics_params,
        "reward": state.reward_params,
        "q_ensemble": state.q_ensemble_params,
        "policy": state.policy_params,
    }

    covs = []
    succs = []
    key = jax.random.PRNGKey(13)
    for ep in range(n_episodes):
        bundle = _make_bundle_for_shape(cfg, block_shape, seed=base_seed + ep)
        env = bundle.eval_env
        obs_np, _ = env.reset(seed=base_seed + ep)
        prev_mean = jnp.zeros((1, cfg.horizon, cfg.action_dim))
        t0 = jnp.ones(1, dtype=jnp.bool_)
        info = {}
        for _ in range(max_steps):
            key, plan_key = jax.random.split(key)
            obs = jnp.asarray(obs_np)
            z_0 = encoder.apply(plan_params["encoder"], obs)
            plan_keys = jax.random.split(plan_key, 1)
            action, prev_mean = plan_fn(
                plan_params, z_0, prev_mean, t0, cfg, plan_keys, True,
            )
            action_np = np.asarray(action, dtype=np.float32)
            obs_np, r, term, trunc, info = env.step(action_np)
            t0 = jnp.asarray(np.asarray(term | trunc, dtype=bool))
            if bool(np.asarray(term).any() or np.asarray(trunc).any()):
                break
        # PushT vec_env wraps info into batched arrays of len 1
        cov = float(np.asarray(info.get("coverage", 0.0)).reshape(-1)[0])
        succ = 1 if bool(np.asarray(info.get("is_success", False)).reshape(-1)[0]) else 0
        covs.append(cov)
        succs.append(succ)
        env.close()
    return np.array(covs), np.array(succs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="N ckpt dirs in TRAIN_SHAPES order (subset OK; passes empty/missing rows as skip)")
    ap.add_argument("--rows", nargs="+", default=None,
                    help="Which TRAIN_SHAPES to include (default: as many as --ckpts given)")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-eval-seed", type=int, default=2000,
                    help="Base seed for env resets (avoids hardcoded 2000 which has S clip)")
    args = ap.parse_args()

    rows = args.rows if args.rows else list(TRAIN_SHAPES[:len(args.ckpts)])
    if len(rows) != len(args.ckpts):
        ap.error(f"--rows ({len(rows)}) must match --ckpts ({len(args.ckpts)})")

    cfg = get_tdmpc2_preset("PushT")
    # Force num_envs=1 for state init (matches eval bundle).
    cfg = dataclasses.replace(cfg, num_envs=1)

    # Init template state (will reload params per row)
    # Need obs_dim. Build a temp bundle for tee to get it.
    tmp_bundle = _make_bundle_for_shape(cfg, "tee", seed=args.seed)
    obs_dim = tmp_bundle.obs_dim
    print(f"[matrix] obs_dim={obs_dim} action_dim={cfg.action_dim}")

    modules = build_modules(cfg)
    _, dynamics, reward_net, q_ensemble, policy = modules
    plan_fn = make_plan_batched(
        dynamics=dynamics, reward_net=reward_net,
        q_ensemble_net=q_ensemble, policy_net=policy,
    )

    mean = np.zeros((len(rows), len(TEST_SHAPES)))
    mn = np.zeros_like(mean)
    mx = np.zeros_like(mean)
    sc = np.zeros_like(mean)

    for i, tr in enumerate(rows):
        print(f"\n=== train={tr.upper()} (ckpt={args.ckpts[i]}) ===")
        state, _wo, _po = init_train_state(cfg, obs_dim, args.seed)
        state = load_params_into_state(state, args.ckpts[i])
        for j, te in enumerate(TEST_SHAPES):
            covs, succs = eval_cell(
                state, plan_fn, modules, cfg, te,
                n_episodes=args.n_episodes,
                base_seed=args.base_eval_seed,
            )
            mean[i, j] = covs.mean()
            mn[i, j] = covs.min()
            mx[i, j] = covs.max()
            sc[i, j] = succs.mean()
            print(f"  → {te:4s} cov={covs.mean()*100:5.1f}% "
                  f"({covs.min()*100:.1f}–{covs.max()*100:.1f}) s={succs.mean()*100:.0f}%")

    # Print matrix
    lines = ["\n# TDMPC2 letter matrix (MPPI eval)\n",
             f"*n_episodes={args.n_episodes}.*\n",
             "| train ↓ / test → | " + " | ".join(TEST_SHAPES) + " |",
             "|---|" + "---|" * len(TEST_SHAPES)]
    for i, tr in enumerate(rows):
        row = [tr]
        for j in range(len(TEST_SHAPES)):
            mark = "✱ " if (tr in TEST_SHAPES and tr == TEST_SHAPES[j]) else ""
            row.append(f"{mark}{mean[i,j]*100:.1f}% ({mn[i,j]*100:.1f}–{mx[i,j]*100:.1f}) s={sc[i,j]*100:.0f}%")
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines) + "\n"
    out_path = Path("projects/pusht/artifacts/tdmpc2_letter_matrix.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(md)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
