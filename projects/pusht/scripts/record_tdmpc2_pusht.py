"""Record TDMPC2 MPPI rollout on PushT.

Loads TDMPC2 ckpt, rolls out N episodes with MPPI planning, saves mp4.
"""
import argparse
import dataclasses
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

import jax
import jax.numpy as jnp
import numpy as np
import imageio

from jax_rl.algos.tdmpc2 import make_plan_batched
from jax_rl.algos.tdmpc2.runtime import (
    build_modules,
    build_train_config_from_tdmpc2,
    init_train_state,
    load_params_into_state,
)
from jax_rl.configs.env_presets import get_tdmpc2_preset
from jax_rl.training import make_env_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--shape", default="tee")
    ap.add_argument("--num-episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2000)
    args = ap.parse_args()

    cfg = get_tdmpc2_preset("PushT")
    cfg = dataclasses.replace(cfg, num_envs=1)

    train_cfg = build_train_config_from_tdmpc2(cfg, "PushT", total_timesteps=0, seed=args.seed)
    train_cfg = dataclasses.replace(
        train_cfg,
        num_envs=1,
        env_kwargs={
            "obs_type": "keypoints",
            "block_shape": args.shape,
            "reward_mode": "contact_gated",
            "coverage_shape": "log_barrier",
            "coverage_eps": 0.01,
            "render_mode": "rgb_array",
        },
    )
    bundle = make_env_bundle(train_cfg, args.seed)
    obs_dim = bundle.obs_dim
    print(f"[record] obs_dim={obs_dim} action_dim={cfg.action_dim}")

    state, _, _ = init_train_state(cfg, obs_dim, args.seed)
    state = load_params_into_state(state, args.ckpt_dir)

    modules = build_modules(cfg)
    encoder, dynamics, reward_net, q_ensemble, policy = modules
    plan_fn = make_plan_batched(
        dynamics=dynamics, reward_net=reward_net,
        q_ensemble_net=q_ensemble, policy_net=policy,
    )
    plan_params = {
        "encoder": state.encoder_params,
        "dynamics": state.dynamics_params,
        "reward": state.reward_params,
        "q_ensemble": state.q_ensemble_params,
        "policy": state.policy_params,
    }

    env = bundle.eval_env
    frames = []
    key = jax.random.PRNGKey(13)
    for ep in range(args.num_episodes):
        obs_np, _ = env.reset(seed=args.seed + ep)
        prev_mean = jnp.zeros((1, cfg.horizon, cfg.action_dim))
        t0 = jnp.ones(1, dtype=jnp.bool_)
        info = {}
        ep_frames = []
        for step in range(args.max_steps):
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
            frame = bundle.render_fn(None, env_idx=0)
            if frame is not None:
                ep_frames.append(np.asarray(frame))
            if bool(np.asarray(term).any() or np.asarray(trunc).any()):
                break
        cov = float(np.asarray(info.get("coverage", 0.0)).reshape(-1)[0])
        succ = bool(np.asarray(info.get("is_success", False)).reshape(-1)[0])
        print(f"[record] ep={ep} steps={len(ep_frames)} cov={cov*100:.1f}% succ={succ}")
        frames.extend(ep_frames)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=args.fps)
    print(f"[record] saved {len(frames)} frames → {out_path}")


if __name__ == "__main__":
    main()
