"""Record TDMPC2 policy videos in MPPI and prior modes.

Two-phase pattern (per .context/lessons/infrastructure.md):
  Phase 1: jit-scan rollout on GPU, capture qpos/qvel per step.
  Phase 2: reuse single mujoco.Renderer + MjData on CPU; render frames.

Usage:
    uv run python scripts/record_video_tdmpc2.py \\
        --env HumanoidRun --load-ckpt .temp/tdmpc2_j4_v3/best
"""

import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
sys.stdout.reconfigure(line_buffering=True)

import argparse
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np
import jax
import jax.numpy as jnp

from jax_rl.algos.tdmpc2 import make_plan_batched
from jax_rl.algos.tdmpc2_runtime import (
    build_modules, init_train_state, build_train_config_from_tdmpc2,
    load_params_into_state, _pipe_obs,
)
from jax_rl.configs.env_presets import get_tdmpc2_preset
from jax_rl.training.env_setup import make_env_bundle


def scan_mppi(env_step, env_state, key, n_steps, plan_fn, plan_params, encoder, cfg, dict_obs):
    """JIT-scan an MPPI rollout. Returns (qpos[n,DOF], qvel[n,DOF], reward[n])."""
    init_pm = jnp.zeros((1, cfg.horizon, cfg.action_dim))
    init_t0 = jnp.ones(1, dtype=jnp.bool_)

    def step(carry, _):
        es, pm, k, t0 = carry
        k, sk = jax.random.split(k)
        obs = _pipe_obs(es.obs, dict_obs)
        z = encoder.apply(plan_params["encoder"], obs)
        keys = jax.random.split(sk, 1)
        a, npm = plan_fn(plan_params, z, pm, t0, cfg, keys, True)
        es = env_step(es, a)
        return (es, npm, k, jnp.zeros(1, dtype=jnp.bool_)), (es.data.qpos, es.data.qvel, es.reward[0])

    _, (qpos, qvel, rew) = jax.lax.scan(
        step, (env_state, init_pm, key, init_t0), None, length=n_steps,
    )
    return qpos, qvel, rew


def scan_prior(env_step, env_state, key, n_steps, encoder, policy, enc_p, pol_p, dict_obs):
    """JIT-scan policy-prior rollout (greedy mean action)."""
    dummy = jax.random.PRNGKey(0)

    def step(carry, _):
        es, _ = carry
        obs = _pipe_obs(es.obs, dict_obs)
        z = encoder.apply(enc_p, obs)
        _, ex = policy.apply(pol_p, z, dummy)
        a = jnp.tanh(ex["mean"])
        es = env_step(es, a)
        return (es, dummy), (es.data.qpos, es.data.qvel, es.reward[0])

    _, (qpos, qvel, rew) = jax.lax.scan(
        step, (env_state, dummy), None, length=n_steps,
    )
    return qpos, qvel, rew


def render_qpos_qvel(mj_model, qpos_seq, qvel_seq, n, label, width, height,
                      track_body_id=1, distance=4.0, azimuth=135.0, elevation=-20.0):
    """Phase 2 — reuse one Renderer + MjData. Camera tracks `track_body_id`
    (typically the torso/root body, id=1). Pattern from scripts/record_video.py."""
    print(f"[render] {label}: rendering {n} frames...")
    t0 = time.time()
    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    mj_data = mujoco.MjData(mj_model)
    qpos_np = np.asarray(qpos_seq)
    qvel_np = np.asarray(qvel_seq)
    if qpos_np.ndim == 3:  # strip num_envs dim
        qpos_np = qpos_np[:, 0, :]
        qvel_np = qvel_np[:, 0, :]

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = track_body_id
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    frames = []
    for h in range(n):
        mj_data.qpos[:] = qpos_np[h]
        mj_data.qvel[:] = qvel_np[h]
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()
    elapsed = time.time() - t0
    print(f"[render] {label}: {len(frames)} frames in {elapsed:.1f}s ({elapsed/n*1000:.1f}ms/frame)")
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--load-ckpt", required=True, help="Path to ckpt dir (actor_params.npz + world_model_params.npz)")
    p.add_argument("--mode", choices=["mppi", "prior", "both"], default="both")
    p.add_argument("--num-steps", type=int, default=500, help="Rollout length (= episode_length for DMC default 500)")
    p.add_argument("--out-dir", default=None, help="Defaults to <ckpt>/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--num-envs", type=int, default=None, help="Override TDMPC2Config.num_envs (state-shape only)")
    args = p.parse_args()

    cfg = get_tdmpc2_preset(args.env)
    if args.num_envs is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, num_envs=args.num_envs)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.load_ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[record] env={args.env} ckpt={args.load_ckpt} mode={args.mode}")
    train_cfg = build_train_config_from_tdmpc2(cfg, args.env, total_timesteps=0, seed=args.seed)
    env_bundle = make_env_bundle(train_cfg, args.seed)
    print(f"[record] obs_dim={env_bundle.obs_dim} action_dim={env_bundle.action_dim}")

    state, _, _ = init_train_state(cfg, env_bundle.obs_dim, args.seed)
    state = load_params_into_state(state, args.load_ckpt)
    print(f"[record] params loaded")

    encoder, dynamics, reward_net, q_ensemble, policy = build_modules(cfg)
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

    eval_env = env_bundle.eval_env
    env_step = env_bundle.env_step
    dict_obs = env_bundle.dict_obs
    mj_model = eval_env.mj_model  # raw mujoco model (mujoco_playground exposes this)

    modes = ["mppi", "prior"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"\n[record] === mode={mode} ===")
        key = jax.random.PRNGKey(args.seed)
        key, reset_key = jax.random.split(key)
        env_state = eval_env.reset(jax.random.split(reset_key, 1))

        print(f"[scan] {mode}: rolling {args.num_steps} steps on GPU...")
        t0 = time.time()
        if mode == "mppi":
            qpos, qvel, rew = scan_mppi(
                env_step, env_state, key, args.num_steps, plan_fn, plan_params, encoder, cfg, dict_obs,
            )
        else:
            qpos, qvel, rew = scan_prior(
                env_step, env_state, key, args.num_steps, encoder, policy,
                plan_params["encoder"], plan_params["policy"], dict_obs,
            )
        rew.block_until_ready()
        total_reward = float(jnp.sum(rew))
        print(f"[scan] {mode}: done in {time.time()-t0:.1f}s, total_reward={total_reward:.2f}")

        frames = render_qpos_qvel(mj_model, qpos, qvel, args.num_steps, mode, args.width, args.height)
        # Source ctrl_dt=0.025 (40Hz). With action_repeat=2, agent decisions at 20Hz.
        # Render at 20fps so playback = real-time agent decisions.
        fps = 20
        out_path = out_dir / f"{mode}.mp4"
        imageio.mimsave(out_path.as_posix(), frames, fps=fps)
        print(f"[save] {out_path} ({len(frames)} frames @ {fps}fps)")


if __name__ == "__main__":
    main()
