"""Record a video of PPO policy on CartpoleBalance.

Usage:
  MUJOCO_GL=egl uv run python record_video.py                    # random policy
  MUJOCO_GL=egl uv run python record_video.py --checkpoint ckpt  # trained policy

Two-phase approach:
  1. JIT-scan the rollout on GPU (fast — collects all states)
  2. Render frames on CPU from saved states (slow but unavoidable)
"""

import argparse
from datetime import datetime
import os
import time

import imageio
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from mujoco_playground import dm_control_suite

from jax_rl.algos.ppo_scan import PPO
from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.utils.normalization import (
    init as norm_init,
    update as norm_update,
    normalize as norm_normalize,
)


ENV_DEFAULTS = {
    "CartpoleBalance": ((64, 64), "fixed"),
    "CheetahRun":      ((256, 256), "side"),
}


def record(env_name: str | None = None, checkpoint: str | None = None,
           out: str = "rollout.mp4", max_steps: int = 1000,
           hidden_dim: tuple[int, ...] | None = None, camera: str | None = None):
    # Load config from checkpoint metadata if available
    if checkpoint is not None:
        meta_path = os.path.join(checkpoint, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            env_name = env_name or meta["env_name"]
            hidden_dim = hidden_dim or tuple(meta["hidden_dim"])
            print(f"Loaded meta: env={env_name}, hidden_dim={hidden_dim}")

    env_name = env_name or "CartpoleBalance"
    defaults = ENV_DEFAULTS.get(env_name, ((256, 256), None))
    hidden_dim = hidden_dim or defaults[0]
    camera = camera or defaults[1]

    env = dm_control_suite.load(env_name)
    env_step = jax.jit(env.step)

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    env_state = env.reset(reset_key)

    obs_dim = env_state.obs.shape[-1]
    action_dim = env.action_size

    config = PPOConfig(
        encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=hidden_dim),
        policy_head=PolicyHeadConfig(action_dim=action_dim, squash=False),
        num_envs=1,
        num_steps=64,
    )

    # Dummy optimizers — only needed for init shapes, not training
    dummy_opt = optax.adam(1e-3)
    ppo = PPO(config, obs_dim, action_dim, dummy_opt, dummy_opt)
    key, init_key = jax.random.split(key)
    training_state = ppo.init(init_key)

    if checkpoint is not None:
        print(f"Loading checkpoint: {checkpoint}")
        target = {"training_state": training_state, "norm_state": norm_init(obs_dim)}
        ckpt = ocp.StandardCheckpointer().restore(os.path.abspath(checkpoint), target=target)
        training_state = ckpt["training_state"]
        norm_state = ckpt["norm_state"]
        print("Loaded trained params + normalization stats")
    else:
        norm_state = norm_init(obs_dim)
        print("Using random (untrained) policy")

    # ── Phase 1: JIT-scan rollout on GPU ──────────────────────────────────
    def rollout_step(carry, _):
        env_state, norm_state, key = carry
        obs = env_state.obs[None]  # (1, obs_dim) — policy expects batch dim
        norm_state = norm_update(norm_state, obs)
        normed_obs = norm_normalize(norm_state, obs)

        key, action_key = jax.random.split(key)
        action, _, _ = ppo.select_action(
            training_state, normed_obs, action_key, deterministic=True
        )
        clipped_action = jnp.clip(action, -1.0, 1.0).squeeze(0)

        env_state = env_step(env_state, clipped_action)
        return (env_state, norm_state, key), env_state

    print("JIT-compiling rollout scan...")
    t0 = time.time()
    (final_state, _, _), trajectory = jax.lax.scan(
        rollout_step, (env_state, norm_state, key), None, length=max_steps
    )
    jax.block_until_ready(trajectory.obs)
    t_rollout = time.time() - t0
    print(f"Rollout done: {max_steps} steps in {t_rollout:.2f}s (includes JIT compilation)")

    total_reward = float(trajectory.reward.sum())
    print(f"Total reward: {total_reward:.1f}")

    # Find first done step (unwrapped env doesn't auto-reset)
    dones = np.asarray(trajectory.done)
    done_indices = np.where(dones > 0.5)[0]
    if len(done_indices) > 0:
        num_frames = int(done_indices[0]) + 1
        print(f"Episode ended at step {num_frames}")
    else:
        num_frames = max_steps
        print(f"Episode ran full {max_steps} steps (no termination)")

    # ── Phase 2: Render frames on CPU ─────────────────────────────────────
    # Unstack trajectory into list of individual states for rendering
    # Include initial state + trajectory states up to done
    print(f"Rendering {num_frames + 1} frames (CPU)...")
    t0 = time.time()

    # Render initial state
    states = [env_state]
    for i in range(num_frames):
        state_i = jax.tree.map(lambda x: x[i], trajectory)
        states.append(state_i)

    frames = env.render(states, camera=camera)
    t_render = time.time() - t0
    print(f"Render done: {len(frames)} frames in {t_render:.2f}s "
          f"({t_render / len(frames) * 1000:.1f}ms/frame)")

    # ── Save video ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint is not None:
        video_path = os.path.join(checkpoint, f"{timestamp}_rollout.mp4")
    else:
        video_path = out if out != "rollout.mp4" else f"{timestamp}_rollout.mp4"

    print(f"Saving to {video_path}...")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Done: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="rollout.mp4")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=None)
    parser.add_argument("--camera", type=str, default=None)
    args = parser.parse_args()
    record(
        env_name=args.env, checkpoint=args.checkpoint, out=args.out,
        max_steps=args.max_steps,
        hidden_dim=tuple(args.hidden_dim) if args.hidden_dim else None,
        camera=args.camera,
    )
