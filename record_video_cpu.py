#!/usr/bin/env python3
"""Record video of a trained policy on the CPU Go2 env.

Same MJCF + overrides as MJX training env, running on CPU mj_step.
Tests MJX→CPU transfer without any MJCF model differences.

Usage:
    MUJOCO_GL=egl uv run python record_video_cpu.py \
        --checkpoint checkpoints/.../best \
        --vx 0.5 --duration 10
"""
import argparse
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import imageio

from jax_rl.envs.locomotion.go2_cpu import Go2CpuEnv
from deploy.policy_runner import PolicyRunner


def main():
    parser = argparse.ArgumentParser(description="Record CPU env video")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    env = Go2CpuEnv()
    runner = PolicyRunner(args.checkpoint)
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)

    print(f"Policy: {runner.algo}, obs={runner.obs_dim}d, act={runner.action_dim}d")
    print(f"Command: vx={args.vx}, vy={args.vy}, yaw={args.yaw}")

    obs = env.reset()
    renderer = mujoco.Renderer(env.model, width=1280, height=720)
    frames = []

    total_steps = int(args.duration / 0.02)  # 50Hz policy
    frame_skip = max(1, int(1.0 / args.fps / 0.02))

    print(f"Running {total_steps} steps ({args.duration}s)...")
    for step in range(total_steps):
        action = runner.get_action(obs)
        obs, base_z = env.step(action, command)

        if step % frame_skip == 0:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.model.body('base').id
            cam.distance = 2.0
            cam.azimuth = 135
            cam.elevation = -20
            renderer.update_scene(env.data, camera=cam)

            # Command arrow overlay (green arrow showing velocity direction)
            vx, vy = float(command[0]), float(command[1])
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 0.05:
                quat = env.data.qpos[3:7]
                w, x, y, z = quat
                fwd_x = 1 - 2*(y*y + z*z)
                fwd_y = 2*(x*y + w*z)
                right_x = 2*(x*y - w*z)
                right_y = 1 - 2*(x*x + z*z)
                world_vx = vx * fwd_x + vy * right_x
                world_vy = vx * fwd_y + vy * right_y

                base_pos = env.data.qpos[:3].copy()
                base_pos[2] = 0.4
                end_pos = base_pos.copy()
                end_pos[0] += world_vx * 0.3
                end_pos[1] += world_vy * 0.3
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[renderer.scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3), np.zeros(3), np.zeros(9), np.zeros(4),
                )
                mujoco.mjv_connector(
                    renderer.scene.geoms[renderer.scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.015,
                    base_pos.astype(np.float64),
                    end_pos.astype(np.float64),
                )
                renderer.scene.geoms[renderer.scene.ngeom].rgba = np.array([0, 1, 0, 0.8], dtype=np.float32)
                renderer.scene.ngeom += 1

            frames.append(renderer.render().copy())

        if step % 50 == 0:
            print(f"  Step {step:4d} | z={base_z:.3f} | pos=[{env.data.qpos[0]:.2f},{env.data.qpos[1]:.2f}]")

    renderer.close()

    out = args.out or os.path.join(
        os.path.dirname(args.checkpoint),
        f"cpu_rollout_{int(args.vx*10):+d}vx_{int(args.yaw*10):+d}yaw.mp4"
    )
    imageio.mimwrite(out, frames, fps=args.fps)
    print(f"Done: {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
