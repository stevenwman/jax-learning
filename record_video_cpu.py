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
    parser.add_argument("--vx", type=float, default=None, help="Fixed vx (default: random resampling)")
    parser.add_argument("--vy", type=float, default=None)
    parser.add_argument("--yaw", type=float, default=None)
    parser.add_argument("--duration", type=float, default=20.0)  # match MJX episode length
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--fps", type=int, default=50)  # 50Hz policy = 50fps for real-time
    args = parser.parse_args()

    env = Go2CpuEnv()
    runner = PolicyRunner(args.checkpoint)
    fixed_cmd = args.vx is not None or args.vy is not None or args.yaw is not None
    if fixed_cmd:
        command = np.array([args.vx or 0.0, args.vy or 0.0, args.yaw or 0.0], dtype=np.float32)
    else:
        command = np.zeros(3, dtype=np.float32)

    print(f"Policy: {runner.algo}, obs={runner.obs_dim}d, act={runner.action_dim}d")
    print(f"Command: {'fixed ' + str(command) if fixed_cmd else 'random resampling (like training)'}")

    obs = env.reset()
    renderer = mujoco.Renderer(env.model, width=1280, height=720)
    frames = []

    # Trajectory buffers
    traj_qpos = []
    traj_qvel = []
    traj_actions = []
    traj_commands = []

    total_steps = int(args.duration / 0.02)  # 50Hz policy
    frame_skip = max(1, int(1.0 / args.fps / 0.02))

    # Command resampling (matches training env: resample every ~5s)
    # Probability of non-zero per axis: [0.9, 0.25, 0.5] from default_config.command_config.b
    rng = np.random.default_rng(42)
    cmd_max = np.array([1.5, 0.8, 1.2])  # from default_config command_config.a
    cmd_prob = np.array([0.9, 0.25, 0.5])  # probability of non-zero per axis
    steps_until_resample = 0 if not fixed_cmd else total_steps + 1

    print(f"Running {total_steps} steps ({args.duration}s)...")
    for step in range(total_steps):
        if not fixed_cmd and steps_until_resample <= 0:
            # Sample command with per-axis zero probability (matches training)
            raw = rng.uniform(-cmd_max, cmd_max).astype(np.float32)
            mask = (rng.random(3) < cmd_prob).astype(np.float32)
            command = raw * mask
            steps_until_resample = int(rng.exponential(5.0) / 0.02)
            print(f"  New cmd: vx={command[0]:.2f} vy={command[1]:.2f} yaw={command[2]:.2f} "
                  f"(next in {steps_until_resample} steps)")
        steps_until_resample -= 1

        action = runner.get_action(obs)

        traj_qpos.append(env.data.qpos.copy())
        traj_qvel.append(env.data.qvel.copy())
        traj_actions.append(action.copy())
        traj_commands.append(command.copy())

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

            # Yaw command indicator (yellow arc arrow)
            yaw_cmd = float(command[2])
            if abs(yaw_cmd) > 0.05:
                base_pos = env.data.qpos[:3].copy()
                base_pos[2] = 0.45
                quat = env.data.qpos[3:7]
                w, x, y, z = quat
                # Robot forward direction
                fwd_x = 1 - 2*(y*y + z*z)
                fwd_y = 2*(x*y + w*z)
                # Draw arc: start from front of robot, curve left (positive yaw) or right
                radius = 0.2
                sign = 1.0 if yaw_cmd > 0 else -1.0
                # Perpendicular to forward (left = positive yaw)
                perp_x = -fwd_y * sign
                perp_y = fwd_x * sign
                # Arc start: slightly ahead
                arc_start = base_pos.copy()
                arc_start[0] += fwd_x * radius
                arc_start[1] += fwd_y * radius
                # Arc end: rotated by yaw magnitude (clamped)
                arc_angle = min(abs(yaw_cmd) * 0.5, 0.8)
                arc_end = base_pos.copy()
                arc_end[0] += (fwd_x * np.cos(arc_angle * sign) - fwd_y * np.sin(arc_angle * sign)) * radius
                arc_end[1] += (fwd_x * np.sin(arc_angle * sign) + fwd_y * np.cos(arc_angle * sign)) * radius
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[renderer.scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3), np.zeros(3), np.zeros(9), np.zeros(4),
                )
                mujoco.mjv_connector(
                    renderer.scene.geoms[renderer.scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.01,
                    arc_start.astype(np.float64),
                    arc_end.astype(np.float64),
                )
                renderer.scene.geoms[renderer.scene.ngeom].rgba = np.array([1, 1, 0, 0.8], dtype=np.float32)
                renderer.scene.ngeom += 1

            frames.append(renderer.render().copy())

        if step % 50 == 0:
            print(f"  Step {step:4d} | z={base_z:.3f} | pos=[{env.data.qpos[0]:.2f},{env.data.qpos[1]:.2f}]")

    renderer.close()

    if args.out:
        out = args.out
    elif fixed_cmd:
        vx_s = int((args.vx or 0) * 10)
        yaw_s = int((args.yaw or 0) * 10)
        out = os.path.join(os.path.dirname(args.checkpoint), f"cpu_rollout_{vx_s:+d}vx_{yaw_s:+d}yaw.mp4")
    else:
        out = os.path.join(os.path.dirname(args.checkpoint), "cpu_rollout_random_cmd.mp4")
    imageio.mimwrite(out, frames, fps=args.fps)
    print(f"Done: {out} ({len(frames)} frames)")

    # Save trajectory
    traj_path = out.replace(".mp4", "_traj.npz")
    np.savez_compressed(traj_path,
        qpos=np.array(traj_qpos),
        qvel=np.array(traj_qvel),
        actions=np.array(traj_actions),
        commands=np.array(traj_commands),
    )
    print(f"Trajectory: {traj_path}")


if __name__ == "__main__":
    main()
