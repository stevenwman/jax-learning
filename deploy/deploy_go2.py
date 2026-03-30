#!/usr/bin/env python3
# deploy/deploy_go2.py
"""Deploy trained JAX RL policy on Go2 robot (sim or real).

Usage:
    # Sim2sim (unitree_mujoco must be running):
    deploy/.venv/bin/python deploy/deploy_go2.py --checkpoint checkpoints/.../best --sim

    # Sim2real (Go2 EDU connected via ethernet):
    deploy/.venv/bin/python deploy/deploy_go2.py --checkpoint checkpoints/.../best --interface enp2s0

FSM: IDLE -> STAND (interpolation) -> HOLD -> POLICY (runs until Ctrl+C)
"""
import argparse
import os
import sys
import time
import numpy as np

# Ensure project root is on path when running as `deploy/.venv/bin/python deploy/deploy_go2.py`
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deploy.policy_runner import PolicyRunner
from deploy.obs_builder import ObsBuilder, _quat_rotate_inverse
from deploy.robot_interface import Go2Interface
from deploy.go2_constants import POLICY_DT, DEFAULT_POSE_SDK, NUM_JOINTS


class VelocityEstimator:
    """Estimate local linear velocity from IMU accelerometer + leaky integration.

    Integrates body-frame acceleration (minus gravity) at policy rate.
    Leaky factor prevents drift: v = alpha * (v + a*dt).
    """
    def __init__(self, dt: float = 0.02, alpha: float = 0.95):
        self.dt = dt
        self.alpha = alpha
        self.velocity = np.zeros(3, dtype=np.float32)
        self.gravity_world = np.array([0.0, 0.0, 9.81], dtype=np.float32)

    def update(self, accelerometer: np.ndarray) -> np.ndarray:
        """Update velocity estimate. Returns local (body-frame) linear velocity."""
        # MuJoCo's accelerometer sensor reads specific force (accel - gravity)
        # in body frame, so raw accel is body-frame linear acceleration.
        self.velocity = self.alpha * (self.velocity + accelerometer * self.dt)
        return self.velocity.copy()

    def reset(self):
        self.velocity = np.zeros(3, dtype=np.float32)


def interpolate_to_stand(iface: Go2Interface, duration: float = 2.0, dt: float = 0.002):
    """Smoothly interpolate from current pose to default standing pose."""
    print(f"  Interpolating to stand ({duration}s)...")
    state = None
    while state is None:
        state = iface.get_state()
        time.sleep(0.01)

    start_pos = state["joint_pos_sdk"]
    steps = int(duration / dt)

    for step in range(steps):
        t = (step + 1) / steps
        alpha = 0.5 * (1 - np.cos(np.pi * t))  # smooth cosine interpolation
        target = start_pos + alpha * (DEFAULT_POSE_SDK - start_pos)
        iface.send_joint_targets(target)
        time.sleep(dt)


def run_policy_loop(
    runner: PolicyRunner,
    obs_builder: ObsBuilder,
    iface: Go2Interface,
    vel_estimator: 'VelocityEstimator',
    command: np.ndarray,
    save_traj: str | None = None,
    max_steps: int = 0,
):
    """Run policy at 50Hz until Ctrl+C or max_steps."""
    print(f"  Policy running — cmd: vx={command[0]:.1f} vy={command[1]:.1f} yaw={command[2]:.1f}")
    if max_steps > 0:
        print(f"  Running {max_steps} steps ({max_steps * POLICY_DT:.1f}s)")
    else:
        print("  Press Ctrl+C to stop")

    # Trajectory buffers
    traj_obs = []
    traj_actions = []
    traj_joint_pos = []
    traj_joint_vel = []
    traj_gyro = []
    traj_quat = []

    step = 0
    try:
        while max_steps <= 0 or step < max_steps:
            t_start = time.monotonic()

            state = iface.get_state()
            if state is None:
                time.sleep(POLICY_DT)
                continue

            # Estimate local velocity from IMU accelerometer
            local_linvel = vel_estimator.update(state["accelerometer"])

            obs = obs_builder.build(
                joint_pos_sdk=state["joint_pos_sdk"],
                joint_vel_sdk=state["joint_vel_sdk"],
                gyroscope=state["gyroscope"],
                quaternion=state["quaternion"],
                command=command,
                linvel=local_linvel,
            )

            action = runner.get_action(obs)
            iface.send_action(action)
            obs_builder.update_last_action(action)

            # Record trajectory
            traj_obs.append(obs.copy())
            traj_actions.append(action.copy())
            traj_joint_pos.append(state["joint_pos_sdk"].copy())
            traj_joint_vel.append(state["joint_vel_sdk"].copy())
            traj_gyro.append(state["gyroscope"].copy())
            traj_quat.append(state["quaternion"].copy())

            step += 1
            if step % 50 == 0:
                q = state["joint_pos_sdk"]
                print(f"  Step {step:5d} | action [{action.min():.2f}, {action.max():.2f}] | "
                      f"q [{q.min():.2f}, {q.max():.2f}] | "
                      f"obs [{obs.min():.2f}, {obs.max():.2f}]")

            elapsed = time.monotonic() - t_start
            sleep_time = POLICY_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n  Stopping policy loop")

    # Save trajectory
    if save_traj and traj_obs:
        np.savez_compressed(save_traj,
            obs=np.array(traj_obs),
            actions=np.array(traj_actions),
            joint_pos_sdk=np.array(traj_joint_pos),
            joint_vel_sdk=np.array(traj_joint_vel),
            gyroscope=np.array(traj_gyro),
            quaternion=np.array(traj_quat),
        )
        print(f"  Trajectory saved: {save_traj} ({step} steps)")


def main():
    parser = argparse.ArgumentParser(description="Deploy Go2 RL policy")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir (with actor_params.npy)")
    parser.add_argument("--sim", action="store_true", help="Use sim (unitree_mujoco, domain=1, lo)")
    parser.add_argument("--interface", default=None, help="Network interface for real robot (e.g. enp2s0)")
    parser.add_argument("--vx", type=float, default=0.5, help="Forward velocity cmd (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity cmd (m/s)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rate cmd (rad/s)")
    parser.add_argument("--stand-duration", type=float, default=2.0, help="Stand-up time (s)")
    parser.add_argument("--hold-duration", type=float, default=1.0, help="Hold standing time (s)")
    parser.add_argument("--save-traj", type=str, default=None, help="Save trajectory .npz (obs, actions, joints)")
    parser.add_argument("--max-steps", type=int, default=0, help="Max policy steps (0=run forever)")
    args = parser.parse_args()

    if not args.sim and args.interface is None:
        parser.error("Must specify --sim or --interface <name>")

    # Determine mode
    sim = args.sim
    interface = "lo" if sim else args.interface
    mode = "SIM" if sim else "REAL"

    # Load policy
    print(f"[1/4] Loading policy from {args.checkpoint}")
    runner = PolicyRunner(args.checkpoint)
    print(f"  algo={runner.algo}, obs={runner.obs_dim}d, act={runner.action_dim}d")
    print(f"  hidden={runner.hidden_dim}, activation={runner.activation}")
    print(f"  obs_norm={'yes' if runner.use_obs_norm else 'no'} (n={runner.norm_count})")

    obs_builder = ObsBuilder()
    vel_estimator = VelocityEstimator(dt=POLICY_DT)
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)

    # Connect
    print(f"\n[2/4] Connecting to Go2 ({mode}) on '{interface}'")
    iface = Go2Interface(sim=sim, interface=interface)
    iface.start()

    print("  Waiting for robot state...")
    t_wait = time.monotonic()
    while iface.get_state() is None:
        if time.monotonic() - t_wait > 10.0:
            print("  ERROR: No robot state after 10s. Is the simulator/robot running?")
            return
        time.sleep(0.1)
    print("  State received")

    # Safety gate for real robot
    if not sim:
        print(f"\n  === REAL ROBOT MODE ===")
        print(f"  Kp={iface.kp}, Kd={iface.kd}")
        input("  Press Enter to start (robot will move!)... ")

    # FSM
    print(f"\n[3/4] Standing up")
    interpolate_to_stand(iface, duration=args.stand_duration)

    print(f"  Holding ({args.hold_duration}s)...")
    t_hold = time.monotonic()
    while time.monotonic() - t_hold < args.hold_duration:
        iface.send_stand()
        time.sleep(0.002)

    print(f"\n[4/4] Running policy")
    run_policy_loop(runner, obs_builder, iface, vel_estimator, command,
                    save_traj=args.save_traj, max_steps=args.max_steps)

    # Cleanup
    print("  Returning to stand...")
    for _ in range(500):
        iface.send_stand()
        time.sleep(0.002)
    print("Done.")


if __name__ == "__main__":
    main()
