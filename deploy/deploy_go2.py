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
import time
import numpy as np

from deploy.policy_runner import PolicyRunner
from deploy.obs_builder import ObsBuilder
from deploy.robot_interface import Go2Interface
from deploy.go2_constants import POLICY_DT, DEFAULT_POSE_SDK, NUM_JOINTS


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
    command: np.ndarray,
):
    """Run policy at 50Hz until Ctrl+C."""
    print(f"  Policy running — cmd: vx={command[0]:.1f} vy={command[1]:.1f} yaw={command[2]:.1f}")
    print("  Press Ctrl+C to stop")

    step = 0
    try:
        while True:
            t_start = time.monotonic()

            state = iface.get_state()
            if state is None:
                time.sleep(POLICY_DT)
                continue

            obs = obs_builder.build(
                joint_pos_sdk=state["joint_pos_sdk"],
                joint_vel_sdk=state["joint_vel_sdk"],
                gyroscope=state["gyroscope"],
                quaternion=state["quaternion"],
                command=command,
            )

            action = runner.get_action(obs)
            iface.send_action(action)
            obs_builder.update_last_action(action)

            step += 1
            if step % 50 == 0:
                q = state["joint_pos_sdk"]
                print(f"  Step {step:5d} | action [{action.min():.2f}, {action.max():.2f}] | "
                      f"q [{q.min():.2f}, {q.max():.2f}]")

            elapsed = time.monotonic() - t_start
            sleep_time = POLICY_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n  Stopping policy loop")


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
    run_policy_loop(runner, obs_builder, iface, command)

    # Cleanup
    print("  Returning to stand...")
    for _ in range(500):
        iface.send_stand()
        time.sleep(0.002)
    print("Done.")


if __name__ == "__main__":
    main()
