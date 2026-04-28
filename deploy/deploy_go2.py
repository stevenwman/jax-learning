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
from deploy.go2_constants import POLICY_DT, NUM_JOINTS


def interpolate_to_stand(iface: Go2Interface, duration: float = 2.0, dt: float = 0.002):
    """Smoothly interpolate from current pose to checkpoint default pose."""
    print(f"  Interpolating to stand ({duration}s)...")
    state = None
    while state is None:
        state = iface.get_state()
        time.sleep(0.01)

    start_pos = state["joint_pos_sdk"]
    target_pose = iface.default_pose_sdk
    steps = int(duration / dt)

    for step in range(steps):
        t = (step + 1) / steps
        alpha = 0.5 * (1 - np.cos(np.pi * t))  # smooth cosine interpolation
        target = start_pos + alpha * (target_pose - start_pos)
        iface.send_joint_targets(target)
        time.sleep(dt)


def run_policy_loop(
    runner: PolicyRunner,
    obs_builder: ObsBuilder,
    iface: Go2Interface,
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

            obs = obs_builder.build(
                joint_pos_sdk=state["joint_pos_sdk"],
                joint_vel_sdk=state["joint_vel_sdk"],
                gyroscope=state["gyroscope"],
                accelerometer=state["accelerometer"],
                quaternion=state["quaternion"],
                command=command,
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

    # Load checkpoint metadata once — drives obs schema, default pose, and PD gains.
    import json
    meta_path = os.path.join(args.checkpoint, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    control_meta = meta.get("control")

    # Construct ObsBuilder with strict schema for real arm; legacy fallback only
    # in --sim mode. Real robot refuses to load a ckpt without obs_schema.
    obs_builder = ObsBuilder.from_checkpoint(
        args.checkpoint,
        n_frame_stack=runner.n_frame_stack,
        strict=not sim,
    )
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)

    # Connect — pass control_meta when present so robot_interface uses
    # checkpoint-derived Kp/Kd/action_scale/default_pose.
    print(f"\n[2/4] Connecting to Go2 ({mode}) on '{interface}'")
    if control_meta is None and not sim:
        raise RuntimeError(
            f"meta['control'] missing from {meta_path} — refuse to arm real "
            f"robot. Re-train with current code or pass --sim for legacy fallback."
        )
    iface = Go2Interface(sim=sim, interface=interface, control_meta=control_meta)
    iface.start()

    print("  Waiting for robot state...")
    t_wait = time.monotonic()
    while iface.get_state() is None:
        if time.monotonic() - t_wait > 10.0:
            print("  ERROR: No robot state after 10s. Is the simulator/robot running?")
            return
        time.sleep(0.1)
    print("  State received")

    # Pre-arm sanity: validate raw IMU + quat against expected stance values.
    # Catches the .temp/obs_prinout.txt failure mode (gravity_z and accel_z same
    # sign → IMU/quat frame mismatch). See deploy/test_time_validate.md §3.
    # WARN-only on first cycle to avoid bricking a deploy on an over-strict
    # check; promote to assert after one clean run.
    _state = iface.get_state()
    _accel = _state["accelerometer"]
    _gravity = _quat_rotate_inverse(_state["quaternion"], np.array([0.0, 0.0, -1.0], dtype=np.float32))
    _accel_norm = float(np.linalg.norm(_accel))
    _grav_norm = float(np.linalg.norm(_gravity))
    print(f"\n  Pre-arm sanity:")
    print(f"    quat        = {_state['quaternion']}")
    print(f"    accel       = {_accel}  (norm={_accel_norm:.2f}, expect ~9.8)")
    print(f"    gravity     = {_gravity}  (norm={_grav_norm:.3f}, expect ~1.0)")
    if _accel_norm < 9.0:
        print(f"    WARN: |accel|={_accel_norm:.2f} < 9.0 — units bug? (g vs m/s²)")
    if abs(_grav_norm - 1.0) > 0.05:
        print(f"    WARN: |gravity|={_grav_norm:.3f} ≠ 1.0 — quat not normalized")
    if np.sign(_accel[2]) == np.sign(_gravity[2]):
        print(f"    WARN: sign(accel_z)={int(np.sign(_accel[2]))} == sign(gravity_z)={int(np.sign(_gravity[2]))}")
        print(f"          — at upright stance these should be OPPOSITE.")
        print(f"          Likely IMU mount frame vs quat convention mismatch.")
        print(f"          See deploy/test_time_validate.md §3.C.")
    if _grav_norm > 0.05:
        _tilt_deg = float(np.degrees(np.arccos(np.clip(-_gravity[2] / _grav_norm, -1.0, 1.0))))
        print(f"    tilt        = {_tilt_deg:.1f}° (expect <5° at stance)")
        if _tilt_deg > 15.0:
            print(f"    WARN: tilt {_tilt_deg:.1f}° > 15° — robot not upright or gravity decode wrong")
    if not sim:
        input("  Pre-arm sanity printed. Enter to proceed (or Ctrl+C to abort)... ")

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
    run_policy_loop(runner, obs_builder, iface, command,
                    save_traj=args.save_traj, max_steps=args.max_steps)

    # Cleanup
    print("  Returning to stand...")
    for _ in range(500):
        iface.send_stand()
        time.sleep(0.002)
    print("Done.")


if __name__ == "__main__":
    main()
