#!/usr/bin/env python3
"""Direct sim2sim: run policy in unitree_mujoco's Go2 model WITHOUT DDS.

This is the correct sim2sim approach (matching unitree_rl_gym's deploy_mujoco.py):
- Load unitree_mujoco's Go2 MJCF directly
- Policy at 50Hz (every `decimation` physics steps)
- PD recomputed at physics rate (every mj_step) from fresh joint state
- No DDS bridge, no command holding artifacts

Usage:
    deploy/.venv/bin/python deploy/sim2sim_direct.py \
        --checkpoint checkpoints/.../best \
        --vx 0.5 --duration 10 --record /tmp/sim2sim.mp4
"""
import argparse
import os
import sys
import time
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

# Project root on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deploy.policy_runner import PolicyRunner
from deploy.obs_builder import ObsBuilder, _quat_rotate_inverse
from deploy.go2_constants import (
    DEFAULT_POSE_POLICY, POLICY_TO_SDK, SDK_TO_POLICY,
    ACTION_SCALE, NUM_JOINTS, KP_SIM, KD_SIM,
)

UNITREE_MUJOCO = os.environ.get(
    "UNITREE_MUJOCO",
    os.path.expanduser("~/.local/share/unitree/unitree_mujoco")
)


def pd_control(
    target_q: np.ndarray,
    current_q: np.ndarray,
    current_dq: np.ndarray,
    kp: float,
    kd: float,
) -> np.ndarray:
    """Compute PD torque. All arrays in SDK joint order."""
    return kp * (target_q - current_q) + kd * (0.0 - current_dq)


def build_obs_from_mj(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    obs_builder: ObsBuilder,
    command: np.ndarray,
) -> np.ndarray:
    """Build obs from mj_data. Sensors are in SDK order (FR,FL,RR,RL)."""
    num_motor = model.nu  # 12

    # Joint pos/vel from sensordata (SDK order)
    joint_pos_sdk = np.array(data.sensordata[0:num_motor], dtype=np.float32)
    joint_vel_sdk = np.array(data.sensordata[num_motor:2*num_motor], dtype=np.float32)

    # IMU from sensordata (after 3*num_motor = 36 joint sensors)
    offset = 3 * num_motor  # skip pos(12) + vel(12) + torque(12)
    quat = np.array(data.sensordata[offset:offset+4], dtype=np.float32)      # [w,x,y,z]
    gyro = np.array(data.sensordata[offset+4:offset+7], dtype=np.float32)    # [wx,wy,wz]

    # Accelerometer (specific force, body frame). unitree mujoco scene exposes
    # this via the standard `accelerometer` sensor on the IMU site.
    accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'accelerometer')
    if accel_id >= 0:
        acc_adr = model.sensor_adr[accel_id]
        accel = np.array(data.sensordata[acc_adr:acc_adr+3], dtype=np.float32)
    else:
        # Fallback: zero accel. Sim2sim parity will degrade — fix the scene to
        # include the accelerometer sensor.
        accel = np.zeros(3, dtype=np.float32)

    return obs_builder.build(
        joint_pos_sdk=joint_pos_sdk,
        joint_vel_sdk=joint_vel_sdk,
        gyroscope=gyro,
        accelerometer=accel,
        quaternion=quat,
        command=command,
    )


def run_sim2sim(
    checkpoint: str,
    command: np.ndarray,
    duration: float = 10.0,
    record_path: str | None = None,
    fps: int = 30,
):
    # Load policy
    print(f"Loading policy from {checkpoint}")
    runner = PolicyRunner(checkpoint)
    print(f"  algo={runner.algo}, obs={runner.obs_dim}d, act={runner.action_dim}d")
    print(f"  hidden={runner.hidden_dim}, activation={runner.activation}")
    print(f"  obs_norm={'yes' if runner.use_obs_norm else 'no'} (n={runner.norm_count})")

    # Schema-driven obs builder: reads obs term layout from meta.json.
    obs_builder = ObsBuilder.from_checkpoint(checkpoint, n_frame_stack=runner.n_frame_stack)
    print(f"  obs_schema={obs_builder.state_schema} (raw_dim={obs_builder.raw_dim})")

    # Load unitree_mujoco Go2 model
    scene_path = os.path.join(UNITREE_MUJOCO, "unitree_robots", "go2", "scene.xml")
    if not os.path.exists(scene_path):
        print(f"ERROR: {scene_path} not found. Set UNITREE_MUJOCO env var.")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Match training env physics (go2_warp_base.py overrides).
    # Contact: unitree defaults condim=6, friction=[0.4,0.02,0.01], solimp=Menagerie.
    for foot_name in ["FL", "FR", "RL", "RR"]:
        gid = model.geom(foot_name).id
        model.geom_solimp[gid, :3] = [0.9, 0.95, 0.023]
        model.geom_condim[gid] = 3
        model.geom_friction[gid] = [0.6, 0.005, 0.0001]
    # Match what we CAN without breaking unitree's solver stability.
    # Contact properties and CCD match training. Solver stays at unitree defaults
    # (elliptic cone, 100 iterations) because their collision mesh requires it.
    model.opt.ccd_iterations = 20
    model.geom_friction[model.geom('floor').id] = [0.6, 0.005, 0.0001]
    # Force limits: unitree has forcerange=[0,0] (unlimited). Match training.
    import mujoco as _mj
    for i in range(model.nu):
        name = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_ACTUATOR, i)
        if 'calf' in name.lower():
            model.actuator_forcerange[i] = [-45.43, 45.43]
        else:
            model.actuator_forcerange[i] = [-23.7, 23.7]

    # Match training env: sim_dt=0.004, 5 substeps per ctrl_dt=0.02.
    # unitree_mujoco XML default is 0.002 but we override to match training.
    # Native dt (0.002 × 10) was tested — worse transfer due to amplified contact mismatch.
    physics_dt = 0.004
    model.opt.timestep = physics_dt
    decimation = 5  # 5 * 0.004 = 0.02s = 50Hz policy
    policy_dt = physics_dt * decimation
    policy_dt = physics_dt * decimation

    print(f"\nSimulator: {scene_path}")
    print(f"  physics_dt={physics_dt}s, decimation={decimation}, policy_dt={policy_dt}s")
    print(f"  PD gains: Kp={KP_SIM}, Kd={KD_SIM}")

    # Reset to home keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Video recording
    renderer = None
    frames = []
    if record_path:
        renderer = mujoco.Renderer(model, width=640, height=480)
        print(f"  Recording to {record_path}")

    # Trajectory buffers
    traj_obs = []
    traj_actions = []
    traj_base_pos = []
    traj_joint_vel = []

    total_physics_steps = int(duration / physics_dt)
    total_policy_steps = total_physics_steps // decimation

    # Default pose in SDK order (for PD targets)
    default_pose_sdk = DEFAULT_POSE_POLICY[POLICY_TO_SDK]
    current_target_sdk = default_pose_sdk.copy()

    print(f"\nRunning {total_policy_steps} policy steps ({duration}s)...")
    print(f"  command: vx={command[0]:.1f}, vy={command[1]:.1f}, yaw={command[2]:.1f}")

    policy_step = 0
    for phys_step in range(total_physics_steps):
        # Policy step (every `decimation` physics steps)
        if phys_step % decimation == 0:
            obs = build_obs_from_mj(data, model, obs_builder, command)
            action = runner.get_action(obs)
            obs_builder.update_last_action(action)

            # Convert action to joint targets (SDK order)
            action_sdk = action[POLICY_TO_SDK]
            current_target_sdk = default_pose_sdk + action_sdk * ACTION_SCALE

            # Log
            traj_obs.append(obs.copy())
            traj_actions.append(action.copy())
            traj_base_pos.append(data.qpos[:3].copy())
            num_motor = model.nu
            traj_joint_vel.append(data.sensordata[num_motor:2*num_motor].copy())

            policy_step += 1
            if policy_step % 50 == 0:
                bz = data.qpos[2]
                bx, by = data.qpos[0], data.qpos[1]
                jvel_max = abs(data.sensordata[model.nu:2*model.nu]).max()
                print(f"  Step {policy_step:4d} | z={bz:.3f}m | pos=[{bx:.2f},{by:.2f}] | "
                      f"jvel_max={jvel_max:.1f} | action [{action.min():.2f},{action.max():.2f}]")

        # PD control at physics rate (every mj_step)
        q_current = data.sensordata[0:model.nu]
        dq_current = data.sensordata[model.nu:2*model.nu]
        tau = pd_control(current_target_sdk, q_current, dq_current, KP_SIM, KD_SIM)
        data.ctrl[:] = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

        mujoco.mj_step(model, data)

        # Capture frame
        if renderer and phys_step % int(1.0 / fps / physics_dt) == 0:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = model.body('base_link').id
            cam.distance = 2.0
            cam.azimuth = 135
            cam.elevation = -20
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    # Final stats
    print(f"\nDone: {policy_step} policy steps")
    if traj_obs:
        obs_arr = np.array(traj_obs)
        act_arr = np.array(traj_actions)
        jvel_arr = np.array(traj_joint_vel)
        print(f"  obs range: [{obs_arr.min():.1f}, {obs_arr.max():.1f}]")
        print(f"  jvel range: [{jvel_arr.min():.1f}, {jvel_arr.max():.1f}]")
        print(f"  action range: [{act_arr.min():.2f}, {act_arr.max():.2f}]")
        print(f"  final base_z: {data.qpos[2]:.3f}m")

    # Save trajectory
    traj_path = (record_path or "/tmp/sim2sim_direct").replace(".mp4", "_traj.npz")
    np.savez_compressed(traj_path,
        obs=np.array(traj_obs),
        actions=np.array(traj_actions),
        base_pos=np.array(traj_base_pos),
        joint_vel_sdk=np.array(traj_joint_vel),
    )
    print(f"  Trajectory: {traj_path}")

    # Save video
    if renderer and frames:
        try:
            import imageio
            imageio.mimwrite(record_path, frames, fps=fps)
            print(f"  Video: {record_path}")
        except ImportError:
            npz = record_path.replace(".mp4", "_frames.npz")
            np.savez_compressed(npz, frames=np.array(frames))
            print(f"  Frames saved: {npz} (install imageio for mp4)")
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Direct sim2sim (no DDS)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    run_sim2sim(
        checkpoint=args.checkpoint,
        command=np.array([args.vx, args.vy, args.yaw], dtype=np.float32),
        duration=args.duration,
        record_path=args.record,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
