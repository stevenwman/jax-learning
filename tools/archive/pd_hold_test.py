#!/usr/bin/env python
"""Warp Go2: hold keyframe pose with PD (zero action), render video.

Simple sanity check — does the robot stand still when PD holds home pose?

Usage:
    MUJOCO_GL=egl uv run python tools/pd_hold_test.py
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env


def main():
    from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick

    env = WarpJoystick()

    # Clean reset — no random perturbation, just keyframe
    qpos = jnp.array(env._mj_model.keyframe("home").qpos)
    qvel = jnp.zeros(env.mjx_model.nv)
    default_pose = jnp.array(env._mj_model.keyframe("home").qpos[7:])

    data = mjx_env.make_data(
        env.mj_model,
        qpos=qpos, qvel=qvel,
        ctrl=jnp.zeros(env.mjx_model.nu),
        impl=env.mjx_model.impl.value,
        naconmax=512,
        naccdmax=512,
        njmax=100,
    )
    data = mjx.forward(env.mjx_model, data)

    model = env.mjx_model
    kp, kd = 20.0, 0.5  # unitree_rl_gym gains (matches Warp env config)
    action_scale = env._config.action_scale  # 0.5
    n_steps = 500
    n_substeps = 5

    rng = jax.random.PRNGKey(42)

    def one_ctrl_step(carry, step_idx):
        data, rng = carry

        # Random action every step — sample desired position uniformly in [-1, 1]
        rng, action_key = jax.random.split(rng)
        action = jax.random.uniform(action_key, (12,), minval=-1.0, maxval=1.0)
        motor_targets = default_pose + action * action_scale

        a2j = env._act_to_joint

        def substep(data, _):
            q = data.qpos[7:]
            dq = data.qvel[6:]
            tau_joint = kp * (motor_targets - q) + kd * (0.0 - dq)
            tau_act = tau_joint[a2j]
            data = data.replace(ctrl=tau_act)
            return mjx.step(model, data), None
        data, _ = jax.lax.scan(substep, data, None, length=n_substeps)
        return (data, rng), (data.qpos, data.qvel, data.ctrl, motor_targets)

    print("JIT compiling...")
    _, (all_qpos, all_qvel, all_ctrl, all_targets) = jax.lax.scan(
        one_ctrl_step, (data, rng), jnp.arange(n_steps), length=n_steps
    )
    jax.block_until_ready(all_qpos)
    print("Done.\n")

    trajectory = [np.array(data.qpos)] + [np.array(all_qpos[i]) for i in range(n_steps)]
    targets_np = np.array(all_targets)  # (n_steps, 12) — varies per step

    # Save everything to npz
    q_joints = np.array(all_qpos[:, 7:])        # (n_steps, 12)
    dq_joints = np.array(all_qvel[:, 6:])        # (n_steps, 12)
    ctrl_out = np.array(all_ctrl)                 # (n_steps, 12)
    pos_err = targets_np - q_joints
    base_z = np.array(all_qpos[:, 2])             # (n_steps,)

    npz_path = "/tmp/pd_hold_debug.npz"
    np.savez_compressed(npz_path,
        base_z=base_z,
        joint_pos=q_joints,
        joint_vel=dq_joints,
        pos_err=pos_err,
        ctrl=ctrl_out,
        targets=targets_np,
        joint_names=["FR_hip","FR_thigh","FR_calf","FL_hip","FL_thigh","FL_calf",
                      "RR_hip","RR_thigh","RR_calf","RL_hip","RL_thigh","RL_calf"],
        mode="random_actions",
        kp=np.array(kp),
        kd=np.array(kd),
    )
    print(f"Saved {npz_path}")
    print(f"  {n_steps} steps, random actions")
    print(f"  base_z: start={base_z[0]:.4f} min={base_z.min():.4f} max={base_z.max():.4f} end={base_z[-1]:.4f}")

    # Render
    print(f"\nRendering {len(trajectory)} frames...")
    renderer = mujoco.Renderer(env.mj_model, width=640, height=480)
    mj_data = mujoco.MjData(env.mj_model)
    frames = []

    for qp in trajectory:
        mj_data.qpos[:] = qp
        mujoco.mj_forward(env.mj_model, mj_data)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance = 2.0
        cam.azimuth = 135
        cam.elevation = -20
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()

    import imageio
    out = "/tmp/warp_pd_hold.mp4"
    imageio.mimsave(out, frames, fps=50)
    print(f"Saved {out} ({len(frames)} frames, {len(frames)/50:.1f}s)")


if __name__ == "__main__":
    main()
