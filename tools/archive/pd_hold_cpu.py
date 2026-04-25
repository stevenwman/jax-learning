#!/usr/bin/env python
"""PD hold test on CPU MuJoCo with same unitree XML + same overrides as Warp env.

Exact parity check: same model, same overrides, same PD gains, same kicks.
If CPU behaves differently from Warp, the issue is in the Warp backend.

Usage:
    MUJOCO_GL=egl uv run python tools/pd_hold_cpu.py
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import imageio


def main():
    # Load the EXACT same XML as the Warp env
    from pathlib import Path
    from mujoco_playground._src import mjx_env

    vendor_path = Path("jax_rl/envs/locomotion/xmls/unitree_go2")
    xmls_path = Path("jax_rl/envs/locomotion/xmls")

    assets = {}
    mjx_env.update_assets(assets, vendor_path, "*.xml")
    mjx_env.update_assets(assets, vendor_path / "assets")
    mjx_env.update_assets(assets, xmls_path, "*.xml")

    scene_xml = (xmls_path / "go2_warp_scene_flat.xml").read_text()
    m = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)

    # Apply EXACT same overrides as go2_warp_base.py
    m.opt.timestep = 0.004
    m.opt.ccd_iterations = 100

    # contact_mode="training" overrides
    for foot_name in ["FL", "FR", "RL", "RR"]:
        gid = m.geom(foot_name).id
        m.geom_solimp[gid, :3] = [0.9, 0.95, 0.023]
        m.geom_condim[gid] = 3
        m.geom_friction[gid] = [0.6, 0.005, 0.0001]

    # forcerange = ctrlrange
    for i in range(m.nu):
        m.actuator_forcerange[i] = m.actuator_ctrlrange[i]

    # Print solver settings for verification
    print("=== CPU Model Settings (should match Warp) ===")
    print(f"  timestep:       {m.opt.timestep}")
    print(f"  iterations:     {m.opt.iterations}")
    print(f"  ls_iterations:  {m.opt.ls_iterations}")
    print(f"  ccd_iterations: {m.opt.ccd_iterations}")
    print(f"  cone:           {m.opt.cone} (0=pyramidal, 1=elliptic)")
    print(f"  eulerdamp:      disabled={bool(m.opt.disableflags & 32768)}")
    print(f"  dof_damping[6]: {m.dof_damping[6]}")
    print(f"  dof_frictionloss[6]: {m.dof_frictionloss[6]}")
    print(f"  armature[6]:    {m.dof_armature[6]}")
    print(f"  forcerange[0]:  [{m.actuator_forcerange[0,0]:.1f}, {m.actuator_forcerange[0,1]:.1f}]")
    print(f"  forcerange[2]:  [{m.actuator_forcerange[2,0]:.1f}, {m.actuator_forcerange[2,1]:.1f}]")
    print()

    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # PD parameters — same as Warp env
    kp, kd = 20.0, 0.5
    n_steps = 500
    n_substeps = 5  # ctrl_dt=0.02 / sim_dt=0.004
    kick_interval = 50
    kick_xy = 1.5
    kick_z = 2.0
    default_pose = np.array(m.keyframe("home").qpos[7:])

    rng = np.random.RandomState(42)

    # Storage
    all_qpos = np.zeros((n_steps, m.nq))
    all_qvel = np.zeros((n_steps, m.nv))
    all_ctrl = np.zeros((n_steps, m.nu))

    for i in range(n_steps):
        # Kick logic — same as Warp test
        kick_count = i // kick_interval
        is_up_kick = kick_count % 2 == 1
        do_kick = (i > 0) and (i % kick_interval == 0)

        if do_kick:
            if is_up_kick:
                d.qvel[2] += kick_z
                print(f"  Step {i}: UP KICK (z_vel += {kick_z})")
            else:
                xy = rng.uniform(-kick_xy, kick_xy, size=2)
                d.qvel[0:2] += xy
                print(f"  Step {i}: XY KICK ({xy})")

        # PD hold with substeps
        for _ in range(n_substeps):
            q = d.qpos[7:]
            dq = d.qvel[6:]
            tau = kp * (default_pose - q) + kd * (0.0 - dq)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

        all_qpos[i] = d.qpos.copy()
        all_qvel[i] = d.qvel.copy()
        all_ctrl[i] = d.ctrl.copy()

    # Save npz
    joint_pos = all_qpos[:, 7:]
    joint_vel = all_qvel[:, 6:]
    base_z = all_qpos[:, 2]
    pos_err = np.tile(default_pose, (n_steps, 1)) - joint_pos

    npz_path = "/tmp/pd_hold_cpu.npz"
    np.savez_compressed(npz_path,
        base_z=base_z,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        pos_err=pos_err,
        ctrl=all_ctrl,
        targets=np.tile(default_pose, (n_steps, 1)),
        joint_names=["FR_hip","FR_thigh","FR_calf","FL_hip","FL_thigh","FL_calf",
                      "RR_hip","RR_thigh","RR_calf","RL_hip","RL_thigh","RL_calf"],
    )
    print(f"\nSaved {npz_path}")
    print(f"  base_z: start={base_z[0]:.4f} min={base_z.min():.4f} max={base_z.max():.4f} end={base_z[-1]:.4f}")

    # Render video
    print(f"\nRendering {n_steps+1} frames...")
    renderer = mujoco.Renderer(m, width=640, height=480)
    frames = []

    # Render from saved qpos
    mj_render = mujoco.MjData(m)
    # Initial frame
    mujoco.mj_resetDataKeyframe(m, mj_render, 0)
    mujoco.mj_forward(m, mj_render)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance = 2.0
    cam.azimuth = 135
    cam.elevation = -20
    renderer.update_scene(mj_render, camera=cam)
    frames.append(renderer.render().copy())

    for i in range(n_steps):
        mj_render.qpos[:] = all_qpos[i]
        mj_render.qvel[:] = all_qvel[i]
        mujoco.mj_forward(m, mj_render)
        renderer.update_scene(mj_render, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()

    vid_path = "/tmp/pd_hold_cpu.mp4"
    imageio.mimsave(vid_path, frames, fps=50)
    print(f"Saved {vid_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
