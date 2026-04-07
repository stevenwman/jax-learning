#!/usr/bin/env python
"""Compare Warp vs MJX Go2 physics with zero torque (ragdoll test).

Bypasses PD controller entirely — ctrl=0 every substep. Robot should
collapse under gravity. If the collapse is wildly different between
backends, the physics config (solver, contacts) is the issue.
If similar, the PD/actuation config is the issue.

Usage:
    MUJOCO_GL=egl uv run python tools/ragdoll_test.py
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


def run_ragdoll(env, label, n_steps=200):
    """Run env with zero torque, return qpos trajectory."""
    qpos = jnp.array(env._mj_model.keyframe("home").qpos)
    qvel = jnp.zeros(env.mjx_model.nv)

    make_data_kwargs = dict(
        qpos=qpos, qvel=qvel,
        ctrl=jnp.zeros(env.mjx_model.nu),
        impl=env.mjx_model.impl.value,
    )
    if hasattr(env._config, "naccdmax"):
        make_data_kwargs["naccdmax"] = env._config.naccdmax
    if hasattr(env._config, "naconmax"):
        make_data_kwargs["naconmax"] = env._config.naconmax
    if hasattr(env._config, "njmax"):
        make_data_kwargs["njmax"] = env._config.njmax

    data = mjx_env.make_data(env.mj_model, **make_data_kwargs)
    data = mjx.forward(env.mjx_model, data)

    model = env.mjx_model
    trajectory = [np.array(data.qpos)]

    print(f"\n=== {label} (zero torque ragdoll) ===")
    for i in range(n_steps):
        # Zero ctrl every substep — no actuator forces at all
        data = data.replace(ctrl=jnp.zeros(env.mjx_model.nu))
        data = mjx.step(model, data)
        trajectory.append(np.array(data.qpos))
        if i < 15 or i % 50 == 0:
            z = float(data.qpos[2])
            jvel = float(jnp.abs(data.qvel[6:]).max())
            joints = np.array(data.qpos[7:])
            print(f"  Step {i+1:3d}: z={z:.4f}m  jvel_max={jvel:.1f}  "
                  f"joints=[{' '.join(f'{v:.2f}' for v in joints)}]")

    return trajectory


def render_trajectory(env, trajectory, path):
    """Render qpos trajectory to mp4."""
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
    imageio.mimsave(path, frames, fps=50)
    print(f"  Saved {path} ({len(frames)} frames)")


def main():
    from jax_rl.envs.locomotion.go2_joystick import Joystick as MJXJoystick
    from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick

    mjx_env_inst = MJXJoystick()
    warp_env_inst = WarpJoystick()

    # Print physics config comparison
    print("=== Physics config ===")
    for name, env in [("MJX", mjx_env_inst), ("Warp", warp_env_inst)]:
        m = env.mj_model
        print(f"  {name}:")
        print(f"    cone={m.opt.cone}  iterations={m.opt.iterations}  "
              f"ls_iterations={m.opt.ls_iterations}")
        print(f"    ccd_iterations={m.opt.ccd_iterations}  "
              f"eulerdamp_disabled={bool(m.opt.disableflags & 32768)}")
        print(f"    timestep={m.opt.timestep}  damping[6]={m.dof_damping[6]}")
        print(f"    forcerange[0]=[{m.actuator_forcerange[0,0]:.1f}, "
              f"{m.actuator_forcerange[0,1]:.1f}]")

    # Run ragdoll tests
    mjx_traj = run_ragdoll(mjx_env_inst, "MJX (Menagerie go2_mjx.xml)")
    warp_traj = run_ragdoll(warp_env_inst, "Warp (unitree go2.xml)")

    # Render videos
    print("\nRendering...")
    render_trajectory(mjx_env_inst, mjx_traj, "/tmp/ragdoll_mjx.mp4")
    render_trajectory(warp_env_inst, warp_traj, "/tmp/ragdoll_warp.mp4")

    print("\nDone. Compare:")
    print("  /tmp/ragdoll_mjx.mp4")
    print("  /tmp/ragdoll_warp.mp4")


if __name__ == "__main__":
    main()
