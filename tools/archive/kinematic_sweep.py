"""Kinematic sweep diagnostic — test if a robot's action space covers walking positions.

Uses CPU MuJoCo (not MJX) — no GPU needed.
Sweeps joint actions through [-1, 1], reports foot positions achieved.

Usage:
  uv run python tools/kinematic_sweep.py --env Go2JoystickFlat
  uv run python tools/kinematic_sweep.py --env Go1  # shortcut for Go1JoystickFlatTerrain
"""

import argparse

import mujoco
import numpy as np


def load_model(env_name):
    """Load MuJoCo model and extract config from the env."""
    if 'Go2' in env_name:
        from jax_rl.envs.locomotion.go2_joystick import Joystick
        env = Joystick.__new__(Joystick)
        from jax_rl.envs.locomotion.go2_joystick import default_config
        from jax_rl.envs.locomotion import go2_base, go2_constants as consts
        cfg = default_config()
        assets = go2_base.get_assets()
        from etils import epath
        m = mujoco.MjModel.from_xml_string(
            epath.Path(consts.SCENE_FLAT_XML.as_posix()).read_text(), assets=assets
        )
        # Apply same overrides as go2_base.py
        m.opt.timestep = cfg.sim_dt
        m.dof_damping[6:] = cfg.Kd
        m.actuator_gainprm[:, 0] = cfg.Kp
        m.actuator_biasprm[:, 1] = -cfg.Kp
        m.actuator_biasprm[:, 2] = 0.0  # Current fix
        action_scale = cfg.action_scale
        default_pose = np.array(m.keyframe('home').qpos[7:])
        foot_sites = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    else:
        # Go1 from Playground
        from mujoco_playground import registry
        pg_env = registry.load('Go1JoystickFlatTerrain')
        m = pg_env.mj_model
        action_scale = pg_env._config.action_scale
        default_pose = np.array(m.keyframe('home').qpos[7:])
        foot_sites = ['FR', 'FL', 'RR', 'RL']

    d = mujoco.MjData(m)
    foot_site_ids = [m.site(name).id for name in foot_sites]
    return m, d, action_scale, default_pose, foot_site_ids


def reset(m, d, elevate=0.15):
    """Reset to home keyframe, elevated so legs have room to move."""
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe('home').id)
    d.qpos[2] += elevate  # Lift base up
    mujoco.mj_forward(m, d)


def step_to_target(m, d, action, default_pose, action_scale, n_steps=200):
    """Apply action as PD target for n_steps."""
    ctrl = default_pose + action * action_scale
    ctrl = np.clip(ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    d.ctrl[:] = ctrl
    for _ in range(n_steps):
        mujoco.mj_step(m, d)


def sweep_joint(m, d, joint_idx, joint_name, default_pose, action_scale, foot_site_id):
    """Sweep one joint action from -1 to +1."""
    print(f'\n--- Sweep {joint_name} (action[{joint_idx}]) ---')
    for a in np.linspace(-1, 1, 11):
        reset(m, d)
        action = np.zeros(m.nu)
        action[joint_idx] = a
        step_to_target(m, d, action, default_pose, action_scale)
        foot_pos = d.site_xpos[foot_site_id]
        joint_pos = d.qpos[7 + joint_idx]
        target = default_pose[joint_idx] + a * action_scale
        print(f'  act={a:+.1f} → target={target:.3f} actual={joint_pos:.3f} '
              f'foot=[{foot_pos[0]:.3f}, {foot_pos[1]:.3f}, z={foot_pos[2]:.4f}]')


def test_walking_combos(m, d, default_pose, action_scale, foot_site_id):
    """Test combined thigh+calf actions for walking phases."""
    print('\n--- Walking phase combinations (FL leg) ---')
    combos = [
        ('stance (0,0)', 0.0, 0.0),
        ('slight swing', -0.3, -0.3),
        ('mid-swing', -0.5, -0.5),
        ('full-swing', -1.0, -1.0),
        ('slight extend', 0.3, 0.3),
        ('mid-extend', 0.5, 0.5),
        ('full-extend', 1.0, 1.0),
        ('lift (thigh fwd, calf retract)', -0.5, 0.5),
        ('push (thigh back, calf extend)', 0.5, -0.5),
    ]
    for name, t, c in combos:
        reset(m, d)
        action = np.zeros(m.nu)
        action[1] = t  # FL_thigh
        action[2] = c  # FL_calf
        step_to_target(m, d, action, default_pose, action_scale)
        foot_pos = d.site_xpos[foot_site_id]
        print(f'  {name:35s}: thigh={t:+.1f} calf={c:+.1f} → '
              f'foot=[{foot_pos[0]:.3f}, {foot_pos[1]:.3f}, z={foot_pos[2]:.4f}]')


def main():
    parser = argparse.ArgumentParser(description="Kinematic sweep (CPU, no GPU)")
    parser.add_argument("--env", default="Go2JoystickFlat")
    args = parser.parse_args()

    m, d, action_scale, default_pose, foot_site_ids = load_model(args.env)
    fl_foot = foot_site_ids[0]

    print(f'Environment: {args.env}')
    print(f'action_scale: {action_scale}')
    print(f'default_pose (per leg): [{default_pose[0]:.2f}, {default_pose[1]:.2f}, {default_pose[2]:.2f}]')
    print(f'Kp={m.actuator_gainprm[0,0]:.0f}, dof_damping={m.dof_damping[6]:.1f}, '
          f'actuator_Kd={abs(m.actuator_biasprm[0,2]):.1f}')
    print(f'Total mass: {sum(m.body_mass):.1f} kg')

    # Disable gravity so the robot doesn't fall — pure kinematics test
    m.opt.gravity[:] = 0
    print('Gravity disabled, base elevated 0.15m')

    sweep_joint(m, d, 0, 'FL_hip', default_pose, action_scale, fl_foot)
    sweep_joint(m, d, 1, 'FL_thigh', default_pose, action_scale, fl_foot)
    sweep_joint(m, d, 2, 'FL_calf', default_pose, action_scale, fl_foot)

    test_walking_combos(m, d, default_pose, action_scale, fl_foot)

    # Foot z range summary
    print('\n=== FOOT Z RANGE (all thigh x calf combos) ===')
    z_min, z_max = float('inf'), float('-inf')
    for t in np.linspace(-1, 1, 9):
        for c in np.linspace(-1, 1, 9):
            reset(m, d)
            action = np.zeros(m.nu)
            action[1] = t
            action[2] = c
            step_to_target(m, d, action, default_pose, action_scale)
            fz = d.site_xpos[fl_foot, 2]
            z_min = min(z_min, fz)
            z_max = max(z_max, fz)
    print(f'  Min foot z: {z_min:.4f}m')
    print(f'  Max foot z: {z_max:.4f}m')
    print(f'  Z range: {z_max - z_min:.4f}m')
    print(f'  (Need ~0.05-0.10m clearance for walking)')


if __name__ == "__main__":
    main()
