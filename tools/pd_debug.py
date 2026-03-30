#!/usr/bin/env python
"""Debug PD controller on Warp vs MJX — hold keyframe pose, print everything.

Both envs try to hold the home pose with zero action. Prints per-step:
- joint position error (target - actual)
- joint velocity
- computed PD torque (before and after clamp)
- base height

Usage:
    uv run python tools/pd_debug.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env


def run_pd_hold(env, label, n_steps=20):
    """Hold keyframe pose with zero action, print PD internals."""
    qpos = jnp.array(env._mj_model.keyframe("home").qpos)
    qvel = jnp.zeros(env.mjx_model.nv)
    default_pose = jnp.array(env._mj_model.keyframe("home").qpos[7:])

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
    kp = env._kp
    kd = env._kd
    action_scale = env._config.action_scale

    # motor_targets = default_pose + action * action_scale
    # With zero action: motor_targets = default_pose
    motor_targets = default_pose

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Kp={kp}, Kd={kd}, action_scale={action_scale}")
    print(f"  motor_targets (home pose): [{' '.join(f'{v:.3f}' for v in np.array(motor_targets))}]")
    print(f"  forcerange[0]: [{env.mj_model.actuator_forcerange[0,0]:.1f}, {env.mj_model.actuator_forcerange[0,1]:.1f}]")
    print(f"  ctrlrange[0]:  [{env.mj_model.actuator_ctrlrange[0,0]:.1f}, {env.mj_model.actuator_ctrlrange[0,1]:.1f}]")
    print(f"  solver: iterations={env.mj_model.opt.iterations}, cone={env.mj_model.opt.cone}")
    print(f"{'='*80}")

    n_substeps = int(env._config.ctrl_dt / env._config.sim_dt)
    print(f"  substeps per ctrl: {n_substeps}")
    print()

    for step in range(n_steps):
        # Print state BEFORE stepping
        q = np.array(data.qpos[7:])
        dq = np.array(data.qvel[6:])
        z = float(data.qpos[2])

        pos_err = np.array(motor_targets) - q
        vel_err = 0.0 - dq
        tau_raw = kp * pos_err + kd * vel_err

        # Clamp to ctrlrange (what the env does via forcerange)
        ctrl_lo = np.array(env.mj_model.actuator_ctrlrange[:, 0])
        ctrl_hi = np.array(env.mj_model.actuator_ctrlrange[:, 1])
        tau_clamped = np.clip(tau_raw, ctrl_lo, ctrl_hi)

        print(f"Step {step:3d} | z={z:.4f}m")
        print(f"  pos_err:     [{' '.join(f'{v:7.3f}' for v in pos_err)}]")
        print(f"  joint_vel:   [{' '.join(f'{v:7.2f}' for v in dq)}]")
        print(f"  tau_raw:     [{' '.join(f'{v:7.2f}' for v in tau_raw)}]")
        print(f"  tau_clamped: [{' '.join(f'{v:7.2f}' for v in tau_clamped)}]")
        if not np.allclose(tau_raw, tau_clamped, atol=0.01):
            saturated = np.where(np.abs(tau_raw - tau_clamped) > 0.01)[0]
            print(f"  ** SATURATED joints: {saturated.tolist()}")
        print()

        # Actually step the env (PD applied internally in substeps)
        # We use the env's step which does PD per substep
        action = jnp.zeros(12)

        # Manual PD substep loop (matching env.step behavior)
        for sub in range(n_substeps):
            current_q = data.qpos[7:]
            current_dq = data.qvel[6:]
            tau = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
            data = data.replace(ctrl=tau)
            data = mjx.step(model, data)

    return data


def main():
    from jax_rl.envs.locomotion.go2_joystick import Joystick as MJXJoystick
    from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick

    mjx_env_inst = MJXJoystick()
    warp_env_inst = WarpJoystick()

    run_pd_hold(mjx_env_inst, "MJX (Menagerie go2_mjx.xml)", n_steps=15)
    run_pd_hold(warp_env_inst, "Warp (unitree go2.xml)", n_steps=15)


if __name__ == "__main__":
    main()
