"""Live interactive viewer with WASD keyboard control.

Loads a trained policy checkpoint and runs it in MuJoCo's passive viewer.
WASD controls the velocity commands sent to the joystick env.

Controls:
  W/S  — forward/backward velocity
  A/D  — left/right velocity
  Q/E  — yaw left/right
  Space — pause/resume
  R     — reset env
  Esc   — quit
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.5")

import argparse
import copy
import time

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import mujoco.viewer

from jax_rl.training.checkpointing import load_actor_for_inference
from jax_rl.algos.ppo import PPO
from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.utils.normalization import normalize as norm_normalize, init as norm_init, update as norm_update

import optax


# GLFW key codes
KEY_W = 87
KEY_S = 83
KEY_A = 65
KEY_D = 68
KEY_Q = 81
KEY_E = 69
KEY_R = 82
KEY_SPACE = 32

# Global state for keyboard callback
_STATE = {
    'running': True,
    'command': [0.0, 0.0, 0.0],  # [vx, vy, yaw_rate]
    'reset': False,
    'keys_held': set(),
}

# Command limits (match Go1/Go2 joystick env)
VX_RANGE = (-1.0, 2.0)
VY_RANGE = (-0.5, 0.5)
YAW_RANGE = (-1.0, 1.0)
CMD_STEP = 0.1


def key_callback(key: int) -> None:
    if key == KEY_SPACE:
        _STATE['running'] = not _STATE['running']
        print(f"{'Running' if _STATE['running'] else 'Paused'}")
    elif key == KEY_R:
        _STATE['reset'] = True
        print("Reset requested")


def update_command_from_keys():
    """Update command based on currently held keys (polled each frame)."""
    vx, vy, yaw = 0.0, 0.0, 0.0
    # We can't detect held keys via callback alone, so we use toggle approach:
    # Press W once → set forward velocity. Press S to stop or go backward.
    # This is simpler and works with MuJoCo's key_callback (press events only).
    pass


def main():
    parser = argparse.ArgumentParser(description="Live viewer with WASD control")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    parser.add_argument("--vx", type=float, default=1.0, help="Forward velocity command")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rate command")
    args = parser.parse_args()

    # Load checkpoint
    meta, actor_params, norm_state = load_actor_for_inference(args.checkpoint)
    env_name = meta.get("env_name", meta.get("train_config", {}).get("env_name", ""))
    print(f"Loaded: {args.checkpoint} (env={env_name})")

    tc = meta.get("train_config", {})
    ppo_cfg = tc.get("ppo", tc)
    obs_dim = meta["obs_dim"]
    action_dim = meta["action_dim"]

    policy_hidden = tuple(ppo_cfg.get("policy_hidden_dim", [512, 256, 128]))
    value_hidden = tuple(ppo_cfg.get("value_hidden_dim", [512, 256, 128]))
    activation = ppo_cfg.get("activation", "swish")

    # Build PPO actor (we only need the actor, not critic)
    config = PPOConfig(
        encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=policy_hidden, activation=activation),
        critic_encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=value_hidden, activation=activation),
        policy_head=PolicyHeadConfig(
            action_dim=action_dim,
            squash=ppo_cfg.get("squash", True),
            state_dependent_std=ppo_cfg.get("state_dependent_std", False),
        ),
        num_envs=1,
    )
    ppo = PPO(config, obs_dim, action_dim, optax.adam(1e-3), optax.adam(1e-3))
    ts = ppo.init(jax.random.PRNGKey(0))
    ts = ts.replace(actor_params=actor_params)

    # Load the MuJoCo model (CPU-side for viewer)
    from mujoco_playground import registry as pg_registry
    env = pg_registry.load(env_name)
    m = env.mj_model
    d = mujoco.MjData(m)

    # MJX for physics
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    # JIT compile env step and policy
    print("JIT compiling physics + policy...")
    jit_step = jax.jit(mjx.step)

    @jax.jit
    def get_action(actor_params, obs, norm_s):
        norm_s = norm_update(norm_s, obs[None])
        normed = norm_normalize(norm_s, obs[None])
        mean, log_std = ppo.actor.apply(actor_params, normed)
        action = jnp.clip(mean, -1.0, 1.0)
        return action.squeeze(0), norm_s

    # Warmup JIT
    dummy_obs = jnp.zeros(obs_dim)
    ns = norm_state
    action, ns = get_action(actor_params, dummy_obs, ns)
    dx = jit_step(mx, dx)
    print("JIT done.")

    # Reset
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    dx = mjx.put_data(m, d)

    # Set initial command
    _STATE['command'] = [args.vx, args.vy, args.yaw]
    print(f"\nCommand: vx={args.vx:.1f} vy={args.vy:.1f} yaw={args.yaw:.1f}")
    print("Controls: W/S=forward/back, A/D=left/right, Q/E=yaw, Space=pause, R=reset")
    print("(Key presses toggle command increments)\n")

    ns = norm_state
    last_action = jnp.zeros(action_dim)
    ctrl_dt = m.opt.timestep * 5  # 5 physics steps per control step (0.02s at 0.004 dt)
    step_count = 0

    def _key_cb(key):
        """Handle key presses to adjust command."""
        cmd = _STATE['command']
        if key == KEY_W:
            cmd[0] = min(cmd[0] + CMD_STEP, VX_RANGE[1])
        elif key == KEY_S:
            cmd[0] = max(cmd[0] - CMD_STEP, VX_RANGE[0])
        elif key == KEY_A:
            cmd[1] = min(cmd[1] + CMD_STEP, VY_RANGE[1])
        elif key == KEY_D:
            cmd[1] = max(cmd[1] - CMD_STEP, VY_RANGE[0])
        elif key == KEY_Q:
            cmd[2] = min(cmd[2] + CMD_STEP, YAW_RANGE[1])
        elif key == KEY_E:
            cmd[2] = max(cmd[2] - CMD_STEP, YAW_RANGE[0])
        elif key == KEY_SPACE:
            _STATE['running'] = not _STATE['running']
            print(f"{'Running' if _STATE['running'] else 'Paused'}")
            return
        elif key == KEY_R:
            _STATE['reset'] = True
            return
        else:
            return
        print(f"  cmd: vx={cmd[0]:+.1f} vy={cmd[1]:+.1f} yaw={cmd[2]:+.1f}")

    viewer = mujoco.viewer.launch_passive(m, d, key_callback=_key_cb)

    with viewer:
        while viewer.is_running():
            t0 = time.time()

            if _STATE['reset']:
                mujoco.mj_resetData(m, d)
                mujoco.mj_forward(m, d)
                dx = mjx.put_data(m, d)
                ns = norm_state
                last_action = jnp.zeros(action_dim)
                step_count = 0
                _STATE['reset'] = False
                print("Reset done.")

            if _STATE['running']:
                # Build obs from MJX data (simplified — extract from dx)
                # For now we use the raw qpos/qvel to construct obs matching the env
                qpos = jax.device_get(dx.qpos)
                qvel = jax.device_get(dx.qvel)

                # Construct obs matching Go1/Go2 joystick format:
                # [linvel(3), gyro(3), gravity(3), joint_pos-default(12), joint_vel(12), last_action(12), command(3)]
                # = 48 dims for state obs
                cmd = jnp.array(_STATE['command'])

                # Get sensor-like data from qpos/qvel
                # Base orientation (quaternion at qpos[3:7])
                quat = jnp.array(qpos[3:7])
                # Rotate gravity to local frame
                grav_world = jnp.array([0, 0, -1.0])
                # Simple quat rotation (w, x, y, z format in MuJoCo)
                w, x, y, z = quat
                # Rotation matrix from quaternion
                rot = jnp.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
                ])
                local_grav = rot.T @ grav_world

                # Local linear velocity
                base_vel = jnp.array(qvel[:3])
                local_linvel = rot.T @ base_vel

                # Angular velocity (already in local frame for MuJoCo)
                gyro = jnp.array(qvel[3:6])

                # Joint positions (qpos[7:]) relative to default
                joint_pos = jnp.array(qpos[7:7+action_dim])
                # Default pose — zeros for now (should match env default)
                default_pos = jnp.zeros(action_dim)
                joint_pos_offset = joint_pos - default_pos

                # Joint velocities
                joint_vel = jnp.array(qvel[6:6+action_dim])

                # Build obs
                obs = jnp.concatenate([
                    local_linvel,       # 3
                    gyro,               # 3
                    local_grav,         # 3
                    joint_pos_offset,   # 12
                    joint_vel,          # 12
                    last_action,        # 12
                    cmd,                # 3
                ])

                action, ns = get_action(actor_params, obs, ns)
                last_action = action

                # Apply action as PD target: ctrl = default_pos + action * action_scale
                action_scale = 0.5  # Match PG config
                ctrl = jax.device_get(default_pos + action * action_scale)
                d.ctrl[:action_dim] = ctrl

                # Step physics (multiple substeps for ctrl_dt)
                for _ in range(5):
                    dx = jit_step(mx, dx)
                step_count += 1

                # Sync MJX → MuJoCo for viewer
                mjx.get_data_into(d, m, dx)

            viewer.sync()

            # Realtime pacing
            elapsed = time.time() - t0
            if elapsed < ctrl_dt:
                time.sleep(ctrl_dt - elapsed)

    print(f"\nViewer closed after {step_count} steps.")


if __name__ == "__main__":
    main()
