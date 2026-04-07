"""Live interactive viewer for Go2 joystick locomotion policies.

Loads a trained PPO checkpoint and runs the policy in MuJoCo's passive viewer
with real-time keyboard control of velocity commands.

Usage:
    uv run python live_viewer.py --checkpoint checkpoints/<go2_checkpoint>
    uv run python live_viewer.py --checkpoint checkpoints/<go2_checkpoint> --vx 1.5

Controls:
    W/S   — increase/decrease forward velocity
    A/D   — increase/decrease lateral velocity
    Q/E   — increase/decrease yaw rate
    Space — pause/resume simulation
    R     — reset environment
    Esc   — quit

Limitations:
    - Only supports PPO checkpoints (not SAC/TD3)
    - Only supports Go2-style joystick envs (hardcoded obs construction)
    - Obs construction is manually reimplemented here (must match env's _get_obs)
    - default_pos is loaded from the Go2 env constants, not from checkpoint

If the env's observation format changes, this script WILL break silently.
See .context/integration_debt.md item #4 for the proper fix.
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.5")

import argparse
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


# ── GLFW key codes ──────────────────────────────────────────────────────────
KEY_W, KEY_S, KEY_A, KEY_D = 87, 83, 65, 68
KEY_Q, KEY_E, KEY_R, KEY_SPACE = 81, 69, 82, 32

# ── Command configuration ──────────────────────────────────────────────────
# Ranges match Go2WarpJoystickFlat env command sampling
VX_RANGE = (-1.0, 2.0)     # m/s forward/backward
VY_RANGE = (-0.5, 0.5)     # m/s lateral
YAW_RANGE = (-1.0, 1.0)    # rad/s yaw
CMD_STEP = 0.1             # increment per keypress

# Physics stepping: 5 substeps per control step matches Go2 env's
# ctrl_dt = sim_dt * n_frames = 0.004 * 5 = 0.02s (50 Hz control)
N_SUBSTEPS = 5

# Action scaling: must match the env's action_scale config
# Go2WarpJoystickFlat uses 0.5 (from go2_warp_joystick.py)
ACTION_SCALE = 0.5

# ── Global mutable state (for keyboard callback) ───────────────────────────
_STATE = {
    'running': True,
    'command': [0.0, 0.0, 0.0],  # [vx, vy, yaw_rate]
    'reset': False,
}


def main():
    parser = argparse.ArgumentParser(
        description="Live viewer for Go2 joystick locomotion policies")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--vx", type=float, default=1.0,
                        help="Initial forward velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0,
                        help="Initial lateral velocity command (m/s)")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="Initial yaw rate command (rad/s)")
    args = parser.parse_args()

    # ── Load checkpoint ─────────────────────────────────────────────────
    meta, actor_params, norm_state = load_actor_for_inference(args.checkpoint)
    env_name = meta.get("env_name", meta.get("train_config", {}).get("env_name", ""))
    obs_dim = meta["obs_dim"]
    action_dim = meta["action_dim"]
    print(f"Loaded: {args.checkpoint}")
    print(f"  env={env_name}, obs_dim={obs_dim}, action_dim={action_dim}")

    if "go2" not in env_name.lower() and "go1" not in env_name.lower():
        print(f"WARNING: This viewer is designed for Go2/Go1 joystick envs, "
              f"but checkpoint is from '{env_name}'. Obs construction may be wrong.")

    # ── Reconstruct PPO actor from checkpoint metadata ──────────────────
    tc = meta.get("train_config", {})
    ppo_cfg = tc.get("ppo", tc)
    policy_hidden = tuple(ppo_cfg.get("policy_hidden_dim", [512, 256, 128]))
    value_hidden = tuple(ppo_cfg.get("value_hidden_dim", [512, 256, 128]))
    activation = ppo_cfg.get("activation", "swish")

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

    # ── Set up MuJoCo model + MJX physics ───────────────────────────────
    from mujoco_playground import registry as pg_registry
    env = pg_registry.load(env_name)
    m = env.mj_model
    d = mujoco.MjData(m)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    # ── JIT compile physics + policy ────────────────────────────────────
    print("JIT compiling physics + policy...")
    jit_step = jax.jit(mjx.step)

    @jax.jit
    def get_action(actor_params, obs, norm_s):
        """Run policy: normalize obs → actor forward → clip to [-1, 1]."""
        norm_s = norm_update(norm_s, obs[None])
        normed = norm_normalize(norm_s, obs[None])
        mean, _log_std = ppo.actor.apply(actor_params, normed)
        action = jnp.clip(mean, -1.0, 1.0)
        return action.squeeze(0), norm_s

    # Warmup JIT
    dummy_obs = jnp.zeros(obs_dim)
    ns = norm_state
    _action, ns = get_action(actor_params, dummy_obs, ns)
    dx = jit_step(mx, dx)
    print("JIT done.")

    # ── Load Go2 default joint positions ────────────────────────────────
    # This MUST match the env's default_angles. If it doesn't, the
    # joint_pos_offset obs and action application will be wrong.
    try:
        from jax_rl.envs.locomotion.go2_constants import DEFAULT_JOINT_ANGLES
        default_pos = jnp.array(DEFAULT_JOINT_ANGLES)
        print(f"  Loaded Go2 default joint angles from constants")
    except ImportError:
        default_pos = jnp.zeros(action_dim)
        print(f"  WARNING: Could not load Go2 default angles, using zeros")

    # ── Reset ───────────────────────────────────────────────────────────
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    dx = mjx.put_data(m, d)

    _STATE['command'] = [args.vx, args.vy, args.yaw]
    print(f"\nCommand: vx={args.vx:.1f} vy={args.vy:.1f} yaw={args.yaw:.1f}")
    print("Controls: W/S=forward/back, A/D=left/right, Q/E=yaw, Space=pause, R=reset\n")

    ns = norm_state
    last_action = jnp.zeros(action_dim)
    ctrl_dt = m.opt.timestep * N_SUBSTEPS
    step_count = 0

    # ── Keyboard callback ───────────────────────────────────────────────
    def _key_cb(key):
        """Adjust velocity command on keypress."""
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

    # ── Main loop ───────────────────────────────────────────────────────
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
                # ── Build observation ───────────────────────────────
                # WARNING: This manually reimplements the env's _get_obs().
                # If the env's obs format changes, this WILL break silently.
                # TODO: Use the env directly instead of manual reconstruction.
                qpos = jax.device_get(dx.qpos)
                qvel = jax.device_get(dx.qvel)
                cmd = jnp.array(_STATE['command'])

                # Base orientation quaternion (MuJoCo: w, x, y, z)
                quat = jnp.array(qpos[3:7])
                w, x, y, z = quat

                # Rotation matrix from quaternion → project gravity to body frame
                rot = jnp.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
                ])
                local_grav = rot.T @ jnp.array([0, 0, -1.0])
                local_linvel = rot.T @ jnp.array(qvel[:3])

                # Angular velocity (local frame in MuJoCo)
                gyro = jnp.array(qvel[3:6])

                # Joint state relative to default standing pose
                joint_pos = jnp.array(qpos[7:7+action_dim])
                joint_pos_offset = joint_pos - default_pos
                joint_vel = jnp.array(qvel[6:6+action_dim])

                # Assemble obs vector (must match env's _get_obs order)
                obs = jnp.concatenate([
                    local_linvel,       # 3 — base velocity in body frame
                    gyro,               # 3 — angular velocity
                    local_grav,         # 3 — gravity direction in body frame
                    joint_pos_offset,   # 12 — joint angles minus default
                    joint_vel,          # 12 — joint velocities
                    last_action,        # 12 — previous action (for temporal info)
                    cmd,                # 3 — velocity command [vx, vy, yaw]
                ])

                # ── Run policy ──────────────────────────────────────
                action, ns = get_action(actor_params, obs, ns)
                last_action = action

                # Apply action as PD target: ctrl = default_pos + action * scale
                ctrl = jax.device_get(default_pos + action * ACTION_SCALE)
                d.ctrl[:action_dim] = ctrl

                # Step physics (N_SUBSTEPS to match env's ctrl_dt)
                for _ in range(N_SUBSTEPS):
                    dx = jit_step(mx, dx)
                step_count += 1

                # Sync MJX state → CPU MuJoCo data for viewer rendering
                mjx.get_data_into(d, m, dx)

            viewer.sync()

            # Realtime pacing — sleep to maintain ctrl_dt
            elapsed = time.time() - t0
            if elapsed < ctrl_dt:
                time.sleep(ctrl_dt - elapsed)

    print(f"\nViewer closed after {step_count} steps.")


if __name__ == "__main__":
    main()
