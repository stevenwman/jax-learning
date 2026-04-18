"""Go2-specific rendering helpers for video recording.

- Velocity kick perturbations (for robustness evaluation)
- Command arrow overlays (velocity + yaw rate visualization)
"""

import jax
import jax.numpy as jnp
import numpy as np


def apply_kicks(env_state, step_idx, key, kick_interval=75, kick_strength=1.5):
    """Zero command and apply velocity kicks every kick_interval steps.

    Suitable as a ``kicks_fn`` argument to the rollout step builders.

    Args:
        env_state: MJX env state with .info['command'] and .data.qvel.
        step_idx: Current rollout step index (scalar).
        key: JAX PRNG key.
        kick_interval: Steps between kicks (default 75 = 1.5s at 50 Hz).
        kick_strength: Max velocity perturbation in m/s per axis.

    Returns:
        (env_state, key) with zeroed command and optional velocity kick applied.
    """
    # Zero out velocity command — robot should just stand
    info = {**env_state.info, "command": jnp.zeros(3)}
    env_state = env_state.replace(info=info)

    # Apply velocity kick
    kick_key, key = jax.random.split(key)
    kick_vel = jax.random.uniform(
        kick_key, (3,), minval=-kick_strength, maxval=kick_strength,
    )
    do_kick = (step_idx > 0) & (step_idx % kick_interval == 0)
    new_qvel = env_state.data.qvel.at[0:3].set(
        jnp.where(do_kick,
                   env_state.data.qvel[0:3] + kick_vel,
                   env_state.data.qvel[0:3])
    )
    new_data = env_state.data.replace(qvel=new_qvel)
    env_state = env_state.replace(data=new_data)
    return env_state, key


def render_command_overlays(renderer, mj_data, cmd, idx, goal_xy=None):
    """Draw velocity and yaw-rate command arrows onto the renderer scene.

    Args:
        renderer: mujoco.Renderer with an active scene.
        mj_data: mujoco.MjData positioned at the current frame.
        cmd: Command array of shape (3,) — [vx, vy, yaw_rate] in local frame.
        idx: Frame index (used only for guard; caller should ensure idx > 0).
        goal_xy: Optional (2,) world-frame goal. If provided, draws a BLUE arrow
            from robot base toward the goal (for curriculum-env debugging).
    """
    import mujoco

    vx, vy = float(cmd[0]), float(cmd[1])
    speed = np.sqrt(vx**2 + vy**2)

    # ── Green velocity arrow ──────────────────────────────────────────────
    if speed > 0.05:
        # Rotate command from local to world frame using robot's yaw
        quat = mj_data.qpos[3:7]
        w, x, y, z = quat
        fwd_x = 1 - 2 * (y * y + z * z)
        fwd_y = 2 * (x * y + w * z)
        right_x = 2 * (x * y - w * z)
        right_y = 1 - 2 * (x * x + z * z)
        world_vx = vx * fwd_x + vy * right_x
        world_vy = vx * fwd_y + vy * right_y

        base_pos = mj_data.qpos[:3].copy()
        base_pos[2] = 0.4
        end_pos = base_pos.copy()
        end_pos[0] += world_vx * 0.3
        end_pos[1] += world_vy * 0.3

        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9), np.zeros(4),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.015,
            base_pos.astype(np.float64), end_pos.astype(np.float64),
        )
        geom.rgba = np.array([0, 1, 0, 0.8], dtype=np.float32)
        renderer.scene.ngeom += 1

    # ── Yellow yaw-rate arrow ─────────────────────────────────────────────
    yaw_rate = float(cmd[2])
    if abs(yaw_rate) > 0.05:
        base_pos = mj_data.qpos[:3].copy()
        base_pos[2] = 0.45
        yaw_end = base_pos.copy()
        yaw_end[2] += yaw_rate * 0.2

        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9), np.zeros(4),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.012,
            base_pos.astype(np.float64), yaw_end.astype(np.float64),
        )
        geom.rgba = np.array([1, 0.9, 0, 0.8], dtype=np.float32)
        renderer.scene.ngeom += 1

    # ── Red target marker at goal position (curriculum env only) ────────
    # Flat cylinder (disc) at world (goal_x, goal_y, 0) — marks "go here".
    if goal_xy is not None:
        goal_xy = np.asarray(goal_xy, dtype=np.float64)
        if goal_xy.shape == (2,):
            goal_pos = np.array([goal_xy[0], goal_xy[1], 0.02], dtype=np.float64)
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            # Cylinder: size = (radius, half-height, 0). Flat disc = small half-height.
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([0.5, 0.02, 0.0], dtype=np.float64),  # size
                goal_pos,                                        # pos
                np.eye(3, dtype=np.float64).flatten(),           # mat (identity)
                np.array([1.0, 0.1, 0.1, 0.7], dtype=np.float32),
            )
            renderer.scene.ngeom += 1
