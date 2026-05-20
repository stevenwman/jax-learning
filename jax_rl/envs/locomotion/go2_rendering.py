"""Go2-specific rendering helpers for video recording.

- Velocity kick perturbations (for robustness evaluation)
- Command arrow overlays (velocity + yaw rate visualization)
"""

import jax
import jax.numpy as jnp
import numpy as np


def make_varied_cmd_fn(period_steps: int = 75, cmd_max=(1.5, 0.8, 1.2)):
    """Build a `kicks_fn` that resamples uniform velocity command every N steps.

    Default 75 steps = 1.5s at 50 Hz. Use to stress-test command-tracking by
    forcing the policy through a faster command schedule than the default
    env's ~5s exponential resample.

    Args:
        period_steps: Steps between command resamples.
        cmd_max: (vx_max, vy_max, yaw_max) symmetric ranges for uniform sample.

    Returns:
        kicks_fn(env_state, step_idx, key) -> (env_state, key).
    """
    cmd_max_arr = jnp.asarray(cmd_max, dtype=jnp.float32)

    def fn(env_state, step_idx, key):
        do_resample = (step_idx > 0) & (step_idx % period_steps == 0)
        sample_key, key = jax.random.split(key)
        new_cmd = jax.random.uniform(
            sample_key, (3,), minval=-cmd_max_arr, maxval=cmd_max_arr
        )
        cmd = jnp.where(do_resample, new_cmd, env_state.info["command"])
        info = {**env_state.info, "command": cmd}
        return env_state.replace(info=info), key

    return fn


def make_locked_cmd_fn(cmd_vec):
    """Build a kicks_fn that pins env.info['command'] to a fixed value every step.

    Args:
        cmd_vec: (3,) array-like — [vx, vy, yaw] to lock. Use 0 for any axis
            you want disabled, non-zero for the axis to test.

    Returns:
        kicks_fn(env_state, step_idx, key) -> (env_state, key).
    """
    locked = jnp.asarray(cmd_vec, dtype=jnp.float32)

    def fn(env_state, step_idx, key):
        info = {**env_state.info, "command": locked}
        return env_state.replace(info=info), key

    return fn


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

    # Scale arrow size by robot height — Go2 pelvis ~0.3, G1 pelvis ~0.76.
    # Reference Go2 (pelvis_z 0.3) → scale 1.0; G1 → scale ~2.0.
    pelvis_z = float(mj_data.qpos[2])
    scale = max(pelvis_z / 0.30, 1.0)

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

        # Anchor at pelvis height + clearance so arrow sits above torso.
        # Use unit direction × constant display length so arrow stays
        # legible even with small cmd magnitudes (G1 cmd_a=0.1).
        # Magnitude visible via shaft thickness modulated by speed.
        base_pos = mj_data.qpos[:3].copy()
        base_pos[2] = pelvis_z + 0.25 * scale
        # Cap at cmd_max ~ 1.0 m/s for normalization; if higher, arrow grows.
        cmd_max = 1.0
        disp_len = 0.6 * scale * min(speed / cmd_max, 1.0) ** 0.5  # sqrt → small cmd still visible
        # Floor minimum length so tiny cmd still visible
        disp_len = max(disp_len, 0.3 * scale)
        end_pos = base_pos.copy()
        end_pos[0] += (world_vx / speed) * disp_len
        end_pos[1] += (world_vy / speed) * disp_len

        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9), np.zeros(4),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.022 * scale,
            base_pos.astype(np.float64), end_pos.astype(np.float64),
        )
        geom.rgba = np.array([0, 1, 0, 0.9], dtype=np.float32)
        renderer.scene.ngeom += 1

    # ── Yellow yaw-rate arrow ─────────────────────────────────────────────
    yaw_rate = float(cmd[2])
    if abs(yaw_rate) > 0.05:
        base_pos = mj_data.qpos[:3].copy()
        base_pos[2] = pelvis_z + 0.35 * scale
        # Same length-normalization treatment as velocity arrow.
        yaw_max = 1.0
        yaw_len = 0.5 * scale * min(abs(yaw_rate) / yaw_max, 1.0) ** 0.5
        yaw_len = max(yaw_len, 0.25 * scale)
        yaw_end = base_pos.copy()
        yaw_end[2] += np.sign(yaw_rate) * yaw_len

        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9), np.zeros(4),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.018 * scale,
            base_pos.astype(np.float64), yaw_end.astype(np.float64),
        )
        geom.rgba = np.array([1, 0.9, 0, 0.9], dtype=np.float32)
        renderer.scene.ngeom += 1

    # ── Red target marker floating above goal position (curriculum env) ─
    # Sphere at world (goal_x, goal_y, 2.5) — elevated so it clears any
    # pyramid apex (max ~2m) and stays visible on tilted/bowl terrains.
    if goal_xy is not None:
        goal_xy = np.asarray(goal_xy, dtype=np.float64)
        if goal_xy.shape == (2,):
            goal_pos = np.array([goal_xy[0], goal_xy[1], 2.5], dtype=np.float64)
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.3, 0.0, 0.0], dtype=np.float64),   # radius
                goal_pos,
                np.eye(3, dtype=np.float64).flatten(),
                np.array([1.0, 0.15, 0.15, 0.85], dtype=np.float32),
            )
            renderer.scene.ngeom += 1
