"""Frame stacking utility — shared by Go2 (state obs) and vision RL (pixel obs).

Maintains a FIFO buffer of the last N observations via jnp.roll.
JIT-compatible, works with any obs shape.

Usage:
    stack = init_frame_stack(obs_dim=31, n_frames=3, num_envs=1024)
    stack = update_frame_stack(stack, new_obs)   # push new, drop oldest
    flat_obs = get_stacked_obs(stack)             # (num_envs, obs_dim * n_frames)
"""

import jax.numpy as jnp


def init_frame_stack(obs_dim: int, n_frames: int, num_envs: int) -> jnp.ndarray:
    """Initialize frame stack buffer with zeros.

    Returns:
        (num_envs, n_frames * obs_dim) array — newest obs at index [0:obs_dim].
    """
    return jnp.zeros((num_envs, n_frames * obs_dim))


def update_frame_stack(stack: jnp.ndarray, new_obs: jnp.ndarray, obs_dim: int) -> jnp.ndarray:
    """Push new observation, drop oldest.

    Args:
        stack: (num_envs, n_frames * obs_dim) current stack
        new_obs: (num_envs, obs_dim) new observation
        obs_dim: single-frame observation dimension

    Returns:
        Updated stack with new_obs at front, oldest frame dropped.
    """
    # Roll right by obs_dim (shift all frames one slot older), then overwrite front
    return stack.at[:, :obs_dim].set(new_obs).at[:, obs_dim:].set(stack[:, :-obs_dim])


def get_stacked_obs(stack: jnp.ndarray) -> jnp.ndarray:
    """Return the full stacked observation.

    The stack IS the observation — newest frame first, oldest last.
    Returns (num_envs, n_frames * obs_dim).
    """
    return stack


def reset_frame_stack(stack: jnp.ndarray, new_obs: jnp.ndarray, obs_dim: int,
                      done: jnp.ndarray) -> jnp.ndarray:
    """Reset stack for done envs, filling all frames with the new obs.

    Args:
        stack: (num_envs, n_frames * obs_dim)
        new_obs: (num_envs, obs_dim) — the reset observation
        obs_dim: single-frame dimension
        done: (num_envs,) bool — which envs just reset

    Returns:
        Stack with done envs filled with repeated new_obs, others unchanged.
    """
    n_frames = stack.shape[1] // obs_dim
    # Tile new_obs n_frames times for reset envs
    tiled = jnp.tile(new_obs, (1, n_frames))
    return jnp.where(done[:, None], tiled, stack)
