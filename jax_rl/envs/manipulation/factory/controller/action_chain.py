"""EMA + denormalization + clipping for Factory pose-delta action.

Action is (6,) in [-1, 1]: [dx, dy, dz, drx, dry, drz].
Pipeline: raw → EMA-smooth → denorm to pose delta → clip relative to anchor.

State.info auto-reset trap: Brax-style auto-reset does NOT clear info between
episodes. EMA actions and prev_actions carry across resets if not explicitly
cleared. Use `reset_on_done` at the start of every env.step.
"""
from typing import Tuple

import jax.numpy as jp


def apply_ema(raw: jp.ndarray, prev_ema: jp.ndarray, ema_factor: float = 0.2) -> jp.ndarray:
    """Smooth raw action via exponential moving average."""
    return ema_factor * raw + (1.0 - ema_factor) * prev_ema


def denormalize(
    ema: jp.ndarray,
    pos_threshold: jp.ndarray,
    rot_threshold: jp.ndarray,
) -> Tuple[jp.ndarray, jp.ndarray]:
    """Convert EMA action [-1, 1]⁶ to (pos_delta, rot_delta).

    rot_delta is a rotation-vector (axis*angle) in [-rot_threshold, rot_threshold].
    """
    pos_delta = ema[:3] * pos_threshold
    rot_delta = ema[3:6] * rot_threshold
    return pos_delta, rot_delta


def rotvec_to_quat(v: jp.ndarray) -> jp.ndarray:
    """Convert rotation vector v (axis*angle, rad) → quat (w, x, y, z).

    Safe at |v|→0: returns identity. Used to compose target_quat deltas
    from policy rot actions in env.step.
    """
    norm = jp.linalg.norm(v)
    safe = jp.where(norm > 1e-8, norm, 1.0)
    axis = v / safe
    half = 0.5 * norm
    qw = jp.cos(half)
    qxyz = jp.where(norm > 1e-8, axis * jp.sin(half), jp.zeros_like(v))
    return jp.concatenate([qw[None], qxyz])


def clip_to_bounds(target: jp.ndarray, anchor: jp.ndarray, bounds: jp.ndarray) -> jp.ndarray:
    """Clip target so |target - anchor| ≤ bounds, elementwise.

    Used to keep the controller's commanded position close to the bolt/hole,
    preventing the policy from wandering during exploration.
    """
    delta = jp.clip(target - anchor, -bounds, bounds)
    return anchor + delta


def reset_on_done(ema: jp.ndarray, done: jp.ndarray) -> jp.ndarray:
    """Zero out EMA action on done=True per-env, preserve otherwise.

    ema: (..., 6) batched EMA action state stored in state.info.
    done: (...,) bool — True when episode just terminated.
    """
    return jp.where(done[..., None], jp.zeros_like(ema), ema)
