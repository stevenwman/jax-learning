"""Offline gait analysis for splitbelt rollouts.

Pure numpy. Loads `splitbelt_traj.npz` (per-step primitives) + schedule_table,
returns adaptation metrics per spec S§9.3.
"""

from __future__ import annotations

import numpy as np


# Foot index convention: FL=0, FR=1, RL=2, RR=3 (matches go2_constants.FEET_GEOMS order).
_LEFT_FEET = (0, 2)
_RIGHT_FEET = (1, 3)


def detect_step_events(contact: np.ndarray) -> dict[str, list[list[int]]]:
    """Detect touchdown (rising-edge) and liftoff (falling-edge) per foot.

    Args:
        contact: bool array (T, F) where F is number of feet.

    Returns:
        dict with keys "touchdown_steps", "liftoff_steps", each a list of length F,
        each element a list of step indices.
    """
    T, F = contact.shape
    diffs = np.diff(contact.astype(np.int8), axis=0)  # (T-1, F)
    touchdown = [[int(t + 1) for t in np.flatnonzero(diffs[:, f] > 0)] for f in range(F)]
    liftoff = [[int(t + 1) for t in np.flatnonzero(diffs[:, f] < 0)] for f in range(F)]
    return {"touchdown_steps": touchdown, "liftoff_steps": liftoff}


def step_lengths(events: dict, foot_xy: np.ndarray) -> list[list[float]]:
    """Per-foot list of forward stride distances (touchdown_x[k+1] - touchdown_x[k])."""
    out: list[list[float]] = []
    for f, td_steps in enumerate(events["touchdown_steps"]):
        if len(td_steps) < 2:
            out.append([])
            continue
        xs = foot_xy[td_steps, f, 0]
        out.append([float(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)])
    return out


def step_length_asymmetry(events: dict, foot_xy: np.ndarray) -> float:
    """(SL_R - SL_L) / (SL_R + SL_L), averaged across all strides per side."""
    sl = step_lengths(events, foot_xy)
    left = [s for f in _LEFT_FEET for s in sl[f]]
    right = [s for f in _RIGHT_FEET for s in sl[f]]
    if not left or not right:
        return float("nan")
    sl_l = float(np.mean(left))
    sl_r = float(np.mean(right))
    if sl_l + sl_r == 0:
        return 0.0
    return (sl_r - sl_l) / (sl_r + sl_l)
