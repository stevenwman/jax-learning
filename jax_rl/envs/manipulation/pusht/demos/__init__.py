"""Load LeRobot pusht expert demos packed in pusht_demos.npz."""
from pathlib import Path
from typing import Optional

import numpy as np

_DEMO_PATH = Path(__file__).parent / "pusht_demos.npz"


def load_demos(demo_path: Optional[str] = None) -> dict:
    """Load all demos. Returns dict with keys:
        obs_state, actions, rewards, dones, successes,
        episode_idx, frame_idx, ep_bounds.
    """
    path = Path(demo_path) if demo_path else _DEMO_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Demo file not found: {path}. Run demos/download.py to fetch."
        )
    return dict(np.load(path))


def get_episode(ep_idx: int, demo_path: Optional[str] = None) -> dict:
    """Return one episode's data as a dict of (T_i, ...) arrays."""
    d = load_demos(demo_path)
    start, end = d["ep_bounds"][ep_idx]
    return {k: v[start:end] for k, v in d.items() if v.ndim > 0 and v.shape[0] == len(d["obs_state"])}


def n_episodes(demo_path: Optional[str] = None) -> int:
    return len(load_demos(demo_path)["ep_bounds"])
