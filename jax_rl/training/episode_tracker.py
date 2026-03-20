"""Episode return tracking — replaces the 7-line block duplicated across all train scripts."""

import numpy as np


class EpisodeTracker:
    """Track per-env episode rewards, compute completed returns.

    Replaces:
        episode_rewards = np.zeros(num_envs)
        completed_returns: list[float] = []
        # ... and the step_rewards/done_mask block in each training loop
    """

    def __init__(self, num_envs: int):
        self.episode_rewards = np.zeros(num_envs)
        self.completed_returns: list[float] = []

    def step(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        """Update episode rewards and track completed episodes."""
        self.episode_rewards += rewards
        done_mask = dones.astype(bool)
        if done_mask.any():
            self.completed_returns.extend(self.episode_rewards[done_mask].tolist())
            self.episode_rewards[done_mask] = 0.0

    def recent_stats(self, n: int = 100) -> dict:
        """Compute avg/min/max return over last n completed episodes."""
        if not self.completed_returns:
            return {"avg": float("nan"), "min": float("nan"),
                    "max": float("nan"), "n_eps": 0}
        recent = self.completed_returns[-n:]
        return {
            "avg": float(np.mean(recent)),
            "min": float(np.min(recent)),
            "max": float(np.max(recent)),
            "n_eps": len(self.completed_returns),
        }

    @property
    def n_episodes(self) -> int:
        return len(self.completed_returns)
