"""Numpy circular replay buffer for off-policy RL (SAC, TD3).

Stores transitions on CPU as numpy arrays. GPU only sees sampled minibatches.
Works with any env — jittable or not.
"""

import numpy as np


class ReplayBuffer:
    """Circular FIFO replay buffer with uniform random sampling.

    Stores raw (unnormalized) observations. Normalization happens at gradient
    time inside the algorithm's update step.

    Args:
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        max_size: Maximum number of transitions to store.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.ptr = 0       # write head
        self.size = 0      # current number of stored transitions

        self.obs         = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.next_obs    = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self.dones       = np.zeros((max_size, 1),          dtype=np.float32)
        self.truncations = np.zeros((max_size, 1),          dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float | np.ndarray,
        next_obs: np.ndarray,
        done: float | np.ndarray,
        truncation: float | np.ndarray = 0.0,
    ) -> None:
        """Add a single transition. All inputs are scalars or 1-D arrays."""
        self.obs[self.ptr]         = obs
        self.next_obs[self.ptr]    = next_obs
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.dones[self.ptr]       = done
        self.truncations[self.ptr] = truncation

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        truncation: np.ndarray | None = None,
    ) -> None:
        """Add a batch of transitions (shape: (batch, dim))."""
        n = obs.shape[0]
        if truncation is None:
            truncation = np.zeros((n, 1), dtype=np.float32)

        indices = np.arange(self.ptr, self.ptr + n) % self.max_size
        self.obs[indices]         = obs
        self.next_obs[indices]    = next_obs
        self.actions[indices]     = action
        self.rewards[indices]     = reward.reshape(n, 1)
        self.dones[indices]       = done.reshape(n, 1)
        self.truncations[indices] = truncation.reshape(n, 1)

        self.ptr  = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random minibatch. Returns a dict of numpy arrays."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":         self.obs[idx],
            "action":      self.actions[idx],
            "reward":      self.rewards[idx],
            "next_obs":    self.next_obs[idx],
            "done":        self.dones[idx],
            "truncation":  self.truncations[idx],
        }

    def __len__(self) -> int:
        return self.size
