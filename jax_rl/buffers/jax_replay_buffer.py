"""JAX-native circular replay buffer for off-policy RL.

All data lives as jax.Array on GPU. Eliminates CPU↔GPU transfer at sample time.
add_batch accepts both jax.Array or numpy (auto-converts once).
Both add_batch and sample are JIT'd — all GPU ops compile into single XLA graphs.

Same interface as the numpy ReplayBuffer for drop-in replacement.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np


class JaxReplayBuffer:
    """GPU-resident circular FIFO replay buffer with uniform random sampling.

    Args:
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        max_size: Maximum number of transitions to store.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        self._jit_cache: dict = {}  # batch_size → compiled sample fn

        # Pre-allocate on GPU
        self.obs         = jnp.zeros((max_size, obs_dim),    dtype=jnp.float32)
        self.next_obs    = jnp.zeros((max_size, obs_dim),    dtype=jnp.float32)
        self.actions     = jnp.zeros((max_size, action_dim), dtype=jnp.float32)
        self.rewards     = jnp.zeros((max_size, 1),          dtype=jnp.float32)
        self.dones       = jnp.zeros((max_size, 1),          dtype=jnp.float32)
        self.truncations = jnp.zeros((max_size, 1),          dtype=jnp.float32)

    def add_batch(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        truncation=None,
    ) -> None:
        """Add a batch of transitions. Accepts jax.Array or numpy (auto-converts).

        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim)
            reward: (batch,) or (batch, 1)
            next_obs: (batch, obs_dim)
            done: (batch,) or (batch, 1)
            truncation: (batch,) or (batch, 1), optional
        """
        obs = jnp.asarray(obs)
        action = jnp.asarray(action)
        reward = jnp.asarray(reward).reshape(-1, 1)
        next_obs = jnp.asarray(next_obs)
        done = jnp.asarray(done).reshape(-1, 1)
        if truncation is None:
            truncation = jnp.zeros_like(done)
        else:
            truncation = jnp.asarray(truncation).reshape(-1, 1)

        n = obs.shape[0]
        ptr = jnp.array(self.ptr)

        # Single JIT'd scatter for all 6 arrays — compiles once, reuses every step.
        # Without JIT, each .at[].set() is a separate XLA dispatch that accumulates
        # command buffers and eventually OOMs on long runs.
        (self.obs, self.next_obs, self.actions,
         self.rewards, self.dones, self.truncations) = self._jit_add(
            self.obs, self.next_obs, self.actions,
            self.rewards, self.dones, self.truncations,
            obs, next_obs, action, reward, done, truncation, ptr,
        )

        self.ptr  = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    @functools.cached_property
    def _jit_add(self):
        """JIT'd scatter — compiled once on first call, cached permanently."""
        max_size = self.max_size

        @jax.jit
        def _add(buf_obs, buf_next, buf_act, buf_rew, buf_done, buf_trunc,
                 new_obs, new_next, new_act, new_rew, new_done, new_trunc, ptr):
            n = new_obs.shape[0]
            indices = (jnp.arange(n) + ptr) % max_size
            return (
                buf_obs.at[indices].set(new_obs),
                buf_next.at[indices].set(new_next),
                buf_act.at[indices].set(new_act),
                buf_rew.at[indices].set(new_rew),
                buf_done.at[indices].set(new_done),
                buf_trunc.at[indices].set(new_trunc),
            )
        return _add

    def sample(self, batch_size: int, key: jax.Array | None = None) -> dict[str, jax.Array]:
        """Sample a random minibatch. Returns dict of jax.Array (already on GPU).

        Args:
            batch_size: Number of transitions to sample.
            key: PRNG key for sampling (required for JIT'd fast path).
        """
        if key is None:
            idx = jnp.array(np.random.randint(0, self.size, size=batch_size))
            return {
                "obs":        self.obs[idx],
                "action":     self.actions[idx],
                "reward":     self.rewards[idx],
                "next_obs":   self.next_obs[idx],
                "done":       self.dones[idx],
                "truncation": self.truncations[idx],
            }
        # JIT'd fast path: compile gather per batch_size (reuses across calls)
        if batch_size not in self._jit_cache:
            self._jit_cache[batch_size] = self._make_jit_sample(batch_size)
        return self._jit_cache[batch_size](
            self.obs, self.actions, self.rewards,
            self.next_obs, self.dones, self.truncations,
            self.size, key,
        )

    def _make_jit_sample(self, batch_size: int):
        """Create a JIT'd sample function for a specific batch_size."""
        @jax.jit
        def _sample(obs, actions, rewards, next_obs, dones, truncations, size, key):
            idx = jax.random.randint(key, (batch_size,), 0, size)
            return {
                "obs":        obs[idx],
                "action":     actions[idx],
                "reward":     rewards[idx],
                "next_obs":   next_obs[idx],
                "done":       dones[idx],
                "truncation": truncations[idx],
            }
        return _sample

    def __len__(self) -> int:
        return self.size
