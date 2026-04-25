"""JAX-native circular replay buffer for off-policy RL.

All data lives as jax.Array on GPU. Eliminates CPU↔GPU transfer at sample time.
add_batch accepts both jax.Array or numpy (auto-converts once).
Both add_batch and sample are JIT'd — all GPU ops compile into single XLA graphs.

Same interface as the numpy ReplayBuffer for drop-in replacement.
"""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class FrameStackConfig:
    """Config for sample-time frame stack reconstruction."""
    n_frames: int    # e.g., 3
    raw_dim: int     # e.g., 48 (single-frame obs dim)
    num_envs: int    # e.g., 1024 (stride for same-env lookback)


class JaxReplayBuffer:
    """GPU-resident circular FIFO replay buffer with uniform random sampling.

    Args:
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        max_size: Maximum number of transitions to store.
        frame_stack_config: When set, stores only raw (single-frame) obs and
            reconstructs stacked obs/next_obs at sample time.
            When None, behavior is identical to the original buffer.
        extra_obs_dims: Optional dict mapping name -> dim for extra observation
            fields (e.g. {"critic_obs": 122} for asymmetric critic). Allocates
            both `name` and `next_{name}` buffers. Stored/sampled alongside
            the main obs/action/reward arrays.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 1_000_000,
                 frame_stack_config: FrameStackConfig | None = None,
                 extra_obs_dims: dict[str, int] | None = None):
        self.max_size = max_size
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        self._jit_cache: dict = {}  # batch_size → compiled sample fn
        self._fsc = frame_stack_config

        if frame_stack_config is not None:
            self.obs_dim = frame_stack_config.raw_dim
            self._stacked_dim = frame_stack_config.raw_dim * frame_stack_config.n_frames
        else:
            self.obs_dim = obs_dim
            self._stacked_dim = obs_dim

        # Pre-allocate on GPU
        self.obs         = jnp.zeros((max_size, self.obs_dim), dtype=jnp.float32)
        self.actions     = jnp.zeros((max_size, action_dim),   dtype=jnp.float32)
        self.rewards     = jnp.zeros((max_size, 1),            dtype=jnp.float32)
        self.dones       = jnp.zeros((max_size, 1),            dtype=jnp.float32)
        self.truncations = jnp.zeros((max_size, 1),            dtype=jnp.float32)
        # Per-transition episode ID for TD-MPC2 per-episode sequence sampling (B2).
        # Supplied externally via add_batch(..., episode_ids=...); defaults to zeros.
        self.episode_ids = jnp.zeros((max_size,), dtype=jnp.int32)

        if frame_stack_config is None:
            self.next_obs = jnp.zeros((max_size, self.obs_dim), dtype=jnp.float32)
        # When frame stacking, next_obs is derived at sample time — no allocation

        # Extra obs buffers (e.g., critic_obs for asymmetric critic)
        self._extra_obs_dims = extra_obs_dims or {}
        self._extra_bufs: dict[str, jax.Array] = {}
        self._extra_next_keys: dict[str, str] = {}
        for name, dim in self._extra_obs_dims.items():
            # Convention: "critic_obs" → also allocates "critic_next_obs"
            next_name = name.replace("_obs", "_next_obs") if "_obs" in name else f"next_{name}"
            self._extra_bufs[name] = jnp.zeros((max_size, dim), dtype=jnp.float32)
            self._extra_bufs[next_name] = jnp.zeros((max_size, dim), dtype=jnp.float32)
            self._extra_next_keys[name] = next_name

    def add_batch(
        self,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        done: jax.Array,
        truncation: jax.Array | None = None,
        episode_ids: jax.Array | None = None,
        **extra: jax.Array,
    ) -> None:
        """Add a batch of transitions. Accepts jax.Array or numpy (auto-converts).

        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim)
            reward: (batch,) or (batch, 1)
            next_obs: (batch, obs_dim)
            done: (batch,) or (batch, 1)
            truncation: (batch,) or (batch, 1), optional
            episode_ids: (batch,) int32, optional. Per-transition episode IDs for
                TD-MPC2 per-episode sequence sampling (B2). Defaults to zeros if
                not supplied (backward compatible).
            **extra: Extra obs fields matching extra_obs_dims keys
                (e.g., critic_obs, next_critic_obs).
        """
        obs = jnp.asarray(obs)
        action = jnp.asarray(action)
        reward = jnp.asarray(reward).reshape(-1, 1)
        done = jnp.asarray(done).reshape(-1, 1)
        if truncation is None:
            truncation = jnp.zeros_like(done)
        else:
            truncation = jnp.asarray(truncation).reshape(-1, 1)

        # Frame stack mode: extract raw (newest) frame from stacked obs
        if self._fsc is not None:
            obs = obs[:, :self.obs_dim]
            # next_obs not stored — derived at sample time
        else:
            next_obs = jnp.asarray(next_obs)

        n = obs.shape[0]
        ptr = jnp.array(self.ptr)

        if self._fsc is not None:
            (self.obs, self.actions, self.rewards,
             self.dones, self.truncations) = self._jit_add_fs(
                self.obs, self.actions, self.rewards,
                self.dones, self.truncations,
                obs, action, reward, done, truncation, ptr,
            )
        else:
            # Single JIT'd scatter for all 6 arrays — compiles once, reuses every step.
            # Without JIT, each .at[].set() is a separate XLA dispatch that accumulates
            # command buffers and eventually OOMs on long runs.
            (self.obs, self.next_obs, self.actions,
             self.rewards, self.dones, self.truncations) = self._jit_add(
                self.obs, self.next_obs, self.actions,
                self.rewards, self.dones, self.truncations,
                obs, next_obs, action, reward, done, truncation, ptr,
            )

        # Scatter episode_ids via Python-level update (same pattern as extras below).
        # Avoids touching _jit_add / _jit_add_fs closures.
        if episode_ids is None:
            episode_ids = jnp.zeros(n, dtype=jnp.int32)
        else:
            episode_ids = jnp.asarray(episode_ids, dtype=jnp.int32)
        indices = (jnp.arange(n) + ptr) % self.max_size
        self.episode_ids = self.episode_ids.at[indices].set(episode_ids)

        # Store extra obs (e.g., critic_obs, next_critic_obs) via non-JIT scatter.
        # Overhead is tiny (<0.1ms per extra field) — not worth a separate JIT path.
        if self._extra_obs_dims and extra:
            indices = (jnp.arange(n) + ptr) % self.max_size
            for name in self._extra_obs_dims:
                if name in extra:
                    val = jnp.asarray(extra[name])
                    self._extra_bufs[name] = self._extra_bufs[name].at[indices].set(val)
                next_key = self._extra_next_keys[name]
                if next_key in extra:
                    val = jnp.asarray(extra[next_key])
                    self._extra_bufs[next_key] = self._extra_bufs[next_key].at[indices].set(val)

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

    @functools.cached_property
    def _jit_add_fs(self):
        """JIT'd scatter for frame-stack mode (no next_obs)."""
        max_size = self.max_size

        @jax.jit
        def _add(buf_obs, buf_act, buf_rew, buf_done, buf_trunc,
                 new_obs, new_act, new_rew, new_done, new_trunc, ptr):
            n = new_obs.shape[0]
            indices = (jnp.arange(n) + ptr) % max_size
            return (
                buf_obs.at[indices].set(new_obs),
                buf_act.at[indices].set(new_act),
                buf_rew.at[indices].set(new_rew),
                buf_done.at[indices].set(new_done),
                buf_trunc.at[indices].set(new_trunc),
            )
        return _add

    def _gather_extra(self, batch: dict, idx: jax.Array) -> dict:
        """Add extra obs fields to batch dict using pre-computed indices."""
        for name in self._extra_obs_dims:
            next_key = self._extra_next_keys[name]
            batch[name] = self._extra_bufs[name][idx]
            batch[next_key] = self._extra_bufs[next_key][idx]
        return batch

    def sample(self, batch_size: int, key: jax.Array | None = None) -> dict[str, jax.Array]:
        """Sample a random minibatch. Returns dict of jax.Array (already on GPU).

        Args:
            batch_size: Number of transitions to sample.
            key: PRNG key for sampling (required for JIT'd fast path).
        """
        has_extra = bool(self._extra_obs_dims)

        if key is None:
            # Non-JIT path (rarely used)
            idx = jnp.array(np.random.randint(0, self.size, size=batch_size))
            if self._fsc is not None:
                batch = {
                    "obs":        self._reconstruct(self.obs, self.dones, idx),
                    "action":     self.actions[idx],
                    "reward":     self.rewards[idx],
                    "next_obs":   self._reconstruct(self.obs, self.dones, (idx + self._fsc.num_envs) % self.max_size),
                    "done":       self.dones[idx],
                    "truncation": self.truncations[idx],
                }
            else:
                batch = {
                    "obs":        self.obs[idx],
                    "action":     self.actions[idx],
                    "reward":     self.rewards[idx],
                    "next_obs":   self.next_obs[idx],
                    "done":       self.dones[idx],
                    "truncation": self.truncations[idx],
                }
            return self._gather_extra(batch, idx) if has_extra else batch

        # JIT'd fast path: compile gather per batch_size (reuses across calls)
        if batch_size not in self._jit_cache:
            if self._fsc is not None:
                self._jit_cache[batch_size] = self._make_jit_sample_fs(batch_size)
            elif has_extra:
                self._jit_cache[batch_size] = self._make_jit_sample_with_idx(batch_size)
            else:
                self._jit_cache[batch_size] = self._make_jit_sample(batch_size)

        if self._fsc is not None:
            batch, idx = self._jit_cache[batch_size](
                self.obs, self.actions, self.rewards,
                self.dones, self.truncations,
                self.size, key,
            )
            return self._gather_extra(batch, idx) if has_extra else batch
        if has_extra:
            batch, idx = self._jit_cache[batch_size](
                self.obs, self.actions, self.rewards,
                self.next_obs, self.dones, self.truncations,
                self.size, key,
            )
            return self._gather_extra(batch, idx)
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

    def _make_jit_sample_with_idx(self, batch_size: int):
        """Create a JIT'd sample that also returns indices (for extra obs gather)."""
        @jax.jit
        def _sample(obs, actions, rewards, next_obs, dones, truncations, size, key):
            idx = jax.random.randint(key, (batch_size,), 0, size)
            batch = {
                "obs":        obs[idx],
                "action":     actions[idx],
                "reward":     rewards[idx],
                "next_obs":   next_obs[idx],
                "done":       dones[idx],
                "truncation": truncations[idx],
            }
            return batch, idx
        return _sample

    def _reconstruct(self, obs_buf, dones_buf, indices):
        """Reconstruct stacked obs from raw frames at given indices."""
        fsc = self._fsc
        frames = [obs_buf[indices]]
        prev_idx = indices
        valid = jnp.ones(indices.shape[0], dtype=jnp.bool_)

        for k in range(1, fsc.n_frames):
            cand_idx = (prev_idx - fsc.num_envs) % self.max_size
            boundary = dones_buf[cand_idx].squeeze(-1) > 0.5
            valid = valid & ~boundary
            frame = jnp.where(valid[:, None], obs_buf[cand_idx], frames[-1])
            frames.append(frame)
            prev_idx = cand_idx

        return jnp.concatenate(frames, axis=-1)

    def _make_jit_sample_fs(self, batch_size: int):
        """Create JIT'd sample function for frame-stack mode."""
        n_frames = self._fsc.n_frames
        num_envs = self._fsc.num_envs
        max_size = self.max_size

        @jax.jit
        def _sample(obs, actions, rewards, dones, truncations, size, key):
            # Sample valid indices (exclude first (n_frames-1)*num_envs entries
            # which lack history, and last num_envs which lack next_obs)
            min_idx = (n_frames - 1) * num_envs
            max_idx = size - num_envs
            # Clamp: if buffer is too small, sample from what's available
            max_idx = jnp.maximum(max_idx, min_idx + 1)

            raw_idx = jax.random.randint(key, (batch_size,), 0, max_idx - min_idx)
            idx = raw_idx + min_idx

            # Reconstruct obs stack
            def reconstruct(indices):
                frames = [obs[indices]]
                prev = indices
                valid = jnp.ones(batch_size, dtype=jnp.bool_)
                for kk in range(1, n_frames):
                    cand = (prev - num_envs) % max_size
                    boundary = dones[cand].squeeze(-1) > 0.5
                    valid = valid & ~boundary
                    frame = jnp.where(valid[:, None], obs[cand], frames[-1])
                    frames.append(frame)
                    prev = cand
                return jnp.concatenate(frames, axis=-1)

            stacked_obs = reconstruct(idx)
            next_idx = (idx + num_envs) % max_size
            stacked_next_obs = reconstruct(next_idx)

            batch = {
                "obs":        stacked_obs,
                "action":     actions[idx],
                "reward":     rewards[idx],
                "next_obs":   stacked_next_obs,
                "done":       dones[idx],
                "truncation": truncations[idx],
            }
            return batch, idx
        return _sample

    def sample_sequence(self, batch: int, H: int, key: jax.Array, stride: int = 1) -> dict:
        """Sample `batch` contiguous H+1 sequence windows, all within a single episode.

        Uses rejection sampling: oversample candidate start indices 4x and filter by
        constant-episode-id-across-window. Rejection rate expected <1% in normal training.

        `stride` handles multi-env round-robin storage. add_batch writes a (num_envs,)
        batch into consecutive buffer indices, so env e's transitions are at indices
        ptr+e, ptr+e+num_envs, ptr+e+2*num_envs, ... Set stride=num_envs so a window
        of H+1 indices stays within a single env's trajectory; default stride=1 matches
        single-env source behavior.

        Args:
            batch: number of sequences to return.
            H: horizon (actions/rewards per window; obs has H+1 steps).
            key: JAX PRNGKey.
            stride: index stride between consecutive timesteps (=num_envs for multi-env).

        Returns:
            Dict with:
              obs          (H+1, batch, obs_dim)
              actions      (H,   batch, action_dim)
              rewards      (H,   batch, 1)
              dones        (H,   batch, 1)        -- "terminated" flags
              truncations  (H,   batch, 1)

        Raises:
            ValueError: if buffer has fewer than H+1 transitions, or if oversample
                cannot find `batch` valid windows.
        """
        span = stride * H  # last offset = stride*H; need start + span < size
        if self.size < span + 1:
            raise ValueError(f"Buffer has {self.size} transitions; need >= {span + 1}")

        max_start = self.size - span - 1
        n_candidates = 4 * batch

        candidates = jax.random.randint(key, (n_candidates,), 0, max_start + 1)
        offsets = jnp.arange(H + 1) * stride
        eids = self.episode_ids[candidates[:, None] + offsets[None, :]]
        valid = jnp.all(eids == eids[:, :1], axis=1)

        n_valid = int(valid.sum())
        if n_valid < batch:
            raise ValueError(
                f"Only {n_valid}/{batch} valid sequence windows in {n_candidates} candidates; "
                f"buffer may be full of very short episodes."
            )

        order = jnp.argsort(-valid.astype(jnp.int32))
        starts = candidates[order][:batch]

        obs_idx = starts[:, None] + jnp.arange(H + 1)[None, :] * stride
        trans_idx = starts[:, None] + jnp.arange(H)[None, :] * stride

        obs = self.obs[obs_idx].transpose(1, 0, 2)
        actions = self.actions[trans_idx].transpose(1, 0, 2)
        rewards = self.rewards[trans_idx].transpose(1, 0, 2)
        dones = self.dones[trans_idx].transpose(1, 0, 2)
        truncations = self.truncations[trans_idx].transpose(1, 0, 2)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "truncations": truncations,
        }

    def __len__(self) -> int:
        return self.size
