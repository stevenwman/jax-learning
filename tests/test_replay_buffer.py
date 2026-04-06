"""Tests for JaxReplayBuffer — GPU-resident replay buffer."""

import os
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer

OBS_DIM = 17
ACTION_DIM = 6
KEY = jax.random.PRNGKey(0)


def test_init_empty():
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    assert len(buf) == 0


def test_add_batch():
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    buf.add_batch(
        obs=jnp.zeros((10, OBS_DIM)),
        action=jnp.zeros((10, ACTION_DIM)),
        reward=jnp.zeros(10),
        next_obs=jnp.zeros((10, OBS_DIM)),
        done=jnp.zeros(10),
        truncation=jnp.zeros(10),
    )
    assert len(buf) == 10


def test_add_multiple_batches():
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    for i in range(5):
        buf.add_batch(
            obs=jnp.ones((10, OBS_DIM)) * i,
            action=jnp.zeros((10, ACTION_DIM)),
            reward=jnp.ones(10) * i,
            next_obs=jnp.ones((10, OBS_DIM)) * i,
            done=jnp.zeros(10),
            truncation=jnp.zeros(10),
        )
    assert len(buf) == 50


def test_wrap_around():
    """Buffer should wrap when exceeding max_size."""
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=20)
    for i in range(5):
        buf.add_batch(
            obs=jnp.ones((10, OBS_DIM)) * i,
            action=jnp.zeros((10, ACTION_DIM)),
            reward=jnp.ones(10) * i,
            next_obs=jnp.ones((10, OBS_DIM)) * i,
            done=jnp.zeros(10),
            truncation=jnp.zeros(10),
        )
    # Should cap at max_size
    assert len(buf) == 20


def test_sample_shape():
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    buf.add_batch(
        obs=jax.random.normal(KEY, (50, OBS_DIM)),
        action=jax.random.normal(KEY, (50, ACTION_DIM)),
        reward=jnp.zeros(50),
        next_obs=jax.random.normal(KEY, (50, OBS_DIM)),
        done=jnp.zeros(50),
        truncation=jnp.zeros(50),
    )

    batch = buf.sample(32, key=KEY)
    assert batch["obs"].shape == (32, OBS_DIM)
    assert batch["action"].shape == (32, ACTION_DIM)
    assert batch["reward"].shape == (32, 1)
    assert batch["next_obs"].shape == (32, OBS_DIM)
    assert batch["done"].shape == (32, 1)
    assert batch["truncation"].shape == (32, 1)


def test_sample_values_from_buffer():
    """Sampled values should come from data that was actually added."""
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
    buf.add_batch(
        obs=jnp.ones((50, OBS_DIM)) * 7.0,
        action=jnp.ones((50, ACTION_DIM)) * 3.0,
        reward=jnp.ones(50) * 2.0,
        next_obs=jnp.ones((50, OBS_DIM)) * 7.0,
        done=jnp.zeros(50),
        truncation=jnp.zeros(50),
    )

    batch = buf.sample(16, key=KEY)
    assert jnp.allclose(batch["obs"], 7.0)
    assert jnp.allclose(batch["action"], 3.0)
    assert jnp.allclose(batch["reward"], 2.0)


def test_sample_different_keys_give_different_batches():
    """Different random keys should produce different samples."""
    buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=1000)
    buf.add_batch(
        obs=jax.random.normal(KEY, (500, OBS_DIM)),
        action=jax.random.normal(KEY, (500, ACTION_DIM)),
        reward=jax.random.normal(KEY, (500,)),
        next_obs=jax.random.normal(KEY, (500, OBS_DIM)),
        done=jnp.zeros(500),
        truncation=jnp.zeros(500),
    )

    b1 = buf.sample(32, key=jax.random.PRNGKey(1))
    b2 = buf.sample(32, key=jax.random.PRNGKey(2))
    # Very unlikely to be identical with different keys
    assert not jnp.allclose(b1["obs"], b2["obs"])


# ── Frame-stack reconstruction tests ─────────────────────────────────────

from jax_rl.buffers.jax_replay_buffer import FrameStackConfig


class TestFrameStackBuffer:
    """Tests for sample-time frame stack reconstruction."""

    def _make_buffer(self, max_size=200, num_envs=4, n_frames=3, raw_dim=8):
        fsc = FrameStackConfig(n_frames=n_frames, raw_dim=raw_dim, num_envs=num_envs)
        return JaxReplayBuffer(raw_dim, ACTION_DIM, max_size=max_size, frame_stack_config=fsc), fsc

    def test_stores_raw_dim(self):
        buf, fsc = self._make_buffer(raw_dim=8)
        assert buf.obs.shape == (200, 8)
        assert not hasattr(buf, 'next_obs') or buf.__dict__.get('next_obs') is None

    def test_no_next_obs_allocated(self):
        buf, _ = self._make_buffer()
        # next_obs should not be in __dict__ when frame stacking
        assert 'next_obs' not in buf.__dict__

    def test_add_extracts_newest_frame(self):
        buf, fsc = self._make_buffer(num_envs=4, raw_dim=8)
        # Simulate stacked obs: [frame0=1s, frame1=2s, frame2=3s]
        stacked = jnp.concatenate([
            jnp.ones((4, 8)) * 1.0,  # newest
            jnp.ones((4, 8)) * 2.0,
            jnp.ones((4, 8)) * 3.0,
        ], axis=-1)  # (4, 24)
        buf.add_batch(
            obs=stacked,
            action=jnp.zeros((4, ACTION_DIM)),
            reward=jnp.zeros(4),
            next_obs=stacked,  # ignored
            done=jnp.zeros(4),
        )
        # Should store only the first 8 dims (newest frame)
        assert jnp.allclose(buf.obs[0], 1.0)
        assert jnp.allclose(buf.obs[1], 1.0)

    def test_sample_returns_stacked_shape(self):
        buf, fsc = self._make_buffer(max_size=200, num_envs=4, n_frames=3, raw_dim=8)
        # Add enough transitions for valid sampling
        for i in range(20):
            obs = jnp.ones((4, 8)) * float(i)
            stacked = jnp.tile(obs, (1, 3))  # (4, 24)
            buf.add_batch(
                obs=stacked, action=jnp.zeros((4, ACTION_DIM)),
                reward=jnp.zeros(4), next_obs=stacked,
                done=jnp.zeros(4),
            )
        batch = buf.sample(16, key=jax.random.PRNGKey(0))
        assert batch["obs"].shape == (16, 24)  # 3 * 8
        assert batch["next_obs"].shape == (16, 24)

    def test_reconstruction_correctness(self):
        """Verify frames are reconstructed in correct order (newest first)."""
        buf, fsc = self._make_buffer(max_size=200, num_envs=2, n_frames=3, raw_dim=4)
        # Add 5 batches with known values (env0 and env1 get same value per batch)
        for t in range(5):
            obs = jnp.ones((2, 4)) * float(t)
            stacked = jnp.tile(obs, (1, 3))
            buf.add_batch(
                obs=stacked, action=jnp.zeros((2, ACTION_DIM)),
                reward=jnp.zeros(2), next_obs=stacked,
                done=jnp.zeros(2),
            )
        # Sample index for env 0 at t=4: index = 4*2 + 0 = 8
        # Lookback: t=4 (idx=8), t=3 (idx=6), t=2 (idx=4)
        # Expected stack: [4.0, 3.0, 2.0] (each repeated 4 times)
        idx = jnp.array([8])
        reconstructed = buf._reconstruct(buf.obs, buf.dones, idx)
        assert reconstructed.shape == (1, 12)  # 3 * 4
        assert jnp.allclose(reconstructed[0, :4], 4.0)   # newest: t=4
        assert jnp.allclose(reconstructed[0, 4:8], 3.0)   # t=3
        assert jnp.allclose(reconstructed[0, 8:12], 2.0)  # oldest: t=2

    def test_episode_boundary_tiles(self):
        """At episode boundaries, older frames should be tiled from latest valid frame."""
        buf, fsc = self._make_buffer(max_size=200, num_envs=2, n_frames=3, raw_dim=4)
        # t=0: obs=0, done=0
        buf.add_batch(obs=jnp.tile(jnp.zeros((2, 4)), (1, 3)),
                      action=jnp.zeros((2, ACTION_DIM)),
                      reward=jnp.zeros(2), next_obs=jnp.zeros((2, 12)),
                      done=jnp.zeros(2))
        # t=1: obs=1, done=1 (episode ends!)
        buf.add_batch(obs=jnp.tile(jnp.ones((2, 4)), (1, 3)),
                      action=jnp.zeros((2, ACTION_DIM)),
                      reward=jnp.zeros(2), next_obs=jnp.ones((2, 12)),
                      done=jnp.ones(2))
        # t=2: obs=2, done=0 (new episode, after auto-reset)
        buf.add_batch(obs=jnp.tile(jnp.ones((2, 4)) * 2, (1, 3)),
                      action=jnp.zeros((2, ACTION_DIM)),
                      reward=jnp.zeros(2), next_obs=jnp.ones((2, 12)) * 2,
                      done=jnp.zeros(2))
        # t=3: obs=3, done=0
        buf.add_batch(obs=jnp.tile(jnp.ones((2, 4)) * 3, (1, 3)),
                      action=jnp.zeros((2, ACTION_DIM)),
                      reward=jnp.zeros(2), next_obs=jnp.ones((2, 12)) * 3,
                      done=jnp.zeros(2))

        # Sample t=3 for env 0: idx = 3*2 + 0 = 6
        # Lookback: t=3 (idx=6), t=2 (idx=4), t=1 (idx=2, done=1 -> boundary!)
        # Expected: frame0=3, frame1=2, frame2=2 (tiled from frame1 due to boundary)
        idx = jnp.array([6])
        reconstructed = buf._reconstruct(buf.obs, buf.dones, idx)
        assert jnp.allclose(reconstructed[0, :4], 3.0)    # newest: t=3
        assert jnp.allclose(reconstructed[0, 4:8], 2.0)    # t=2 (valid)
        assert jnp.allclose(reconstructed[0, 8:12], 2.0)   # t=1 has done=1, tile t=2

    def test_next_obs_derived_correctly(self):
        """next_obs should be the stack at index + num_envs."""
        buf, fsc = self._make_buffer(max_size=200, num_envs=2, n_frames=3, raw_dim=4)
        for t in range(6):
            obs = jnp.ones((2, 4)) * float(t)
            stacked = jnp.tile(obs, (1, 3))
            buf.add_batch(
                obs=stacked, action=jnp.zeros((2, ACTION_DIM)),
                reward=jnp.zeros(2), next_obs=stacked,
                done=jnp.zeros(2),
            )
        # Sample t=3 for env 0: idx = 3*2+0 = 6
        # obs stack: [3, 2, 1]
        # next_obs = stack at idx+num_envs = 8 -> [4, 3, 2]
        idx = jnp.array([6])
        obs_stack = buf._reconstruct(buf.obs, buf.dones, idx)
        next_idx = (idx + fsc.num_envs) % buf.max_size
        next_stack = buf._reconstruct(buf.obs, buf.dones, next_idx)
        assert jnp.allclose(obs_stack[0, :4], 3.0)
        assert jnp.allclose(next_stack[0, :4], 4.0)
        assert jnp.allclose(next_stack[0, 4:8], 3.0)

    def test_plain_buffer_unchanged(self):
        """Non-frame-stack buffer should work identically to before."""
        buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
        buf.add_batch(
            obs=jnp.ones((10, OBS_DIM)) * 5.0,
            action=jnp.zeros((10, ACTION_DIM)),
            reward=jnp.ones(10),
            next_obs=jnp.ones((10, OBS_DIM)) * 6.0,
            done=jnp.zeros(10),
        )
        batch = buf.sample(8, key=jax.random.PRNGKey(0))
        assert batch["obs"].shape == (8, OBS_DIM)
        assert batch["next_obs"].shape == (8, OBS_DIM)
        assert jnp.allclose(batch["obs"], 5.0)
        assert jnp.allclose(batch["next_obs"], 6.0)


# ── Extra obs (asymmetric critic) tests ──────────────────────────────────

CRITIC_DIM = 32


class TestExtraObsBuffer:
    """Tests for extra_obs_dims (asymmetric critic support)."""

    def test_extra_next_keys_mapping(self):
        """_extra_next_keys should map each extra name to its next-obs key."""
        buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100,
                              extra_obs_dims={"critic_obs": CRITIC_DIM})
        assert buf._extra_next_keys == {"critic_obs": "critic_next_obs"}

    def test_extra_next_keys_multiple(self):
        """Multiple extra obs dims should all get correct next-key mappings."""
        buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100,
                              extra_obs_dims={"critic_obs": CRITIC_DIM,
                                              "aux_obs": 8})
        assert buf._extra_next_keys == {"critic_obs": "critic_next_obs",
                                        "aux_obs": "aux_next_obs"}

    def test_extra_next_keys_no_obs_suffix(self):
        """Names without '_obs' should use 'next_' prefix convention."""
        buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100,
                              extra_obs_dims={"privileged": 64})
        assert buf._extra_next_keys == {"privileged": "next_privileged"}

    def test_extra_bufs_allocated(self):
        """Both current and next buffers should be allocated with correct shapes."""
        buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100,
                              extra_obs_dims={"critic_obs": CRITIC_DIM})
        assert "critic_obs" in buf._extra_bufs
        assert "critic_next_obs" in buf._extra_bufs
        assert buf._extra_bufs["critic_obs"].shape == (100, CRITIC_DIM)
        assert buf._extra_bufs["critic_next_obs"].shape == (100, CRITIC_DIM)

    def test_add_and_sample_extra_obs(self):
        """Extra obs should round-trip through add_batch → sample correctly."""
        buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100,
                              extra_obs_dims={"critic_obs": CRITIC_DIM})
        n = 20
        buf.add_batch(
            obs=jnp.ones((n, OBS_DIM)),
            action=jnp.zeros((n, ACTION_DIM)),
            reward=jnp.zeros(n),
            next_obs=jnp.ones((n, OBS_DIM)),
            done=jnp.zeros(n),
            critic_obs=jnp.ones((n, CRITIC_DIM)) * 3.0,
            critic_next_obs=jnp.ones((n, CRITIC_DIM)) * 4.0,
        )
        batch = buf.sample(8, key=KEY)
        assert batch["critic_obs"].shape == (8, CRITIC_DIM)
        assert batch["critic_next_obs"].shape == (8, CRITIC_DIM)
        assert jnp.allclose(batch["critic_obs"], 3.0)
        assert jnp.allclose(batch["critic_next_obs"], 4.0)

    def test_no_extra_obs_unchanged(self):
        """Buffer without extra_obs_dims should have empty mappings."""
        buf = JaxReplayBuffer(OBS_DIM, ACTION_DIM, max_size=100)
        assert not hasattr(buf, '_extra_next_keys') or buf._extra_obs_dims == {}
