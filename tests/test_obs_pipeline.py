"""Tests for ObsPipeline — stateless obs extraction + normalization."""

import jax
import jax.numpy as jnp

from jax_rl.training.obs_pipeline import ObsPipeline
from jax_rl.utils.normalization import init as norm_init


# ── Obs extraction ────────────────────────────────────────────────────────


def test_flat_obs_passthrough():
    """Non-dict obs returns unchanged."""
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=False)
    obs = jnp.ones((4, 8))
    result = pipe.get_obs(obs)
    assert result is obs


def test_dict_obs_extracts_state():
    """Dict obs returns 'state' key."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=False, use_obs_norm=False)
    state_obs = jnp.ones((4, 8))
    obs = {"state": state_obs, "other": jnp.zeros((4, 3))}
    result = pipe.get_obs(obs)
    assert result is state_obs


def test_critic_obs_privileged():
    """Returns 'privileged_state' when has_privileged=True."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False)
    priv = jnp.ones((4, 12))
    obs = {"state": jnp.zeros((4, 8)), "privileged_state": priv}
    result = pipe.get_critic_obs(obs)
    assert result is priv


def test_critic_obs_fallback():
    """Falls back to actor obs when has_privileged=False."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=False, use_obs_norm=False)
    state_obs = jnp.ones((4, 8))
    obs = {"state": state_obs}
    result = pipe.get_critic_obs(obs)
    assert result is state_obs


# ── Normalization stats ───────────────────────────────────────────────────


def test_update_stats_noop_when_disabled():
    """Returns norm_state unchanged when obs_norm is disabled."""
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=False)
    ns = norm_init(8)
    obs = jnp.ones((4, 8))
    result = pipe.update_stats(obs, ns)
    assert result is ns
    assert result.count == 0


def test_update_stats_updates_when_enabled():
    """norm_state.count increases when obs_norm is enabled."""
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=True)
    ns = norm_init(8)
    obs = jnp.ones((4, 8))
    result = pipe.update_stats(obs, ns)
    assert result.count == 4
    assert not jnp.allclose(result.mean, ns.mean)


# ── Normalize for action ─────────────────────────────────────────────────


def test_normalize_for_action_noop():
    """Passthrough when obs_norm is disabled."""
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=False)
    ns = norm_init(8)
    obs = jnp.ones((4, 8)) * 5.0
    result = pipe.normalize_for_action(obs, ns)
    assert result is obs


def test_normalize_for_action_transforms():
    """Obs is actually normalized when enabled."""
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=True)
    key = jax.random.PRNGKey(0)

    # Build up some stats
    ns = norm_init(4)
    for _ in range(20):
        key, sk = jax.random.split(key)
        ns = pipe.update_stats(jax.random.normal(sk, (32, 4)) * 3.0 + 5.0, ns)

    obs = jnp.ones((8, 4)) * 5.0
    result = pipe.normalize_for_action(obs, ns)
    # Mean is ~5, so obs=5 should normalize to ~0
    assert jnp.allclose(result, 0.0, atol=0.3)


# ── Batch normalization ──────────────────────────────────────────────────


def test_normalize_batch_sets_critic_keys():
    """Non-privileged batch gets critic_obs = obs after normalization."""
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=False)
    ns = norm_init(4)
    batch = {
        "obs": jnp.ones((8, 4)),
        "next_obs": jnp.ones((8, 4)) * 2.0,
    }
    result = pipe.normalize_batch(batch, ns)
    assert "critic_obs" in result
    assert "critic_next_obs" in result
    assert jnp.allclose(result["critic_obs"], result["obs"])
    assert jnp.allclose(result["critic_next_obs"], result["next_obs"])


def test_normalize_batch_preserves_critic_keys():
    """Privileged batch keeps existing critic_obs (does not overwrite)."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False)
    ns = norm_init(4)
    critic_data = jnp.ones((8, 6)) * 99.0
    batch = {
        "obs": jnp.ones((8, 4)),
        "next_obs": jnp.ones((8, 4)) * 2.0,
        "critic_obs": critic_data,
        "critic_next_obs": critic_data,
    }
    result = pipe.normalize_batch(batch, ns)
    # critic_obs should be untouched (has_privileged=True, so no aliasing)
    assert jnp.allclose(result["critic_obs"], critic_data)


# ── Buffer factory ────────────────────────────────────────────────────────


def test_make_buffer_basic():
    """Buffer created with correct obs_dim."""
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=False)
    buf = pipe.make_buffer(obs_dim=8, action_dim=3, buffer_size=1000)
    assert buf.obs_dim == 8
    assert buf.action_dim == 3
    assert buf.max_size == 1000


def test_make_buffer_frame_stack():
    """Buffer created with frame stack config stores raw_dim."""
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=False,
                       n_frame_stack=3)
    buf = pipe.make_buffer(obs_dim=24, action_dim=3, buffer_size=1000, num_envs=16)
    # Buffer stores raw frames (24 / 3 = 8)
    assert buf.obs_dim == 8
    assert buf._fsc is not None
    assert buf._fsc.n_frames == 3


# ── Eval obs_norm_fn ──────────────────────────────────────────────────────


def test_make_obs_norm_fn_none_when_disabled():
    """Returns None when no obs_norm and flat obs."""
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=False)
    ns = norm_init(8)
    fn = pipe.make_obs_norm_fn(ns)
    assert fn is None


def test_make_obs_norm_fn_extracts_dict():
    """Returns a function that extracts 'state' from dict obs even without norm."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=False, use_obs_norm=False)
    ns = norm_init(4)
    fn = pipe.make_obs_norm_fn(ns)
    assert fn is not None
    state_obs = jnp.ones((4, 4))
    result = fn({"state": state_obs})
    assert jnp.allclose(result, state_obs)
