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


def test_make_buffer_without_critic_obs_dim():
    """make_buffer with critic_obs_dim=None → no extra critic buffer."""
    from jax_rl.training import ObsPipeline
    pipe = ObsPipeline(dict_obs=False, has_privileged=False, use_obs_norm=False)
    buffer = pipe.make_buffer(obs_dim=17, action_dim=6, buffer_size=1000)
    assert buffer is not None
    assert buffer._extra_obs_dims is None or buffer._extra_obs_dims == {}


def test_make_buffer_with_critic_obs_dim():
    """make_buffer with critic_obs_dim=64 → allocates critic_obs buffer."""
    from jax_rl.training import ObsPipeline
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False)
    buffer = pipe.make_buffer(obs_dim=17, action_dim=6, buffer_size=1000,
                              critic_obs_dim=64)
    assert buffer is not None
    assert "critic_obs" in buffer._extra_obs_dims
    assert buffer._extra_obs_dims["critic_obs"] == 64


def test_make_buffer_frame_stack_with_critic_obs_dim():
    """Frame stacking × privileged critic — this is the Go2Warp production path."""
    from jax_rl.training import ObsPipeline
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False,
                       n_frame_stack=3)
    # obs_dim = raw_dim * n_frame_stack = 16 * 3 = 48 (mimics Go2Warp)
    buffer = pipe.make_buffer(obs_dim=48, action_dim=12, buffer_size=1000,
                              critic_obs_dim=120, num_envs=4)
    assert buffer is not None
    assert "critic_obs" in buffer._extra_obs_dims
    assert buffer._extra_obs_dims["critic_obs"] == 120
    # Frame stack config should be populated.
    assert buffer._fsc is not None
    assert buffer._fsc.n_frames == 3
    assert buffer._fsc.raw_dim == 16


def test_make_buffer_privileged_without_critic_obs_dim_raises():
    """has_privileged=True with critic_obs_dim=None should raise ValueError."""
    import pytest
    from jax_rl.training import ObsPipeline
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False)
    with pytest.raises(ValueError, match="critic_obs_dim required"):
        pipe.make_buffer(obs_dim=17, action_dim=6, buffer_size=1000)


# ── Critic obs normalization (privileged path) ───────────────────────────


def test_init_critic_norm_state_fresh_when_enabled():
    """Enabled obs_norm → fresh running stats at given critic_obs_dim."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=True)
    cns = pipe.init_critic_norm_state(critic_obs_dim=16)
    assert cns.mean.shape == (16,)
    assert int(cns.count) == 0


def test_init_critic_norm_state_identity_when_disabled():
    """Disabled obs_norm → identity stats (mean=0, var=1, count=1)."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False)
    cns = pipe.init_critic_norm_state(critic_obs_dim=16)
    assert cns.mean.shape == (16,)
    assert int(cns.count) >= 1
    assert jnp.allclose(cns.mean, 0.0)


def test_update_critic_stats_noop_when_disabled():
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=False)
    cns = pipe.init_critic_norm_state(critic_obs_dim=8)
    obs = jnp.ones((4, 8))
    out = pipe.update_critic_stats(obs, cns)
    assert out is cns


def test_update_critic_stats_accumulates():
    """Count goes up; mean moves toward data."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=True)
    cns = pipe.init_critic_norm_state(critic_obs_dim=8)
    obs = jnp.ones((4, 8)) * 3.0
    cns = pipe.update_critic_stats(obs, cns)
    assert int(cns.count) == 4
    assert jnp.allclose(cns.mean, 3.0, atol=1e-5)


def test_normalize_critic_whitens():
    """After many updates, normalize_critic produces mean≈0, std≈1."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=True)
    cns = pipe.init_critic_norm_state(critic_obs_dim=4)
    key = jax.random.PRNGKey(0)
    for _ in range(50):
        key, sk = jax.random.split(key)
        obs = jax.random.normal(sk, (64, 4)) * 3.0 + 5.0
        cns = pipe.update_critic_stats(obs, cns)
    key, sk = jax.random.split(key)
    obs = jax.random.normal(sk, (1024, 4)) * 3.0 + 5.0
    whitened = pipe.normalize_critic(obs, cns)
    assert jnp.allclose(whitened.mean(axis=0), 0.0, atol=0.2)
    assert jnp.allclose(whitened.std(axis=0), 1.0, atol=0.2)


def test_normalize_critic_never_stacked():
    """normalize_critic ignores n_frame_stack — privileged obs is never stacked."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=True,
                       n_frame_stack=3)
    cns = pipe.init_critic_norm_state(critic_obs_dim=16)
    # If it tried to treat as stacked (16/3), shape math would break.
    obs = jnp.ones((8, 16)) * 5.0
    cns = pipe.update_critic_stats(obs, cns)
    out = pipe.normalize_critic(obs, cns)
    assert out.shape == (8, 16)


def test_normalize_batch_with_critic_norm_whitens_critic():
    """has_privileged + critic_norm_state → critic_obs/critic_next_obs normalized."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=True)
    ns = norm_init(4)
    cns = pipe.init_critic_norm_state(critic_obs_dim=8)
    key = jax.random.PRNGKey(1)
    for _ in range(50):
        key, sk = jax.random.split(key)
        cns = pipe.update_critic_stats(
            jax.random.normal(sk, (32, 8)) * 2.0 + 7.0, cns)
    batch = {
        "obs": jnp.ones((16, 4)),
        "next_obs": jnp.ones((16, 4)) * 2.0,
        "critic_obs": jnp.ones((16, 8)) * 7.0,   # = mean, should go to ≈0
        "critic_next_obs": jnp.ones((16, 8)) * 7.0,
    }
    out = pipe.normalize_batch(batch, ns, critic_norm_state=cns)
    assert jnp.allclose(out["critic_obs"], 0.0, atol=0.3)
    assert jnp.allclose(out["critic_next_obs"], 0.0, atol=0.3)


def test_normalize_batch_without_critic_norm_passes_critic_through():
    """has_privileged but no critic_norm_state → critic_obs unchanged."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=True)
    ns = norm_init(4)
    critic_data = jnp.ones((16, 8)) * 99.0
    batch = {
        "obs": jnp.ones((16, 4)),
        "next_obs": jnp.ones((16, 4)) * 2.0,
        "critic_obs": critic_data,
        "critic_next_obs": critic_data,
    }
    out = pipe.normalize_batch(batch, ns, critic_norm_state=None)
    assert jnp.allclose(out["critic_obs"], critic_data)
    assert jnp.allclose(out["critic_next_obs"], critic_data)


def test_make_critic_norm_fn_whitens_eval_input():
    """make_critic_norm_fn closure normalizes single eval-time critic obs."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=True, use_obs_norm=True)
    cns = pipe.init_critic_norm_state(critic_obs_dim=8)
    key = jax.random.PRNGKey(2)
    for _ in range(50):
        key, sk = jax.random.split(key)
        cns = pipe.update_critic_stats(
            jax.random.normal(sk, (32, 8)) * 2.0 + 4.0, cns)
    fn = pipe.make_critic_norm_fn(cns)
    obs = jnp.ones((4, 8)) * 4.0
    assert jnp.allclose(fn(obs), 0.0, atol=0.3)


def test_make_critic_norm_fn_none_when_no_privileged():
    """has_privileged=False → no critic-norm fn needed."""
    pipe = ObsPipeline(dict_obs=True, has_privileged=False, use_obs_norm=True)
    cns = pipe.init_critic_norm_state(critic_obs_dim=8)
    fn = pipe.make_critic_norm_fn(cns)
    assert fn is None
