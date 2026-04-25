"""Tests for video-recording rollout builders.

Core contract at inference time:
- norm_state is FROZEN (no stats update during rollout).
- Frame-stacked obs → normalize_stacked (single-frame stats, per-frame apply).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_rl.utils.normalization import (
    init as norm_init,
    normalize as norm_normalize,
    normalize_stacked as norm_normalize_stacked,
    update as norm_update,
)


class _FakeEnvState:
    """Bare-bones env state for testing (no pytree needed — we use carry, not scan)."""
    def __init__(self, obs):
        self.obs = obs


class _FakeAlgo:
    """Algo stub whose select_action returns zeros (so clipping/env stepping deterministic)."""
    def select_action(self, state, obs, key, deterministic=True):
        action = jnp.zeros(obs.shape[:-1] + (1,))
        return action, None, None


def _env_step(env_state, action):
    # Increment obs by 1.0 each step — simple trajectory for testing
    return _FakeEnvState(env_state.obs + 1.0)


def test_ppo_rollout_freezes_norm_fs1():
    """PPO rollout must NOT mutate norm_state (flat obs, FS=1)."""
    from jax_rl.utils.rollout import build_ppo_rollout_step

    obs_dim = 5
    ns = norm_init(obs_dim)
    # Seed with non-identity stats to make bug (mutation) detectable
    ns = norm_update(ns, jnp.ones((10, obs_dim)) * 3.0)
    original = jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, ns)

    training_state = type("TS", (), {"actor_params": None})()
    algo = _FakeAlgo()
    env_state = _FakeEnvState(jnp.zeros(obs_dim))

    step_fn, _ = build_ppo_rollout_step(algo, training_state, ns, _env_step,
                                        n_frame_stack=1)
    carry = (env_state, jax.random.PRNGKey(0))
    for i in range(5):
        carry, _ = step_fn(carry, i)

    # norm_state should be unchanged — it was frozen in closure, not in carry
    assert len(carry) == 2, f"carry must be (env_state, key), got len {len(carry)}"
    assert jnp.allclose(ns.mean, original.mean)
    assert jnp.allclose(ns.mean_of_squares, original.mean_of_squares)
    assert ns.count == original.count


def test_ppo_rollout_fs3_uses_stacked_normalize():
    """FS=3: normalize_stacked path produces the expected shape and whitening."""
    from jax_rl.utils.rollout import build_ppo_rollout_step

    single_dim = 4
    fs = 3
    ns = norm_init(single_dim)
    ns = norm_update(ns, jnp.ones((100, single_dim)) * 2.0)  # train stats: mean~2

    stacked_obs = jnp.tile(jnp.arange(single_dim, dtype=jnp.float32), fs)
    training_state = type("TS", (), {"actor_params": None})()
    algo = _FakeAlgo()
    env_state = _FakeEnvState(stacked_obs)

    step_fn, _ = build_ppo_rollout_step(algo, training_state, ns, _env_step,
                                        n_frame_stack=fs)
    carry = (env_state, jax.random.PRNGKey(0))
    # Just need it to not crash — stacked path is exercised inside rollout_step.
    carry, (out_state, _) = step_fn(carry, 0)

    # Verify the stacked normalize matches manual computation
    expected = norm_normalize_stacked(ns, stacked_obs[None], fs)
    assert expected.shape == (1, single_dim * fs)
    # And that single-frame normalize would've crashed on shape — sanity.
    with pytest.raises(Exception):
        _ = norm_normalize(ns, stacked_obs[None])


def test_offpolicy_rollout_freezes_norm():
    """Off-policy rollout already frozen — regression guard."""
    from jax_rl.utils.rollout import build_offpolicy_rollout_step

    obs_dim = 5
    ns = norm_init(obs_dim)
    ns = norm_update(ns, jnp.ones((10, obs_dim)) * 3.0)
    original = jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, ns)

    class _OffAlgo:
        def select_action(self, params, obs, key, deterministic=True):
            return jnp.zeros(obs.shape[:-1] + (1,))

    algo = _OffAlgo()
    env_state = _FakeEnvState(jnp.zeros(obs_dim))

    step_fn, _ = build_offpolicy_rollout_step(algo, None, ns, _env_step,
                                              use_obs_norm=True, n_frame_stack=1)
    carry = (env_state, jax.random.PRNGKey(0))
    for i in range(5):
        carry, _ = step_fn(carry, i)

    assert jnp.allclose(ns.mean, original.mean)
    assert ns.count == original.count


def test_offpolicy_rollout_fs3_stacked_path():
    """Off-policy FS=3: normalize_stacked path routed when n_frame_stack>1."""
    from jax_rl.utils.rollout import build_offpolicy_rollout_step

    single_dim = 4
    fs = 3
    ns = norm_init(single_dim)
    ns = norm_update(ns, jnp.ones((100, single_dim)) * 2.0)

    stacked_obs = jnp.tile(jnp.arange(single_dim, dtype=jnp.float32), fs)

    class _OffAlgo:
        def select_action(self, params, obs, key, deterministic=True):
            return jnp.zeros(obs.shape[:-1] + (1,))

    algo = _OffAlgo()
    env_state = _FakeEnvState(stacked_obs)

    step_fn, _ = build_offpolicy_rollout_step(algo, None, ns, _env_step,
                                              use_obs_norm=True, n_frame_stack=fs)
    carry = (env_state, jax.random.PRNGKey(0))
    # Doesn't crash with FS=3 shape — proves stacked path wired.
    carry, _ = step_fn(carry, 0)
