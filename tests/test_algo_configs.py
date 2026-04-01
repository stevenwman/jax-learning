"""Tests for algorithm config compatibility and correctness.

Verifies:
- Every algo can init + run 1 update step with both Adam and AdamW
- critic_hidden_dim produces different Q network shapes than hidden_dim
- FastSAC policy_delay only updates actor every N steps
- All algos respect alpha_init config
- Gradient clipping is configurable
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.algos.sac import SAC
from jax_rl.algos.td3 import TD3
from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.algos.fast_sac import FastSAC
from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.fast_sac_config import FastSACConfig
from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.fast_td3_config import FastTD3Config


OBS_DIM = 10
ACTION_DIM = 4
KEY = jax.random.PRNGKey(42)


def _make_batch(obs_dim, action_dim, batch_size=64):
    """Create a fake batch for testing."""
    obs = jnp.ones((batch_size, obs_dim))
    next_obs = jnp.ones((batch_size, obs_dim))
    return {
        "obs": obs,
        "action": jnp.zeros((batch_size, action_dim)),
        "reward": jnp.ones((batch_size, 1)),
        "next_obs": next_obs,
        "done": jnp.zeros((batch_size, 1)),
        "truncation": jnp.zeros((batch_size, 1)),
        "critic_obs": obs,
        "critic_next_obs": next_obs,
    }


# ── Optimizer compatibility ───────────────────────────────────────────────


@pytest.mark.parametrize("opt_fn", [
    lambda lr: optax.adam(lr),
    lambda lr: optax.adamw(lr, weight_decay=0.001),
    lambda lr: optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr)),
    lambda lr: optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=0.001)),
])
def test_sac_optimizer_compat(opt_fn):
    """SAC should work with any optax optimizer."""
    cfg = SACConfig(hidden_dim=(32, 32), batch_size=64)
    opt = opt_fn(1e-3)
    sac = SAC(cfg, OBS_DIM, ACTION_DIM, opt, opt, gamma=0.99)
    state = sac.init(KEY)
    batch = _make_batch(OBS_DIM, ACTION_DIM)
    new_state, metrics = sac.update(state, batch)
    assert not jnp.isnan(metrics["q1_mean"]), "Q1 is NaN"


@pytest.mark.parametrize("opt_fn", [
    lambda lr: optax.adam(lr),
    lambda lr: optax.adamw(lr, weight_decay=0.001),
])
def test_td3_optimizer_compat(opt_fn):
    """TD3 should work with any optax optimizer."""
    cfg = TD3Config(hidden_dim=(32, 32), batch_size=64, min_buffer_size=1)
    opt = opt_fn(1e-3)
    td3 = TD3(cfg, OBS_DIM, ACTION_DIM, opt, opt, gamma=0.99)
    state = td3.init(KEY)
    batch = _make_batch(OBS_DIM, ACTION_DIM)
    new_state, metrics = td3.update(state, batch)
    assert not jnp.isnan(metrics["q1_mean"]), "Q1 is NaN"


@pytest.mark.parametrize("opt_fn", [
    lambda lr: optax.adam(lr),
    lambda lr: optax.adamw(lr, weight_decay=0.001),
])
def test_fast_td3_optimizer_compat(opt_fn):
    """FastTD3 should work with any optax optimizer."""
    cfg = FastTD3Config(hidden_dim=(32, 16), critic_hidden_dim=(48, 24),
                        batch_size=64, min_buffer_size=1)
    opt = opt_fn(1e-3)
    td3 = FastTD3(cfg, OBS_DIM, ACTION_DIM, opt, opt, gamma=0.99)
    state = td3.init(KEY)
    batch = _make_batch(OBS_DIM, ACTION_DIM)
    new_state, metrics = td3.update(state, batch)
    assert not jnp.isnan(metrics["q1_mean"]), "Q1 is NaN"


@pytest.mark.parametrize("opt_fn", [
    lambda lr: optax.adam(lr),
    lambda lr: optax.adamw(lr, weight_decay=0.001),
])
def test_fast_sac_optimizer_compat(opt_fn):
    """FastSAC should work with any optax optimizer."""
    cfg = FastSACConfig(hidden_dim=(32, 16), critic_hidden_dim=(48, 24),
                        batch_size=64, min_buffer_size=1, policy_delay=1,
                        num_atoms=11, v_min=-5.0, v_max=5.0)
    opt = opt_fn(1e-3)
    alpha_opt = opt_fn(1e-3)
    sac = FastSAC(cfg, OBS_DIM, ACTION_DIM, opt, alpha_opt, gamma=0.99)
    state = sac.init(KEY)
    batch = _make_batch(OBS_DIM, ACTION_DIM)
    new_state, metrics = sac.update(state, batch)
    assert not jnp.isnan(metrics["q1_mean"]), "Q1 is NaN"


# ── Critic hidden dim ────────────────────────────────────────────────────


def test_sac_critic_hidden_dim():
    """SAC with critic_hidden_dim should have different Q param count."""
    cfg_same = SACConfig(hidden_dim=(32, 32), critic_hidden_dim=None)
    cfg_diff = SACConfig(hidden_dim=(32, 32), critic_hidden_dim=(64, 64))

    sac_same = SAC(cfg_same, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3), gamma=0.99)
    sac_diff = SAC(cfg_diff, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3), gamma=0.99)

    state_same = sac_same.init(KEY)
    state_diff = sac_diff.init(KEY)

    q_params_same = sum(x.size for x in jax.tree.leaves(state_same.q1_params))
    q_params_diff = sum(x.size for x in jax.tree.leaves(state_diff.q1_params))

    assert q_params_diff > q_params_same, \
        f"critic_hidden_dim=(64,64) should have more Q params than (32,32): {q_params_diff} vs {q_params_same}"


def test_td3_critic_hidden_dim():
    """TD3 with critic_hidden_dim should have different Q param count."""
    cfg_same = TD3Config(hidden_dim=(32, 32), critic_hidden_dim=None)
    cfg_diff = TD3Config(hidden_dim=(32, 32), critic_hidden_dim=(64, 64))

    td3_same = TD3(cfg_same, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3), gamma=0.99)
    td3_diff = TD3(cfg_diff, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3), gamma=0.99)

    state_same = td3_same.init(KEY)
    state_diff = td3_diff.init(KEY)

    q_same = sum(x.size for x in jax.tree.leaves(state_same.q1_params))
    q_diff = sum(x.size for x in jax.tree.leaves(state_diff.q1_params))

    assert q_diff > q_same, \
        f"critic_hidden_dim=(64,64) should have more Q params: {q_diff} vs {q_same}"


# ── Policy delay ─────────────────────────────────────────────────────────


def test_fast_sac_policy_delay():
    """FastSAC with policy_delay=4 should only update actor every 4th step."""
    cfg = FastSACConfig(hidden_dim=(32, 16), critic_hidden_dim=(48, 24),
                        batch_size=64, min_buffer_size=1, policy_delay=4,
                        num_atoms=11, v_min=-5.0, v_max=5.0)
    opt = optax.adam(1e-3)
    sac = FastSAC(cfg, OBS_DIM, ACTION_DIM, opt, opt, gamma=0.99)
    state = sac.init(KEY)
    batch = _make_batch(OBS_DIM, ACTION_DIM)

    actor_losses = []
    for i in range(8):
        state, metrics = sac.update(state, batch)
        actor_losses.append(float(metrics["actor_loss"]))

    # Steps 1,2,3 should have actor_loss=0 (skipped), step 4 should be nonzero
    assert actor_losses[0] == 0.0, f"Step 1 should skip actor (got {actor_losses[0]})"
    assert actor_losses[1] == 0.0, f"Step 2 should skip actor (got {actor_losses[1]})"
    assert actor_losses[2] == 0.0, f"Step 3 should skip actor (got {actor_losses[2]})"
    assert actor_losses[3] != 0.0, f"Step 4 should update actor (got {actor_losses[3]})"
    assert actor_losses[7] != 0.0, f"Step 8 should update actor (got {actor_losses[7]})"


def test_fast_sac_no_policy_delay():
    """FastSAC with policy_delay=1 should update actor every step."""
    cfg = FastSACConfig(hidden_dim=(32, 16), batch_size=64,
                        min_buffer_size=1, policy_delay=1,
                        num_atoms=11, v_min=-5.0, v_max=5.0)
    opt = optax.adam(1e-3)
    sac = FastSAC(cfg, OBS_DIM, ACTION_DIM, opt, opt, gamma=0.99)
    state = sac.init(KEY)
    batch = _make_batch(OBS_DIM, ACTION_DIM)

    for i in range(4):
        state, metrics = sac.update(state, batch)
        assert float(metrics["actor_loss"]) != 0.0, \
            f"Step {i+1} should update actor with policy_delay=1"


# ── Alpha init ───────────────────────────────────────────────────────────


def test_sac_alpha_init_default():
    """Vanilla SAC default alpha_init=1.0."""
    cfg = SACConfig(hidden_dim=(32, 32))
    sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3), gamma=0.99)
    state = sac.init(KEY)
    alpha = float(jnp.exp(state.log_alpha))
    assert abs(alpha - 1.0) < 1e-5, f"Default alpha should be 1.0, got {alpha}"


def test_sac_alpha_init_custom():
    """Vanilla SAC respects alpha_init=0.001."""
    cfg = SACConfig(hidden_dim=(32, 32), alpha_init=0.001)
    sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3), gamma=0.99)
    state = sac.init(KEY)
    alpha = float(jnp.exp(state.log_alpha))
    assert abs(alpha - 0.001) < 1e-5, f"Alpha should be 0.001, got {alpha}"


def test_fast_sac_alpha_init():
    """FastSAC respects alpha_init from config."""
    cfg = FastSACConfig(hidden_dim=(32, 16), alpha_init=0.01, batch_size=64,
                        num_atoms=11, v_min=-5.0, v_max=5.0)
    opt = optax.adam(1e-3)
    sac = FastSAC(cfg, OBS_DIM, ACTION_DIM, opt, opt, gamma=0.99)
    state = sac.init(KEY)
    alpha = float(jnp.exp(state.log_alpha))
    assert abs(alpha - 0.01) < 1e-4, f"Alpha should be 0.01, got {alpha}"


# ── Tapered network dims ─────────────────────────────────────────────────


def test_tapered_network_dims():
    """Tapered hidden_dim (512, 256, 128) should produce 3-layer actor."""
    cfg = SACConfig(hidden_dim=(64, 32, 16))
    sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3), gamma=0.99)
    state = sac.init(KEY)

    # Count actor param layers — should have 3 Dense layers in encoder
    actor_param_count = sum(x.size for x in jax.tree.leaves(state.actor_params))

    cfg_flat = SACConfig(hidden_dim=(64, 64))
    sac_flat = SAC(cfg_flat, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3), gamma=0.99)
    state_flat = sac_flat.init(KEY)
    flat_param_count = sum(x.size for x in jax.tree.leaves(state_flat.actor_params))

    # Tapered (64,32,16) has fewer params than flat (64,64) due to shrinking layers
    assert actor_param_count != flat_param_count, \
        "Tapered and flat networks should have different param counts"


# ── Observation normalization tests ──────────────────────────────────────


from jax_rl.utils.normalization import (
    NormalizationState, init as norm_init, update as norm_update, normalize as norm_normalize,
)
from jax_rl.training.env_setup import make_identity_norm_state


def test_obs_normalization_config_exists():
    """All off-policy configs have obs_normalization field, default False."""
    for cfg_cls in [SACConfig, FastSACConfig, TD3Config, FastTD3Config]:
        cfg = cfg_cls()
        assert hasattr(cfg, "obs_normalization"), f"{cfg_cls.__name__} missing obs_normalization"
        assert cfg.obs_normalization is False, f"{cfg_cls.__name__} should default to False"
        assert hasattr(cfg, "obs_norm_eps"), f"{cfg_cls.__name__} missing obs_norm_eps"
        assert cfg.obs_norm_eps == 1e-2, f"{cfg_cls.__name__} obs_norm_eps should be 1e-2"


def test_normalize_with_large_eps():
    """Sample-time normalization with eps=1e-2 stays bounded even with near-zero variance."""
    state = NormalizationState(
        mean=jnp.zeros(4),
        mean_of_squares=jnp.full(4, 1e-16),  # var ≈ 0
        count=100,
    )
    obs = jnp.ones((32, 4))
    result = norm_normalize(state, obs, eps=1e-2)
    # With eps=1e-2: result ≈ 1.0 / 0.01 = 100 (bounded)
    # With eps=1e-8: result ≈ 1.0 / 1e-8 = 1e8 (explosion!)
    assert jnp.all(jnp.abs(result) < 200), f"Normalized values too large: {result.max()}"


def test_normalize_with_small_eps_explodes():
    """Verify that small eps causes explosion (the bug we're preventing)."""
    state = NormalizationState(
        mean=jnp.zeros(4),
        mean_of_squares=jnp.full(4, 1e-16),  # var ≈ 0
        count=100,
    )
    obs = jnp.ones((32, 4))
    result = norm_normalize(state, obs, eps=1e-8)
    # With eps=1e-8, values should be huge
    assert jnp.any(jnp.abs(result) > 1e6), "Small eps should produce large values"


def test_identity_norm_is_passthrough():
    """Identity norm state leaves obs unchanged."""
    state = make_identity_norm_state(4)
    obs = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    result = norm_normalize(state, obs)
    assert jnp.allclose(result, obs, atol=1e-6), f"Expected {obs}, got {result}"


def test_normalize_eps_parameter():
    """The eps parameter actually changes the result."""
    state = norm_init(4)
    state = norm_update(state, jnp.ones((100, 4)))  # constant obs → var=0
    obs = jnp.ones((1, 4)) * 2.0

    result_large_eps = norm_normalize(state, obs, eps=1e-2)
    result_small_eps = norm_normalize(state, obs, eps=1e-8)
    # Different eps should give different results when var≈0
    assert not jnp.allclose(result_large_eps, result_small_eps), \
        "eps parameter should affect normalization when variance is near-zero"


# ── NaN-safe env step tests ──────────────────────────────────────────────


from jax_rl.training.env_setup import _make_nan_safe_step


def test_nan_guard_catches_nan_obs():
    """NaN obs → zeroed obs, zeroed reward, done=True."""
    # Fake env state with a replace() method (like Brax State)
    import flax

    @flax.struct.dataclass
    class FakeState:
        obs: jnp.ndarray
        reward: jnp.ndarray
        done: jnp.ndarray

    def fake_step(state, action):
        # Simulate MJX producing NaN in some envs
        nan_obs = state.obs.at[1].set(jnp.nan)  # env 1 gets NaN
        nan_reward = state.reward.at[1].set(jnp.nan)
        return state.replace(obs=nan_obs, reward=nan_reward)

    safe_step = _make_nan_safe_step(fake_step)

    state = FakeState(
        obs=jnp.ones((4, 10)),
        reward=jnp.ones(4),
        done=jnp.zeros(4),
    )
    action = jnp.zeros((4, 3))

    result = safe_step(state, action)

    # Env 1 should be guarded
    assert not jnp.any(jnp.isnan(result.obs)), "NaN obs should be zeroed"
    assert jnp.allclose(result.obs[1], 0.0), "NaN env obs should be zeros"
    assert result.reward[1] == 0.0, "NaN env reward should be zero"
    assert result.done[1] == 1.0, "NaN env should be marked done"

    # Other envs should be unaffected
    assert jnp.allclose(result.obs[0], 1.0), "Clean env obs should be unchanged"
    assert result.reward[0] == 1.0, "Clean env reward should be unchanged"
    assert result.done[0] == 0.0, "Clean env done should be unchanged"


def test_nan_guard_catches_nan_action():
    """NaN actions get zeroed before env.step (verified via obs echo)."""
    import flax

    @flax.struct.dataclass
    class FakeState:
        obs: jnp.ndarray
        reward: jnp.ndarray
        done: jnp.ndarray

    def fake_step(state, action):
        # Echo the received action back as obs so we can inspect it
        padded = jnp.zeros_like(state.obs)
        padded = padded.at[:, :action.shape[-1]].set(action)
        return state.replace(obs=padded)

    safe_step = _make_nan_safe_step(fake_step)

    state = FakeState(
        obs=jnp.ones((4, 10)),
        reward=jnp.ones(4),
        done=jnp.zeros(4),
    )
    nan_action = jnp.ones((4, 3)).at[2].set(jnp.nan)

    result = safe_step(state, nan_action)

    # Env 2's action was NaN → should have been zeroed before env.step
    # The echo puts the action into obs[:, :3], so check those columns
    assert jnp.allclose(result.obs[0, :3], 1.0), "Clean action should pass through"
    assert jnp.allclose(result.obs[2, :3], 0.0), "NaN action should be zeroed"
