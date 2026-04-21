"""Tests for TDMPC2Config + factory."""
import pytest

from jax_rl.configs.tdmpc2_config import TDMPC2Config, compute_discount, make_tdmpc2_config


def test_tdmpc2_config_defaults_match_spec():
    c = make_tdmpc2_config(action_dim=6, episode_length=500)
    # Bin size derived, not hardcoded
    assert (c.vmax - c.vmin) / (c.num_bins - 1) == 0.2
    # action_dim wired through
    assert c.action_dim == 6
    # Discount heuristic
    assert compute_discount(500, 5, 0.95, 0.995) == 0.99
    assert compute_discount(1000, 5, 0.95, 0.995) == 0.995
    assert compute_discount(10, 5, 0.95, 0.995) == 0.95
    # Factory wires discount onto cfg
    assert make_tdmpc2_config(action_dim=6, episode_length=500).discount == 0.99
    assert make_tdmpc2_config(action_dim=6, episode_length=1000).discount == 0.995


def test_tdmpc2_config_rejects_zero_action_dim():
    with pytest.raises(AssertionError):
        make_tdmpc2_config(action_dim=0)


def test_tdmpc2_config_is_hashable():
    """frozen=True → hashable → safe as jax.jit static_argnames arg."""
    c = make_tdmpc2_config(action_dim=6)
    hash(c)  # must not raise
