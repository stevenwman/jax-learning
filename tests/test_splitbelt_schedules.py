"""Hermetic tests for splitbelt schedule samplers (S§10.1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.locomotion import splitbelt_schedules as sched


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


def test_tied_shape_and_value(rng):
    table = sched.tied(rng, T=100, v=0.7)
    assert table.shape == (100, 2)
    assert jnp.allclose(table, 0.7)


def test_split_constant(rng):
    table = sched.split_constant(rng, T=50, vL=0.5, vR=1.0)
    assert table.shape == (50, 2)
    assert jnp.allclose(table[:, 0], 0.5)
    assert jnp.allclose(table[:, 1], 1.0)


def test_tied_split_tied_phase_values(rng):
    T, t1, t2 = 1000, 200, 600
    table = sched.tied_split_tied(
        rng, T=T,
        v_warm=0.5, vL_split=0.5, vR_split=1.0, t1=t1, t2=t2,
    )
    assert table.shape == (T, 2)
    assert jnp.allclose(table[:t1], 0.5)
    assert jnp.allclose(table[t1:t1+t2, 0], 0.5)
    assert jnp.allclose(table[t1:t1+t2, 1], 1.0)
    assert jnp.allclose(table[t1+t2:], 0.5)


def test_tied_split_tied_phase_boundaries(rng):
    T, t1, t2 = 100, 30, 40
    table = sched.tied_split_tied(
        rng, T=T,
        v_warm=0.5, vL_split=0.3, vR_split=0.9, t1=t1, t2=t2,
    )
    assert table[t1 - 1, 0] == table[t1 - 1, 1]
    assert table[t1, 0] != table[t1, 1]
    assert table[t1 + t2 - 1, 0] != table[t1 + t2 - 1, 1]
    assert table[t1 + t2, 0] == table[t1 + t2, 1]


def test_random_per_episode_range(rng):
    table = sched.random_per_episode(
        rng, T=10, v_range=(0.3, 1.5), ratio_range=(0.5, 2.0),
    )
    assert table.shape == (10, 2)
    assert jnp.allclose(table[0], table[-1])
    vL, vR = table[0]
    assert 0.3 <= vL <= 1.5
    assert 0.3 * 0.5 <= vR <= 1.5 * 2.0


def test_random_per_episode_determinism(rng):
    a = sched.random_per_episode(rng, T=20)
    b = sched.random_per_episode(rng, T=20)
    assert jnp.allclose(a, b)
    different = sched.random_per_episode(jax.random.PRNGKey(1), T=20)
    assert not jnp.allclose(a, different)


def test_continual_phase_warmup_is_tied(rng):
    table = sched.continual_phase(
        rng, T=50, phase_id=0, v_warm=0.5, vL_split=0.5, vR_split=1.0,
    )
    assert jnp.allclose(table, 0.5)


def test_continual_phase_split_is_split(rng):
    table = sched.continual_phase(
        rng, T=50, phase_id=1, v_warm=0.5, vL_split=0.5, vR_split=1.0,
    )
    assert jnp.allclose(table[:, 0], 0.5)
    assert jnp.allclose(table[:, 1], 1.0)


def test_sample_schedule_dispatch(rng):
    table = sched.sample_schedule(rng, T=50, kind="tied", params={"v": 0.4})
    assert jnp.allclose(table, 0.4)
    table2 = sched.sample_schedule(
        rng, T=20, kind="split_constant", params={"vL": 0.3, "vR": 0.6}
    )
    assert jnp.allclose(table2[:, 0], 0.3)


def test_sample_schedule_unknown_kind(rng):
    with pytest.raises(ValueError, match="unknown schedule kind"):
        sched.sample_schedule(rng, T=10, kind="not_a_real_kind", params={})
