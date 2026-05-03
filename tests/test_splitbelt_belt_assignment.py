"""Hermetic test for foot-to-belt id mapping (S§5 risk callout)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jax_rl.envs.locomotion import splitbelt_geom as geom


@pytest.fixture
def belt_layout():
    return geom.BeltLayout(
        left_y_min=-0.30, left_y_max=-0.025,
        right_y_min=0.025, right_y_max=0.30,
    )


def test_foot_in_left_belt(belt_layout):
    foot_xy = jnp.array([[0.0, -0.15]])
    assert int(geom.foot_belt_id(foot_xy, belt_layout)[0]) == 0


def test_foot_in_right_belt(belt_layout):
    foot_xy = jnp.array([[0.0, 0.15]])
    assert int(geom.foot_belt_id(foot_xy, belt_layout)[0]) == 1


def test_foot_in_gap(belt_layout):
    foot_xy = jnp.array([[0.0, 0.0]])
    assert int(geom.foot_belt_id(foot_xy, belt_layout)[0]) == -1


def test_foot_off_belt_y(belt_layout):
    foot_xy = jnp.array([[0.0, 0.5]])
    assert int(geom.foot_belt_id(foot_xy, belt_layout)[0]) == -1


def test_vectorized_over_4_feet(belt_layout):
    foot_xy = jnp.array([
        [0.0, -0.15],   # left
        [0.0, 0.15],    # right
        [0.0, 0.0],     # gap
        [0.0, 0.5],     # off
    ])
    ids = geom.foot_belt_id(foot_xy, belt_layout)
    assert tuple(int(i) for i in ids) == (0, 1, -1, -1)
