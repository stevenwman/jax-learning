"""Hermetic tests for splitbelt offline analysis (S§9.3, S§10.1)."""

from __future__ import annotations

import numpy as np

from jax_rl.envs.locomotion import splitbelt_analysis as sba


def test_detect_step_events_simple():
    # Foot 0: touches down at t=2, lifts at t=5, touches down again at t=8, lifts at t=11.
    contact = np.zeros((15, 1), dtype=bool)
    contact[2:5, 0] = True
    contact[8:11, 0] = True
    events = sba.detect_step_events(contact)
    assert events["touchdown_steps"][0] == [2, 8]
    assert events["liftoff_steps"][0] == [5, 11]


def test_detect_step_events_starts_in_contact():
    # Already in contact at t=0; first touchdown should not register at t=0.
    contact = np.zeros((10, 1), dtype=bool)
    contact[0:3, 0] = True
    contact[6:9, 0] = True
    events = sba.detect_step_events(contact)
    # First detected touchdown is the t=6 rising edge, not t=0.
    assert events["touchdown_steps"][0] == [6]
    assert events["liftoff_steps"][0] == [3, 9]


def test_step_length_symmetric():
    # 4 feet (FL, FR, RL, RR). Stride 0.4 m, all feet step at same length.
    T = 200
    contact = np.zeros((T, 4), dtype=bool)
    foot_xy = np.zeros((T, 4, 2))
    stride_len = 0.4
    for f in range(4):
        for k in range(5):
            t_td = 20 + k * 30 + (10 if f in (1, 2) else 0)  # stagger diagonal feet
            t_lo = t_td + 15
            if t_lo < T:
                contact[t_td:t_lo, f] = True
                foot_xy[t_td, f, 0] = k * stride_len  # x at touchdown
    events = sba.detect_step_events(contact)
    sl = sba.step_lengths(events, foot_xy)
    for f in range(4):
        if len(sl[f]) > 0:
            assert all(abs(s - stride_len) < 1e-6 for s in sl[f])


def test_step_length_asymmetry_nonzero():
    # Right legs (FR=1, RR=3) walk twice as fast as left legs (FL=0, RL=2).
    T = 400
    contact = np.zeros((T, 4), dtype=bool)
    foot_xy = np.zeros((T, 4, 2))
    # Left feet: 0.3 m strides
    for f in (0, 2):
        for k in range(4):
            t_td = 20 + k * 50
            if t_td + 25 < T:
                contact[t_td:t_td + 25, f] = True
                foot_xy[t_td, f, 0] = k * 0.3
    # Right feet: 0.6 m strides (matching belt 2x faster)
    for f in (1, 3):
        for k in range(4):
            t_td = 20 + k * 50
            if t_td + 25 < T:
                contact[t_td:t_td + 25, f] = True
                foot_xy[t_td, f, 0] = k * 0.6
    events = sba.detect_step_events(contact)
    asym = sba.step_length_asymmetry(events, foot_xy)
    # Right > left → asym > 0
    assert asym > 0.2  # well outside zero
