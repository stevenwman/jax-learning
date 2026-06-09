"""Unit tests for decoupled stiffness/damping in the variable-impedance OSC env.

Pure-function tests (CPU, no Go2/Warp) for `impedance_gains`, which maps a policy
action tail to per-leg Cartesian gains:

    kp = s · kp_base                 s = log map of stiffness action  ∈ [s_min, s_max]
    kd = ζ · √s · kd_base            ζ = log map of damping  action   ∈ [z_min, z_max]

Key properties:
  - kd_base is already critical damping (2√kp_base), so ζ=1 ⇒ critically damped.
  - ζ log-range [0.5, 2.0] has midpoint 1.0, so a NEUTRAL (0) damping action is
    critical — i.e. enabling damping with neutral actions reproduces the old
    locked-critical behavior exactly.
  - ζ scales kd only; kp is untouched by the damping action (true decoupling).
"""
import jax.numpy as jp
import numpy as np
import pytest

from jax_rl.envs.locomotion.go2_warp_osc_var_impedance import (
    impedance_gains, var_action_size,
)

KP = jp.array([3000.0, 3000.0, 4000.0])
KD = jp.array([110.0, 110.0, 130.0])
SMIN, SMAX = 0.25, 2.0
ZMIN, ZMAX = 0.5, 2.0
S_NEUTRAL = (SMIN * SMAX) ** 0.5     # stiffness action 0 → log-mid ≈ 0.707


def _gains(action, gran, damp):
    return impedance_gains(
        jp.array(action), KP, KD, granularity=gran,
        s_min=SMIN, s_max=SMAX, damping_action=damp, z_min=ZMIN, z_max=ZMAX)


def test_no_damping_is_critical_per_foot():
    kp, kd = _gains([0.0] * 12 + [0.0] * 4, "per_foot", False)
    np.testing.assert_allclose(kp, S_NEUTRAL * np.array(KP) * np.ones((4, 1)), rtol=1e-5)
    np.testing.assert_allclose(kd, (S_NEUTRAL ** 0.5) * np.array(KD) * np.ones((4, 1)), rtol=1e-5)


def test_neutral_damping_action_reproduces_critical():
    # damping ON but ζ-action = 0 → ζ = √(0.5·2) = 1.0 → kd identical to no-damping
    kp, kd = _gains([0.0] * 12 + [0.0] * 4 + [0.0] * 4, "per_foot", True)
    np.testing.assert_allclose(kd, (S_NEUTRAL ** 0.5) * np.array(KD) * np.ones((4, 1)), rtol=1e-5)


@pytest.mark.parametrize("z_act,zeta", [(1.0, 2.0), (-1.0, 0.5)])
def test_zeta_extremes_scale_kd_only(z_act, zeta):
    kp, kd = _gains([0.0] * 12 + [0.0] * 4 + [z_act] * 4, "per_foot", True)
    np.testing.assert_allclose(kd, zeta * (S_NEUTRAL ** 0.5) * np.array(KD) * np.ones((4, 1)), rtol=1e-5)
    # kp is unaffected by the damping action — that's the decoupling.
    np.testing.assert_allclose(kp, S_NEUTRAL * np.array(KP) * np.ones((4, 1)), rtol=1e-5)


def test_per_axis_shapes_and_decoupling():
    # per_axis: 12 stiffness + 12 damping; ζ-action +1 everywhere → ζ=2.
    kp, kd = _gains([0.0] * 12 + [0.0] * 12 + [1.0] * 12, "per_axis", True)
    assert kp.shape == (4, 3) and kd.shape == (4, 3)
    expect = 2.0 * (S_NEUTRAL ** 0.5) * np.array(KD) * np.ones((4, 1))   # (4,3)
    np.testing.assert_allclose(kd, expect, rtol=1e-5)


def test_action_size():
    assert var_action_size("per_foot", False) == 16   # current behavior
    assert var_action_size("per_foot", True) == 20     # +4 damping
    assert var_action_size("per_axis", True) == 36      # +12 damping
