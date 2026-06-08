"""Task 2.1 — hermetic CPU test: site Jacobian matches MuJoCo's mj_jacSite.

Verifies `compute_site_jacobian` returns (6, nv) with rows 0:3 linear, 3:6
angular, both in world frame — matching MuJoCo's reference implementation.
"""
from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np
import pytest


@pytest.fixture(scope="module")
def panda_cpu():
    xml_path = (
        Path(__file__).parents[3]
        / "jax_rl" / "envs" / "manipulation" / "factory"
        / "assets" / "franka_panda" / "panda.xml"
    )
    m = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    d = mujoco.MjData(m)
    # Choose a non-degenerate config inside joint limits.
    d.qpos[:7] = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5])
    d.qvel[:7] = np.array([0.1, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0])
    mujoco.mj_forward(m, d)
    return m, d


def test_jacobian_shape_and_frame(panda_cpu):
    """Shape (6, nv), block structure [linear; angular]."""
    from mujoco import mjx
    from jax_rl.envs.manipulation.factory.controller.osc import (
        compute_site_jacobian,
    )
    m, d = panda_cpu
    site_id = m.site("fingertip_centered").id
    mjx_model = mjx.put_model(m)
    mjx_data = mjx.put_data(m, d)
    J = compute_site_jacobian(mjx_model, mjx_data, site_id)
    assert J.shape == (6, m.nv)


def test_jacobian_linear_matches_mj_jacSite(panda_cpu):
    """Row block 0:3 matches mj_jacSite linear part."""
    from mujoco import mjx
    from jax_rl.envs.manipulation.factory.controller.osc import (
        compute_site_jacobian,
    )
    m, d = panda_cpu
    site_id = m.site("fingertip_centered").id

    # MuJoCo reference Jacobian
    jacp_ref = np.zeros((3, m.nv))
    jacr_ref = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp_ref, jacr_ref, site_id)

    mjx_model = mjx.put_model(m)
    mjx_data = mjx.put_data(m, d)
    J = np.asarray(compute_site_jacobian(mjx_model, mjx_data, site_id))

    np.testing.assert_allclose(J[:3], jacp_ref, atol=1e-6)
    np.testing.assert_allclose(J[3:], jacr_ref, atol=1e-6)


def test_jacobian_times_qvel_matches_site_velocity(panda_cpu):
    """J @ qvel reproduces the analytic site velocity from mj_objectVelocity.

    mj_objectVelocity with flg_local=0 returns world-frame (angvel, linvel).
    """
    from mujoco import mjx
    from jax_rl.envs.manipulation.factory.controller.osc import (
        compute_site_jacobian,
    )
    m, d = panda_cpu
    site_id = m.site("fingertip_centered").id

    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, site_id, vel6, 0)
    # mj_objectVelocity returns [angvel(3), linvel(3)] in world frame (flg_local=0).
    expected = np.concatenate([vel6[3:6], vel6[0:3]])  # [linvel, angvel]

    mjx_model = mjx.put_model(m)
    mjx_data = mjx.put_data(m, d)
    J = compute_site_jacobian(mjx_model, mjx_data, site_id)
    computed = np.asarray(J @ jp.asarray(d.qvel))

    np.testing.assert_allclose(computed, expected, atol=1e-5)
