"""GPU/Warp integration tests for the Go2 Cartesian-impedance OSC env.

These pin the Go2-specific wiring that the synthetic-leg unit tests
(`test_go2_osc.py`) cannot: the per-leg DoF index mapping, the foot-site
ordering, and the trunk-frame nominal-foot bookkeeping.
"""
import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco import mjx

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]

from jax_rl.envs.locomotion.go2_warp_osc_joystick import WarpOscJoystick


@pytest.fixture(scope="module")
def env():
    return WarpOscJoystick(task="flat_terrain")


def test_builds_and_steps_without_nan(env):
    s = env.reset(jax.random.PRNGKey(0))
    assert env.action_size == 12
    assert s.obs["state"].shape[-1] == 48
    for a in (jp.zeros(12), 0.3 * jp.ones(12), -0.3 * jp.ones(12)):
        s = env.step(s, a)
    assert not bool(jp.isnan(s.reward))
    assert not bool(jp.any(jp.isnan(s.data.qpos)))


def test_leg_dof_block_isolation(env):
    """Each foot's linear Jacobian is nonzero ONLY on its own 3 leg DoFs.

    Validates the assumption baked into `_leg_dof_ids` that qvel[6+3i:6+3i+3]
    drives foot i. Base DoFs (0:6) legitimately move every foot, so they are
    excluded from the check.
    """
    s = env.reset(jax.random.PRNGKey(0))
    d = mjx.forward(env.mjx_model, s.data)
    leg_dofs = np.asarray(env._leg_dof_ids)          # (4,3)
    all_leg = set(range(6, 18))
    for i in range(4):
        site = int(env._osc_foot_site_ids[i])
        jacp, _ = mjx.jac(env.mjx_model, d, d.site_xpos[site],
                          env.mjx_model.site_bodyid[site])
        jacp = np.asarray(jacp)                       # (nv, 3)
        own = set(int(x) for x in leg_dofs[i])
        others = sorted(all_leg - own)
        off_block = jacp[others]                      # (9, 3) other legs' dofs
        assert np.allclose(off_block, 0.0, atol=1e-6), (
            f"foot {i}: nonzero Jacobian on other legs' dofs:\n{off_block}"
        )
        # And its own block is non-trivial (foot actually moves with its leg).
        assert np.linalg.norm(jacp[sorted(own)]) > 1e-3


def test_leg_dof_site_names_match_by_leg(env):
    """Foot site i and leg DoFs i belong to the SAME leg, checked by MJCF name.

    The block-isolation test pairs the env's own site_id[i] with leg_dofs[i],
    so it cannot catch a *consistent* site-order/dof-order swap (both reordered
    together still isolate). This pins the mapping against ground-truth names.
    """
    from jax_rl.envs.locomotion import go2_constants as consts
    m = env.mj_model
    leg_dofs = np.asarray(env._leg_dof_ids)        # (4,3) qvel indices
    for i in range(4):
        site_leg = consts.FEET_SITES[i].split("_")[0]   # e.g. "FL"
        for d in leg_dofs[i]:
            jid = int(m.dof_jntid[int(d)])
            jname = m.jnt(jid).name                      # e.g. "FL_thigh_joint"
            assert jname.startswith(site_leg + "_"), (
                f"foot {consts.FEET_SITES[i]} dof {d} drives joint {jname} "
                f"(leg mismatch)"
            )


def test_nominal_foot_body_matches_reset_pose(env):
    """Trunk-frame feet at reset == stored nominal.

    Reset keeps the leg joints at the home pose (only the floating base is
    randomized), and trunk-frame foot positions are invariant to base
    translation/rotation — so the live mjx `_feet_in_body` path must reproduce
    the numpy-FK nominal. This cross-checks the two FK paths against each other.
    """
    s = env.reset(jax.random.PRNGKey(3))
    feet_body = np.asarray(env._feet_in_body(s.data))   # (4,3) live mjx path
    nominal = np.asarray(env._nominal_foot_body)         # (4,3) numpy-FK path
    assert np.allclose(feet_body, nominal, atol=1e-5), (feet_body - nominal)
