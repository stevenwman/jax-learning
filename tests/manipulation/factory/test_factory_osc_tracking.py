"""Task 2.4 — Warp OSC tracking sanity test.

End-to-end: actuator_mode='motor' + OSC controller wired into env.step.

NOTE on action_chain clipping: env.step clips OSC target to fixed_pos ± 5cm
each step (clip_to_bounds with anchor=hole, bounds=0.05). Initial fingertip
sits at ~(0.5, 0, 0.45) but the hole is at (0.6, 0, 0.05), so even action=0
drives the target down toward the hole. Tests below account for this — they
check directional response (action=+x lands EE farther +x than action=-x)
rather than "zero action holds initial pose."
"""
import jax
import jax.numpy as jp
import numpy as np
import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.warp]


@pytest.fixture(scope="module")
def env_osc():
    from jax_rl.envs.manipulation.factory.factory_peg_insert import (
        FactoryPegInsert,
        default_config,
    )
    cfg = default_config()
    cfg.actuator_mode = "motor"
    return FactoryPegInsert(cfg)


def _rollout(env, action, n_outer=60):
    state = env.reset(jax.random.PRNGKey(0))
    fingertip = [np.asarray(state.data.site_xpos[env._fingertip_site_id])]
    for _ in range(n_outer):
        state = env.step(state, action)
        fingertip.append(np.asarray(state.data.site_xpos[env._fingertip_site_id]))
    return np.stack(fingertip)


def test_no_nan_with_osc_motor(env_osc):
    """OSC + motor + gravity comp must not produce NaN over 60 outer steps."""
    traj = _rollout(env_osc, jp.zeros(6), n_outer=60)
    assert np.all(np.isfinite(traj)), f"NaN in fingertip trajectory: {traj}"


def test_pos_y_action_drives_fingertip_more_plus_y_than_neg_y(env_osc):
    """+y action lands EE farther +y than -y action (differential tracking).

    Why not test x or z too? Initial fingertip is at x≈0.5, z≈0.45 but the
    target is clipped to hole_pos ± 5cm = ([0.55, 0.65], [-0.05, 0.05],
    [0, 0.1]). Initial x and z are OUTSIDE those bounds, so every per-step
    target proposal in x/z clips to the lower bound regardless of action
    sign — annihilating the differential. y is the only axis where init
    (0.0) is centered on hole_y (0.0) and lives inside the bounds, so the
    differential survives clipping. (Phase 3 fixes this by starting the
    arm near the bolt rather than at DEFAULT_ARM_QPOS.)
    """
    traj_plus = _rollout(env_osc, jp.array([0., 1., 0., 0., 0., 0.]), n_outer=60)
    traj_neg  = _rollout(env_osc, jp.array([0., -1., 0., 0., 0., 0.]), n_outer=60)
    dy = traj_plus[-1, 1] - traj_neg[-1, 1]
    assert dy > 0.05, (
        f"+y must end farther +y than -y by ≥5cm; got dy={dy:.4f}m"
    )


def test_zero_action_converges_to_hole_xy(env_osc):
    """action=0 → OSC tracks clipped target (hole xy ± 0 = hole xy).

    With pos_delta=0 the per-step target = current_fingertip_pos, then
    clip_to_bounds caps it within hole ± 5cm. After enough steps the EE
    settles inside the hole's xy bounding box. Loose tolerance (10cm) — we
    just want to confirm the controller is non-divergent.
    """
    traj = _rollout(env_osc, jp.zeros(6), n_outer=60)
    hole_xy = np.array([0.6, 0.0])
    end_xy = traj[-1, :2]
    xy_err = np.linalg.norm(end_xy - hole_xy)
    assert xy_err < 0.10, (
        f"fingertip xy must end within 10cm of hole; got xy_err={xy_err:.4f}m, "
        f"end={end_xy}"
    )
