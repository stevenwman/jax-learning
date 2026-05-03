"""Control metadata test for SplitbeltTreadmill env (lesson §7.2).

Catches drift between env XML keyframe / scale and deploy/go2_constants.py —
the same drift class that bricked the deploy stack 2026-04-10 → 2026-04-24.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2, pytest.mark.deploy]


def test_splitbelt_metadata_matches_deploy_constants():
    pytest.importorskip("mujoco")
    pytest.importorskip("warp")

    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    from deploy.go2_constants import (
        DEFAULT_POSE_POLICY, DEFAULT_POSE_SDK, POLICY_TO_SDK, SDK_TO_POLICY,
        ACTION_SCALE,
    )

    env = Go2WarpSplitbeltEnv()
    m = env.get_control_metadata()

    np.testing.assert_array_almost_equal(
        m["default_pose_policy"], DEFAULT_POSE_POLICY,
        err_msg="splitbelt_spawn keyframe ≠ deploy/go2_constants.DEFAULT_POSE_POLICY",
    )
    np.testing.assert_array_almost_equal(
        m["default_pose_sdk"], DEFAULT_POSE_SDK,
        err_msg="env-derived default_pose_sdk drifted from constants",
    )
    np.testing.assert_array_equal(m["policy_to_sdk"], POLICY_TO_SDK)
    np.testing.assert_array_equal(m["sdk_to_policy"], SDK_TO_POLICY)
    assert np.isclose(m["action_scale"], ACTION_SCALE), (
        f"action_scale drift: env={m['action_scale']} vs constants={ACTION_SCALE}"
    )
