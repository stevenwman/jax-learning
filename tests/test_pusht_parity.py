"""Parity: vendored PushTEnv (reward_mode='coverage') matches upstream gym-pusht."""
import numpy as np
import pytest


def test_vendored_matches_upstream_coverage_mode():
    """Same seed → identical obs/reward/info over 100 steps."""
    import gymnasium
    import gym_pusht  # pip
    from jax_rl.envs.manipulation.pusht import PushTEnv

    env_pip = gymnasium.make("gym_pusht/PushT-v0", obs_type="state").unwrapped
    env_ours = PushTEnv(obs_type="state", reward_mode="coverage")

    obs_p, info_p = env_pip.reset(seed=42)
    obs_o, info_o = env_ours.reset(seed=42)
    assert np.allclose(obs_p, obs_o), f"reset obs differ:\n  pip={obs_p}\n  ours={obs_o}"

    rng = np.random.default_rng(0)
    for i in range(100):
        a = rng.uniform(50, 450, size=2).astype(np.float32)
        op, rp, tp, trp, ip = env_pip.step(a)
        oo, ro, to, tro, io = env_ours.step(a)
        assert np.allclose(op, oo, atol=1e-6), f"step {i} obs differ"
        assert abs(rp - ro) < 1e-6, f"step {i} reward differ pip={rp} ours={ro}"
        assert abs(ip["coverage"] - io["coverage"]) < 1e-6, f"step {i} coverage differ"
        if tp or to:
            break


def test_reward_modes_produce_different_values():
    """All 5 reward modes produce distinct rewards on same action."""
    from jax_rl.envs.manipulation.pusht import PushTEnv

    rewards = {}
    for mode in ("coverage", "sparse", "shaped", "approach", "dense"):
        env = PushTEnv(obs_type="state", reward_mode=mode)
        env.reset(seed=42)
        _, r, *_ = env.step(np.array([250.0, 250.0], dtype=np.float32))
        rewards[mode] = r

    # Sparse = 0 when not solved
    assert rewards["sparse"] == 0.0
    # Shaped ≠ coverage when there are contacts or nonzero distance
    assert rewards["shaped"] != rewards["coverage"]
    # Approach includes proximity bonus
    assert rewards["approach"] != rewards["coverage"]
    # Dense combines coverage + multiple shaping terms — always > coverage alone
    # when not solved (proximity bonus is always positive).
    assert rewards["dense"] > rewards["coverage"]


def test_dense_info_components_present():
    """Dense mode exposes per-component breakdown in info dict."""
    from jax_rl.envs.manipulation.pusht import PushTEnv

    env = PushTEnv(obs_type="state", reward_mode="dense")
    env.reset(seed=42)
    _, r, _, _, info = env.step(np.array([250.0, 250.0], dtype=np.float32))

    for key in ("r_coverage", "r_pos", "r_angle", "r_approach",
                "r_block_vel", "r_contact", "r_success",
                "block_to_goal", "angle_err", "block_vel_toward"):
        assert key in info, f"missing info[{key!r}]"

    # All shaping terms in [0, 1] or reasonable range
    assert 0.0 <= info["r_pos"] <= 1.0
    assert 0.0 <= info["r_angle"] <= 1.0
    assert 0.0 <= info["r_approach"] <= 1.0
    assert -1.0 <= info["r_block_vel"] <= 1.0


def test_unknown_reward_mode_raises():
    from jax_rl.envs.manipulation.pusht import PushTEnv
    with pytest.raises(ValueError, match="Unknown reward_mode"):
        PushTEnv(reward_mode="nonsense")
