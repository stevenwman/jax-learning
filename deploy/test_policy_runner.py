"""Test PolicyRunner loads checkpoints and produces valid actions."""
import numpy as np
import os
import sys


def test_policy_runner_loads_and_infers():
    """PolicyRunner should accept obs and produce action in [-1, 1]."""
    from deploy.policy_runner import PolicyRunner

    # Find any checkpoint with actor_params.npy
    ckpt_dir = None
    if os.path.isdir("checkpoints"):
        for d in sorted(os.listdir("checkpoints")):
            best = os.path.join("checkpoints", d, "best")
            if os.path.isdir(best) and os.path.exists(os.path.join(best, "actor_params.npy")):
                ckpt_dir = best
                break

    if ckpt_dir is None:
        print("SKIP: No checkpoint found in checkpoints/")
        return

    runner = PolicyRunner(ckpt_dir)
    print(f"Loaded: algo={runner.algo}, obs_dim={runner.obs_dim}, action_dim={runner.action_dim}")
    print(f"  hidden_dim={runner.hidden_dim}, activation={runner.activation}, squash={runner.squash}")
    print(f"  obs_norm={'yes' if runner.use_obs_norm else 'no'} (count={runner.norm_count})")
    print(f"  encoder_layers={len(runner.encoder_layers)}, layer_norm={runner.has_layer_norm}")

    # Run inference with zeros
    obs = np.zeros(runner.obs_dim, dtype=np.float32)
    action = runner.get_action(obs)

    assert action.shape == (runner.action_dim,), f"Expected ({runner.action_dim},), got {action.shape}"
    assert np.all(np.abs(action) <= 1.0 + 1e-6), f"Actions outside [-1,1]: min={action.min()}, max={action.max()}"
    assert not np.any(np.isnan(action)), "NaN in actions"
    print(f"  zero_obs action: [{action.min():.4f}, {action.max():.4f}]")

    # Run inference with random obs
    obs_rand = np.random.randn(runner.obs_dim).astype(np.float32)
    action_rand = runner.get_action(obs_rand)
    assert action_rand.shape == (runner.action_dim,)
    assert np.all(np.abs(action_rand) <= 1.0 + 1e-6)
    assert not np.any(np.isnan(action_rand))
    print(f"  rand_obs action: [{action_rand.min():.4f}, {action_rand.max():.4f}]")

    # Actions should differ for different inputs
    assert not np.allclose(action, action_rand), "Same action for different obs — network may not be loaded correctly"

    print("PASS")


def test_obs_builder_shape_and_layout():
    """ObsBuilder should produce 48d obs with correct layout."""
    from deploy.obs_builder import ObsBuilder
    from deploy.go2_constants import NUM_JOINTS

    builder = ObsBuilder()

    joint_pos_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
    joint_vel_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
    gyro = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity
    command = np.array([0.5, 0.0, 0.1], dtype=np.float32)

    obs = builder.build(
        joint_pos_sdk=joint_pos_sdk,
        joint_vel_sdk=joint_vel_sdk,
        gyroscope=gyro,
        quaternion=quat,
        command=command,
    )

    assert obs.shape == (48,), f"Expected (48,), got {obs.shape}"
    assert not np.any(np.isnan(obs)), "NaN in obs"

    # Verify linvel zeroed (dims 0:3)
    np.testing.assert_array_equal(obs[0:3], [0, 0, 0])

    # Verify gyro (dims 3:6)
    np.testing.assert_array_almost_equal(obs[3:6], [0.1, 0.2, 0.3])

    # Verify projected gravity for identity quaternion = [0, 0, -1]
    np.testing.assert_array_almost_equal(obs[6:9], [0.0, 0.0, -1.0], decimal=5)

    # Verify command at end (dims 45:48)
    np.testing.assert_array_almost_equal(obs[45:48], [0.5, 0.0, 0.1])

    # Verify joint offsets: pos=0 minus default=[0, 0.9, -1.8]*4
    expected_offsets = np.array([0, -0.9, 1.8] * 4, dtype=np.float32)
    np.testing.assert_array_almost_equal(obs[9:21], expected_offsets)

    print("test_obs_builder_shape_and_layout PASS")


def test_obs_builder_gravity_tilted():
    """Projected gravity should change when robot is tilted."""
    from deploy.obs_builder import ObsBuilder
    from deploy.go2_constants import NUM_JOINTS

    builder = ObsBuilder()
    zeros12 = np.zeros(NUM_JOINTS, dtype=np.float32)
    zeros3 = np.zeros(3, dtype=np.float32)
    cmd = np.zeros(3, dtype=np.float32)

    # 90-degree pitch forward: quat = [cos(45), 0, sin(45), 0] = [0.707, 0, 0.707, 0]
    angle = np.pi / 2
    quat_pitched = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0], dtype=np.float32)

    obs = builder.build(zeros12, zeros12, zeros3, quat_pitched, cmd)
    proj_grav = obs[6:9]

    # When pitched 90 deg forward, gravity in body frame should be roughly [1, 0, 0] or [-1, 0, 0]
    # (depending on convention — the z component should be ~0)
    assert abs(proj_grav[2]) < 0.1, f"Expected near-zero z gravity when pitched 90deg, got {proj_grav}"
    assert np.linalg.norm(proj_grav) > 0.9, f"Gravity magnitude should be ~1, got {np.linalg.norm(proj_grav)}"

    print(f"test_obs_builder_gravity_tilted PASS (proj_gravity={proj_grav})")


if __name__ == "__main__":
    test_policy_runner_loads_and_infers()
    test_obs_builder_shape_and_layout()
    test_obs_builder_gravity_tilted()
