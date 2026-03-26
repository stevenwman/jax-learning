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


if __name__ == "__main__":
    test_policy_runner_loads_and_infers()
