"""Test script to verify PPO implementation."""

import jax
import jax.numpy as jnp
from flax import nnx

from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.algos import PPO
from jax_rl.buffers import RolloutBatch


def test_ppo_initialization():
    """Test that PPO initializes correctly."""
    print("Testing PPO initialization...")

    # Setup
    obs_dim = 17
    action_dim = 6
    batch_size = 64
    num_envs = 4
    num_steps = 32

    # Create config
    encoder_config = EncoderConfig(
        obs_dim=obs_dim,
        hidden_dim=(256, 256),
    )

    policy_config = PolicyHeadConfig(
        action_dim=action_dim,
        squash=True,  # Test with squashing to debug the real issue
    )

    config = PPOConfig(
        encoder=encoder_config,
        policy_head=policy_config,
        num_envs=num_envs,
        num_steps=num_steps,
    )

    # Initialize PPO
    rngs = nnx.Rngs(0)
    ppo = PPO(config, obs_dim, action_dim, rngs)

    print(f"✓ PPO initialized successfully")
    print(f"  - Actor encoder: {encoder_config.hidden_dim}")
    print(f"  - Critic encoder: {encoder_config.hidden_dim}")
    print(f"  - Action dim: {action_dim}")

    return ppo, config


def test_action_selection(ppo, num_envs=4, obs_dim=17):
    """Test action selection."""
    print("\nTesting action selection...")

    # Create dummy observation
    key = jax.random.PRNGKey(42)
    obs = jax.random.normal(key, (num_envs, obs_dim))

    # Select actions
    key, subkey = jax.random.split(key)
    action, log_prob, value = ppo.select_action(obs, subkey, deterministic=False)

    print(f"✓ Action selection successful")
    print(f"  - Action shape: {action.shape}")
    print(f"  - Log prob shape: {log_prob.shape}")
    print(f"  - Value shape: {value.shape}")
    print(f"  - Action range: [{action.min():.3f}, {action.max():.3f}]")

    return action, log_prob, value


def test_ppo_update(ppo: PPO, config):
    """Test PPO update."""
    print("\nTesting PPO update...")

    # Create dummy rollout batch
    num_steps = config.num_steps
    num_envs = config.num_envs
    obs_dim = config.encoder.obs_dim
    action_dim = config.policy_head.action_dim

    key = jax.random.PRNGKey(123)
    key, obs_key, reward_key, adv_key, ret_key, update_key = jax.random.split(key, 6)
    keys = jax.random.split(key, num_steps)
    obs = jax.random.normal(obs_key, (num_steps, num_envs, obs_dim))

    # Sample stochastically from policy
    batched_action_select = jax.vmap(ppo.select_action, in_axes=(0, 0, None))
    actions, log_probs, values = batched_action_select(obs, keys, False)

    # Generate random rollout data
    batch = RolloutBatch(
        obs=obs,
        actions=actions,
        rewards=jax.random.normal(reward_key, (num_steps, num_envs)),
        dones=jnp.zeros((num_steps, num_envs)),
        log_probs=log_probs,
        values=values,
        advantages=jax.random.normal(adv_key, (num_steps, num_envs)),
        returns=jax.random.normal(ret_key, (num_steps, num_envs)),
    )

    # Run update
    metrics = ppo.update(batch, update_key)

    print(f"✓ PPO update successful")
    print(f"  - Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  - Value loss: {metrics['value_loss']:.4f}")
    print(f"  - Entropy: {metrics['entropy']:.4f}")
    print(f"  - Approx KL: {metrics['approx_kl']:.4f}")
    print(f"  - Clip fraction: {metrics['clip_fraction']:.4f}")

    return metrics


def test_buffer_and_gae(config):
    """Test rollout buffer and GAE computation."""
    print("\nTesting RolloutBuffer and GAE...")

    from jax_rl.buffers import RolloutBuffer

    num_steps = config.num_steps
    num_envs = config.num_envs
    obs_dim = config.encoder.obs_dim
    action_dim = config.policy_head.action_dim

    # Create buffer
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, action_dim)

    # Fill buffer with dummy data
    key = jax.random.PRNGKey(456)
    for step in range(num_steps):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, 5)
        buffer.add(
            obs=jax.random.normal(keys[0], (num_envs, obs_dim)),
            action=jax.random.normal(keys[1], (num_envs, action_dim)),
            reward=jax.random.normal(keys[2], (num_envs,)),
            done=jnp.zeros((num_envs,)),
            log_prob=jax.random.normal(keys[3], (num_envs,)),
            value=jax.random.normal(keys[4], (num_envs,)),
        )

    # Compute advantages
    key, subkey = jax.random.split(key)
    next_value = jax.random.normal(subkey, (num_envs,))
    batch = buffer.get(next_value, gamma=config.gamma, gae_lambda=config.gae_lambda)

    print(f"✓ Buffer and GAE computation successful")
    print(f"  - Advantages shape: {batch.advantages.shape}")
    print(f"  - Returns shape: {batch.returns.shape}")
    print(f"  - Advantage mean: {batch.advantages.mean():.4f}")
    print(f"  - Advantage std: {batch.advantages.std():.4f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PPO Implementation Test Suite")
    print("=" * 60)

    # Test 1: Initialization
    ppo, config = test_ppo_initialization()

    # Test 2: Action selection
    action, log_prob, value = test_action_selection(ppo)

    # Test 3: Buffer and GAE
    test_buffer_and_gae(config)

    # Test 4: Update
    metrics = test_ppo_update(ppo, config)

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
