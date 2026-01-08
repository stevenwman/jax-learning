"""
Test script to verify shared buffer implementation
"""
import jax
import jax.numpy as jnp
from jax_rl.algos.sac.shared_buffer import SharedReplayBuffer
from jax_rl.algos.sac.sac import Transition

def test_buffer():
    capacity = 100
    obs_dim = 5
    act_dim = 2
    num_envs = 10
    
    buffer = SharedReplayBuffer(capacity=capacity, obs_dim=obs_dim, act_dim=act_dim)
    state = buffer.init()
    
    print(f"Initial state:")
    print(f"  ptr: {state.ptr}")
    print(f"  count: {state.count}")
    print(f"  capacity: {state.capacity}")
    
    # Add some transitions
    rng = jax.random.PRNGKey(42)
    for step in range(5):
        rng, key = jax.random.split(rng)
        # Create fake transitions
        obs = jax.random.normal(key, (num_envs, obs_dim))
        act = jax.random.normal(key, (num_envs, act_dim))
        rew = jax.random.normal(key, (num_envs,))
        next_obs = jax.random.normal(key, (num_envs, obs_dim))
        done = jax.random.bernoulli(key, 0.1, (num_envs,))
        
        batch = Transition(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)
        state = buffer.add(state, batch)
        
        print(f"\nAfter step {step + 1}:")
        print(f"  ptr: {state.ptr}")
        print(f"  count: {state.count}")
        print(f"  Expected count: {min((step + 1) * num_envs, capacity)}")
        
        # Sample and check
        rng, sample_key = jax.random.split(rng)
        sample = buffer.sample(state, sample_key, batch_size=5)
        print(f"  Sampled obs shape: {sample.obs.shape}")
        print(f"  Sampled obs mean: {jnp.mean(sample.obs)}")
        print(f"  Sampled obs std: {jnp.std(sample.obs)}")
        
        # Check if sampled data is non-zero (should be after first add)
        if state.count > 0:
            non_zero = jnp.any(jnp.abs(sample.obs) > 1e-6)
            print(f"  Sampled data is non-zero: {non_zero}")
            if not non_zero:
                print("  ⚠️ WARNING: Sampled data is all zeros!")

if __name__ == "__main__":
    test_buffer()

