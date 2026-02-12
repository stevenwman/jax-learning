"""Rollout buffer for on-policy algorithms (PPO)."""

import jax
import jax.numpy as jnp


class RolloutBuffer:
    """Fixed-size buffer for storing rollout data.
    
    Stores (num_steps, num_envs, dim) arrays for each field.
    Uses Python loop for simplicity — jitted version with lax.scan later.
    """
    
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.ptr = 0  # Current write position
        self.reset()
    
    def reset(self):
        """Allocate fresh arrays."""
        self.obs = jnp.zeros((self.num_steps, self.num_envs, self.obs_dim))
        self.actions = jnp.zeros((self.num_steps, self.num_envs, self.action_dim))
        self.rewards = jnp.zeros((self.num_steps, self.num_envs))
        self.dones = jnp.zeros((self.num_steps, self.num_envs))
        self.log_probs = jnp.zeros((self.num_steps, self.num_envs))
        self.values = jnp.zeros((self.num_steps, self.num_envs))
        self.ptr = 0
    
    def add(
        self,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        log_prob: jax.Array,
        value: jax.Array,
    ):
        """Add a single transition (all envs at once).
        
        Args:
            obs: (num_envs, obs_dim)
            action: (num_envs, action_dim)
            reward: (num_envs,)
            done: (num_envs,)
            log_prob: (num_envs,)
            value: (num_envs,)
        """
        self.obs = self.obs.at[self.ptr].set(obs)
        self.actions = self.actions.at[self.ptr].set(action)
        self.rewards = self.rewards.at[self.ptr].set(reward)
        self.dones = self.dones.at[self.ptr].set(done)
        self.log_probs = self.log_probs.at[self.ptr].set(log_prob)
        self.values = self.values.at[self.ptr].set(value)
        self.ptr += 1
    
    def compute_gae(
        self, 
        next_value: jax.Array, 
        gamma: float = 0.99, 
        lam: float = 0.95,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            next_value: V(s_T) from critic, shape (num_envs,)
            gamma: Discount factor
            lam: GAE lambda (0 = TD, 1 = MC)
        
        Returns:
            advantages: (num_steps, num_envs)
            returns: (num_steps, num_envs) — targets for value function
        """
        # Precompute next_values[t] = V(s_{t+1})
        # For t < T-1: next_values[t] = values[t+1]
        # For t = T-1: next_values[T-1] = next_value (bootstrap)
        next_values = jnp.concatenate(
            [self.values[1:], next_value[None]], 
            axis=0
        )  # (num_steps, num_envs)
        
        def scan_fn(gae, t):
            # δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = (
                self.rewards[t] 
                + gamma * next_values[t] * (1 - self.dones[t]) 
                - self.values[t]
            )
            # A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            return gae, gae
        
        # Scan backwards: t = T-1, T-2, ..., 0
        init_gae = jnp.zeros(self.num_envs)
        _, advantages = jax.lax.scan(
            scan_fn, 
            init_gae, 
            jnp.arange(self.num_steps)[::-1]
        )
        # Reverse back to forward order
        advantages = advantages[::-1]
        
        # Returns = advantages + values (target for critic)
        returns = advantages + self.values
        
        return advantages, returns
    
    def get_batches(
        self, 
        advantages: jax.Array, 
        returns: jax.Array,
        batch_size: int,
        key: jax.Array,
    ) -> tuple[dict, jax.Array]:
        """Flatten and yield random minibatches for PPO update.
        
        Args:
            advantages: (num_steps, num_envs) from compute_gae
            returns: (num_steps, num_envs) from compute_gae
            batch_size: Minibatch size
            key: PRNG key for shuffling
        
        Returns:
            List of batch dicts, each containing:
                obs: (batch_size, obs_dim)
                actions: (batch_size, action_dim)
                log_probs: (batch_size,)
                values: (batch_size,)
                advantages: (batch_size,)
                returns: (batch_size,)
        """
        total_size = self.num_steps * self.num_envs
        
        # Flatten all arrays: (num_steps, num_envs, ...) -> (total_size, ...)
        flat_obs = self.obs.reshape(total_size, self.obs_dim)
        flat_actions = self.actions.reshape(total_size, self.action_dim)
        flat_log_probs = self.log_probs.reshape(total_size)
        flat_values = self.values.reshape(total_size)
        flat_advantages = advantages.reshape(total_size)
        flat_returns = returns.reshape(total_size)
        
        # Shuffle indices
        indices = jax.random.permutation(key, total_size)
        
        # Generate batches
        batches = []
        num_batches = total_size // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batches.append({
                "obs": flat_obs[batch_indices],
                "actions": flat_actions[batch_indices],
                "log_probs": flat_log_probs[batch_indices],
                "values": flat_values[batch_indices],
                "advantages": flat_advantages[batch_indices],
                "returns": flat_returns[batch_indices],
            })
        
        return batches


# Quick test
if __name__ == "__main__":
    buf = RolloutBuffer(num_steps=128, num_envs=4, obs_dim=17, action_dim=6)
    
    # Simulate filling buffer
    for t in range(128):
        buf.add(
            obs=jnp.ones((4, 17)) * t,
            action=jnp.zeros((4, 6)),
            reward=jnp.ones(4) * 0.1,
            done=jnp.zeros(4),
            log_prob=jnp.zeros(4),
            value=jnp.ones(4) * t * 0.01,
        )
    
    # Compute GAE
    next_value = jnp.ones(4) * 1.28
    advantages, returns = buf.compute_gae(next_value, gamma=0.99, lam=0.95)
    
    print(f"advantages shape: {advantages.shape}")  # (128, 4)
    print(f"returns shape: {returns.shape}")        # (128, 4)
    print(f"advantages[0]: {advantages[0]}")        # Should be positive (discounted future rewards)
    print(f"advantages[-1]: {advantages[-1]}")      # Closer to just delta
    
    # Test batching
    batches = buf.get_batches(advantages, returns, batch_size=64, key=jax.random.PRNGKey(0))
    print(f"num batches: {len(batches)}")           # 128*4 / 64 = 8
    print(f"batch obs shape: {batches[0]['obs'].shape}")  # (64, 17)
