import jax
import jax.numpy as jnp


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, act_dim: int) -> None:
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs = jnp.zeros((num_steps, num_envs, obs_dim))
        self.actions = jnp.zeros((num_steps, num_envs, act_dim))
        self.rewards = jnp.zeros((num_steps, num_envs))
        self.dones = jnp.zeros_like(self.rewards)
        self.log_prob = jnp.zeros_like(self.rewards)
        self.values = jnp.zeros_like(self.rewards)
        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value) -> None:
        self.obs = self.obs.at[self.ptr].set(obs)
        self.actions = self.actions.at[self.ptr].set(action)
        self.rewards = self.rewards.at[self.ptr].set(reward)
        self.dones = self.dones.at[self.ptr].set(done)
        self.log_prob = self.log_prob.at[self.ptr].set(log_prob)
        self.values = self.values.at[self.ptr].set(value)
        self.ptr += 1

    def reset(self) -> None:
        self.ptr = 0

    def compute_gae(self, next_value, gamma, lam):
        # Shift values: next_values[t] = V(s_{t+1})
        next_values = jnp.concatenate([self.values[1:], next_value[None]], axis=0)
        
        def scan_fn(gae, t):
            delta = self.rewards[t] + gamma * next_values[t] * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            return gae, gae
        
        # jax.lax.scan performs reverse-time GAE computation for all steps and environments
        _, advantages = jax.lax.scan(scan_fn, jnp.zeros(self.num_envs), jnp.arange(self.num_steps)[::-1])
        advantages = advantages[::-1]
        returns = advantages + self.values
        return advantages, returns