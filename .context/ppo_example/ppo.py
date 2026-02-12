"""Proximal Policy Optimization (PPO) implementation."""

from dataclasses import dataclass, field
from typing import Callable
import jax
import jax.numpy as jnp
import optax
import distrax
from flax import nnx

from configs import EncoderConfig
from encoder import MLPEncoder
from heads import GaussianHead, ValueHead
from policy import Policy
from buffers import RolloutBuffer


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # Environment
    num_envs: int = 4096
    num_steps: int = 32  # Steps per rollout before update
    
    # Learning
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    num_epochs: int = 4  # Epochs per PPO update
    batch_size: int = 2048
    
    # PPO-specific
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Value clipping (optional, can help stability)
    clip_value: bool = True
    value_clip_eps: float = 0.2
    
    # Advantage normalization
    normalize_advantage: bool = True
    
    # Network
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        obs_dim=17,  # Will be overridden
        hidden_dim=(256, 256),
    ))
    

class PPO:
    """PPO algorithm with actor-critic."""
    
    def __init__(
        self,
        config: PPOConfig,
        obs_dim: int,
        action_dim: int,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Update encoder config with actual obs_dim
        config.encoder.obs_dim = obs_dim
        
        # Build networks
        self.encoder = MLPEncoder(config.encoder, rngs=rngs)
        feature_dim = config.encoder.hidden_dim[-1]
        
        self.policy_head = GaussianHead(feature_dim, action_dim, rngs=rngs)
        self.value_head = ValueHead(feature_dim, rngs=rngs)
        
        # Compose policy (encoder + head)
        self.policy = Policy(self.encoder, self.policy_head, squash=False)
        
        # Create a simple container for all modules
        class ActorCritic(nnx.Module):
            def __init__(self, encoder, policy_head, value_head):
                self.encoder = encoder
                self.policy_head = policy_head
                self.value_head = value_head
        
        self.ac = ActorCritic(self.encoder, self.policy_head, self.value_head)
        
        # Optimizer
        self.optimizer = nnx.Optimizer(
            self.ac,
            optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr),
            ),
            wrt=nnx.Param,
        )
        
        # Buffer
        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_envs=config.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    
    def get_value(self, obs: jax.Array) -> jax.Array:
        """Get value estimate for observations."""
        features = self.encoder(obs)
        return self.value_head(features).squeeze(-1)
    
    def get_action_and_value(
        self, 
        obs: jax.Array, 
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample action and get value + log_prob.
        
        Returns:
            action: (num_envs, action_dim)
            log_prob: (num_envs,)
            value: (num_envs,)
        """
        action, log_prob = self.policy.sample(obs, key)
        value = self.get_value(obs)
        return action, log_prob, value
    
    def update(
        self, 
        next_value: jax.Array,
        key: jax.Array,
    ) -> dict:
        """Run PPO update on collected rollout.
        
        Args:
            next_value: V(s_T) for GAE bootstrap
            key: PRNG key
        
        Returns:
            metrics: Dict of logged values (averaged over epochs/batches)
        """
        # Compute advantages and returns
        advantages, returns = self.buffer.compute_gae(
            next_value, 
            gamma=self.config.gamma, 
            lam=self.config.gae_lambda,
        )
        
        # Normalize advantages
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Loss function that takes the actor-critic container
        def loss_fn(ac, batch):
            obs = batch["obs"]
            actions = batch["actions"]
            old_log_probs = batch["log_probs"]
            old_values = batch["values"]
            advs = batch["advantages"]
            rets = batch["returns"]
            
            # Forward pass
            features = ac.encoder(obs)
            mean, log_std = ac.policy_head(features)
            std = jnp.exp(log_std)
            new_values = ac.value_head(features).squeeze(-1)
            
            # Create distribution
            dist = distrax.Independent(
                distrax.Normal(mean, std), 
                reinterpreted_batch_ndims=1
            )
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Policy loss: clipped surrogate
            ratio = jnp.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advs
            surr2 = jnp.clip(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advs
            policy_loss = -jnp.minimum(surr1, surr2).mean()
            
            # Value loss
            if self.config.clip_value:
                value_clipped = old_values + jnp.clip(
                    new_values - old_values, 
                    -self.config.value_clip_eps, 
                    self.config.value_clip_eps
                )
                value_loss1 = (new_values - rets) ** 2
                value_loss2 = (value_clipped - rets) ** 2
                value_loss = 0.5 * jnp.maximum(value_loss1, value_loss2).mean()
            else:
                value_loss = 0.5 * ((new_values - rets) ** 2).mean()
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = (
                policy_loss 
                + self.config.value_coef * value_loss 
                + self.config.entropy_coef * entropy_loss
            )
            
            metrics = {
                "loss/policy": policy_loss,
                "loss/value": value_loss,
                "loss/entropy": -entropy_loss,
                "loss/total": total_loss,
                "policy/entropy": entropy.mean(),
                "policy/std": std.mean(),
                "policy/ratio": ratio.mean(),
                "policy/clip_frac": ((ratio < 1 - self.config.clip_eps) | (ratio > 1 + self.config.clip_eps)).mean(),
                "policy/approx_kl": (0.5 * (old_log_probs - new_log_probs) ** 2).mean(),
            }
            
            return total_loss, metrics
        
        # Multiple epochs over the data
        all_metrics = []
        
        for epoch in range(self.config.num_epochs):
            key, subkey = jax.random.split(key)
            batches = self.buffer.get_batches(
                advantages, returns, 
                batch_size=self.config.batch_size, 
                key=subkey,
            )
            
            for batch in batches:
                # Compute gradients w.r.t. ac container (all Params by default)
                grads, metrics = nnx.grad(loss_fn, has_aux=True)(
                    self.ac, batch
                )
                
                # Apply updates (NNX 0.11+ requires both model and grads)
                self.optimizer.update(self.ac, grads)
                
                all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for k in all_metrics[0].keys():
            avg_metrics[k] = jnp.mean(jnp.array([m[k] for m in all_metrics]))
        
        return avg_metrics


# Quick test
if __name__ == "__main__":
    config = PPOConfig(
        num_envs=4,
        num_steps=32,
        batch_size=16,
    )
    
    ppo = PPO(
        config=config,
        obs_dim=17,
        action_dim=6,
        rngs=nnx.Rngs(0),
    )
    
    # Test get_action_and_value
    obs = jnp.ones((4, 17))
    action, log_prob, value = ppo.get_action_and_value(obs, jax.random.PRNGKey(0))
    print(f"action shape: {action.shape}")      # (4, 6)
    print(f"log_prob shape: {log_prob.shape}")  # (4,)
    print(f"value shape: {value.shape}")        # (4,)
    
    # Fill buffer with dummy data
    for t in range(32):
        ppo.buffer.add(
            obs=jnp.ones((4, 17)),
            action=jnp.zeros((4, 6)),
            reward=jnp.ones(4) * 0.1,
            done=jnp.zeros(4),
            log_prob=jnp.zeros(4),
            value=jnp.ones(4),
        )
    
    # Test update
    next_value = ppo.get_value(obs)
    metrics = ppo.update(next_value, jax.random.PRNGKey(1))
    print(f"\nMetrics after update:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
