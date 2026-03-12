"""Proximal Policy Optimization (PPO) algorithm."""

from typing import Any
import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.configs import PPOConfig
from jax_rl.networks.builders import Actor, Critic
from jax_rl.buffers import RolloutBatch
from jax_rl.networks.distributions import entropy_gaussian, gaussian_log_prob, sample_gaussian


@flax.struct.dataclass
class TrainingState:
    actor_params: Any
    critic_params: Any
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState


class PPO:
    """Proximal Policy Optimization with separate actor and critic.

    Implements PPO with:
    - Clipped surrogate objective
    - Separate actor/critic networks and optimizers (no value_coef)
    - Optional value clipping
    - Entropy bonus
    - Gradient clipping
    - Optional adaptive learning rate based on KL divergence
    """

    def __init__(
        self,
        config: PPOConfig,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        """Initialize PPO.

        Args:
            config: PPO configuration
            obs_dim: Observation dimension
            action_dim: Action dimension
            rngs: Random number generators for initialization
        """
        self.config = config

        # Update encoder config with obs/action dims
        encoder_config = config.encoder
        policy_config = config.policy_head
        encoder_config.obs_dim = obs_dim
        policy_config.action_dim = action_dim

        self.num_envs = config.num_envs
        self.obs_dim = obs_dim

        # Build actor and critic networks
        self.actor = Actor(encoder_config, policy_config)
        self.critic = Critic(encoder_config, config.value_head)
        self.actor_optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.actor_lr))
        self.critic_optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.critic_lr))

    def init(self, key: jax.Array) -> TrainingState:
        actor_key, critic_key = jax.random.split(key, 2)
        dummy_obs = jnp.zeros(self.obs_dim)

        actor_params = self.actor.init(actor_key, dummy_obs)
        critic_params = self.critic.init(critic_key, dummy_obs)
        
        return TrainingState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt_state=self.actor_optimizer.init(actor_params),
            critic_opt_state=self.critic_optimizer.init(critic_params),
        )

    def update(self, state: TrainingState, batch: RolloutBatch, key: jax.Array) -> tuple[TrainingState, dict]:
        
        # Flatten batch: (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
        obs = batch.obs.reshape(-1, batch.obs.shape[-1])
        actions = batch.actions.reshape(-1, batch.actions.shape[-1])
        old_log_probs = batch.log_probs.reshape(-1)
        advantages = batch.advantages.reshape(-1)
        returns = batch.returns.reshape(-1)

        # Track metrics across epochs
        metrics_accum = {}
        
        for epoch in range(self.config.num_epochs):
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, obs.shape[0])
            
            for start in range(0, obs.shape[0], self.config.minibatch_size):
                mb = perm[start:start+self.config.minibatch_size]
                mb_adv = advantages[mb]

                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_returns = returns[mb]
                
                if self.config.normalize_advantage:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                def value_loss_fn(critic_params):
                    values = self.critic.apply(critic_params, mb_obs)
                    value_loss = jnp.mean((values - mb_returns) ** 2)
                    return value_loss

                def actor_loss_fn(actor_params):
                    mb_mean, mb_log_std = self.actor.apply(actor_params, mb_obs)
                    mb_log_probs = gaussian_log_prob(mb_mean, mb_log_std, mb_actions, squash=self.config.policy_head.squash)
                    
                    ratio = jnp.exp(mb_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_adv
                    surr2 = jnp.clip(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * mb_adv
                    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
                    
                    entropy = entropy_gaussian(mb_log_std).mean()
                    entropy_loss = -self.config.entropy_coef * entropy
                    policy_loss += entropy_loss

                    clip_fraction = ((jnp.abs(ratio - 1) > self.config.clip_eps).astype(jnp.float32)).mean()
                    approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()

                    metrics = {
                        "policy_loss": policy_loss,
                        "entropy": entropy,
                        "approx_kl": approx_kl,
                        "clip_fraction": clip_fraction,
                    }
                    
                    return policy_loss, metrics

                (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor_params)
                actor_updates, new_actor_opt_state = self.actor_optimizer.update(actor_grads, state.actor_opt_state)
                new_actor_params = optax.apply_updates(state.actor_params, actor_updates)

                value_loss, value_grads = jax.value_and_grad(value_loss_fn)(state.critic_params)
                value_updates, new_value_opt_state = self.critic_optimizer.update(value_grads, state.critic_opt_state)
                new_critic_params = optax.apply_updates(state.critic_params, value_updates)

                state = state.replace(
                    actor_params=new_actor_params,
                    critic_params=new_critic_params,
                    actor_opt_state=new_actor_opt_state,
                    critic_opt_state=new_value_opt_state,
                )
                
                if not metrics_accum:
                    metrics_accum = {**actor_metrics, "value_loss": value_loss}
                else:
                    for k, v in actor_metrics.items():
                        metrics_accum[k] += v
                    metrics_accum["value_loss"] += value_loss
        
        num_minibatches = obs.shape[0] // self.config.minibatch_size
        num_updates = self.config.num_epochs * num_minibatches
        return state, {k: v / num_updates for k, v in metrics_accum.items()}

    def select_action(
        self,
        state: TrainingState, 
        obs: jax.Array,
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Select action from policy.

        Args:
            obs: Observation, shape (num_envs, obs_dim) or (obs_dim,)
            key: PRNGKey for sampling
            deterministic: If True, return mean action (no sampling)

        Returns:
            (action, log_prob, value) tuple
        """

        mean, log_std = self.actor.apply(state.actor_params, obs)

        if deterministic:
            action = mean
            log_prob = gaussian_log_prob(mean, log_std, action, squash=self.config.policy_head.squash)
        else:
            action, log_prob = sample_gaussian(mean, log_std, key, squash=self.config.policy_head.squash)

        value = self.critic.apply(state.critic_params, obs)
        return action, log_prob, value
