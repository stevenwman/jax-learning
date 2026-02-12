"""Proximal Policy Optimization (PPO) algorithm."""

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple

from jax_rl.configs import PPOConfig
from jax_rl.networks.builders import Actor, Critic, build_actor_critic
from jax_rl.buffers import RolloutBuffer, RolloutBatch
from jax_rl.networks.distributions import entropy_gaussian


class PPOMetrics(NamedTuple):
    """Metrics returned from PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float


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
        rngs: nnx.Rngs,
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
        encoder_config.obs_dim = obs_dim

        policy_config = config.policy_head
        policy_config.action_dim = action_dim

        # Build actor and critic networks
        self.actor, self.critic = build_actor_critic(
            encoder_config,
            policy_config,
            config.value_head,
            rngs,
        )

        # Create separate optimizers for actor and critic
        actor_tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.actor_lr),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.critic_lr),
        )

        self.actor_optimizer = nnx.Optimizer(self.actor, actor_tx, wrt=nnx.Param)
        self.critic_optimizer = nnx.Optimizer(self.critic, critic_tx, wrt=nnx.Param)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_envs=config.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

    def update(self, batch: RolloutBatch, key: jax.Array) -> dict[str, float]:
        """Update policy and value function using PPO.

        Args:
            batch: Rollout batch with computed advantages and returns

        Returns:
            Dictionary of training metrics
        """
        # Flatten batch: (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
        obs = batch.obs.reshape(-1, batch.obs.shape[-1])
        actions = batch.actions.reshape(-1, batch.actions.shape[-1])
        old_log_probs = batch.log_probs.reshape(-1)
        advantages = batch.advantages.reshape(-1)
        returns = batch.returns.reshape(-1)

        # Track metrics across epochs
        metrics_accum = {
            "policy_loss": jnp.array([0]),
            "value_loss": jnp.array([0]),
            "entropy": jnp.array([0]),
            "approx_kl": jnp.array([0]),
            "clip_fraction": jnp.array([0]),
        }

        def epoch_loop(epoch, metrics_accum=metrics_accum):
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey)

            def minibatch_loop(minibatch):
                mb = perm[minibatch : minibatch + self.config.minibatch_size]
                mb_adv = advantages[mb]

                if self.config.normalize_advantage:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                policy_metrics = self._update_policy(
                    obs[mb], actions[mb], old_log_probs[mb], mb_adv
                )

                value_metrics = self._update_critic(obs[mb], returns[mb])

                # Accumulate metrics
                metrics_accum["policy_loss"] += policy_metrics["policy_loss"]
                metrics_accum["value_loss"] += value_metrics["value_loss"]
                metrics_accum["entropy"] += policy_metrics["entropy"]
                metrics_accum["approx_kl"] += policy_metrics["approx_kl"]
                metrics_accum["clip_fraction"] += policy_metrics["clip_fraction"]


            return jax.lax.fori_loop(0, obs.shape[0] // self.config.minibatch_size, minibatch_loop)

        return jax.lax.fori_loop(0, self.config.num_epochs , epoch_loop)

    @nnx.jit
    def _update_policy(
        self,
        obs: jax.Array,
        actions: jax.Array,
        old_log_probs: jax.Array,
        advantages: jax.Array,
    ) -> dict[str, float]:
        """Update policy network using clipped surrogate objective.

        Args:
            obs: Observations, shape (batch, obs_dim)
            actions: Actions taken, shape (batch, action_dim)
            old_log_probs: Log probs under old policy, shape (batch,)
            advantages: Advantages, shape (batch,)

        Returns:
            Dictionary of policy metrics
        """

        def policy_loss_fn(actor: Actor) -> tuple[jax.Array, dict]:
            """Compute PPO clipped surrogate loss."""
            # Get current policy distribution
            mean, log_std = actor(obs)

            # Compute log prob of actions under current policy
            log_probs = actor.log_prob(obs, actions)

            # Compute ratio: π(a|s) / π_old(a|s)
            ratio = jnp.exp(log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            # Entropy bonus (encourages exploration)
            entropy = entropy_gaussian(log_std).mean()
            entropy_loss = -self.config.entropy_coef * entropy

            # Total loss
            total_loss = policy_loss + entropy_loss

            # Metrics
            approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
            clip_fraction = ((jnp.abs(ratio - 1) > self.config.clip_eps).astype(jnp.float32)).mean()

            metrics = {
                "policy_loss": policy_loss,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            }

            return total_loss, metrics

        # Compute gradients and update
        grad_fn = nnx.value_and_grad(policy_loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(self.actor)
        self.actor_optimizer.update(self.actor, grads)

        return metrics

    @nnx.jit
    def _update_critic(
        self,
        obs: jax.Array,
        returns: jax.Array,
    ) -> dict[str, float]:
        """Update critic network using value loss.

        Args:
            obs: Observations, shape (batch, obs_dim)
            returns: Target returns, shape (batch,)

        Returns:
            Dictionary of value metrics
        """

        def value_loss_fn(critic: Critic) -> jax.Array:
            """Compute value function loss."""
            values = critic(obs)

            if self.config.clip_value_loss:
                # Clipped value loss (like clipped policy loss)
                # Not commonly used, but available
                values_clipped = jnp.clip(
                    values,
                    returns - self.config.clip_eps,
                    returns + self.config.clip_eps,
                )
                loss1 = (values - returns) ** 2
                loss2 = (values_clipped - returns) ** 2
                value_loss = jnp.mean(jnp.maximum(loss1, loss2))
            else:
                # Standard MSE loss
                value_loss = jnp.mean((values - returns) ** 2)

            return value_loss

        # Compute gradients and update
        val_and_grad = nnx.value_and_grad(value_loss_fn)
        loss, grads = val_and_grad(self.critic)
        self.critic_optimizer.update(self.critic, grads)

        return {"value_loss": loss}

    @nnx.jit(static_argnums=(3,))
    def select_action(
        self,
        obs: jax.Array,
        key: jax.random.PRNGKey,
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
        if deterministic:
            mean, _ = self.actor(obs)
            action = mean
            log_prob = self.actor.log_prob(obs, action)
        else:
            action, log_prob = self.actor.sample(obs, key)

        value = self.critic(obs)

        return action, log_prob, value
