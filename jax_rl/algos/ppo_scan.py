"""PPO with fully-JIT'd update via jax.lax.scan.

Same as ppo_jit.py but replaces the Python epoch/minibatch loops in update()
with jax.lax.scan, so the entire update is a single compiled XLA program.

Difference from ppo_jit.py:
  ppo_jit.py:  Python loop calls @jax.jit _minibatch_step 64 times
  ppo_scan.py: @jax.jit _update scans over all 64 minibatches in one kernel
"""

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

    def __init__(
        self,
        config: PPOConfig,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        self.config = config

        encoder_config = config.encoder
        policy_config = config.policy_head
        encoder_config.obs_dim = obs_dim
        policy_config.action_dim = action_dim

        self.num_envs = config.num_envs
        self.obs_dim = obs_dim

        self.actor = Actor(encoder_config, policy_config)
        self.critic = Critic(encoder_config, config.value_head)
        self.actor_optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.actor_lr))
        self.critic_optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.critic_lr))

        # Capture immutable refs for closures
        actor = self.actor
        critic = self.critic
        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        clip_eps = config.clip_eps
        entropy_coef = config.entropy_coef
        normalize_advantage = config.normalize_advantage
        squash = config.policy_head.squash
        num_epochs = config.num_epochs
        minibatch_size = config.minibatch_size

        def _minibatch_step(state, mb_obs, mb_actions, mb_old_log_probs, mb_returns, mb_adv):
            if normalize_advantage:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            def value_loss_fn(critic_params):
                values = critic.apply(critic_params, mb_obs)
                return jnp.mean((values - mb_returns) ** 2)

            def actor_loss_fn(actor_params):
                mb_mean, mb_log_std = actor.apply(actor_params, mb_obs)
                mb_log_probs = gaussian_log_prob(mb_mean, mb_log_std, mb_actions, squash=squash)
                ratio = jnp.exp(mb_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
                entropy = entropy_gaussian(mb_log_std).mean()
                policy_loss += -entropy_coef * entropy
                clip_fraction = (jnp.abs(ratio - 1) > clip_eps).astype(jnp.float32).mean()
                approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
                return policy_loss, {
                    "policy_loss": policy_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                }

            (_, actor_metrics), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor_params)
            actor_updates, new_actor_opt_state = actor_optimizer.update(actor_grads, state.actor_opt_state)
            new_actor_params = optax.apply_updates(state.actor_params, actor_updates)

            value_loss, value_grads = jax.value_and_grad(value_loss_fn)(state.critic_params)
            value_updates, new_value_opt_state = critic_optimizer.update(value_grads, state.critic_opt_state)
            new_critic_params = optax.apply_updates(state.critic_params, value_updates)

            new_state = state.replace(
                actor_params=new_actor_params,
                critic_params=new_critic_params,
                actor_opt_state=new_actor_opt_state,
                critic_opt_state=new_value_opt_state,
            )
            metrics = {**actor_metrics, "value_loss": value_loss}
            return new_state, metrics

        @jax.jit
        def _update(state, obs, actions, old_log_probs, returns, advantages, key):
            N = obs.shape[0]
            num_minibatches = N // minibatch_size

            def epoch_step(carry, _):
                state, key, metrics_sum = carry
                key, subkey = jax.random.split(key)
                perm = jax.random.permutation(subkey, N)

                # Reshape permuted data into (num_minibatches, minibatch_size, ...)
                mb_obs = obs[perm].reshape(num_minibatches, minibatch_size, -1)
                mb_actions = actions[perm].reshape(num_minibatches, minibatch_size, -1)
                mb_old_lp = old_log_probs[perm].reshape(num_minibatches, minibatch_size)
                mb_returns = returns[perm].reshape(num_minibatches, minibatch_size)
                mb_adv = advantages[perm].reshape(num_minibatches, minibatch_size)

                def scan_minibatch(carry, minibatch):
                    state, metrics_sum = carry
                    o, a, olp, r, adv = minibatch
                    state, metrics = _minibatch_step(state, o, a, olp, r, adv)
                    metrics_sum = jax.tree.map(jnp.add, metrics_sum, metrics)
                    return (state, metrics_sum), None

                (state, metrics_sum), _ = jax.lax.scan(
                    scan_minibatch,
                    (state, metrics_sum),
                    (mb_obs, mb_actions, mb_old_lp, mb_returns, mb_adv),
                )
                return (state, key, metrics_sum), None

            init_metrics = {
                "policy_loss": jnp.float32(0),
                "entropy": jnp.float32(0),
                "approx_kl": jnp.float32(0),
                "clip_fraction": jnp.float32(0),
                "value_loss": jnp.float32(0),
            }

            (state, _, metrics_sum), _ = jax.lax.scan(
                epoch_step,
                (state, key, init_metrics),
                None,
                length=num_epochs,
            )

            num_updates = num_epochs * num_minibatches
            metrics_avg = jax.tree.map(lambda x: x / num_updates, metrics_sum)
            return state, metrics_avg

        self._update = _update

        @jax.jit
        def _select_stochastic(actor_params, critic_params, obs, key):
            mean, log_std = actor.apply(actor_params, obs)
            action, log_prob = sample_gaussian(mean, log_std, key, squash=squash)
            value = critic.apply(critic_params, obs)
            return action, log_prob, value

        @jax.jit
        def _select_deterministic(actor_params, critic_params, obs):
            mean, log_std = actor.apply(actor_params, obs)
            log_prob = gaussian_log_prob(mean, log_std, mean, squash=squash)
            value = critic.apply(critic_params, obs)
            return mean, log_prob, value

        self._select_stochastic = _select_stochastic
        self._select_deterministic = _select_deterministic

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
        obs = batch.obs.reshape(-1, batch.obs.shape[-1])
        actions = batch.actions.reshape(-1, batch.actions.shape[-1])
        old_log_probs = batch.log_probs.reshape(-1)
        advantages = batch.advantages.reshape(-1)
        returns = batch.returns.reshape(-1)

        return self._update(state, obs, actions, old_log_probs, returns, advantages, key)

    def select_action(
        self,
        state: TrainingState,
        obs: jax.Array,
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if deterministic:
            return self._select_deterministic(state.actor_params, state.critic_params, obs)
        return self._select_stochastic(state.actor_params, state.critic_params, obs, key)
