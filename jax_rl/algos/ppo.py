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
from jax_rl.networks.builders import Actor, VCritic
from jax_rl.buffers import RolloutBatch
from jax_rl.buffers.rollout import compute_gae
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
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
    ) -> None:
        self.config = config

        encoder_config = config.encoder
        critic_encoder_config = config.critic_encoder or encoder_config
        policy_config = config.policy_head
        encoder_config.obs_dim = obs_dim
        critic_encoder_config.obs_dim = obs_dim
        policy_config.action_dim = action_dim

        self.num_envs = config.num_envs
        self.obs_dim = obs_dim

        self.actor = Actor(encoder_config, policy_config)
        self.critic = VCritic(critic_encoder_config, config.value_head)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Capture immutable refs for closures
        actor = self.actor
        critic = self.critic
        clip_eps = config.clip_eps
        entropy_coef = config.entropy_coef
        normalize_advantage = config.normalize_advantage
        squash = config.policy_head.squash
        num_epochs = config.num_epochs
        minibatch_size = config.minibatch_size
        gamma = config.gamma
        gae_lambda = config.gae_lambda

        def _minibatch_step(state, mb_obs, mb_actions, mb_old_log_probs, mb_returns, mb_adv, key):
            # Per-minibatch advantage normalization (matches Brax)
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
                entropy = entropy_gaussian(mb_log_std, mean=mb_mean, key=key, squash=squash).mean()
                policy_loss += -entropy_coef * entropy
                clip_fraction = (jnp.abs(ratio - 1) > clip_eps).astype(jnp.float32).mean()
                approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
                return policy_loss, {
                    "policy_loss": policy_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "log_std_mean": mb_log_std.mean(),
                    "log_std_min": mb_log_std.min(),
                    "log_std_max": mb_log_std.max(),
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
        def _update(state, obs_t, actions_t, old_log_probs_t,
                    rewards_t, dones_t, truncations_t, next_obs, key):
            """PPO update with per-epoch GAE recomputation.

            Args have temporal shape: (T, E, ...) where T=num_steps, E=num_envs.
            Each epoch recomputes V(s) from current critic and re-runs GAE.
            """
            T, E = rewards_t.shape
            N = T * E
            num_minibatches = N // minibatch_size
            obs_dim = obs_t.shape[-1]
            action_dim = actions_t.shape[-1]

            # Flatten actions and old_log_probs (these don't change across epochs)
            actions_flat = actions_t.reshape(N, action_dim)
            old_lp_flat = old_log_probs_t.reshape(N)

            def epoch_step(carry, _):
                state, key, metrics_sum = carry

                # Recompute values from current critic (per-epoch, matches Brax)
                obs_flat = obs_t.reshape(N, obs_dim)
                values_flat = critic.apply(state.critic_params, obs_flat)
                values = values_flat.reshape(T, E)
                next_value = critic.apply(state.critic_params, next_obs)

                # Recompute GAE with fresh values
                advantages, returns = compute_gae(
                    rewards_t, values, dones_t, truncations_t,
                    next_value, gamma, gae_lambda,
                )
                adv_flat = advantages.reshape(N)
                ret_flat = returns.reshape(N)

                key, subkey = jax.random.split(key)
                perm = jax.random.permutation(subkey, N)
                usable = num_minibatches * minibatch_size
                perm = perm[:usable]

                mb_obs = obs_flat[perm].reshape(num_minibatches, minibatch_size, obs_dim)
                mb_actions = actions_flat[perm].reshape(num_minibatches, minibatch_size, action_dim)
                mb_old_lp = old_lp_flat[perm].reshape(num_minibatches, minibatch_size)
                mb_returns = ret_flat[perm].reshape(num_minibatches, minibatch_size)
                mb_adv = adv_flat[perm].reshape(num_minibatches, minibatch_size)

                key, entropy_key = jax.random.split(key)
                mb_keys = jax.random.split(entropy_key, num_minibatches)

                def scan_minibatch(carry, minibatch):
                    state, metrics_sum = carry
                    o, a, olp, r, adv, mb_key = minibatch
                    state, metrics = _minibatch_step(state, o, a, olp, r, adv, mb_key)
                    metrics_sum = jax.tree.map(jnp.add, metrics_sum, metrics)
                    return (state, metrics_sum), None

                (state, metrics_sum), _ = jax.lax.scan(
                    scan_minibatch,
                    (state, metrics_sum),
                    (mb_obs, mb_actions, mb_old_lp, mb_returns, mb_adv, mb_keys),
                )
                return (state, key, metrics_sum), None

            init_metrics = {
                "policy_loss": jnp.float32(0),
                "entropy": jnp.float32(0),
                "approx_kl": jnp.float32(0),
                "clip_fraction": jnp.float32(0),
                "value_loss": jnp.float32(0),
                "log_std_mean": jnp.float32(0),
                "log_std_min": jnp.float32(0),
                "log_std_max": jnp.float32(0),
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

    def update(self, state: TrainingState, batch: RolloutBatch, key: jax.Array,
               next_obs: jax.Array = None) -> tuple[TrainingState, dict]:
        return self._update(
            state, batch.obs, batch.actions, batch.log_probs,
            batch.rewards, batch.dones, batch.truncations, next_obs, key,
        )

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
