"""PPO with contraction-theory metric training + reward-augmentation hook.

Fork of ppo.py. Baseline PPO untouched so ablations (contraction=None) remain
bit-identical to pre-port. See .superpowers/plans/2026-04-21-contraction-ppo.md.

Algorithmic core (ref: /tmp/cppo/contraction_ppo.py):
- Separate metric network M(c) = L(c) L(c)^T trained per-minibatch alongside
  actor/critic. Loss = mean(ReLU(V̇ + αV + ε)) where V(c) = c^T M(c) c,
  V̇ = ∇_c V · ċ, ċ supplied as env obs (no dynamics Jacobian).
- PPO policy loss is UNCHANGED (ref :661 commented out). Policy influence is
  via reward augmentation during rollout (ref :905-908, implemented in Task 7).
- Metric loss is 2nd-order: outer grad differentiates through jax.grad(V).
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.buffers import RolloutBatch
from jax_rl.buffers.rollout import compute_gae
from jax_rl.configs import PPOConfig
from jax_rl.networks.builders import Actor, VCritic
from jax_rl.networks.contraction_metric import ContractionMetric
from jax_rl.networks.distributions import (
    entropy_gaussian,
    gaussian_log_prob,
    sample_gaussian,
)


@flax.struct.dataclass
class TrainingState:
    actor_params: Any
    critic_params: Any
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    metric_params: Any = None
    metric_opt_state: Any = None


class PPOContraction:
    """PPO + contraction metric training. contraction=None falls back to baseline PPO."""

    def __init__(
        self,
        config: PPOConfig,
        obs_dim: int,
        action_dim: int,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        critic_obs_dim: int | None = None,
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim or obs_dim

        encoder_config = config.encoder
        critic_encoder_config = config.critic_encoder or encoder_config
        policy_config = config.policy_head
        encoder_config.obs_dim = obs_dim
        critic_encoder_config.obs_dim = self.critic_obs_dim
        policy_config.action_dim = action_dim

        self.num_envs = config.num_envs

        self.actor = Actor(encoder_config, policy_config)
        self.critic = VCritic(critic_encoder_config)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self._contraction_enabled = config.contraction is not None
        if self._contraction_enabled:
            config.contraction.validate()
            self.metric = ContractionMetric(
                constraint_dim=config.contraction.constraint_dim,
                hidden_dims=tuple(config.contraction.hidden_dims),
                activation=config.contraction.activation,
                min_diagonal_value=config.contraction.min_diagonal_value,
            )
            self.metric_optimizer = optax.adam(config.contraction.metric_lr)
        else:
            self.metric = None
            self.metric_optimizer = None

        actor = self.actor
        critic = self.critic
        metric = self.metric
        clip_eps = config.clip_eps
        entropy_coef = config.entropy_coef
        normalize_advantage = config.normalize_advantage
        squash = config.policy_head.squash
        num_epochs = config.num_epochs
        minibatch_size = config.minibatch_size
        gamma = config.gamma
        gae_lambda = config.gae_lambda
        c_obs_dim = self.critic_obs_dim
        contraction_enabled = self._contraction_enabled
        alpha = config.contraction.alpha if contraction_enabled else 0.0
        eps_c = config.contraction.epsilon_contraction if contraction_enabled else 0.0
        constraint_coef = config.contraction.constraint_coef if contraction_enabled else 1.0
        metric_optimizer = self.metric_optimizer

        def _metric_loss_fn(metric_params, c_batch, c_dot_batch):
            """mean(ReLU(V̇ + αV + ε)) with V = cᵀM(c)c, V̇ = ∇V·ċ."""
            c_scaled = c_batch * constraint_coef
            c_dot_scaled = c_dot_batch * constraint_coef

            def V_single(c):
                M = metric.apply(metric_params, c[None])[0]
                return c @ M @ c

            V = jax.vmap(V_single)(c_scaled)
            grad_V = jax.vmap(jax.grad(V_single))(c_scaled)
            V_dot = jnp.sum(grad_V * c_dot_scaled, axis=-1)
            penalty = jax.nn.relu(V_dot + alpha * V + eps_c)
            aux = {
                "contraction_penalty": penalty.mean(),
                "V_mean": V.mean(),
                "V_dot_mean": V_dot.mean(),
            }
            return jnp.mean(penalty), aux

        def _minibatch_step(state, mb_obs, mb_critic_obs, mb_actions,
                            mb_old_log_probs, mb_returns, mb_adv,
                            mb_c, mb_c_dot, key):
            def value_loss_fn(critic_params):
                values = critic.apply(critic_params, mb_critic_obs)
                return jnp.mean((values - mb_returns) ** 2) * 0.5 * 0.5

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

            if contraction_enabled:
                (metric_loss, metric_aux), metric_grads = jax.value_and_grad(
                    _metric_loss_fn, has_aux=True, argnums=0
                )(state.metric_params, mb_c, mb_c_dot)
                metric_updates, new_metric_opt_state = metric_optimizer.update(
                    metric_grads, state.metric_opt_state
                )
                new_metric_params = optax.apply_updates(state.metric_params, metric_updates)
                new_state = state.replace(
                    actor_params=new_actor_params,
                    critic_params=new_critic_params,
                    actor_opt_state=new_actor_opt_state,
                    critic_opt_state=new_value_opt_state,
                    metric_params=new_metric_params,
                    metric_opt_state=new_metric_opt_state,
                )
                metrics = {**actor_metrics, "value_loss": value_loss, **metric_aux}
            else:
                new_state = state.replace(
                    actor_params=new_actor_params,
                    critic_params=new_critic_params,
                    actor_opt_state=new_actor_opt_state,
                    critic_opt_state=new_value_opt_state,
                )
                metrics = {**actor_metrics, "value_loss": value_loss}
            return new_state, metrics

        @jax.jit
        def _update(state, obs_t, critic_obs_t, actions_t, old_log_probs_t,
                    rewards_t, dones_t, truncations_t, next_obs, critic_next_obs,
                    c_t, c_dot_t, key):
            T, E = rewards_t.shape
            N = T * E
            num_minibatches = N // minibatch_size
            obs_dim_ = obs_t.shape[-1]
            action_dim_ = actions_t.shape[-1]

            actions_flat = actions_t.reshape(N, action_dim_)
            old_lp_flat = old_log_probs_t.reshape(N)

            if contraction_enabled:
                constraint_dim_ = c_t.shape[-1]
                c_flat = c_t.reshape(N, constraint_dim_)
                c_dot_flat = c_dot_t.reshape(N, constraint_dim_)
            else:
                constraint_dim_ = 1
                c_flat = jnp.zeros((N, 1))
                c_dot_flat = jnp.zeros((N, 1))

            def epoch_step(carry, _):
                state, key, metrics_sum = carry

                c_obs_flat = critic_obs_t.reshape(N, c_obs_dim)
                values_flat = critic.apply(state.critic_params, c_obs_flat)
                values = values_flat.reshape(T, E)
                next_value = critic.apply(state.critic_params, critic_next_obs)

                advantages, returns = compute_gae(
                    rewards_t, values, dones_t, truncations_t,
                    next_value, gamma, gae_lambda,
                )
                adv_flat = advantages.reshape(N)
                if normalize_advantage:
                    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
                ret_flat = returns.reshape(N)

                key, subkey = jax.random.split(key)
                perm = jax.random.permutation(subkey, N)
                usable = num_minibatches * minibatch_size
                perm = perm[:usable]

                obs_flat = obs_t.reshape(N, obs_dim_)
                mb_obs = obs_flat[perm].reshape(num_minibatches, minibatch_size, obs_dim_)
                mb_critic_obs = c_obs_flat[perm].reshape(num_minibatches, minibatch_size, c_obs_dim)
                mb_actions = actions_flat[perm].reshape(num_minibatches, minibatch_size, action_dim_)
                mb_old_lp = old_lp_flat[perm].reshape(num_minibatches, minibatch_size)
                mb_returns = ret_flat[perm].reshape(num_minibatches, minibatch_size)
                mb_adv = adv_flat[perm].reshape(num_minibatches, minibatch_size)
                mb_c = c_flat[perm].reshape(num_minibatches, minibatch_size, constraint_dim_)
                mb_c_dot = c_dot_flat[perm].reshape(num_minibatches, minibatch_size, constraint_dim_)

                key, entropy_key = jax.random.split(key)
                mb_keys = jax.random.split(entropy_key, num_minibatches)

                def scan_minibatch(carry, minibatch):
                    state, metrics_sum = carry
                    o, co, a, olp, r, adv, c, c_dot, mb_key = minibatch
                    state, metrics = _minibatch_step(
                        state, o, co, a, olp, r, adv, c, c_dot, mb_key
                    )
                    metrics_sum = jax.tree.map(jnp.add, metrics_sum, metrics)
                    return (state, metrics_sum), None

                (state, metrics_sum), _ = jax.lax.scan(
                    scan_minibatch,
                    (state, metrics_sum),
                    (mb_obs, mb_critic_obs, mb_actions, mb_old_lp,
                     mb_returns, mb_adv, mb_c, mb_c_dot, mb_keys),
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
            if contraction_enabled:
                init_metrics.update({
                    "contraction_penalty": jnp.float32(0),
                    "V_mean": jnp.float32(0),
                    "V_dot_mean": jnp.float32(0),
                })

            (state, _, metrics_sum), _ = jax.lax.scan(
                epoch_step, (state, key, init_metrics), None, length=num_epochs,
            )
            num_updates = num_epochs * num_minibatches
            metrics_avg = jax.tree.map(lambda x: x / num_updates, metrics_sum)
            return state, metrics_avg

        self._update = _update

        @jax.jit
        def _select_stochastic(actor_params, critic_params, obs, critic_obs, key):
            mean, log_std = actor.apply(actor_params, obs)
            action, log_prob = sample_gaussian(mean, log_std, key, squash=squash)
            value = critic.apply(critic_params, critic_obs)
            return action, log_prob, value

        @jax.jit
        def _select_deterministic(actor_params, critic_params, obs, critic_obs):
            mean, log_std = actor.apply(actor_params, obs)
            log_prob = gaussian_log_prob(mean, log_std, mean, squash=squash)
            value = critic.apply(critic_params, critic_obs)
            return mean, log_prob, value

        @jax.jit
        def _select_eval(actor_params, obs):
            mean, _log_std = actor.apply(actor_params, obs)
            if squash:
                return jnp.tanh(mean)
            return mean

        self._select_stochastic = _select_stochastic
        self._select_deterministic = _select_deterministic
        self._select_eval = _select_eval

    def init(self, key: jax.Array) -> TrainingState:
        actor_key, critic_key, metric_key = jax.random.split(key, 3)
        dummy_obs = jnp.zeros(self.obs_dim)
        dummy_critic_obs = jnp.zeros(self.critic_obs_dim)

        actor_params = self.actor.init(actor_key, dummy_obs)
        critic_params = self.critic.init(critic_key, dummy_critic_obs)

        if self._contraction_enabled:
            cdim = self.config.contraction.constraint_dim
            metric_params = self.metric.init(metric_key, jnp.zeros((1, cdim)))
            metric_opt_state = self.metric_optimizer.init(metric_params)
        else:
            metric_params = None
            metric_opt_state = None

        return TrainingState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt_state=self.actor_optimizer.init(actor_params),
            critic_opt_state=self.critic_optimizer.init(critic_params),
            metric_params=metric_params,
            metric_opt_state=metric_opt_state,
        )

    def update(self, state: TrainingState, batch: RolloutBatch, key: jax.Array,
               next_obs: jax.Array = None, critic_obs: jax.Array = None,
               critic_next_obs: jax.Array = None) -> tuple[TrainingState, dict]:
        if critic_obs is None:
            critic_obs = batch.obs
        if critic_next_obs is None:
            critic_next_obs = next_obs

        if self._contraction_enabled:
            assert batch.contraction_c is not None, "contraction enabled but batch.contraction_c is None"
            assert batch.contraction_c_dot is not None, "contraction enabled but batch.contraction_c_dot is None"
            c_t = batch.contraction_c
            c_dot_t = batch.contraction_c_dot
        else:
            # Dummy arrays: not read when contraction disabled (gated via contraction_enabled flag).
            T, E = batch.rewards.shape
            c_t = jnp.zeros((T, E, 1))
            c_dot_t = jnp.zeros((T, E, 1))

        return self._update(
            state, batch.obs, critic_obs, batch.actions, batch.log_probs,
            batch.rewards, batch.dones, batch.truncations,
            next_obs, critic_next_obs, c_t, c_dot_t, key,
        )

    def select_action(self, state, obs, key, deterministic=False, critic_obs=None):
        if critic_obs is None:
            critic_obs = obs
        if deterministic:
            return self._select_deterministic(state.actor_params, state.critic_params, obs, critic_obs)
        return self._select_stochastic(state.actor_params, state.critic_params, obs, critic_obs, key)

    def select_action_eval(self, actor_params, obs):
        return self._select_eval(actor_params, obs)

    def compute_contraction_reward(self, metric_params, c, c_dot):
        """Per-step reward augmentation: (ε − penalty) * penalty_coef.
        Used by the rollout collector (Task 7). Expects batched (E, constraint_dim)."""
        cfg = self.config.contraction
        c_scaled = c * cfg.constraint_coef
        c_dot_scaled = c_dot * cfg.constraint_coef

        def V_single(ci):
            M = self.metric.apply(metric_params, ci[None])[0]
            return ci @ M @ ci

        V = jax.vmap(V_single)(c_scaled)
        grad_V = jax.vmap(jax.grad(V_single))(c_scaled)
        V_dot = jnp.sum(grad_V * c_dot_scaled, axis=-1)
        penalty = jax.nn.relu(V_dot + cfg.alpha * V + cfg.epsilon_contraction)
        return (cfg.epsilon_contraction - penalty) * cfg.penalty_coef
