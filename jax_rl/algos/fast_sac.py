"""FastSAC — SAC with C51 distributional critic.

Same as SAC but with:
  1. C51 distributional critic (101 atoms by default, cross-entropy loss)
  2. Q averaging instead of min (configurable)
  3. LR cosine decay
  4. Designed for large batch sizes + parallel envs

The actor (stochastic Gaussian) and alpha (auto-tuned temperature) are unchanged.
Only the critic representation and loss change.
"""

import math
from typing import Any
import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.configs.fast_sac_config import FastSACConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.networks.builders import Actor
from jax_rl.networks.heads.q_distributional import DistributionalQHead
from jax_rl.networks.distributions import sample_gaussian
from jax_rl.utils.distributional import (
    make_support,
    logits_to_q,
    project_distribution,
    cross_entropy_categorical,
)
from jax_rl.utils.polyak import soft_update as polyak_update


@flax.struct.dataclass
class TrainingState:
    actor_params: Any
    actor_opt_state: optax.OptState
    q1_params: Any
    q2_params: Any
    q_opt_state: Any
    target_q1_params: Any
    target_q2_params: Any
    log_alpha: jnp.ndarray
    alpha_opt_state: optax.OptState
    key: jax.Array
    update_count: jnp.ndarray  # for policy delay


class FastSAC:
    """FastSAC — SAC with C51 distributional critic and auto-tuned temperature."""

    def __init__(
        self,
        config: FastSACConfig,
        obs_dim: int,
        action_dim: int,
        optimizer: optax.GradientTransformation,
        alpha_optimizer: optax.GradientTransformation,
        gamma: float = 0.99,
        critic_obs_dim: int | None = None,
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim or obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_entropy = -config.target_entropy_scale * action_dim

        # Networks — actor is same as SAC, critic is distributional
        enc_cfg = EncoderConfig(
            obs_dim=obs_dim,
            hidden_dim=config.hidden_dim,
            activation=config.activation,
        )
        log_std_max = math.log(config.max_std) if config.max_std is not None else 2.0
        pol_cfg = PolicyHeadConfig(
            action_dim=action_dim,
            state_dependent_std=True,
            min_std=0.001,
            log_std_max=log_std_max,
            squash=True,
        )
        self.actor = Actor(enc_cfg, pol_cfg)
        critic_dim = config.critic_hidden_dim or config.hidden_dim
        self.q1 = DistributionalQHead(
            critic_dim, config.num_atoms, config.activation, config.q_layer_norm,
        )
        self.q2 = DistributionalQHead(
            critic_dim, config.num_atoms, config.activation, config.q_layer_norm,
        )

        self.optimizer = optimizer
        self.alpha_optimizer = alpha_optimizer

        # C51 support
        support = make_support(config.v_min, config.v_max, config.num_atoms)
        self._support = support

        # Freeze refs
        actor = self.actor
        q1 = self.q1
        q2 = self.q2
        tau = config.tau
        target_entropy = self.target_entropy
        use_avg = config.q_aggregation == "avg"

        # ── Actor forward ────────────────────────────────────────────────
        def _actor_forward(actor_params, obs, key):
            mean, log_std = actor.apply(actor_params, obs)
            action, log_prob = sample_gaussian(mean, log_std, key, squash=True)
            return action, log_prob

        # ── Critic loss (C51 distributional + entropy) ───────────────────
        def _critic_loss(q_params, actor_params, target_q1_params, target_q2_params,
                         log_alpha, batch, key):
            q1_params_, q2_params_ = q_params
            critic_obs = batch["critic_obs"]
            action = batch["action"]
            reward = batch["reward"].squeeze(-1)
            critic_next_obs = batch["critic_next_obs"]
            next_obs = batch["next_obs"]
            done = batch["done"].squeeze(-1)
            truncation = batch["truncation"].squeeze(-1)

            alpha = jnp.exp(log_alpha)

            # Next action from current policy (actor sees 48d obs)
            next_action, next_log_prob = _actor_forward(actor_params, next_obs, key)

            # Target Q distributions (critic sees privileged obs)
            tq1_logits = q1.apply(target_q1_params, critic_next_obs, next_action)
            tq2_logits = q2.apply(target_q2_params, critic_next_obs, next_action)
            tq1_probs = jax.nn.softmax(tq1_logits, axis=-1)
            tq2_probs = jax.nn.softmax(tq2_logits, axis=-1)

            # Q aggregation
            if use_avg:
                target_probs = 0.5 * (tq1_probs + tq2_probs)
            else:
                tq1_val = jnp.sum(tq1_probs * support, axis=-1)
                tq2_val = jnp.sum(tq2_probs * support, axis=-1)
                use_q1 = (tq1_val < tq2_val)[:, None]
                target_probs = jnp.where(use_q1, tq1_probs, tq2_probs)

            # SAC entropy-adjusted reward: r - alpha * log_prob
            adjusted_reward = reward - alpha * next_log_prob

            # Brax/Playground truncation convention (matches SAC/TD3):
            # - done = terminated OR truncated (from EpisodeWrapper)
            # - truncation = truncated AND NOT terminated
            # Target zeros bootstrap on both (via done). Loss mask drops pure-
            # timeout rows so the r-only target doesn't teach Q=r at timeout.
            projected = jax.lax.stop_gradient(
                project_distribution(target_probs, adjusted_reward, done,
                                     self.gamma, support)
            )

            # Online Q logits (critic sees privileged obs)
            q1_logits = q1.apply(q1_params_, critic_obs, action)
            q2_logits = q2.apply(q2_params_, critic_obs, action)

            # Cross-entropy loss with truncation mask.
            # cross_entropy_categorical clamps log_softmax to avoid 0 * -inf
            # = NaN (see safe_log_softmax docstring).
            mask = 1.0 - truncation
            q1_per_sample = cross_entropy_categorical(projected, q1_logits)
            q2_per_sample = cross_entropy_categorical(projected, q2_logits)
            q1_loss = jnp.mean(q1_per_sample * mask)
            q2_loss = jnp.mean(q2_per_sample * mask)

            # Metrics
            q1_val = logits_to_q(q1_logits, support)
            q2_val = logits_to_q(q2_logits, support)

            metrics = {
                "q1_mean": q1_val.mean(),
                "q2_mean": q2_val.mean(),
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
            }
            return q1_loss + q2_loss, metrics

        # ── Actor loss (uses expected Q from distribution) ───────────────
        def _actor_loss(actor_params, q1_params_, q2_params_, log_alpha, batch, key):
            obs = batch["obs"]
            critic_obs = batch["critic_obs"]
            alpha = jnp.exp(log_alpha)

            action, log_prob = _actor_forward(actor_params, obs, key)
            q1_logits = q1.apply(q1_params_, critic_obs, action)
            q2_logits = q2.apply(q2_params_, critic_obs, action)
            q1_val = logits_to_q(q1_logits, support)
            q2_val = logits_to_q(q2_logits, support)

            if use_avg:
                min_q = 0.5 * (q1_val + q2_val)
            else:
                min_q = jnp.minimum(q1_val, q2_val)

            loss = jnp.mean(alpha * log_prob - min_q)
            metrics = {
                "actor_loss": loss,
                "entropy": -log_prob.mean(),
            }
            return loss, metrics

        # ── Alpha loss (same as vanilla SAC) ─────────────────────────────
        def _alpha_loss(log_alpha, actor_params, batch, key):
            obs = batch["obs"]
            _, log_prob = _actor_forward(actor_params, obs, key)
            loss = jnp.exp(log_alpha) * jax.lax.stop_gradient(
                -log_prob - target_entropy
            ).mean()
            return loss, {"alpha_loss": loss, "alpha": jnp.exp(log_alpha)}

        # ── Full update step ─────────────────────────────────────────────
        policy_delay = config.policy_delay

        @jax.jit
        def update(state: TrainingState, batch: dict) -> tuple[TrainingState, dict]:
            key, k1, k2, k3 = jax.random.split(state.key, 4)
            new_count = state.update_count + 1

            # Critic (every step)
            q_params = (state.q1_params, state.q2_params)
            critic_grad_fn = jax.value_and_grad(_critic_loss, argnums=0, has_aux=True)
            (_, critic_metrics), q_grads = critic_grad_fn(
                q_params, state.actor_params, state.target_q1_params,
                state.target_q2_params, state.log_alpha, batch, k1,
            )
            q_updates, new_q_opt_state = optimizer.update(
                q_grads, state.q_opt_state, params=q_params)
            new_q1_params, new_q2_params = optax.apply_updates(q_params, q_updates)

            # Actor + Alpha (delayed by policy_delay steps, like TD3)
            def _do_actor_alpha_update(args):
                (actor_params, actor_opt, log_alpha, alpha_opt,
                 new_q1, new_q2, target_q1, target_q2) = args

                # Update actor: maximize Q - alpha * log_prob
                actor_grad_fn = jax.value_and_grad(_actor_loss, argnums=0, has_aux=True)
                (_, actor_metrics), actor_grads = actor_grad_fn(
                    actor_params, new_q1, new_q2, log_alpha, batch, k2,
                )
                actor_updates, new_actor_opt = optimizer.update(
                    actor_grads, actor_opt, params=actor_params)
                new_actor_params = optax.apply_updates(actor_params, actor_updates)

                # Update alpha (entropy temperature): track target entropy
                alpha_grad_fn = jax.value_and_grad(_alpha_loss, argnums=0, has_aux=True)
                (_, alpha_metrics), alpha_grads = alpha_grad_fn(
                    log_alpha, actor_params, batch, k3,
                )
                alpha_updates, new_alpha_opt = alpha_optimizer.update(
                    alpha_grads, alpha_opt, params=log_alpha)
                new_log_alpha = optax.apply_updates(log_alpha, alpha_updates)

                # Polyak-average target Q networks
                new_target_q1 = polyak_update(new_q1, target_q1, tau)
                new_target_q2 = polyak_update(new_q2, target_q2, tau)

                return (new_actor_params, new_actor_opt, new_log_alpha,
                        new_alpha_opt, new_target_q1, new_target_q2,
                        {**actor_metrics, **alpha_metrics})

            def _skip_actor_alpha_update(args):
                (actor_params, actor_opt, log_alpha, alpha_opt,
                 _new_q1, _new_q2, target_q1, target_q2) = args
                dummy = {"actor_loss": jnp.float32(0.0), "entropy": jnp.float32(0.0),
                         "alpha_loss": jnp.float32(0.0), "alpha": jnp.exp(log_alpha)}
                return (actor_params, actor_opt, log_alpha,
                        alpha_opt, target_q1, target_q2, dummy)

            do_update = (new_count % policy_delay) == 0
            (new_actor_params, new_actor_opt_state, new_log_alpha,
             new_alpha_opt_state, new_tq1, new_tq2,
             actor_alpha_metrics) = jax.lax.cond(
                do_update, _do_actor_alpha_update, _skip_actor_alpha_update,
                (state.actor_params, state.actor_opt_state,
                 state.log_alpha, state.alpha_opt_state,
                 new_q1_params, new_q2_params,
                 state.target_q1_params, state.target_q2_params),
            )

            new_state = state.replace(
                actor_params=new_actor_params,
                actor_opt_state=new_actor_opt_state,
                q1_params=new_q1_params,
                q2_params=new_q2_params,
                q_opt_state=new_q_opt_state,
                target_q1_params=new_tq1,
                target_q2_params=new_tq2,
                log_alpha=new_log_alpha,
                alpha_opt_state=new_alpha_opt_state,
                key=key,
                update_count=new_count,
            )
            metrics = {**critic_metrics, **actor_alpha_metrics}
            return new_state, metrics

        @jax.jit
        def select_action(
            actor_params: Any,
            obs: jax.Array,
            key: jax.Array,
            deterministic: bool = False,
        ) -> jax.Array:
            mean, log_std = actor.apply(actor_params, obs)
            action, _ = sample_gaussian(mean, log_std, key, squash=True)
            return jax.lax.cond(deterministic, lambda: jnp.tanh(mean), lambda: action)

        self.update = update
        self.select_action = select_action
        self._actor_forward = _actor_forward

    def get_q_value(self, state, obs: jax.Array, action: jax.Array,
                    critic_obs: jax.Array | None = None) -> jax.Array:
        """Return scalar Q1 value (expected value from C51 logits)."""
        q_obs = critic_obs if critic_obs is not None else obs
        logits = self.q1.apply(state.q1_params, q_obs, action)
        return logits_to_q(logits, self._support)

    def init(self, key: jax.Array) -> TrainingState:
        key, k1, k3, k4 = jax.random.split(key, 4)

        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_critic_obs = jnp.zeros((1, self.critic_obs_dim))
        dummy_action = jnp.zeros((1, self.action_dim))

        actor_params = self.actor.init(k1, dummy_obs)

        q1_params = self.q1.init(k3, dummy_critic_obs, dummy_action)
        q2_params = self.q2.init(k4, dummy_critic_obs, dummy_action)

        actor_opt_state = self.optimizer.init(actor_params)
        q_params = (q1_params, q2_params)
        q_opt_state = self.optimizer.init(q_params)
        log_alpha = jnp.array(jnp.log(self.config.alpha_init))
        alpha_opt_state = self.alpha_optimizer.init(log_alpha)

        return TrainingState(
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            q1_params=q1_params,
            q2_params=q2_params,
            q_opt_state=q_opt_state,
            target_q1_params=q1_params,
            target_q2_params=q2_params,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
            key=key,
            update_count=jnp.zeros((), dtype=jnp.int32),
        )
