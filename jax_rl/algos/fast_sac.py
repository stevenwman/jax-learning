"""FastSAC — SAC with C51 distributional critic.

Same as SAC but with:
  1. C51 distributional critic (51 atoms, cross-entropy loss)
  2. Q averaging instead of min (configurable)
  3. LR cosine decay
  4. Designed for large batch sizes + parallel envs

The actor (stochastic Gaussian) and alpha (auto-tuned temperature) are unchanged.
Only the critic representation and loss change.
"""

from typing import Any
import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.networks.encoders.mlp import MlpEncoder
from jax_rl.networks.heads.gaussian import GaussianHead
from jax_rl.networks.heads.q_distributional import DistributionalQHead
from jax_rl.networks.distributions import sample_gaussian
from jax_rl.utils.distributional import (
    make_support,
    logits_to_q,
    project_distribution,
)


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


class FastSAC:
    """FastSAC — SAC with C51 distributional critic and auto-tuned temperature."""

    def __init__(
        self,
        config: SACConfig,
        obs_dim: int,
        action_dim: int,
        optimizer: optax.GradientTransformation,
        alpha_optimizer: optax.GradientTransformation,
        gamma: float = 0.99,
        handle_truncation: bool = True,
        # C51 params (not in SACConfig to avoid breaking vanilla SAC)
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        q_aggregation: str = "avg",
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.handle_truncation = handle_truncation
        self.target_entropy = -config.target_entropy_scale * action_dim

        # Networks — actor is same as SAC, critic is distributional
        enc_cfg = EncoderConfig(
            obs_dim=obs_dim,
            hidden_dim=config.hidden_dim,
            activation=config.activation,
        )
        pol_cfg = PolicyHeadConfig(
            action_dim=action_dim,
            state_dependent_std=True,
            min_std=0.001,
            squash=True,
        )
        self.actor_enc = MlpEncoder(enc_cfg)
        self.actor_head = GaussianHead(pol_cfg)
        self.q1 = DistributionalQHead(
            config.hidden_dim, num_atoms, config.activation, config.q_layer_norm,
        )
        self.q2 = DistributionalQHead(
            config.hidden_dim, num_atoms, config.activation, config.q_layer_norm,
        )

        self.optimizer = optimizer
        self.alpha_optimizer = alpha_optimizer

        # C51 support
        support = make_support(v_min, v_max, num_atoms)

        # Freeze refs
        actor_enc = self.actor_enc
        actor_head = self.actor_head
        q1 = self.q1
        q2 = self.q2
        tau = config.tau
        target_entropy = self.target_entropy
        use_avg = q_aggregation == "avg"

        # ── Actor forward ────────────────────────────────────────────────
        def _actor_forward(actor_params, obs, key):
            enc_params, head_params = actor_params
            features = actor_enc.apply(enc_params, obs)
            mean, log_std = actor_head.apply(head_params, features)
            action, log_prob = sample_gaussian(mean, log_std, key, squash=True)
            return action, log_prob

        # ── Critic loss (C51 distributional + entropy) ───────────────────
        def _critic_loss(q_params, actor_params, target_q1_params, target_q2_params,
                         log_alpha, batch, key):
            q1_params_, q2_params_ = q_params
            obs = batch["obs"]
            action = batch["action"]
            reward = batch["reward"].squeeze(-1)
            next_obs = batch["next_obs"]
            done = batch["done"].squeeze(-1)
            truncation = batch["truncation"].squeeze(-1)

            alpha = jnp.exp(log_alpha)

            # Next action from current policy
            next_action, next_log_prob = _actor_forward(actor_params, next_obs, key)

            # Target Q distributions
            tq1_logits = q1.apply(target_q1_params, next_obs, next_action)
            tq2_logits = q2.apply(target_q2_params, next_obs, next_action)
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
            effective_done = jnp.maximum(done, truncation)

            # C51 projection with entropy-adjusted reward
            projected = jax.lax.stop_gradient(
                project_distribution(target_probs, adjusted_reward, effective_done,
                                     self.gamma, support)
            )

            # Online Q logits
            q1_logits = q1.apply(q1_params_, obs, action)
            q2_logits = q2.apply(q2_params_, obs, action)

            # Cross-entropy loss
            q1_log_probs = jax.nn.log_softmax(q1_logits, axis=-1)
            q2_log_probs = jax.nn.log_softmax(q2_logits, axis=-1)
            q1_loss = -jnp.mean(jnp.sum(projected * q1_log_probs, axis=-1))
            q2_loss = -jnp.mean(jnp.sum(projected * q2_log_probs, axis=-1))

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
            alpha = jnp.exp(log_alpha)

            action, log_prob = _actor_forward(actor_params, obs, key)
            q1_logits = q1.apply(q1_params_, obs, action)
            q2_logits = q2.apply(q2_params_, obs, action)
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

        # ── Polyak ───────────────────────────────────────────────────────
        def _soft_update(online, target):
            return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)

        # ── Full update step ─────────────────────────────────────────────
        @jax.jit
        def update(state: TrainingState, batch: dict) -> tuple[TrainingState, dict]:
            key, k1, k2, k3 = jax.random.split(state.key, 4)

            # Critic
            q_params = (state.q1_params, state.q2_params)
            (_, critic_metrics), q_grads = jax.value_and_grad(
                _critic_loss, argnums=0, has_aux=True
            )(q_params, state.actor_params, state.target_q1_params,
              state.target_q2_params, state.log_alpha, batch, k1)
            q_updates, new_q_opt_state = optimizer.update(q_grads, state.q_opt_state)
            new_q1_params, new_q2_params = optax.apply_updates(q_params, q_updates)

            # Actor
            (_, actor_metrics), actor_grads = jax.value_and_grad(
                _actor_loss, argnums=0, has_aux=True
            )(state.actor_params, state.q1_params, state.q2_params,
              state.log_alpha, batch, k2)
            actor_updates, new_actor_opt_state = optimizer.update(
                actor_grads, state.actor_opt_state
            )
            new_actor_params = optax.apply_updates(state.actor_params, actor_updates)

            # Alpha
            (_, alpha_metrics), alpha_grads = jax.value_and_grad(
                _alpha_loss, argnums=0, has_aux=True
            )(state.log_alpha, state.actor_params, batch, k3)
            alpha_updates, new_alpha_opt_state = alpha_optimizer.update(
                alpha_grads, state.alpha_opt_state
            )
            new_log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)

            # Polyak
            new_tq1 = _soft_update(new_q1_params, state.target_q1_params)
            new_tq2 = _soft_update(new_q2_params, state.target_q2_params)

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
            )
            metrics = {**critic_metrics, **actor_metrics, **alpha_metrics}
            return new_state, metrics

        @jax.jit
        def select_action(
            actor_params: Any,
            obs: jax.Array,
            key: jax.Array,
            deterministic: bool = False,
        ) -> jax.Array:
            enc_params, head_params = actor_params
            features = actor_enc.apply(enc_params, obs)
            mean, log_std = actor_head.apply(head_params, features)
            action, _ = sample_gaussian(mean, log_std, key, squash=True)
            return jax.lax.cond(deterministic, lambda: jnp.tanh(mean), lambda: action)

        self.update = update
        self.select_action = select_action
        self._actor_forward = _actor_forward

    def init(self, key: jax.Array) -> TrainingState:
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_action = jnp.zeros((1, self.action_dim))

        enc_params = self.actor_enc.init(k1, dummy_obs)
        dummy_features = jnp.zeros((1, self.config.hidden_dim[-1]))
        head_params = self.actor_head.init(k2, dummy_features)
        actor_params = (enc_params, head_params)

        q1_params = self.q1.init(k3, dummy_obs, dummy_action)
        q2_params = self.q2.init(k4, dummy_obs, dummy_action)

        actor_opt_state = self.optimizer.init(actor_params)
        q_params = (q1_params, q2_params)
        q_opt_state = self.optimizer.init(q_params)
        log_alpha = jnp.zeros(())
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
        )
