"""FastTD3 — TD3 with C51 distributional critic.

Paper: https://arxiv.org/abs/2505.22642 (Seo et al., 2025)

Key modifications over vanilla TD3:
  1. C51 distributional critic (51 atoms, cross-entropy loss)
  2. Q averaging instead of min (reduces underestimation at scale)
  3. Large batch sizes (8K-32K) with LayerNorm for stability
  4. LR cosine decay (3e-4 → 3e-5)

The actor is unchanged — deterministic policy with target smoothing.
Only the critic representation and loss change.

Loss reference: .context/fast_td3_plan.md
"""

from typing import Any
import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.configs.fast_td3_config import FastTD3Config
from jax_rl.configs.networks_config import EncoderConfig
from jax_rl.networks.encoders.mlp import MlpEncoder
from jax_rl.networks.heads.deterministic import DeterministicHead
from jax_rl.networks.heads.q_distributional import DistributionalQHead
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
    target_actor_params: Any
    target_q1_params: Any
    target_q2_params: Any
    key: jax.Array
    update_count: jnp.ndarray


class FastTD3:
    """FastTD3 — TD3 with C51 distributional critic and Q averaging."""

    def __init__(
        self,
        config: FastTD3Config,
        obs_dim: int,
        action_dim: int,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        gamma: float = 0.99,
        handle_truncation: bool = True,
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.handle_truncation = handle_truncation

        # Networks
        enc_cfg = EncoderConfig(
            obs_dim=obs_dim,
            hidden_dim=config.hidden_dim,
            activation=config.activation,
        )
        self.actor_enc = MlpEncoder(enc_cfg)
        self.actor_head = DeterministicHead(action_dim=action_dim)
        critic_dim = config.critic_hidden_dim or config.hidden_dim
        self.q1 = DistributionalQHead(
            critic_dim, config.num_atoms, config.activation, config.q_layer_norm,
        )
        self.q2 = DistributionalQHead(
            critic_dim, config.num_atoms, config.activation, config.q_layer_norm,
        )

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # C51 support atoms (fixed, not trainable)
        support = make_support(config.v_min, config.v_max, config.num_atoms)
        self._support = support

        # Freeze refs
        actor_enc = self.actor_enc
        actor_head = self.actor_head
        q1 = self.q1
        q2 = self.q2
        tau = config.tau
        policy_delay = config.policy_delay
        target_noise_std = config.target_noise_std
        noise_clip = config.noise_clip
        num_atoms = config.num_atoms
        use_avg = config.q_aggregation == "avg"

        # ── Actor forward ────────────────────────────────────────────────
        def _actor_forward(actor_params, obs):
            enc_params, head_params = actor_params
            features = actor_enc.apply(enc_params, obs)
            return actor_head.apply(head_params, features)

        # ── Critic loss (C51 distributional) ─────────────────────────────
        def _critic_loss(q_params, target_actor_params, target_q1_params,
                         target_q2_params, batch, key):
            q1_params_, q2_params_ = q_params
            obs = batch["obs"]
            action = batch["action"]
            reward = batch["reward"].squeeze(-1)
            next_obs = batch["next_obs"]
            done = batch["done"].squeeze(-1)
            truncation = batch["truncation"].squeeze(-1)

            # Target policy smoothing
            next_action = _actor_forward(target_actor_params, next_obs)
            noise = jnp.clip(
                jax.random.normal(key, next_action.shape) * target_noise_std,
                -noise_clip, noise_clip,
            )
            next_action = jnp.clip(next_action + noise, -1.0, 1.0)

            # Target Q distributions
            tq1_logits = q1.apply(target_q1_params, next_obs, next_action)
            tq2_logits = q2.apply(target_q2_params, next_obs, next_action)
            tq1_probs = jax.nn.softmax(tq1_logits, axis=-1)
            tq2_probs = jax.nn.softmax(tq2_logits, axis=-1)

            # Q aggregation: average distributions or use min-Q's distribution
            if use_avg:
                target_probs = 0.5 * (tq1_probs + tq2_probs)
            else:
                # Use distribution from whichever Q has lower expected value
                tq1_val = jnp.sum(tq1_probs * support, axis=-1)
                tq2_val = jnp.sum(tq2_probs * support, axis=-1)
                use_q1 = (tq1_val < tq2_val)[:, None]  # (batch, 1)
                target_probs = jnp.where(use_q1, tq1_probs, tq2_probs)

            # Effective done: treat truncation as terminal (next_obs is wrong)
            effective_done = jnp.maximum(done, truncation)

            # C51 projection: shift target atoms by Bellman operator
            projected = jax.lax.stop_gradient(
                project_distribution(target_probs, reward, effective_done, self.gamma, support)
            )

            # Online Q logits
            q1_logits = q1.apply(q1_params_, obs, action)
            q2_logits = q2.apply(q2_params_, obs, action)

            # Cross-entropy loss for each Q network
            # Clamp log_probs to prevent -inf * 0 = NaN in cross-entropy
            q1_log_probs = jnp.maximum(jax.nn.log_softmax(q1_logits, axis=-1), -30.0)
            q2_log_probs = jnp.maximum(jax.nn.log_softmax(q2_logits, axis=-1), -30.0)
            q1_loss = -jnp.mean(jnp.sum(projected * q1_log_probs, axis=-1))
            q2_loss = -jnp.mean(jnp.sum(projected * q2_log_probs, axis=-1))

            # Metrics (expected Q for logging)
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
        def _actor_loss_fn(actor_params, q1_params_, q2_params_, obs):
            action = _actor_forward(actor_params, obs)
            q1_logits = q1.apply(q1_params_, obs, action)
            q2_logits = q2.apply(q2_params_, obs, action)
            q1_val = logits_to_q(q1_logits, support)
            q2_val = logits_to_q(q2_logits, support)

            if use_avg:
                q_val = 0.5 * (q1_val + q2_val)
            else:
                q_val = q1_val  # original TD3: use Q1 only

            loss = -jnp.mean(q_val)
            return loss, {"actor_loss": loss}

        # ── Polyak ───────────────────────────────────────────────────────
        def _soft_update(online, target):
            return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)

        # ── Full update step ─────────────────────────────────────────────
        @jax.jit
        def update(state: TrainingState, batch: dict) -> tuple[TrainingState, dict]:
            key, k1 = jax.random.split(state.key)
            new_count = state.update_count + 1

            # Critic update (every step)
            q_params = (state.q1_params, state.q2_params)
            (_, critic_metrics), q_grads = jax.value_and_grad(
                _critic_loss, argnums=0, has_aux=True
            )(q_params, state.target_actor_params, state.target_q1_params,
              state.target_q2_params, batch, k1)
            q_updates, new_q_opt_state = critic_optimizer.update(
                q_grads, state.q_opt_state, params=q_params
            )
            new_q1_params, new_q2_params = optax.apply_updates(q_params, q_updates)

            # Actor update (delayed)
            def _do_actor_update(args):
                actor_params, actor_opt_state, q1_p, q2_p, nq1, nq2, ta, tq1, tq2 = args
                obs = batch["obs"]
                (_, actor_metrics), actor_grads = jax.value_and_grad(
                    _actor_loss_fn, argnums=0, has_aux=True
                )(actor_params, q1_p, q2_p, obs)
                actor_updates, new_actor_opt_state = actor_optimizer.update(
                    actor_grads, actor_opt_state, params=actor_params
                )
                new_actor_params = optax.apply_updates(actor_params, actor_updates)
                new_ta = _soft_update(new_actor_params, ta)
                new_tq1 = _soft_update(nq1, tq1)
                new_tq2 = _soft_update(nq2, tq2)
                return (new_actor_params, new_actor_opt_state,
                        new_ta, new_tq1, new_tq2, actor_metrics)

            def _skip_actor_update(args):
                actor_params, actor_opt_state, _, _, _, _, ta, tq1, tq2 = args
                dummy_metrics = {"actor_loss": jnp.float32(0.0)}
                return (actor_params, actor_opt_state,
                        ta, tq1, tq2, dummy_metrics)

            do_update = (new_count % policy_delay) == 0
            (new_actor_params, new_actor_opt_state,
             new_target_actor, new_tq1, new_tq2,
             actor_metrics) = jax.lax.cond(
                do_update,
                _do_actor_update,
                _skip_actor_update,
                (state.actor_params, state.actor_opt_state,
                 state.q1_params, state.q2_params,
                 new_q1_params, new_q2_params,
                 state.target_actor_params, state.target_q1_params,
                 state.target_q2_params),
            )

            new_state = state.replace(
                actor_params=new_actor_params,
                actor_opt_state=new_actor_opt_state,
                q1_params=new_q1_params,
                q2_params=new_q2_params,
                q_opt_state=new_q_opt_state,
                target_actor_params=new_target_actor,
                target_q1_params=new_tq1,
                target_q2_params=new_tq2,
                key=key,
                update_count=new_count,
            )
            metrics = {**critic_metrics, **actor_metrics}
            return new_state, metrics

        @jax.jit
        def select_action(
            actor_params: Any,
            obs: jax.Array,
            key: jax.Array,
            deterministic: bool = False,
            exploration_noise: float = 0.1,
        ) -> jax.Array:
            action = _actor_forward(actor_params, obs)
            noisy_action = action + jax.random.normal(key, action.shape) * exploration_noise
            noisy_action = jnp.clip(noisy_action, -1.0, 1.0)
            return jax.lax.cond(deterministic, lambda: action, lambda: noisy_action)

        self.update = update
        self.select_action = select_action
        self._actor_forward = _actor_forward

    def get_q_value(self, state, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Return scalar Q1 value (expected value from C51 logits)."""
        logits = self.q1.apply(state.q1_params, obs, action)
        return logits_to_q(logits, self._support)

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

        actor_opt_state = self.actor_optimizer.init(actor_params)
        q_params = (q1_params, q2_params)
        q_opt_state = self.critic_optimizer.init(q_params)

        return TrainingState(
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            q1_params=q1_params,
            q2_params=q2_params,
            q_opt_state=q_opt_state,
            target_actor_params=actor_params,
            target_q1_params=q1_params,
            target_q2_params=q2_params,
            key=key,
            update_count=jnp.int32(0),
        )
