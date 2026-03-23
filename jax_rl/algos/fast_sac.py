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
from jax_rl.networks.builders import Actor
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
    update_count: jnp.ndarray  # for policy delay


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
        import math
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
            critic_dim, num_atoms, config.activation, config.q_layer_norm,
        )
        self.q2 = DistributionalQHead(
            critic_dim, num_atoms, config.activation, config.q_layer_norm,
        )

        self.optimizer = optimizer
        self.alpha_optimizer = alpha_optimizer

        # C51 support
        support = make_support(v_min, v_max, num_atoms)
        self._support = support

        # Freeze refs
        actor = self.actor
        q1 = self.q1
        q2 = self.q2
        tau = config.tau
        target_entropy = self.target_entropy
        use_avg = q_aggregation == "avg"

        # ── Actor forward ────────────────────────────────────────────────
        def _actor_forward(actor_params, obs, key):
            mean, log_std = actor.apply(actor_params, obs)
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
            # Clamp log_probs to prevent -inf * 0 = NaN in cross-entropy
            q1_log_probs = jnp.maximum(jax.nn.log_softmax(q1_logits, axis=-1), -30.0)
            q2_log_probs = jnp.maximum(jax.nn.log_softmax(q2_logits, axis=-1), -30.0)
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
        policy_delay = config.policy_delay

        @jax.jit
        def update(state: TrainingState, batch: dict) -> tuple[TrainingState, dict]:
            key, k1, k2, k3 = jax.random.split(state.key, 4)
            new_count = state.update_count + 1

            # Critic (every step)
            q_params = (state.q1_params, state.q2_params)
            (_, critic_metrics), q_grads = jax.value_and_grad(
                _critic_loss, argnums=0, has_aux=True
            )(q_params, state.actor_params, state.target_q1_params,
              state.target_q2_params, state.log_alpha, batch, k1)
            q_updates, new_q_opt_state = optimizer.update(
                q_grads, state.q_opt_state, params=q_params)
            new_q1_params, new_q2_params = optax.apply_updates(q_params, q_updates)

            # Actor + Alpha (delayed by policy_delay)
            def _do_actor_alpha_update(args):
                ap, ao, la, aao, nq1, nq2, tq1, tq2 = args
                # Actor
                (_, am), ag = jax.value_and_grad(
                    _actor_loss, argnums=0, has_aux=True
                )(ap, nq1, nq2, la, batch, k2)
                au, nao = optimizer.update(ag, ao, params=ap)
                nap = optax.apply_updates(ap, au)
                # Alpha
                (_, alm), alg = jax.value_and_grad(
                    _alpha_loss, argnums=0, has_aux=True
                )(la, ap, batch, k3)
                alu, naao = alpha_optimizer.update(alg, aao, params=la)
                nla = optax.apply_updates(la, alu)
                # Polyak
                ntq1 = _soft_update(nq1, tq1)
                ntq2 = _soft_update(nq2, tq2)
                return nap, nao, nla, naao, ntq1, ntq2, {**am, **alm}

            def _skip_actor_alpha_update(args):
                ap, ao, la, aao, nq1, nq2, tq1, tq2 = args
                dummy = {"actor_loss": jnp.float32(0.0), "entropy": jnp.float32(0.0),
                         "alpha_loss": jnp.float32(0.0), "alpha": jnp.exp(la)}
                return ap, ao, la, aao, tq1, tq2, dummy

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

    def get_q_value(self, state, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Return scalar Q1 value (expected value from C51 logits)."""
        logits = self.q1.apply(state.q1_params, obs, action)
        return logits_to_q(logits, self._support)

    def init(self, key: jax.Array) -> TrainingState:
        key, k1, k3, k4 = jax.random.split(key, 4)

        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_action = jnp.zeros((1, self.action_dim))

        actor_params = self.actor.init(k1, dummy_obs)

        q1_params = self.q1.init(k3, dummy_obs, dummy_action)
        q2_params = self.q2.init(k4, dummy_obs, dummy_action)

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
