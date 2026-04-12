"""SAC (Soft Actor-Critic) algorithm.

Matches MuJoCo Playground reference config:
  - Twin Q-networks with Polyak target update (tau=0.005)
  - Auto-tuned temperature (log_alpha, separate optimizer)
  - State-dependent std via softplus + min_std (NOT exp(log_std))
  - stop_grad on entropy in alpha loss (prevents gradient bleed into policy)
  - Truncation masking in Q-loss (for auto-reset envs like Playground)
  - Layer norm in Q-networks (q_layer_norm=True)

Loss reference: .context/sac_plan.md
"""

from typing import Any
import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.configs.sac_config import SACConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.networks.builders import Actor
from jax_rl.networks.heads.q_head import QHead
from jax_rl.networks.distributions import sample_gaussian


@flax.struct.dataclass
class TrainingState:
    actor_params: Any
    actor_opt_state: optax.OptState
    q1_params: Any
    q2_params: Any
    q_opt_state: Any          # single optimizer shared for q1+q2
    target_q1_params: Any
    target_q2_params: Any
    log_alpha: jnp.ndarray    # scalar; alpha = exp(log_alpha)
    alpha_opt_state: optax.OptState
    key: jax.Array


class SAC:
    """SAC algorithm with auto-tuned temperature."""

    def __init__(
        self,
        config: SACConfig,
        obs_dim: int,
        action_dim: int,
        optimizer: optax.GradientTransformation,  # shared lr for actor + Q
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

        # Networks — use builders for actor (encoder swappable), QHead directly for critics
        enc_cfg = EncoderConfig(
            obs_dim=obs_dim,
            hidden_dim=config.hidden_dim,
            activation=config.activation,
        )
        pol_cfg = PolicyHeadConfig(
            action_dim=action_dim,
            state_dependent_std=True,    # SAC always uses state-dependent std
            min_std=0.001,
            squash=True,
        )
        self.actor = Actor(enc_cfg, pol_cfg)
        critic_dim = config.critic_hidden_dim or config.hidden_dim
        self.q1 = QHead(critic_dim, config.activation, config.q_layer_norm)
        self.q2 = QHead(critic_dim, config.activation, config.q_layer_norm)

        self.optimizer = optimizer
        self.alpha_optimizer = alpha_optimizer

        # Freeze refs for closures
        actor = self.actor
        q1 = self.q1
        q2 = self.q2
        tau = config.tau
        target_entropy = self.target_entropy

        # ── Actor forward (used in losses and select_action) ──────────────
        def _actor_forward(actor_params, obs, key):
            """Returns (action, log_prob) using reparameterization."""
            mean, log_std = actor.apply(actor_params, obs)
            action, log_prob = sample_gaussian(mean, log_std, key, squash=True)
            return action, log_prob

        # ── Q-value forward ───────────────────────────────────────────────
        def _q_values(q1_params, q2_params, obs, action):
            """Returns (q1, q2) scalars, shape (batch,)."""
            return q1.apply(q1_params, obs, action), q2.apply(q2_params, obs, action)

        def _target_q_values(tq1, tq2, obs, action):
            return q1.apply(tq1, obs, action), q2.apply(tq2, obs, action)

        # ── Critic loss ───────────────────────────────────────────────────
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

            # Bootstrap: next action + log_prob under current policy (actor sees 48d obs)
            next_action, next_log_prob = _actor_forward(actor_params, next_obs, key)

            # Target Q (min of twin targets — critic sees privileged obs)
            tq1_val, tq2_val = _target_q_values(
                target_q1_params, target_q2_params, critic_next_obs, next_action
            )
            min_tq = jnp.minimum(tq1_val, tq2_val)
            next_v = min_tq - alpha * next_log_prob
            target = reward + self.gamma * (1.0 - done) * next_v

            # Online Q predictions (critic sees privileged obs)
            q1_val = q1.apply(q1_params_, critic_obs, action)
            q2_val = q2.apply(q2_params_, critic_obs, action)

            # TD error, masked at truncation boundaries (auto-reset envs)
            q1_err = q1_val - jax.lax.stop_gradient(target)
            q2_err = q2_val - jax.lax.stop_gradient(target)
            mask = 1.0 - truncation
            q1_loss = 0.5 * jnp.mean((q1_err * mask) ** 2)
            q2_loss = 0.5 * jnp.mean((q2_err * mask) ** 2)

            metrics = {
                "q1_mean": q1_val.mean(),
                "q2_mean": q2_val.mean(),
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "td_target_mean": target.mean(),
            }
            return q1_loss + q2_loss, metrics

        # ── Actor loss ────────────────────────────────────────────────────
        def _actor_loss(actor_params, q1_params_, q2_params_, log_alpha, batch, key):
            obs = batch["obs"]
            critic_obs = batch["critic_obs"]
            alpha = jnp.exp(log_alpha)

            action, log_prob = _actor_forward(actor_params, obs, key)
            q1_val = q1.apply(q1_params_, critic_obs, action)
            q2_val = q2.apply(q2_params_, critic_obs, action)
            min_q = jnp.minimum(q1_val, q2_val)

            loss = jnp.mean(alpha * log_prob - min_q)
            metrics = {
                "actor_loss": loss,
                "entropy": -log_prob.mean(),
                "log_prob_mean": log_prob.mean(),
            }
            return loss, metrics

        # ── Alpha loss ────────────────────────────────────────────────────
        def _alpha_loss(log_alpha, actor_params, batch, key):
            obs = batch["obs"]
            _, log_prob = _actor_forward(actor_params, obs, key)
            # stop_grad prevents alpha gradient from flowing into policy params
            loss = jnp.exp(log_alpha) * jax.lax.stop_gradient(
                -log_prob - target_entropy
            ).mean()
            return loss, {"alpha_loss": loss, "alpha": jnp.exp(log_alpha)}

        # ── Polyak soft update ────────────────────────────────────────────
        def _soft_update(online, target):
            return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)

        # ── Full update step ──────────────────────────────────────────────
        @jax.jit
        def update(state: TrainingState, batch: dict) -> tuple[TrainingState, dict]:
            key, k1, k2, k3 = jax.random.split(state.key, 4)

            # Critic update
            q_params = (state.q1_params, state.q2_params)
            (_, critic_metrics), q_grads = jax.value_and_grad(
                _critic_loss, argnums=0, has_aux=True
            )(q_params, state.actor_params, state.target_q1_params,
              state.target_q2_params, state.log_alpha, batch, k1)
            q_updates, new_q_opt_state = optimizer.update(
                q_grads, state.q_opt_state, params=q_params)
            new_q1_params, new_q2_params = optax.apply_updates(q_params, q_updates)

            # Actor update
            (_, actor_metrics), actor_grads = jax.value_and_grad(
                _actor_loss, argnums=0, has_aux=True
            )(state.actor_params, state.q1_params, state.q2_params,
              state.log_alpha, batch, k2)
            actor_updates, new_actor_opt_state = optimizer.update(
                actor_grads, state.actor_opt_state, params=state.actor_params
            )
            new_actor_params = optax.apply_updates(state.actor_params, actor_updates)

            # Alpha update
            (_, alpha_metrics), alpha_grads = jax.value_and_grad(
                _alpha_loss, argnums=0, has_aux=True
            )(state.log_alpha, state.actor_params, batch, k3)
            alpha_updates, new_alpha_opt_state = alpha_optimizer.update(
                alpha_grads, state.alpha_opt_state, params=state.log_alpha
            )
            new_log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)

            # Polyak target update
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
            """Return tanh-squashed action. Deterministic = tanh(mean)."""
            mean, log_std = actor.apply(actor_params, obs)
            action, _ = sample_gaussian(mean, log_std, key, squash=True)
            return jax.lax.cond(deterministic, lambda: jnp.tanh(mean), lambda: action)

        self.update = update
        self.select_action = select_action
        self._actor_forward = _actor_forward

    def get_q_value(self, state: TrainingState, obs: jax.Array, action: jax.Array,
                    critic_obs: jax.Array | None = None) -> jax.Array:
        """Return scalar Q1 value for (obs, action). Used for Q diagnostics."""
        q_obs = critic_obs if critic_obs is not None else obs
        return self.q1.apply(state.q1_params, q_obs, action)

    def init(self, key: jax.Array) -> TrainingState:
        """Initialize parameters and optimizer states."""
        key, k1, k3, k4 = jax.random.split(key, 4)

        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_critic_obs = jnp.zeros((1, self.critic_obs_dim))
        dummy_action = jnp.zeros((1, self.action_dim))

        # Actor params (single init via composed Actor module)
        actor_params = self.actor.init(k1, dummy_obs)

        # Q params (twin, identical structure but separate init — uses critic obs dim)
        q1_params = self.q1.init(k3, dummy_critic_obs, dummy_action)
        q2_params = self.q2.init(k4, dummy_critic_obs, dummy_action)

        # Opt states
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
        )
