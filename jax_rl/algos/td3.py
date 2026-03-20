"""TD3 (Twin Delayed Deep Deterministic Policy Gradient).

Paper: https://arxiv.org/abs/1802.09477 (Fujimoto et al., 2018)
Reference: Spinning Up, CleanRL

Three key tricks over DDPG:
  1. Twin Q-networks — take min to reduce overestimation
  2. Delayed policy updates — update actor every `policy_delay` critic steps
  3. Target policy smoothing — add clipped noise to target actions

Simpler than SAC: no entropy, no alpha, deterministic policy.

Loss reference: .context/td3_plan.md
"""

from typing import Any
import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.configs.td3_config import TD3Config
from jax_rl.configs.networks_config import EncoderConfig
from jax_rl.networks.encoders.mlp import MlpEncoder
from jax_rl.networks.heads.deterministic import DeterministicHead
from jax_rl.networks.heads.q_head import QHead


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
    update_count: jnp.ndarray  # scalar int — tracks critic updates for policy_delay


class TD3:
    """TD3 algorithm with delayed policy updates and target smoothing."""

    def __init__(
        self,
        config: TD3Config,
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
        self.q1 = QHead(critic_dim, config.activation, config.q_layer_norm)
        self.q2 = QHead(critic_dim, config.activation, config.q_layer_norm)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Freeze refs for closures
        actor_enc = self.actor_enc
        actor_head = self.actor_head
        q1 = self.q1
        q2 = self.q2
        tau = config.tau
        policy_delay = config.policy_delay
        target_noise_std = config.target_noise_std
        noise_clip = config.noise_clip

        # ── Actor forward ────────────────────────────────────────────────
        def _actor_forward(actor_params, obs):
            """Returns deterministic action in [-1, 1]."""
            enc_params, head_params = actor_params
            features = actor_enc.apply(enc_params, obs)
            return actor_head.apply(head_params, features)

        # ── Critic loss ──────────────────────────────────────────────────
        def _critic_loss(q_params, target_actor_params, target_q1_params,
                         target_q2_params, batch, key):
            q1_params_, q2_params_ = q_params
            obs = batch["obs"]
            action = batch["action"]
            reward = batch["reward"].squeeze(-1)
            next_obs = batch["next_obs"]
            done = batch["done"].squeeze(-1)
            truncation = batch["truncation"].squeeze(-1)

            # Target policy smoothing: deterministic action + clipped noise
            next_action = _actor_forward(target_actor_params, next_obs)
            noise = jnp.clip(
                jax.random.normal(key, next_action.shape) * target_noise_std,
                -noise_clip, noise_clip,
            )
            next_action = jnp.clip(next_action + noise, -1.0, 1.0)

            # Target Q (min of twin targets — no entropy term, unlike SAC)
            tq1_val = q1.apply(target_q1_params, next_obs, next_action)
            tq2_val = q2.apply(target_q2_params, next_obs, next_action)
            min_tq = jnp.minimum(tq1_val, tq2_val)
            target = reward + self.gamma * (1.0 - done) * min_tq

            # Online Q predictions
            q1_val = q1.apply(q1_params_, obs, action)
            q2_val = q2.apply(q2_params_, obs, action)

            # TD error with truncation masking (same as SAC)
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

        # ── Actor loss ───────────────────────────────────────────────────
        def _actor_loss(actor_params, q1_params_):
            """Maximize Q1(s, μ(s)). Uses Q1 only (original TD3)."""
            obs = batch_ref[0]  # will be set via closure trick below
            action = _actor_forward(actor_params, obs)
            q1_val = q1.apply(q1_params_, obs, action)
            loss = -jnp.mean(q1_val)
            return loss, {"actor_loss": loss}

        # ── Polyak soft update ───────────────────────────────────────────
        def _soft_update(online, target):
            return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)

        # ── Full update step ─────────────────────────────────────────────
        # We use a closure-free approach: actor_loss_fn takes obs explicitly
        def _actor_loss_fn(actor_params, q1_params_, obs):
            action = _actor_forward(actor_params, obs)
            q1_val = q1.apply(q1_params_, obs, action)
            loss = -jnp.mean(q1_val)
            return loss, {"actor_loss": loss}

        @jax.jit
        def update(state: TrainingState, batch: dict) -> tuple[TrainingState, dict]:
            key, k1 = jax.random.split(state.key)
            new_count = state.update_count + 1

            # ── Critic update (every step) ────────────────────────────────
            q_params = (state.q1_params, state.q2_params)
            (_, critic_metrics), q_grads = jax.value_and_grad(
                _critic_loss, argnums=0, has_aux=True
            )(q_params, state.target_actor_params, state.target_q1_params,
              state.target_q2_params, batch, k1)
            q_updates, new_q_opt_state = critic_optimizer.update(
                q_grads, state.q_opt_state, params=q_params
            )
            new_q1_params, new_q2_params = optax.apply_updates(q_params, q_updates)

            # ── Actor update (delayed — every policy_delay steps) ─────────
            def _do_actor_update(args):
                actor_params, actor_opt_state, q1_p, nq1, nq2, ta, tq1, tq2 = args
                obs = batch["obs"]
                (_, actor_metrics), actor_grads = jax.value_and_grad(
                    _actor_loss_fn, argnums=0, has_aux=True
                )(actor_params, q1_p, obs)
                actor_updates, new_actor_opt_state = actor_optimizer.update(
                    actor_grads, actor_opt_state, params=actor_params
                )
                new_actor_params = optax.apply_updates(actor_params, actor_updates)
                # Polyak update targets (only when actor updates)
                new_ta = _soft_update(new_actor_params, ta)
                new_tq1 = _soft_update(nq1, tq1)
                new_tq2 = _soft_update(nq2, tq2)
                return (new_actor_params, new_actor_opt_state,
                        new_ta, new_tq1, new_tq2, actor_metrics)

            def _skip_actor_update(args):
                actor_params, actor_opt_state, _, _, _, ta, tq1, tq2 = args
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
                 state.q1_params,  # use pre-update Q1 for actor gradient
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
            """Return action. Adds Gaussian noise when not deterministic."""
            action = _actor_forward(actor_params, obs)
            noisy_action = action + jax.random.normal(key, action.shape) * exploration_noise
            noisy_action = jnp.clip(noisy_action, -1.0, 1.0)
            return jax.lax.cond(deterministic, lambda: action, lambda: noisy_action)

        self.update = update
        self.select_action = select_action
        self._actor_forward = _actor_forward

    def init(self, key: jax.Array) -> TrainingState:
        """Initialize parameters and optimizer states."""
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_action = jnp.zeros((1, self.action_dim))

        # Actor params = (encoder_params, head_params)
        enc_params = self.actor_enc.init(k1, dummy_obs)
        dummy_features = jnp.zeros((1, self.config.hidden_dim[-1]))
        head_params = self.actor_head.init(k2, dummy_features)
        actor_params = (enc_params, head_params)

        # Q params (twin)
        q1_params = self.q1.init(k3, dummy_obs, dummy_action)
        q2_params = self.q2.init(k4, dummy_obs, dummy_action)

        # Opt states
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
