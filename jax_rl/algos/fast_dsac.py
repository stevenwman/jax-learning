"""FastDSAC — SAC with continuous Gaussian distributional critic + DEM.

Paper: FastDSAC (arXiv:2603.12612)

Changes from FastSAC:
  1. Gaussian distributional critic (mean + variance) replaces C51
  2. Dimension-wise Entropy Modulation (DEM) in actor
  3. Population diversity via per-env beta scaling
  4. Target entropy = 0 (not -dim(A))
  5. AdamW optimizer
"""

from typing import Any
import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.configs.fast_dsac_config import FastDSACConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.networks.encoders.mlp import MlpEncoder
from jax_rl.networks.heads.gaussian import GaussianHead
from jax_rl.networks.heads.q_gaussian import GaussianQHead
from jax_rl.networks.distributions import sample_gaussian


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
    mean_std1: jnp.ndarray  # EMA of batch mean of Q1 std (for ratio weighting)
    mean_std2: jnp.ndarray  # EMA of batch mean of Q2 std


class FastDSAC:
    """FastDSAC — SAC + Gaussian distributional critic + DEM."""

    def __init__(
        self,
        config: FastDSACConfig,
        obs_dim: int,
        action_dim: int,
        optimizer: optax.GradientTransformation,
        alpha_optimizer: optax.GradientTransformation,
        gamma: float = 0.99,
        handle_truncation: bool = True,
        beta_per_env: jax.Array | None = None,
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.handle_truncation = handle_truncation
        self.target_entropy = config.target_entropy

        # Networks
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
            dem=True,
        )
        self.actor_enc = MlpEncoder(enc_cfg)
        self.actor_head = GaussianHead(pol_cfg)
        critic_dim = getattr(config, 'critic_hidden_dim', None) or config.hidden_dim
        self.q1 = GaussianQHead(
            critic_dim, config.activation, config.q_layer_norm,
        )
        self.q2 = GaussianQHead(
            critic_dim, config.activation, config.q_layer_norm,
        )

        self.optimizer = optimizer
        self.alpha_optimizer = alpha_optimizer

        # Freeze refs for closures
        actor_enc = self.actor_enc
        actor_head = self.actor_head
        q1 = self.q1
        q2 = self.q2
        tau = config.tau
        tau_b = 0.005  # EMA rate for running mean_std (paper default)
        target_entropy = self.target_entropy
        dem_temp = config.dem_temperature
        bias = 1e-6  # small constant for division safety (paper: 0.000001)
        huber_delta = 50.0  # paper uses delta=50 for Huber loss

        # ── DEM weight computation ────────────────────────────────────────
        def _compute_dem_weights(dem_logits, beta=None):
            """Compute per-dimension exploration weights via temperature-scaled softmax.

            Args:
                dem_logits: [batch, action_dim] raw logits from actor head
                beta: [batch, 1] or scalar per-env scaling (population diversity)
            Returns:
                weights: [batch, action_dim] summing to action_dim per row
            """
            if beta is not None:
                scaled = dem_logits * beta / dem_temp
            else:
                scaled = dem_logits / dem_temp
            weights = jax.nn.softmax(scaled, axis=-1) * action_dim
            return weights

        # ── Actor forward with DEM ────────────────────────────────────────
        def _actor_forward(actor_params, obs, key, beta=None):
            enc_params, head_params = actor_params
            features = actor_enc.apply(enc_params, obs)
            mean, log_std, dem_logits = actor_head.apply(head_params, features)

            # Apply DEM: modulate std with per-dim weights
            dem_weights = _compute_dem_weights(dem_logits, beta)
            modulated_log_std = jnp.log(dem_weights) + log_std

            action, log_prob = sample_gaussian(mean, modulated_log_std, key, squash=True)
            return action, log_prob

        # ── Huber-based distributional critic loss (paper's actual formula) ──
        # NOTE: Despite being called "Gaussian distributional", the paper uses
        # Huber loss, NOT Gaussian NLL. The key insight from reading their source:
        # - No 1/variance anywhere (avoids the explosion that killed our NLL)
        # - Per-sample ratio weighting clamped [0.1, 10] (bounded gradients)
        # - Huber loss with delta=50 (bounded error for large TD differences)
        # - z.clamp(-3, 3) for target Q sampling (prevents extreme samples)
        def _critic_loss(q_params, actor_params, target_q1_params, target_q2_params,
                         log_alpha, batch, key, mean_std1, mean_std2):
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

            # Target Q: mean and std from target networks
            tq1_mean, tq1_std = q1.apply(target_q1_params, next_obs, next_action)
            tq2_mean, tq2_std = q2.apply(target_q2_params, next_obs, next_action)

            # Sample from target distribution with clamped noise (paper: z.clamp(-3, 3))
            key, z1_key, z2_key = jax.random.split(key, 3)
            z1 = jnp.clip(jax.random.normal(z1_key, tq1_mean.shape), -3.0, 3.0)
            z2 = jnp.clip(jax.random.normal(z2_key, tq2_mean.shape), -3.0, 3.0)
            next_q1_sample = tq1_mean + z1 * tq1_std
            next_q2_sample = tq2_mean + z2 * tq2_std

            # Min Q for target (paper: use_cdq=True)
            q_next = jnp.minimum(tq1_mean, tq2_mean)
            q_next_sample = jnp.where(tq1_mean < tq2_mean, next_q1_sample, next_q2_sample)

            # SAC entropy-adjusted targets
            effective_done = jnp.maximum(done, truncation)
            bootstrap = 1.0 - effective_done

            # target_q: for mean regression (uses expected Q)
            target_q = reward + bootstrap * gamma * (q_next - alpha * next_log_prob)
            # target_q_sample: for variance regression (uses sampled Q)
            target_q_sample = reward + bootstrap * gamma * (q_next_sample - alpha * next_log_prob)
            target_q = jax.lax.stop_gradient(target_q)
            target_q_sample = jax.lax.stop_gradient(target_q_sample)

            # Online Q predictions (mean and std)
            q1_mean, q1_std = q1.apply(q1_params_, obs, action)
            q2_mean, q2_std = q2.apply(q2_params_, obs, action)

            # Detached std for ratio computation (no gradient through denominator)
            q1_std_det = jax.lax.stop_gradient(jnp.maximum(q1_std, 0.0))
            q2_std_det = jax.lax.stop_gradient(jnp.maximum(q2_std, 0.0))

            # Per-sample ratio: mean_std² / (q_std² + bias), clamped [0.1, 10]
            # This bounds the per-sample gradient contribution
            ratio1 = jnp.clip(mean_std1 ** 2 / (q1_std_det ** 2 + bias), 0.1, 10.0)
            ratio2 = jnp.clip(mean_std2 ** 2 / (q2_std_det ** 2 + bias), 0.1, 10.0)

            # Huber loss for mean regression
            def _huber(pred, target):
                return optax.huber_loss(pred, target, delta=huber_delta)

            # Paper's loss: ratio * (huber_mean + std * (std_det² - huber_var) / (std_det + bias))
            def _critic_term(q_mean, q_std, q_std_det, ratio, target_q_, target_q_sample_):
                mean_term = _huber(q_mean, target_q_)
                var_term = q_std * (q_std_det ** 2 - _huber(jax.lax.stop_gradient(q_mean), target_q_sample_)) / (q_std_det + bias)
                return jnp.mean(ratio * (mean_term + var_term))

            q1_loss = _critic_term(q1_mean, q1_std, q1_std_det, ratio1, target_q, target_q_sample)
            q2_loss = _critic_term(q2_mean, q2_std, q2_std_det, ratio2, target_q, target_q_sample)

            # Update running mean_std (EMA)
            new_mean_std1 = jax.lax.stop_gradient(
                (1 - tau_b) * mean_std1 + tau_b * q1_std.mean()
            )
            new_mean_std2 = jax.lax.stop_gradient(
                (1 - tau_b) * mean_std2 + tau_b * q2_std.mean()
            )

            metrics = {
                "q1_mean": q1_mean.mean(),
                "q2_mean": q2_mean.mean(),
                "q1_std": q1_std.mean(),
                "q2_std": q2_std.mean(),
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "mean_std1": new_mean_std1,
                "mean_std2": new_mean_std2,
            }
            return q1_loss + q2_loss, metrics

        # ── Actor loss ────────────────────────────────────────────────────
        def _actor_loss(actor_params, q1_params_, q2_params_, log_alpha, batch, key):
            obs = batch["obs"]
            alpha = jnp.exp(log_alpha)

            action, log_prob = _actor_forward(actor_params, obs, key)
            q1_mean, _ = q1.apply(q1_params_, obs, action)
            q2_mean, _ = q2.apply(q2_params_, obs, action)
            min_q = jnp.minimum(q1_mean, q2_mean)

            loss = jnp.mean(alpha * log_prob - min_q)
            metrics = {
                "actor_loss": loss,
                "entropy": -log_prob.mean(),
            }
            return loss, metrics

        # ── Alpha loss ────────────────────────────────────────────────────
        def _alpha_loss(log_alpha, actor_params, batch, key):
            obs = batch["obs"]
            _, log_prob = _actor_forward(actor_params, obs, key)
            loss = jnp.exp(log_alpha) * jax.lax.stop_gradient(
                -log_prob - target_entropy
            ).mean()
            return loss, {"alpha_loss": loss, "alpha": jnp.exp(log_alpha)}

        # ── Polyak ────────────────────────────────────────────────────────
        def _soft_update(online, target):
            return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)

        # ── Full update step ──────────────────────────────────────────────
        @jax.jit
        def update(state: TrainingState, batch: dict) -> tuple[TrainingState, dict]:
            key, k1, k2, k3 = jax.random.split(state.key, 4)

            # Critic (pass mean_std for ratio weighting)
            q_params = (state.q1_params, state.q2_params)
            (_, critic_metrics), q_grads = jax.value_and_grad(
                _critic_loss, argnums=0, has_aux=True
            )(q_params, state.actor_params, state.target_q1_params,
              state.target_q2_params, state.log_alpha, batch, k1,
              state.mean_std1, state.mean_std2)
            q_updates, new_q_opt_state = optimizer.update(
                q_grads, state.q_opt_state, params=q_params)
            new_q1_params, new_q2_params = optax.apply_updates(q_params, q_updates)

            # Actor
            (_, actor_metrics), actor_grads = jax.value_and_grad(
                _actor_loss, argnums=0, has_aux=True
            )(state.actor_params, state.q1_params, state.q2_params,
              state.log_alpha, batch, k2)
            actor_updates, new_actor_opt_state = optimizer.update(
                actor_grads, state.actor_opt_state, params=state.actor_params
            )
            new_actor_params = optax.apply_updates(state.actor_params, actor_updates)

            # Alpha
            (_, alpha_metrics), alpha_grads = jax.value_and_grad(
                _alpha_loss, argnums=0, has_aux=True
            )(state.log_alpha, state.actor_params, batch, k3)
            alpha_updates, new_alpha_opt_state = alpha_optimizer.update(
                alpha_grads, state.alpha_opt_state, params=state.log_alpha
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
                mean_std1=critic_metrics["mean_std1"],
                mean_std2=critic_metrics["mean_std2"],
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
            """Eval/recording action — no population diversity beta."""
            enc_params, head_params = actor_params
            features = actor_enc.apply(enc_params, obs)
            mean, log_std, dem_logits = actor_head.apply(head_params, features)
            dem_weights = _compute_dem_weights(dem_logits)
            modulated_log_std = jnp.log(dem_weights) + log_std
            action, _ = sample_gaussian(mean, modulated_log_std, key, squash=True)
            return jax.lax.cond(deterministic, lambda: jnp.tanh(mean), lambda: action)

        @jax.jit
        def collect_action(
            actor_params: Any,
            obs: jax.Array,
            key: jax.Array,
            beta: jax.Array,
        ) -> jax.Array:
            """Training collection — with per-env population diversity beta.

            Args:
                beta: [num_envs, 1] per-env scaling factors for DEM exploration.
            """
            enc_params, head_params = actor_params
            features = actor_enc.apply(enc_params, obs)
            mean, log_std, dem_logits = actor_head.apply(head_params, features)
            dem_weights = _compute_dem_weights(dem_logits, beta)
            modulated_log_std = jnp.log(dem_weights) + log_std
            action, _ = sample_gaussian(mean, modulated_log_std, key, squash=True)
            return action

        self.update = update
        self.select_action = select_action
        self.collect_action = collect_action
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
            mean_std1=jnp.array(1.0),  # initialize to 1.0, will be updated via EMA
            mean_std2=jnp.array(1.0),
        )
