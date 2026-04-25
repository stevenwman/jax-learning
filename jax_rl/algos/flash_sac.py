"""FlashSAC — SAC with inverted residual blocks, BatchNorm, and weight norm.

Key differences from FastSAC:
  1. Inverted residual blocks with BatchNorm (FlashSACBlock)
  2. Weight normalization after each optimizer step
  3. Cross-batch (2B) forward passes for shared BatchNorm statistics
  4. FlashSAC-specific C51 projection (entropy inside gamma term)
  5. Update order: actor → temperature → critic → target EMA
  6. Target critics maintain independent batch_stats via train=True passes
  7. Polyak EMA only on learned params, NOT batch_stats

Reference: Kim et al. 2026 (Holiday-Robot/FlashSAC)
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.networks.flash_blocks import FlashSACActor, FlashSACCritic, normalize_weights
from jax_rl.networks.distributions import sample_gaussian
from jax_rl.utils.distributional import (
    make_support,
    logits_to_q,
    safe_log_softmax,
    cross_entropy_categorical,
)
from jax_rl.utils.polyak import soft_update as polyak_update


@flax.struct.dataclass
class NoiseState:
    noise: jnp.ndarray       # (num_envs, action_dim)
    count: jnp.ndarray       # (num_envs,)
    repeat_n: jnp.ndarray    # (num_envs,)


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
    update_count: jnp.ndarray
    # BatchNorm running stats
    actor_batch_stats: Any
    q1_batch_stats: Any
    q2_batch_stats: Any
    target_q1_batch_stats: Any
    target_q2_batch_stats: Any
    # Exploration noise state
    noise_state: Any
    # Adaptive reward-scaling state (G_r, G_r_max, G_mean, G_var, G_count).
    # Held here so orbax persists it across resume — matches reference
    # FlashSAC which saves reward_normalizer.pt as a first-class artifact.
    reward_norm_state: Any


class FlashSAC:
    """FlashSAC — SAC with inverted residual blocks, BatchNorm, and weight norm.

    Uses the same closure-based JIT pattern as FastSAC. See FastSAC docstring
    for rationale (JAX JIT cannot trace mutable self).
    """

    def __init__(
        self,
        config: FlashSACConfig,
        obs_dim: int,
        action_dim: int,
        optimizer: optax.GradientTransformation,
        alpha_optimizer: optax.GradientTransformation,
        gamma: float = 0.99,
        critic_obs_dim: int | None = None,
        num_envs: int = 1,
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim or obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.num_envs = num_envs

        # Entropy target: 0.5 * D * log(2*pi*e*sigma^2)
        self.target_entropy = 0.5 * action_dim * jnp.log(
            2 * jnp.pi * jnp.e * config.sigma_target ** 2
        )

        # Networks
        self.actor = FlashSACActor(
            hidden_dim=config.actor_hidden_dim,
            num_blocks=config.num_blocks,
            expansion=config.expansion,
            action_dim=action_dim,
        )
        self.q1 = FlashSACCritic(
            hidden_dim=config.critic_hidden_dim,
            num_blocks=config.num_blocks,
            expansion=config.expansion,
            num_atoms=config.num_atoms,
        )
        self.q2 = FlashSACCritic(
            hidden_dim=config.critic_hidden_dim,
            num_blocks=config.num_blocks,
            expansion=config.expansion,
            num_atoms=config.num_atoms,
        )

        self.optimizer = optimizer
        self.alpha_optimizer = alpha_optimizer

        # C51 support
        support = make_support(config.v_min, config.v_max, config.num_atoms)
        self._support = support

        # Freeze refs for closures
        actor = self.actor
        q1 = self.q1
        q2 = self.q2
        tau = config.tau
        target_entropy = self.target_entropy
        bc_alpha = config.bc_alpha
        n_step = config.n_step
        v_min = config.v_min
        v_max = config.v_max
        num_atoms = config.num_atoms
        weight_norm = config.weight_norm
        policy_delay = config.policy_delay
        gamma_val = gamma

        # ── Actor forward (2B cross-batch) ──────────────────────────────
        def _actor_forward_2b(actor_params, actor_bs, obs, next_obs, key):
            """Run actor on [obs; next_obs], return first-half actions/logprobs + updated BN stats."""
            obs_2b = jnp.concatenate([obs, next_obs], axis=0)
            (mean_2b, log_std_2b), new_actor_vars = actor.apply(
                {'params': actor_params, 'batch_stats': actor_bs},
                obs_2b, train=True, mutable=['batch_stats'],
            )
            new_actor_bs = new_actor_vars['batch_stats']
            action_2b, log_prob_2b = sample_gaussian(mean_2b, log_std_2b, key, squash=True)
            b = obs.shape[0]
            return action_2b[:b], log_prob_2b[:b], new_actor_bs

        def _actor_forward_eval(actor_params, actor_bs, obs, key):
            """Run actor in eval mode (single batch, no BN mutation)."""
            mean, log_std = actor.apply(
                {'params': actor_params, 'batch_stats': actor_bs},
                obs, train=False,
            )
            action, log_prob = sample_gaussian(mean, log_std, key, squash=True)
            return action, log_prob, mean

        # ── FlashSAC C51 projection ─────────────────────────────────────
        def _flash_project(target_log_probs, reward, done, alpha_log_prob):
            """FlashSAC C51 projection: entropy inside gamma term.

            target_log_probs: (B, num_atoms) log probabilities from min-Q target
            reward: (B,)
            done: (B,) terminated OR truncated (Brax convention from EpisodeWrapper)
            alpha_log_prob: (B,) = alpha * next_log_prob
            """
            delta_z = (v_max - v_min) / (num_atoms - 1)
            bin_values = support[None, :]  # (1, num_atoms)

            # FlashSAC: reward + gamma^n * (z - alpha*logp) * (1 - done)
            target_bin_values = (
                reward[:, None]
                + (gamma_val ** n_step)
                * (bin_values - alpha_log_prob[:, None])
                * (1.0 - done[:, None])
            )
            target_bin_values = jnp.clip(target_bin_values, v_min, v_max)

            # Standard C51 scatter
            b = (target_bin_values - v_min) / delta_z
            lo = jnp.floor(b).astype(jnp.int32)
            hi = jnp.clip(lo + 1, 0, num_atoms - 1)
            lo = jnp.clip(lo, 0, num_atoms - 1)

            frac = b - lo.astype(jnp.float32)
            target_probs = jnp.exp(target_log_probs)
            m_lo = target_probs * (1.0 - frac)
            m_hi = target_probs * frac

            lo_oh = jax.nn.one_hot(lo, num_atoms)
            hi_oh = jax.nn.one_hot(hi, num_atoms)
            projected = (
                jnp.einsum('bs,bsd->bd', m_lo, lo_oh)
                + jnp.einsum('bs,bsd->bd', m_hi, hi_oh)
            )
            return projected

        # ── Maybe normalize weights ─────────────────────────────────────
        def _maybe_normalize(params):
            if weight_norm:
                return normalize_weights(params)
            return params

        # ── Actor loss ──────────────────────────────────────────────────
        def _actor_loss(actor_params, actor_bs, q1_p, q1_bs, q2_p, q2_bs,
                        log_alpha, batch, key):
            obs = batch["obs"]
            next_obs = batch["next_obs"]
            critic_obs = batch["critic_obs"]
            alpha = jnp.exp(log_alpha)

            # Cross-batch actor forward
            action, log_prob, new_actor_bs = _actor_forward_2b(
                actor_params, actor_bs, obs, next_obs, key
            )

            # Critic in eval mode (don't pollute critic BN stats)
            q1_logits = q1.apply(
                {'params': q1_p, 'batch_stats': q1_bs},
                critic_obs, action, train=False,
            )
            q2_logits = q2.apply(
                {'params': q2_p, 'batch_stats': q2_bs},
                critic_obs, action, train=False,
            )
            q1_val = logits_to_q(q1_logits, support)
            q2_val = logits_to_q(q2_logits, support)
            min_q = jnp.minimum(q1_val, q2_val)

            loss = jnp.mean(alpha * log_prob - min_q)

            # Optional BC regularization
            if bc_alpha > 0:
                q_abs_mean = jax.lax.stop_gradient(jnp.abs(min_q).mean())
                bc_loss = jnp.mean((action - batch["action"]) ** 2)
                loss = loss + bc_alpha * q_abs_mean * bc_loss

            metrics = {
                "actor_loss": loss,
                "entropy": -log_prob.mean(),
            }
            return loss, (new_actor_bs, metrics)

        # ── Alpha loss ──────────────────────────────────────────────────
        def _alpha_loss(log_alpha, entropy):
            """Temperature loss. Uses entropy computed from actor update."""
            loss = jnp.exp(log_alpha) * jax.lax.stop_gradient(
                entropy - target_entropy
            )
            return loss, {"alpha_loss": loss, "alpha": jnp.exp(log_alpha)}

        # ── Critic loss ─────────────────────────────────────────────────
        def _critic_loss(q_params, q1_bs, q2_bs, tq1_p, tq1_bs, tq2_p, tq2_bs,
                         actor_params, actor_bs, log_alpha, batch, key):
            q1_p, q2_p = q_params
            obs = batch["critic_obs"]
            action = batch["action"]
            reward = batch["reward"].squeeze(-1)
            next_obs = batch["critic_next_obs"]
            actor_next_obs = batch["next_obs"]
            done = batch["done"].squeeze(-1)  # terminated OR truncated
            truncation = batch["truncation"].squeeze(-1)  # pure-timeout mask

            alpha = jnp.exp(log_alpha)

            # Next action from current actor (eval mode, no BN mutation)
            next_action, next_log_prob, _ = _actor_forward_eval(
                actor_params, actor_bs, actor_next_obs, key
            )

            alpha_log_prob = alpha * next_log_prob  # (B,)

            # Target critics: cross-batch (2B) with train=True for BN updates
            obs_2b = jnp.concatenate([obs, next_obs], axis=0)
            act_2b = jnp.concatenate([action, next_action], axis=0)

            tq1_logits_2b, new_tq1_vars = q1.apply(
                {'params': tq1_p, 'batch_stats': tq1_bs},
                obs_2b, act_2b, train=True, mutable=['batch_stats'],
            )
            tq2_logits_2b, new_tq2_vars = q2.apply(
                {'params': tq2_p, 'batch_stats': tq2_bs},
                obs_2b, act_2b, train=True, mutable=['batch_stats'],
            )
            new_tq1_bs = new_tq1_vars['batch_stats']
            new_tq2_bs = new_tq2_vars['batch_stats']

            b = obs.shape[0]
            # Next-obs half for target
            tq1_logits_next = tq1_logits_2b[b:]
            tq2_logits_next = tq2_logits_2b[b:]

            # Min-Q selection: pick FULL distribution from critic with lower expected Q
            tq1_val = logits_to_q(tq1_logits_next, support)
            tq2_val = logits_to_q(tq2_logits_next, support)
            use_q1 = (tq1_val < tq2_val)[:, None]  # (B, 1)

            tq1_log_probs = safe_log_softmax(tq1_logits_next)
            tq2_log_probs = safe_log_softmax(tq2_logits_next)
            target_log_probs = jnp.where(use_q1, tq1_log_probs, tq2_log_probs)

            # FlashSAC C51 projection
            projected = jax.lax.stop_gradient(
                _flash_project(target_log_probs, reward, done, alpha_log_prob)
            )

            # Online critics: cross-batch (2B) with train=True
            q1_logits_2b, new_q1_vars = q1.apply(
                {'params': q1_p, 'batch_stats': q1_bs},
                obs_2b, act_2b, train=True, mutable=['batch_stats'],
            )
            q2_logits_2b, new_q2_vars = q2.apply(
                {'params': q2_p, 'batch_stats': q2_bs},
                obs_2b, act_2b, train=True, mutable=['batch_stats'],
            )
            new_q1_bs = new_q1_vars['batch_stats']
            new_q2_bs = new_q2_vars['batch_stats']

            # First half (obs, action) for loss
            q1_logits = q1_logits_2b[:b]
            q2_logits = q2_logits_2b[:b]

            # Truncation mask: drop pure-timeout rows (matches SAC/TD3 Brax
            # convention). target used done = term|trunc to zero bootstrap on
            # both; the mask here prevents the r-only target from teaching
            # Q=r at timeout.
            mask = 1.0 - truncation
            q1_per_sample = cross_entropy_categorical(projected, q1_logits)
            q2_per_sample = cross_entropy_categorical(projected, q2_logits)
            q1_loss = jnp.mean(q1_per_sample * mask)
            q2_loss = jnp.mean(q2_per_sample * mask)

            q1_val = logits_to_q(q1_logits, support)
            q2_val = logits_to_q(q2_logits, support)

            metrics = {
                "q1_mean": q1_val.mean(),
                "q2_mean": q2_val.mean(),
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
            }
            return q1_loss + q2_loss, (new_q1_bs, new_q2_bs, new_tq1_bs, new_tq2_bs, metrics)

        # ── Full update step ────────────────────────────────────────────
        @jax.jit
        def update(state: TrainingState, batch: dict) -> tuple[TrainingState, dict]:
            key, k1, k2, k3 = jax.random.split(state.key, 4)
            new_count = state.update_count + 1

            # FlashSAC update order: actor → temp → critic → target EMA
            # (opposite from FastSAC which does critic first)

            do_actor_update = (new_count % policy_delay) == 0

            # ── Branch: actor + temperature update ──────────────────────
            def _do_actor_alpha(args):
                (a_params, a_bs, a_opt,
                 q1_p, q1_bs, q2_p, q2_bs,
                 log_alpha, alpha_opt) = args

                # Actor gradient
                actor_grad_fn = jax.value_and_grad(_actor_loss, argnums=0, has_aux=True)
                (_, (new_a_bs, actor_metrics)), a_grads = actor_grad_fn(
                    a_params, a_bs, q1_p, q1_bs, q2_p, q2_bs,
                    log_alpha, batch, k1,
                )
                a_updates, new_a_opt = optimizer.update(
                    a_grads, a_opt, params=a_params
                )
                new_a_params = optax.apply_updates(a_params, a_updates)
                new_a_params = _maybe_normalize(new_a_params)

                # Temperature gradient (uses entropy from actor)
                entropy = actor_metrics["entropy"]
                alpha_grad_fn = jax.value_and_grad(_alpha_loss, argnums=0, has_aux=True)
                (_, alpha_metrics), alpha_grads = alpha_grad_fn(log_alpha, entropy)
                alpha_updates, new_alpha_opt = alpha_optimizer.update(
                    alpha_grads, alpha_opt, params=log_alpha
                )
                new_log_alpha = optax.apply_updates(log_alpha, alpha_updates)

                return (new_a_params, new_a_bs, new_a_opt,
                        new_log_alpha, new_alpha_opt,
                        {**actor_metrics, **alpha_metrics})

            def _skip_actor_alpha(args):
                (a_params, a_bs, a_opt,
                 _q1_p, _q1_bs, _q2_p, _q2_bs,
                 log_alpha, alpha_opt) = args

                dummy = {
                    "actor_loss": jnp.float32(0.0),
                    "entropy": jnp.float32(0.0),
                    "alpha_loss": jnp.float32(0.0),
                    "alpha": jnp.exp(log_alpha),
                }
                return (a_params, a_bs, a_opt,
                        log_alpha, alpha_opt, dummy)

            (new_actor_params, new_actor_bs, new_actor_opt,
             new_log_alpha, new_alpha_opt,
             actor_alpha_metrics) = jax.lax.cond(
                do_actor_update,
                _do_actor_alpha,
                _skip_actor_alpha,
                (state.actor_params, state.actor_batch_stats, state.actor_opt_state,
                 state.q1_params, state.q1_batch_stats,
                 state.q2_params, state.q2_batch_stats,
                 state.log_alpha, state.alpha_opt_state),
            )

            # ── Critic update (every step, uses freshly-updated actor) ──
            q_params = (state.q1_params, state.q2_params)
            (_, (new_q1_bs, new_q2_bs, new_tq1_bs, new_tq2_bs, critic_metrics)), q_grads = (
                jax.value_and_grad(_critic_loss, argnums=0, has_aux=True)(
                    q_params,
                    state.q1_batch_stats, state.q2_batch_stats,
                    state.target_q1_params, state.target_q1_batch_stats,
                    state.target_q2_params, state.target_q2_batch_stats,
                    new_actor_params, new_actor_bs,
                    new_log_alpha, batch, k2,
                )
            )
            q_updates, new_q_opt = optimizer.update(
                q_grads, state.q_opt_state, params=q_params
            )
            new_q1_params, new_q2_params = optax.apply_updates(q_params, q_updates)
            new_q1_params = _maybe_normalize(new_q1_params)
            new_q2_params = _maybe_normalize(new_q2_params)

            # ── Target EMA (every step, NOT gated by policy_delay) ──────
            # Reference: update_target_network is called unconditionally
            new_tq1_params = polyak_update(new_q1_params, state.target_q1_params, tau)
            new_tq2_params = polyak_update(new_q2_params, state.target_q2_params, tau)

            new_state = state.replace(
                actor_params=new_actor_params,
                actor_opt_state=new_actor_opt,
                actor_batch_stats=new_actor_bs,
                q1_params=new_q1_params,
                q2_params=new_q2_params,
                q_opt_state=new_q_opt,
                q1_batch_stats=new_q1_bs,
                q2_batch_stats=new_q2_bs,
                target_q1_params=new_tq1_params,
                target_q2_params=new_tq2_params,
                target_q1_batch_stats=new_tq1_bs,
                target_q2_batch_stats=new_tq2_bs,
                log_alpha=new_log_alpha,
                alpha_opt_state=new_alpha_opt,
                key=key,
                update_count=new_count,
            )
            metrics = {**critic_metrics, **actor_alpha_metrics}
            return new_state, metrics

        # ── Select action ───────────────────────────────────────────────
        @jax.jit
        def select_action(
            actor_params: Any,
            obs: jax.Array,
            key: jax.Array,
            deterministic: bool = False,
            actor_batch_stats: Any = None,
        ) -> jax.Array:
            mean, log_std = actor.apply(
                {'params': actor_params, 'batch_stats': actor_batch_stats},
                obs, train=False,
            )
            action, _ = sample_gaussian(mean, log_std, key, squash=True)
            return jax.lax.cond(
                deterministic, lambda: jnp.tanh(mean), lambda: action
            )

        self.update = update
        self._select_action_jit = select_action
        self._actor_forward_2b = _actor_forward_2b

    def select_action(
        self,
        actor_params: Any,
        obs: jax.Array,
        key: jax.Array,
        deterministic: bool = False,
        actor_batch_stats: Any = None,
    ) -> jax.Array:
        """Select action. batch_stats can be passed explicitly or read from last init."""
        bs = actor_batch_stats if actor_batch_stats is not None else self._default_actor_bs
        return self._select_action_jit(actor_params, obs, key, deterministic, bs)

    def get_q_value(
        self, state: TrainingState, obs: jax.Array, action: jax.Array,
        critic_obs: jax.Array | None = None,
    ) -> jax.Array:
        """Return scalar Q1 value (expected value from C51 logits)."""
        q_obs = critic_obs if critic_obs is not None else obs
        logits = self.q1.apply(
            {'params': state.q1_params, 'batch_stats': state.q1_batch_stats},
            q_obs, action, train=False,
        )
        return logits_to_q(logits, self._support)

    def init(self, key: jax.Array) -> TrainingState:
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_critic_obs = jnp.zeros((1, self.critic_obs_dim))
        dummy_action = jnp.zeros((1, self.action_dim))

        # Init actor with batch_stats
        actor_vars = self.actor.init(k1, dummy_obs, train=False)
        actor_params = actor_vars['params']
        actor_batch_stats = actor_vars.get('batch_stats', {})

        # Init critics with batch_stats
        q1_vars = self.q1.init(k2, dummy_critic_obs, dummy_action, train=False)
        q2_vars = self.q2.init(k3, dummy_critic_obs, dummy_action, train=False)
        q1_params = q1_vars['params']
        q1_batch_stats = q1_vars.get('batch_stats', {})
        q2_params = q2_vars['params']
        q2_batch_stats = q2_vars.get('batch_stats', {})

        # Weight normalization after init
        if self.config.weight_norm:
            actor_params = normalize_weights(actor_params)
            q1_params = normalize_weights(q1_params)
            q2_params = normalize_weights(q2_params)

        # Optimizer states
        actor_opt_state = self.optimizer.init(actor_params)
        q_params = (q1_params, q2_params)
        q_opt_state = self.optimizer.init(q_params)
        log_alpha = jnp.array(jnp.log(self.config.alpha_init))
        alpha_opt_state = self.alpha_optimizer.init(log_alpha)

        # Noise state for exploration
        noise_state = NoiseState(
            noise=jnp.zeros((self.num_envs, self.action_dim)),
            count=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            repeat_n=jnp.ones((self.num_envs,), dtype=jnp.int32),
        )

        # Adaptive reward-scaling state. Persists across ckpt resume via
        # TrainingState. Matches reference FlashSAC's reward_normalizer.pt.
        from jax_rl.utils.reward_scaling import init_reward_norm
        reward_norm_state = init_reward_norm(self.num_envs)

        # Store default batch_stats for select_action
        self._default_actor_bs = actor_batch_stats

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
            actor_batch_stats=actor_batch_stats,
            q1_batch_stats=q1_batch_stats,
            q2_batch_stats=q2_batch_stats,
            target_q1_batch_stats=q1_batch_stats,
            target_q2_batch_stats=q2_batch_stats,
            noise_state=noise_state,
            reward_norm_state=reward_norm_state,
        )
