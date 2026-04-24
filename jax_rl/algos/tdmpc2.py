"""TD-MPC2 algorithm.

Networks, loss, MPPI planner, update fns. Pure math — no env knowledge.

All HPs verified against /tmp/tdmpc2/. See .superpowers/specs/2026-04-21-tdmpc2-design.md
for the full paper-audit trail.
"""
from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_rl.utils.simnorm import simnorm
from jax_rl.utils.twohot import two_hot_inv, two_hot_ce_loss
from jax_rl.utils.qscale import QScaleState, qscale_apply


# ------------------ Activations ------------------

def mish(x):
    """Mish activation: x * tanh(softplus(x))."""
    return x * jnp.tanh(nn.activation.softplus(x))


# ------------------ Building blocks ------------------

class NormedLinear(nn.Module):
    """Dense → LayerNorm → Mish. Matches source common/layers.py NormedLinear.

    Optional dropout applied AFTER Mish (default 0.0 — TD-MPC2 uses dropout only on Q heads).
    """
    features: int
    dropout: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=self.features,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        x = nn.LayerNorm()(x)
        x = mish(x)
        if self.dropout > 0:
            x = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(x)
        return x


class Encoder(nn.Module):
    """h(obs) → z with SimNorm output.

    Arch: num_layers × NormedLinear(enc_dim) → Dense(latent_dim) → SimNorm.
    Source: /tmp/tdmpc2/tdmpc2/common/layers.py:enc(), config.yaml num_enc_layers=2, enc_dim=256.
    """
    enc_dim: int
    num_layers: int
    latent_dim: int
    simnorm_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(self.num_layers):
            x = NormedLinear(features=self.enc_dim)(x)
        x = nn.Dense(
            features=self.latent_dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        return simnorm(x, V=self.simnorm_dim)


class Dynamics(nn.Module):
    """d(z, a) → z' with SimNorm output.

    Arch: 2 × NormedLinear(mlp_dim) → Dense(latent_dim) → SimNorm.
    """
    mlp_dim: int
    latent_dim: int
    simnorm_dim: int

    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], axis=-1)
        x = NormedLinear(features=self.mlp_dim)(x)
        x = NormedLinear(features=self.mlp_dim)(x)
        x = nn.Dense(
            features=self.latent_dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        return simnorm(x, V=self.simnorm_dim)


class Reward(nn.Module):
    """R(z, a) → reward logits over num_bins (two-hot target).

    Arch: 2 × NormedLinear(mlp_dim) → Dense(num_bins) with **zero-init output kernel**.
    Source: /tmp/tdmpc2/tdmpc2/common/world_model.py:31.
    """
    mlp_dim: int
    num_bins: int

    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], axis=-1)
        x = NormedLinear(features=self.mlp_dim)(x)
        x = NormedLinear(features=self.mlp_dim)(x)
        # Zero-init output kernel (source common/world_model.py:31)
        return nn.Dense(
            features=self.num_bins,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)


class QHead(nn.Module):
    """Single Q head: NormedLinear (w/ dropout) → NormedLinear → Dense(num_bins).

    Dropout is applied ONLY on the first hidden layer (source world_model.py:30 passes
    dropout=cfg.dropout to Q's mlp() call; dynamics/reward/policy call mlp() without it).
    Final Dense layer is zero-initialized (source world_model.py:32).
    """
    mlp_dim: int
    num_bins: int
    dropout: float

    @nn.compact
    def __call__(self, z, a, deterministic: bool):
        x = jnp.concatenate([z, a], axis=-1)
        x = NormedLinear(
            features=self.mlp_dim,
            dropout=self.dropout,
            deterministic=deterministic,
        )(x)
        x = NormedLinear(features=self.mlp_dim)(x)  # no dropout on second layer
        return nn.Dense(
            features=self.num_bins,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)


class QEnsemble(nn.Module):
    """num_q Q heads via vmap-over-params.

    Output shape: (num_q, batch, num_bins).
    Matches source `nn.ParameterList` semantics via Flax's `linen.vmap` with
    `variable_axes={'params': 0}`.
    """
    mlp_dim: int
    num_bins: int
    num_q: int
    dropout: float

    @nn.compact
    def __call__(self, z, a, deterministic: bool):
        VmappedQ = nn.vmap(
            QHead,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            axis_size=self.num_q,
        )
        return VmappedQ(
            mlp_dim=self.mlp_dim,
            num_bins=self.num_bins,
            dropout=self.dropout,
        )(z, a, deterministic)


# ------------------ Policy-prior helpers ------------------

def bound_log_std(raw: jax.Array, log_std_min: float, log_std_max: float) -> jax.Array:
    """Tanh-based mapping: low + 0.5·(high−low)·(tanh(raw)+1).

    Source: /tmp/tdmpc2/tdmpc2/common/math.py:13-14. NOT a hard clamp.
    """
    return log_std_min + 0.5 * (log_std_max - log_std_min) * (jnp.tanh(raw) + 1.0)


def squash_log_prob_correction(a: jax.Array) -> jax.Array:
    """Jacobian correction for tanh squash: Σ log(relu(1 - a²) + 1e-6) over last dim.

    Returns a NON-POSITIVE scalar (log of quantities ≤ 1). Caller subtracts it:
        log_prob_post = log_prob_pre - squash_log_prob_correction(action)
    Since correction ≤ 0, log_prob_post ≥ log_prob_pre — squashed density concentrates
    on [-1,1]^D as expected. The relu + 1e-6 floor is load-bearing — naive `1 - tanh²`
    hits zero at saturation (|a| → 1) and produces -inf in log.
    Source: /tmp/tdmpc2/tdmpc2/common/math.py `squash()`.
    """
    return jnp.sum(jnp.log(jax.nn.relu(1.0 - a ** 2) + 1e-6), axis=-1)


def gaussian_log_prob(x: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Standard Gaussian log-prob, summed over last dim.

    -0.5 · Σ [((x - mean)/std)² + 2·log_std + log(2π)]
    """
    return -0.5 * jnp.sum(
        ((x - mean) / jnp.exp(log_std)) ** 2 + 2.0 * log_std + jnp.log(2 * jnp.pi),
        axis=-1,
    )


# ------------------ Policy prior ------------------

class PolicyPrior(nn.Module):
    """π(z) → tanh-squashed reparameterized Gaussian action.

    Returns (action, extras) where extras exposes pre/post-squash log-probs separately:
      - log_prob_pre: Gaussian log-prob of the pre-squash sample (used by scaled_entropy).
      - log_prob_post: Jacobian-corrected log-prob of the squashed action (policy log-prob).

    Arch: 2 × NormedLinear(mlp_dim) → Dense(2·action_dim, trunc_normal 0.02, zero bias).
    log_std output is tanh-bounded into [log_std_min, log_std_max]. Squash uses the
    relu + 1e-6 floor for numerical safety.
    """
    mlp_dim: int
    action_dim: int
    log_std_min: float
    log_std_max: float

    @nn.compact
    def __call__(self, z, key):
        x = NormedLinear(features=self.mlp_dim)(z)
        x = NormedLinear(features=self.mlp_dim)(x)
        out = nn.Dense(
            features=2 * self.action_dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        mean, raw_log_std = jnp.split(out, 2, axis=-1)
        log_std = bound_log_std(raw_log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)
        eps = jax.random.normal(key, mean.shape)
        pre = mean + std * eps
        action = jnp.tanh(pre)
        log_prob_pre = gaussian_log_prob(pre, mean, log_std)
        log_prob_post = log_prob_pre - squash_log_prob_correction(action)
        return action, {
            "pre": pre,
            "mean": mean,
            "log_std": log_std,
            "log_prob_pre": log_prob_pre,
            "log_prob_post": log_prob_post,
        }


# ------------------ Latent rollout helper ------------------

def compute_all_latents(
    wm_params,
    obs_0: jax.Array,
    actions: jax.Array,
    *,
    encoder: "Encoder",
    dynamics: "Dynamics",
) -> jax.Array:
    """Encode obs_0 then roll dynamics forward H steps.

    Args:
        wm_params: dict with keys 'encoder', 'dynamics' (Flax param trees).
        obs_0: (B, obs_dim) — initial observation.
        actions: (H, B, action_dim) — actions to roll forward.
        encoder: Encoder module instance.
        dynamics: Dynamics module instance.

    Returns:
        zs of shape (H+1, B, latent_dim):
          zs[0]     = encoder(obs_0)
          zs[h+1]   = dynamics(zs[h], actions[h])  for h = 0..H-1
    """
    z_0 = encoder.apply(wm_params["encoder"], obs_0)  # (B, latent_dim)

    def scan_body(z, a):
        z_next = dynamics.apply(wm_params["dynamics"], z, a)
        return z_next, z_next

    _, zs_rest = jax.lax.scan(scan_body, z_0, actions)  # (H, B, latent_dim)
    zs = jnp.concatenate([z_0[None, :], zs_rest], axis=0)  # (H+1, B, latent_dim)
    return zs


# ------------------ TD target ------------------

def compute_td_target(
    *,
    target_params,
    online_wm_params,
    policy_params,
    batch,
    cfg,
    key: jax.Array,
    encoder: "Encoder",
    policy_net: "PolicyPrior",
    q_ensemble_net: "QEnsemble",
) -> jax.Array:
    """Compute TD target for world-model value loss.

    Path (source tdmpc2.py:253-264):
      1. next_z_h = encoder_online(obs[h+1])  — ONLINE encoder, NOT dynamics rollout
      2. a_next_h = sample π_online(next_z_h) — ONLINE policy, reparameterized sample
      3. Q_target_all = q_ensemble_target(next_z_h, a_next_h) — ALL heads
      4. Random-permute head indices, take 2. Softmax each, decode each via two_hot_inv(..., apply_symexp=True).
      5. Elementwise min across the 2 decoded scalars.
      6. target_q = reward + γ · (1 - terminated) · q_min
      7. stop-grad on the whole thing.

    Shapes:
      obs       (H+1, B, obs_dim)
      rewards   (H, B, 1)
      dones     (H, B, 1)   -- `terminated` flags
      Returns:  (H, B, 1)
    """
    H = cfg.horizon
    gamma = cfg.discount
    obs_next = batch["obs"][1:]         # (H, B, obs_dim)
    rewards = batch["rewards"]          # (H, B, 1)
    terminated = batch["dones"]         # (H, B, 1)
    B = obs_next.shape[1]

    # 1. Online encoder on real next obs. Flatten (H, B) for batched apply; reshape back.
    obs_next_flat = obs_next.reshape(H * B, -1)                # (H*B, obs_dim)
    next_z_flat = encoder.apply(online_wm_params["encoder"], obs_next_flat)
    next_z = next_z_flat.reshape(H, B, -1)                     # (H, B, latent_dim)
    next_z = jax.lax.stop_gradient(next_z)

    # 2. Online policy sample — independent PRNG per (h, b)
    key_pi, key_q = jax.random.split(key, 2)
    pi_keys = jax.random.split(key_pi, H * B).reshape(H, B, 2)

    def _sample_action(z, k):
        a, _ = policy_net.apply(policy_params, z[None, :], k)
        return a[0]

    # vmap over B (inner), then over H (outer)
    a_next = jax.vmap(jax.vmap(_sample_action, in_axes=(0, 0)), in_axes=(0, 0))(next_z, pi_keys)
    a_next = jax.lax.stop_gradient(a_next)                     # (H, B, action_dim)

    # 3. Target Q ensemble, all heads. Flatten (H, B) again.
    next_z_flat = next_z.reshape(H * B, -1)
    a_next_flat = a_next.reshape(H * B, -1)
    q_logits_flat = q_ensemble_net.apply(
        target_params["q_ensemble"],
        next_z_flat, a_next_flat,
        deterministic=True,
    )  # (num_q, H*B, num_bins)
    q_logits = q_logits_flat.reshape(cfg.num_q, H, B, cfg.num_bins)

    # 4-5. Subsample 2 random heads (one permutation per update step, not per element).
    #      Decode FIRST, then elementwise min.
    perm = jax.random.permutation(key_q, cfg.num_q)[:2]        # (2,)
    q_selected = q_logits[perm]                                 # (2, H, B, num_bins)
    probs = jax.nn.softmax(q_selected, axis=-1)                 # (2, H, B, num_bins)
    decoded = two_hot_inv(probs, cfg.vmin, cfg.vmax, cfg.num_bins,
                          apply_symexp=True)                    # (2, H, B, 1)
    q_min = jnp.min(decoded, axis=0)                            # (H, B, 1)

    # 6. Bootstrap target; terminated zeros the Q term, truncated does NOT.
    target = rewards + gamma * (1.0 - terminated) * q_min       # (H, B, 1)
    return jax.lax.stop_gradient(target)


# ------------------ World-model loss ------------------

def world_model_loss(
    params,
    target_params,
    policy_params,
    batch,
    cfg,
    key: jax.Array,
    *,
    encoder: "Encoder",
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
):
    """Compute world-model loss = consistency + reward + value (all unmasked, per-H normalized).

    Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:270-320.

    Load-bearing:
      - Per-H normalization on consistency and reward (/H); value normalized by (H*num_q)
      - Rho discount rho^h applied per step
      - All three losses UNMASKED (source does not mask by terminated or truncated)
      - Consistency target = stop-grad(encoder_online(obs[h+1]))
      - TD target path uses encoder_online(obs[h+1]), not dynamics rollout (handled in compute_td_target)

    Returns (L_total, metrics) for jax.value_and_grad with has_aux=True.
    """
    H = cfg.horizon
    obs_seq = batch["obs"]          # (H+1, B, obs_dim)
    actions = batch["actions"]      # (H, B, action_dim)
    rewards = batch["rewards"]      # (H, B, 1)
    B = obs_seq.shape[1]

    # 1. Encode all observed steps with online encoder, stop-grad → consistency targets.
    obs_flat = obs_seq.reshape((H + 1) * B, -1)
    z_targets_flat = encoder.apply(params["encoder"], obs_flat)
    z_targets = z_targets_flat.reshape(H + 1, B, -1)
    z_targets = jax.lax.stop_gradient(z_targets)               # (H+1, B, latent_dim)

    # 2. Forward-roll dynamics from z_0 (gradient-carrying).
    z_0 = encoder.apply(params["encoder"], obs_seq[0])         # (B, latent_dim), grad

    def scan_body(z, a):
        z_next = dynamics.apply(params["dynamics"], z, a)
        r_logits = reward_net.apply(params["reward"], z, a)
        q_logits = q_ensemble_net.apply(
            params["q_ensemble"], z, a, deterministic=True,
        )
        return z_next, (z_next, r_logits, q_logits)

    _, (z_pred_seq, r_logits_seq, q_logits_seq) = jax.lax.scan(
        scan_body, z_0, actions
    )
    # z_pred_seq: (H, B, latent_dim) — predicted ẑ_{1..H}
    # r_logits_seq: (H, B, num_bins)
    # q_logits_seq: (H, num_q, B, num_bins)

    # Rho discount per step
    rho_powers = cfg.rho ** jnp.arange(H)                      # (H,)

    # 3. Consistency loss: MSE(ẑ_{h+1}, sg(z_targets_{h+1})) for h=0..H-1, UNMASKED.
    z_target_next = z_targets[1:]                              # (H, B, latent_dim)
    consistency_per_h = jnp.mean((z_pred_seq - z_target_next) ** 2, axis=-1)  # (H, B)
    # Normalize: sum over H with rho discount, mean over B, divide by H.
    L_consistency = (rho_powers[:, None] * consistency_per_h).mean(axis=-1).sum() / H

    # 4. Reward loss: CE(r̂_logits_h, twohot(symlog(r_h))), UNMASKED. Normalized by H.
    reward_ce_per_h = two_hot_ce_loss(
        r_logits_seq, rewards,
        cfg.vmin, cfg.vmax, cfg.num_bins, apply_symlog=True,
    )  # (H, B)
    L_reward = (rho_powers[:, None] * reward_ce_per_h).mean(axis=-1).sum() / H

    # 5. Value loss: CE over all num_q heads, summed over heads, rho-discounted, normalized by (H*num_q).
    target_q = compute_td_target(
        target_params=target_params,
        online_wm_params=params,
        policy_params=jax.lax.stop_gradient(policy_params),
        batch=batch, cfg=cfg, key=key,
        encoder=encoder, policy_net=policy_net, q_ensemble_net=q_ensemble_net,
    )  # (H, B, 1)

    # q_logits_seq: (H, num_q, B, num_bins). vmap value_ce over num_q axis (axis=1).
    def value_ce_per_head(q_logits_for_head):
        # q_logits_for_head: (H, B, num_bins)
        return two_hot_ce_loss(
            q_logits_for_head, target_q,
            cfg.vmin, cfg.vmax, cfg.num_bins, apply_symlog=True,
        )  # (H, B)

    ce_all_heads = jax.vmap(value_ce_per_head, in_axes=1, out_axes=1)(q_logits_seq)  # (H, num_q, B)
    value_ce_summed = ce_all_heads.sum(axis=1)                 # (H, B)
    L_value = (rho_powers[:, None] * value_ce_summed).mean(axis=-1).sum() / (H * cfg.num_q)

    # 6. Aggregate
    L_total = (
        cfg.consistency_coef * L_consistency
        + cfg.reward_coef * L_reward
        + cfg.value_coef * L_value
    )
    metrics = {
        "L_consistency_raw": L_consistency,
        "L_reward_raw": L_reward,
        "L_value_raw": L_value,
        "L_world_total": L_total,
    }
    return L_total, metrics


# ------------------ Policy loss ------------------

def compute_scaled_entropy(log_prob_pre: jax.Array, action_dim: int) -> jax.Array:
    """Single-task simplification of source's scaled_entropy formula.

    Source: /tmp/tdmpc2/tdmpc2/common/world_model.py:176-183. Pre-squash log_prob × action_dim.

    scaled_entropy = -log_prob_pre * action_dim
    """
    return -log_prob_pre * action_dim


def policy_loss(
    online_params,       # {"policy": ..., "q_ensemble": ...}
    qscale_state: "QScaleState",
    zs_detached: jax.Array,  # (H+1, B, latent_dim), already stop-gradded
    cfg,
    key: jax.Array,
    *,
    policy_net: "PolicyPrior",
    q_ensemble_net: "QEnsemble",
):
    """Policy loss. Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:219-227.

    Formula (NOTE SIGN):
      L_policy = -(1/(H+1)) · Σ_h rho^h · mean_over_batch(entropy_coef·scaled_entropy + qs_scaled)

    OUTER NEGATIVE wraps both entropy bonus AND Q term.
    scaled_entropy uses pre-squash log_prob (D6's extras["log_prob_pre"]).
    Q path: detached online q_ensemble params, avg-of-2 random heads, decode-per-head.

    Returns (L_policy, metrics). metrics["a_t0"] exposed for Q-scale update reuse.
    """
    H_plus_1, B = zs_detached.shape[0], zs_detached.shape[1]
    rho_powers = cfg.rho ** jnp.arange(H_plus_1)  # (H+1,)

    # Sample action at each (h, b) with independent PRNG.
    key_pi, key_q = jax.random.split(key, 2)
    pi_keys = jax.random.split(key_pi, H_plus_1 * B).reshape(H_plus_1, B, 2)

    def _sample(z, k):
        a, extras = policy_net.apply(online_params["policy"], z[None, :], k)
        return a[0], extras["log_prob_pre"][0]

    a_all, log_prob_pre_all = jax.vmap(
        jax.vmap(_sample, in_axes=(0, 0)), in_axes=(0, 0)
    )(zs_detached, pi_keys)
    # a_all: (H+1, B, action_dim); log_prob_pre_all: (H+1, B)

    scaled_entropy = compute_scaled_entropy(log_prob_pre_all, cfg.action_dim)  # (H+1, B)

    # Detached online Q ensemble on (zs, a_all), subsample 2 heads, decode, average.
    z_flat = zs_detached.reshape(H_plus_1 * B, -1)
    a_flat = a_all.reshape(H_plus_1 * B, -1)
    q_logits_flat = q_ensemble_net.apply(
        jax.lax.stop_gradient(online_params["q_ensemble"]),
        z_flat, a_flat,
        deterministic=True,
    )  # (num_q, (H+1)*B, num_bins)
    q_logits = q_logits_flat.reshape(cfg.num_q, H_plus_1, B, cfg.num_bins)

    perm = jax.random.permutation(key_q, cfg.num_q)[:2]
    q_selected = q_logits[perm]  # (2, H+1, B, num_bins)
    q_probs = jax.nn.softmax(q_selected, axis=-1)
    q_decoded = two_hot_inv(
        q_probs, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True,
    )  # (2, H+1, B, 1)
    q_avg = q_decoded.mean(axis=0).squeeze(-1)  # (H+1, B)

    qs_scaled = qscale_apply(qscale_state, q_avg)  # (H+1, B)

    # Source formula: pi_loss = (-(entropy_coef · scaled_entropy + qs).mean(dim=(1,2)) * rho).mean()
    per_step = cfg.entropy_coef * scaled_entropy + qs_scaled  # (H+1, B)
    per_step_mean_over_batch = per_step.mean(axis=-1)  # (H+1,)
    weighted = rho_powers * per_step_mean_over_batch  # (H+1,)
    L_policy = -weighted.mean()  # outer negative, single .mean() → 1/(H+1)

    metrics = {
        "L_policy": L_policy,
        "scaled_entropy_mean": scaled_entropy.mean(),
        "q_avg_mean": q_avg.mean(),
        "a_t0": jax.lax.stop_gradient(a_all[0]),  # (B, action_dim)
    }
    return L_policy, metrics


# ------------------ MPPI core ------------------


def mppi_rollout(
    plan_params,
    z_0: jax.Array,           # (N, latent_dim) — N = num_samples per env
    actions_seq: jax.Array,   # (horizon, N, action_dim)
    cfg,
    key: jax.Array,
    *,
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
) -> jax.Array:
    """Score N candidate action trajectories by rolling the world model forward.

    Score formula (source tdmpc2.py:128-136):
      score(τ) = Σ_{h=0..H-1} γ^h · r̂(z_h, a_h) + γ^H · Q_avg_of_2(z_H, π(z_H))

    Reward/Q decoded through two_hot_inv(apply_symexp=True).
    Online Q ensemble used (NOT target Q — MPPI is inference-time).

    Returns: (N,) predicted returns.
    """
    N = z_0.shape[0]
    gamma = cfg.discount

    def step(carry, a):
        z, discount_factor, G = carry
        r_logits = reward_net.apply(plan_params["reward"], z, a)
        r_probs = jax.nn.softmax(r_logits, axis=-1)
        r_hat = two_hot_inv(
            r_probs, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True,
        ).squeeze(-1)  # (N,)
        G_next = G + discount_factor * r_hat
        z_next = dynamics.apply(plan_params["dynamics"], z, a)
        return (z_next, discount_factor * gamma, G_next), None

    initial_carry = (z_0, jnp.ones(N), jnp.zeros(N))
    (z_final, discount_final, G_reward), _ = jax.lax.scan(step, initial_carry, actions_seq)

    # Terminal Q bootstrap: γ^H · Q_avg_of_2(z_final, π(z_final))
    key_pi, key_q = jax.random.split(key, 2)
    a_terminal, _ = policy_net.apply(plan_params["policy"], z_final, key_pi)
    q_logits = q_ensemble_net.apply(
        plan_params["q_ensemble"], z_final, a_terminal, deterministic=True,
    )  # (num_q, N, num_bins)
    perm = jax.random.permutation(key_q, cfg.num_q)[:2]
    q_selected = q_logits[perm]  # (2, N, num_bins)
    q_probs = jax.nn.softmax(q_selected, axis=-1)
    q_decoded = two_hot_inv(
        q_probs, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True,
    ).squeeze(-1)  # (2, N)
    q_terminal = q_decoded.mean(axis=0)  # (N,)

    return G_reward + discount_final * q_terminal


def mppi_iteration(
    mean: jax.Array,          # (horizon, action_dim)
    std: jax.Array,           # (horizon, action_dim)
    plan_params,
    z_0: jax.Array,           # (latent_dim,) — single env
    pi_trajs: jax.Array,      # (horizon, num_pi_trajs, action_dim)
    cfg,
    key: jax.Array,
    *,
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One MPPI iteration: sample, score, select elites, update (mean, std).

    Args:
        mean, std: current sampling distribution, shape (horizon, action_dim).
        pi_trajs: (horizon, num_pi_trajs, action_dim) — policy-seeded samples (from F2).
        z_0: single-env starting latent.

    Returns (new_mean, new_std, elite_actions, weights):
        new_mean, new_std: (horizon, action_dim)
        elite_actions: (horizon, num_elites, action_dim) — kept for F4 Gumbel action selection
        weights: (num_elites,) softmax-over-scores
    """
    key_sample, key_rollout = jax.random.split(key, 2)

    # Gaussian samples: i.i.d. across (horizon, num_samples - num_pi_trajs, action_dim)
    N_gauss = cfg.num_samples - cfg.num_pi_trajs
    eps = jax.random.normal(key_sample, (cfg.horizon, N_gauss, cfg.action_dim))
    gauss_actions = jnp.clip(
        mean[:, None, :] + std[:, None, :] * eps,
        -1.0, 1.0,
    )
    # Stack: [pi_trajs, gauss_actions] → (horizon, num_samples, action_dim)
    actions = jnp.concatenate([pi_trajs, gauss_actions], axis=1)

    # Broadcast z_0 to (num_samples, latent_dim) and score
    z_0_broadcast = jnp.broadcast_to(z_0, (cfg.num_samples,) + z_0.shape)
    scores = mppi_rollout(
        plan_params, z_0_broadcast, actions, cfg, key_rollout,
        dynamics=dynamics, reward_net=reward_net,
        q_ensemble_net=q_ensemble_net, policy_net=policy_net,
    )  # (num_samples,)

    # Elite selection: top-K by score
    _, elite_idx = jax.lax.top_k(scores, cfg.num_elites)  # (num_elites,)
    elite_scores = scores[elite_idx]
    elite_actions = actions[:, elite_idx, :]  # (horizon, num_elites, action_dim)

    # Elite weights: softmax with temperature (numerically stable)
    max_score = elite_scores.max()
    exp_scores = jnp.exp((elite_scores - max_score) / cfg.mppi_temperature)
    weights = exp_scores / (exp_scores.sum() + 1e-9)  # (num_elites,)

    # Update mean/std (elite-weighted)
    new_mean = (weights[None, :, None] * elite_actions).sum(axis=1)  # (horizon, action_dim)
    var = (weights[None, :, None] * (elite_actions - new_mean[:, None, :]) ** 2).sum(axis=1)
    new_std = jnp.clip(jnp.sqrt(var), cfg.mppi_min_std, cfg.mppi_max_std)

    return new_mean, new_std, elite_actions, weights
