"""DEPRECATED: being decomposed into tdmpc2/{networks,losses,mppi,agent}.py.

Do not add new code here. New symbols go in the focused modules under
jax_rl/algos/tdmpc2/. This file is deleted in Task 6 once empty."""
from typing import Any, Optional

import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.utils.twohot import two_hot_inv, two_hot_ce_loss
from jax_rl.utils.qscale import QScaleState, qscale_apply, qscale_update

from jax_rl.algos.tdmpc2.networks import (
    Encoder, Dynamics, Reward, QEnsemble, PolicyPrior,
    compute_scaled_entropy,
)


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
    # True termination only — exclude truncations. Our EpisodeWrapper sets done=1
    # at episode timeout (training.py:121), so batch["dones"] = terminated | truncated.
    # Source stores `terminated` separately and uses ONLY it in (1 - terminated)
    # (tdmpc2/tdmpc2.py:258, common/buffer.py:98-106). For non-episodic DMC tasks
    # this means bootstrap is preserved across timeouts. We recover the same mask
    # via dones - truncations (clipped to {0, 1}).
    terminated = jnp.clip(batch["dones"] - batch["truncations"], 0.0, 1.0)  # (H, B, 1)
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

    # Split key: TD-target sampling (pi/Q heads) + dropout for online Q ensemble.
    # Source world_model.py:30 passes dropout=cfg.dropout to _Qs; world_model.py:74-79
    # leaves _Qs in train mode during _update so dropout is ACTIVE in value-loss path.
    key_td, key_drop = jax.random.split(key, 2)
    drop_keys = jax.random.split(key_drop, H)                  # (H, 2) — one per scan step

    # 1. Encode all observed steps with online encoder, stop-grad → consistency targets.
    obs_flat = obs_seq.reshape((H + 1) * B, -1)
    z_targets_flat = encoder.apply(params["encoder"], obs_flat)
    z_targets = z_targets_flat.reshape(H + 1, B, -1)
    z_targets = jax.lax.stop_gradient(z_targets)               # (H+1, B, latent_dim)

    # 2. Forward-roll dynamics from z_0 (gradient-carrying).
    z_0 = encoder.apply(params["encoder"], obs_seq[0])         # (B, latent_dim), grad

    def scan_body(z, scan_inputs):
        a, dk = scan_inputs
        z_next = dynamics.apply(params["dynamics"], z, a)
        r_logits = reward_net.apply(params["reward"], z, a)
        q_logits = q_ensemble_net.apply(
            params["q_ensemble"], z, a,
            deterministic=False,
            rngs={"dropout": dk},
        )
        return z_next, (z_next, r_logits, q_logits)

    _, (z_pred_seq, r_logits_seq, q_logits_seq) = jax.lax.scan(
        scan_body, z_0, (actions, drop_keys)
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
        batch=batch, cfg=cfg, key=key_td,
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
    # Tier B diagnostics
    # Per-h consistency MSE (mean over batch) — pinpoints world-model drift along horizon
    consistency_per_h_batch = consistency_per_h.mean(axis=-1)  # (H,)
    # Max abs reward observed in this batch — saturation watch (vs cfg.vmax)
    max_reward_observed = jnp.max(jnp.abs(rewards))

    metrics = {
        "L_consistency_raw": L_consistency,
        "L_reward_raw": L_reward,
        "L_value_raw": L_value,
        "L_world_total": L_total,
        # Per-h consistency: index h is MSE between predicted ẑ_{h+1} and target z_{h+1}
        "consistency_per_h": consistency_per_h_batch,  # (H,) — caller picks indices
        "max_reward_observed": max_reward_observed,
    }
    return L_total, metrics


# ------------------ Policy loss ------------------

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
    # key_qdrop: dropout RNG for online Q forward — source's _detach_Qs is in train
    # mode during update_pi (world_model.py:74-80, tdmpc2.py:267/314), so dropout
    # IS active on the policy-loss Q evaluation. We previously hard-set
    # deterministic=True which silently disabled this regularizer.
    key_pi, key_q, key_qdrop = jax.random.split(key, 3)
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
        deterministic=False,
        rngs={"dropout": key_qdrop},
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

    # Tier B diagnostic: full entropy = mean(-log_prob_post). Use to detect collapse.
    # log_prob_post is post-tanh-squash log-prob; -mean is the entropy estimate.
    pi_entropy = -log_prob_pre_all.mean()  # pre-squash entropy proxy

    metrics = {
        "L_policy": L_policy,
        "scaled_entropy_mean": scaled_entropy.mean(),
        "pi_entropy": pi_entropy,           # Tier B: collapse detector
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
    # Source: tdmpc2.py:191 — exp(temperature * delta), NOT exp(delta / temperature).
    # With cfg.mppi_temperature=0.5, the multiply form is softer (coef 0.5);
    # the divide form would be 4× sharper (coef 2.0) and over-concentrates on top elite.
    max_score = elite_scores.max()
    exp_scores = jnp.exp(cfg.mppi_temperature * (elite_scores - max_score))
    weights = exp_scores / (exp_scores.sum() + 1e-9)  # (num_elites,)

    # Update mean/std (elite-weighted)
    new_mean = (weights[None, :, None] * elite_actions).sum(axis=1)  # (horizon, action_dim)
    var = (weights[None, :, None] * (elite_actions - new_mean[:, None, :]) ** 2).sum(axis=1)
    new_std = jnp.clip(jnp.sqrt(var), cfg.mppi_min_std, cfg.mppi_max_std)

    return new_mean, new_std, elite_actions, weights


def sample_pi_trajectories(
    plan_params,
    z_0: jax.Array,          # (latent_dim,) — single env
    cfg,
    key: jax.Array,
    *,
    dynamics: "Dynamics",
    policy_net: "PolicyPrior",
) -> jax.Array:
    """Seed MPPI population with num_pi_trajs trajectories from the policy prior.

    Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:155-165.

    Loop structure — CRITICAL:
      For h = 0..horizon-1: sample a_h = π(z_h)
      For h = 0..horizon-2: advance z_{h+1} = dynamics(z_h, a_h)
      Net: `horizon` policy samples, `horizon - 1` dynamics advances.
      A naive range(horizon) both-loop over-advances latent one step and produces OOD
      final samples.

    Returns: (horizon, num_pi_trajs, action_dim) — pi-seeded action sequences.
    """
    N = cfg.num_pi_trajs
    # Broadcast z_0 to (N, latent_dim)
    z = jnp.broadcast_to(z_0, (N,) + z_0.shape)

    def step(carry, h_idx):
        z, key = carry
        key, sk = jax.random.split(key)
        a, _ = policy_net.apply(plan_params["policy"], z, sk)  # (N, action_dim)
        # Advance dynamics ONLY when h < horizon - 1 (last step: sample a but don't advance).
        z_advanced = dynamics.apply(plan_params["dynamics"], z, a)
        z_next = jnp.where(h_idx < cfg.horizon - 1, z_advanced, z)
        return (z_next, key), a

    _, actions = jax.lax.scan(step, (z, key), jnp.arange(cfg.horizon))
    # actions: (horizon, N, action_dim)
    return actions


def init_mppi_mean(prev_mean: jax.Array, t0: jax.Array,
                    horizon: int, action_dim: int) -> jax.Array:
    """Warm-start MPPI mean for a single env.

    If t0 is True, return zeros. Otherwise shift: new[:-1] = prev[1:], new[-1] = 0.
    Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:167-168.

    Args:
        prev_mean: (horizon, action_dim) — last optimized mean.
        t0: scalar bool — True on new episode.
    """
    shifted = jnp.concatenate([prev_mean[1:], jnp.zeros((1, action_dim))], axis=0)
    return jnp.where(t0, jnp.zeros_like(shifted), shifted)


def init_mppi_mean_batched(prev_mean: jax.Array, t0: jax.Array,
                            horizon: int, action_dim: int) -> jax.Array:
    """Per-env warm-start.

    Args:
        prev_mean: (num_envs, horizon, action_dim)
        t0:        (num_envs,) bool
    Returns:
        (num_envs, horizon, action_dim)
    """
    return jax.vmap(
        lambda p, t: init_mppi_mean(p, t, horizon, action_dim)
    )(prev_mean, t0)


# ------------------ F4: plan() + gumbel_sample_elite + plan_batched ------------------


def gumbel_sample_elite(
    key: jax.Array,
    weights: jax.Array,        # (num_elites,)
    elite_actions: jax.Array,  # (horizon, num_elites, action_dim)
) -> jax.Array:
    """Sample a single elite trajectory via Gumbel-softmax argmax, return its t=0 action.

    Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:201-205.
    NOT elite-weighted mean — a SINGLE elite is sampled, its t=0 action is returned.

    Returns: (action_dim,)
    """
    logits = jnp.log(weights + 1e-9)
    gumbels = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-9) + 1e-9)
    idx = jnp.argmax(logits + gumbels)
    return elite_actions[0, idx]


def plan(
    plan_params,
    z_0: jax.Array,          # (latent_dim,)
    prev_mean: jax.Array,    # (horizon, action_dim)
    t0: jax.Array,           # scalar bool
    cfg,
    key: jax.Array,
    eval_mode: bool = False,
    *,
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
) -> tuple[jax.Array, jax.Array]:
    """Full MPPI planner for a single env. Source: /tmp/tdmpc2/tdmpc2/tdmpc2.py:plan().

    Returns (action, new_prev_mean):
      action: (action_dim,) — action to execute this step.
      new_prev_mean: (horizon, action_dim) — stored for next step's warm-start.
    """
    key_pi_seed, key_iter, key_elite, key_noise = jax.random.split(key, 4)

    # 1. Warm-start mean + init std
    mean = init_mppi_mean(prev_mean, t0, cfg.horizon, cfg.action_dim)
    std = jnp.full((cfg.horizon, cfg.action_dim), cfg.mppi_max_std)

    # 2. Sample pi-seed trajectories once at start (source samples once, reuses)
    pi_trajs = sample_pi_trajectories(
        plan_params, z_0, cfg, key_pi_seed,
        dynamics=dynamics, policy_net=policy_net,
    )

    # 3. MPPI iteration loop. iterations = base + 2 if action_dim >= 20 (source line 35).
    iterations = cfg.mppi_iterations + (2 if cfg.action_dim >= 20 else 0)
    iter_keys = jax.random.split(key_iter, iterations)

    def iter_body(carry, k_i):
        mean_c, std_c = carry
        new_mean, new_std, elite_actions, weights = mppi_iteration(
            mean_c, std_c, plan_params, z_0, pi_trajs, cfg, k_i,
            dynamics=dynamics, reward_net=reward_net,
            q_ensemble_net=q_ensemble_net, policy_net=policy_net,
        )
        return (new_mean, new_std), (elite_actions, weights)

    (final_mean, final_std), (all_elites, all_weights) = jax.lax.scan(
        iter_body, (mean, std), iter_keys
    )
    # Use final iteration's elites + weights
    elite_actions = all_elites[-1]  # (horizon, num_elites, action_dim)
    weights = all_weights[-1]       # (num_elites,)

    # 4. Gumbel-sample single elite, take t=0 action
    action = gumbel_sample_elite(key_elite, weights, elite_actions)  # (action_dim,)

    # 5. Exploration noise (skip in eval_mode): std[0] * ε
    noise = jax.random.normal(key_noise, (cfg.action_dim,)) * final_std[0]
    action = jnp.where(eval_mode, action, action + noise)
    action = jnp.clip(action, -1.0, 1.0)

    return action, final_mean


def make_plan_batched(
    *,
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
):
    """Return a jit+vmap'd `plan` over num_envs.

    Module instances are closed over (cannot be vmapped). Returned callable has signature:
        plan_fn(plan_params, z_0_b, prev_mean_b, t0_b, cfg, keys, eval_mode) → (actions, new_prev_means)
    where _b suffix = per-env leading dim.

    JIT is essential: without it, every env-step re-traces the full MPPI scan (measured
    ~800ms/call CPU, ~10-30s/call under GPU contention). With jit, 2nd+ calls drop to <10ms.
    """
    def single_plan(plan_params, z_0, prev_mean, t0, cfg, key, eval_mode):
        return plan(
            plan_params, z_0, prev_mean, t0, cfg, key, eval_mode=eval_mode,
            dynamics=dynamics, reward_net=reward_net,
            q_ensemble_net=q_ensemble_net, policy_net=policy_net,
        )
    # vmap over (z_0, prev_mean, t0, key); plan_params/cfg/eval_mode shared
    vmapped = jax.vmap(single_plan, in_axes=(None, 0, 0, 0, None, 0, None))
    # JIT with cfg (idx 4) and eval_mode (idx 6) as static. Positional-call compatible.
    # (cfg is frozen/hashable; eval_mode is bool; both required to be static for jit cache.)
    return jax.jit(vmapped, static_argnums=(4, 6))


# ------------------ Training state ------------------

@flax.struct.dataclass
class TDMPC2State:
    """TD-MPC2 training state.

    Holds online params, target params (encoder/dynamics/reward/Q — NO target policy),
    optimizer states, Q-scale tracker, per-env MPPI prev_mean, RNG key, step counter.
    Immutable pytree — use `state.replace(...)` to update.
    """
    encoder_params: Any
    dynamics_params: Any
    reward_params: Any
    q_ensemble_params: Any
    policy_params: Any
    encoder_target_params: Any
    dynamics_target_params: Any
    reward_target_params: Any
    q_ensemble_target_params: Any
    world_model_opt_state: Any
    policy_opt_state: Any
    qscale: QScaleState
    prev_mean: jax.Array         # (num_envs, horizon, action_dim)
    key: jax.Array
    step: jax.Array              # scalar int32


# ------------------ Optimizers ------------------


def build_world_model_optimizer(cfg):
    """Single Adam with per-param-group LR via Optax multi_transform.

    Group "a" (scaled LR): encoder → lr · enc_lr_scale.
    Group "b" (default LR): dynamics + reward + q_ensemble (+ task_emb in C-mode).
    Grad clipping by global norm applied before the Adam transform.

    Returns an Optax GradientTransformation. Caller provides `wm_params` tree at init.
    """
    tx_a = optax.adam(cfg.lr * cfg.enc_lr_scale)
    tx_b = optax.adam(cfg.lr)

    def label_fn(params):
        """Label each LEAF based on top-level key 'encoder' vs anything else."""
        def _label(path, _leaf):
            # path[0] is DictKey(key=<top-level-name>); handle both DictKey and bare-str
            top = path[0].key if hasattr(path[0], "key") else str(path[0])
            return "a" if top == "encoder" else "b"
        return jax.tree_util.tree_map_with_path(_label, params)

    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.multi_transform({"a": tx_a, "b": tx_b}, label_fn),
    )


def build_policy_optimizer(cfg):
    """Single Adam for policy prior, with grad clip.

    Source tdmpc2.py:32 — uses Adam with eps=1e-5 for policy optimizer.
    """
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adam(cfg.lr, eps=cfg.pi_optim_eps),
    )


# ------------------ Update step factory ------------------

def make_update_step(
    cfg,
    wm_optimizer,
    policy_optimizer,
    *,
    encoder: "Encoder",
    dynamics: "Dynamics",
    reward_net: "Reward",
    q_ensemble_net: "QEnsemble",
    policy_net: "PolicyPrior",
):
    """Factory: returns a jit'd update_step closed over (non-pytree) optimizers and modules.

    Order of operations:
      1. World model forward+backward (encoder + dynamics + reward + Q ensemble).
      2. Recompute detached latents from updated world model.
      3. Policy forward+backward using detached latents.
      4. Q-scale update from t=0 avg-of-2 Q values.
      5. Target EMA update on encoder/dynamics/reward/q_ensemble (NOT policy).
      6. Pack new state.
    """
    @jax.jit
    def update_step(state: TDMPC2State, batch: dict):
        key_wm, key_pol, key_qscale, key_next = jax.random.split(state.key, 4)

        # 1. World model update
        wm_params = {
            "encoder": state.encoder_params,
            "dynamics": state.dynamics_params,
            "reward": state.reward_params,
            "q_ensemble": state.q_ensemble_params,
        }
        target_params = {
            "encoder": state.encoder_target_params,
            "dynamics": state.dynamics_target_params,
            "reward": state.reward_target_params,
            "q_ensemble": state.q_ensemble_target_params,
        }

        def wm_loss_fn(p):
            return world_model_loss(
                p, target_params, state.policy_params,
                batch, cfg, key_wm,
                encoder=encoder, dynamics=dynamics,
                reward_net=reward_net, q_ensemble_net=q_ensemble_net,
                policy_net=policy_net,
            )

        (wm_loss_val, wm_metrics), wm_grads = jax.value_and_grad(
            wm_loss_fn, has_aux=True,
        )(wm_params)
        # Tier B: world model grad norm (pre-clip; optimizer chain clips at cfg.grad_clip_norm)
        wm_grad_norm = optax.global_norm(wm_grads)
        wm_updates, new_wm_opt_state = wm_optimizer.update(
            wm_grads, state.world_model_opt_state, wm_params,
        )
        wm_params_new = optax.apply_updates(wm_params, wm_updates)

        # 2. Detached latents from updated world model (for policy loss)
        zs_detached = jax.lax.stop_gradient(
            compute_all_latents(
                wm_params_new, batch["obs"][0], batch["actions"],
                encoder=encoder, dynamics=dynamics,
            )
        )  # (H+1, B, latent_dim)

        # 3. Policy update
        pol_params_in = {
            "policy": state.policy_params,
            "q_ensemble": wm_params_new["q_ensemble"],  # stop-gradded inside policy_loss
        }

        def pol_loss_fn(p):
            return policy_loss(
                p, state.qscale, zs_detached, cfg, key_pol,
                policy_net=policy_net, q_ensemble_net=q_ensemble_net,
            )

        (pol_loss_val, pol_metrics), pol_grads = jax.value_and_grad(
            pol_loss_fn, has_aux=True,
        )(pol_params_in)
        # Tier B: policy grad norm (pre-clip)
        pi_grad_norm = optax.global_norm(pol_grads["policy"])

        # Zero the q_ensemble grads — stop_gradient inside policy_loss means they're zero
        # already, but we zero explicitly to be safe and avoid any no-op optimizer state churn.
        pol_grads = {
            "policy": pol_grads["policy"],
            "q_ensemble": jax.tree_util.tree_map(jnp.zeros_like, pol_grads["q_ensemble"]),
        }
        pol_updates, new_pol_opt_state = policy_optimizer.update(
            pol_grads, state.policy_opt_state, pol_params_in,
        )
        pol_params_new_full = optax.apply_updates(pol_params_in, pol_updates)
        policy_params_new = pol_params_new_full["policy"]

        # 4. Q-scale update from t=0 avg-of-2 Q values
        # Same train-mode-dropout semantics as policy_loss above (source uses
        # _detach_Qs which inherits parent's train mode).
        key_qscale_perm, key_qscale_drop = jax.random.split(key_qscale, 2)
        perm_qs = jax.random.permutation(key_qscale_perm, cfg.num_q)[:2]
        q_logits_for_scale = q_ensemble_net.apply(
            jax.lax.stop_gradient(wm_params_new["q_ensemble"]),
            zs_detached[0],                                  # (B, latent_dim)
            jax.lax.stop_gradient(pol_metrics["a_t0"]),      # (B, action_dim)
            deterministic=False,
            rngs={"dropout": key_qscale_drop},
        )  # (num_q, B, num_bins)
        q_sel = q_logits_for_scale[perm_qs]                  # (2, B, num_bins)
        q_dec = two_hot_inv(
            jax.nn.softmax(q_sel, axis=-1),
            cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True,
        ).squeeze(-1)                                         # (2, B)
        q_avg_t0 = q_dec.mean(axis=0)                        # (B,)
        new_qscale = qscale_update(state.qscale, q_avg_t0, tau=cfg.tau)
        # Tier B: raw percentiles for diagnostic (already computed inside qscale_update internally
        # but not exposed; recompute here cheaply). Differs from EMA'd range — instantaneous.
        q_p5_now = jnp.percentile(q_avg_t0, 5.0)
        q_p95_now = jnp.percentile(q_avg_t0, 95.0)

        # 5. Target EMA (encoder, dynamics, reward, q_ensemble only — no policy target)
        def ema_tree(target, online, tau):
            return jax.tree_util.tree_map(lambda t, o: t + tau * (o - t), target, online)

        new_target_encoder = ema_tree(state.encoder_target_params, wm_params_new["encoder"], cfg.tau)
        new_target_dynamics = ema_tree(state.dynamics_target_params, wm_params_new["dynamics"], cfg.tau)
        new_target_reward = ema_tree(state.reward_target_params, wm_params_new["reward"], cfg.tau)
        new_target_q = ema_tree(state.q_ensemble_target_params, wm_params_new["q_ensemble"], cfg.tau)

        # 6. Pack new state
        new_state = state.replace(
            encoder_params=wm_params_new["encoder"],
            dynamics_params=wm_params_new["dynamics"],
            reward_params=wm_params_new["reward"],
            q_ensemble_params=wm_params_new["q_ensemble"],
            policy_params=policy_params_new,
            encoder_target_params=new_target_encoder,
            dynamics_target_params=new_target_dynamics,
            reward_target_params=new_target_reward,
            q_ensemble_target_params=new_target_q,
            world_model_opt_state=new_wm_opt_state,
            policy_opt_state=new_pol_opt_state,
            qscale=new_qscale,
            key=key_next,
            step=state.step + 1,
        )

        metrics = {
            **wm_metrics,
            **pol_metrics,
            "q_scale_range_ema": new_qscale.range_ema,
            # Tier B additions
            "wm_grad_norm": wm_grad_norm,
            "pi_grad_norm": pi_grad_norm,
            "q_p5_batch": q_p5_now,
            "q_p95_batch": q_p95_now,
        }
        return new_state, metrics

    return update_step
