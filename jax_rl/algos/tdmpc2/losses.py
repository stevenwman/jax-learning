"""TD target + world-model + policy losses for TD-MPC2.

Used by `make_update_step` in tdmpc2/agent.py."""
import jax
import jax.numpy as jnp

from jax_rl.utils.twohot import two_hot_inv, two_hot_ce_loss
from jax_rl.utils.qscale import QScaleState, qscale_apply
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


__all__ = [
    "compute_all_latents",
    "compute_td_target",
    "world_model_loss",
    "policy_loss",
]
