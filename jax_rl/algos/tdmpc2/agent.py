"""TD-MPC2 training state + optimizers + update_step factory."""
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from jax_rl.utils.qscale import QScaleState, qscale_update
from jax_rl.utils.twohot import two_hot_inv
from jax_rl.algos.tdmpc2.losses import (
    compute_all_latents, world_model_loss, policy_loss,
)


# ------------------ Training state ------------------

@flax.struct.dataclass
class TDMPC2State:
    """TD-MPC2 training state.

    Holds online params, the q_ensemble target params (NO target encoder/dynamics/
    reward/policy — TD target reads only target_params["q_ensemble"]; see
    losses.compute_td_target), optimizer states, Q-scale tracker, per-env MPPI
    prev_mean, RNG key, step counter.
    Immutable pytree — use `state.replace(...)` to update.
    """
    encoder_params: Any
    dynamics_params: Any
    reward_params: Any
    q_ensemble_params: Any
    policy_params: Any
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
      5. Target EMA update on q_ensemble only (the only target read by compute_td_target).
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
        # Only "q_ensemble" is read by world_model_loss → compute_td_target
        # (losses.py:116). Other heads' targets are never used.
        target_params = {
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

        # 5. Target EMA — q_ensemble only (the only target read by compute_td_target).
        def ema_tree(target, online, tau):
            return jax.tree_util.tree_map(lambda t, o: t + tau * (o - t), target, online)

        new_target_q = ema_tree(state.q_ensemble_target_params, wm_params_new["q_ensemble"], cfg.tau)

        # 6. Pack new state
        new_state = state.replace(
            encoder_params=wm_params_new["encoder"],
            dynamics_params=wm_params_new["dynamics"],
            reward_params=wm_params_new["reward"],
            q_ensemble_params=wm_params_new["q_ensemble"],
            policy_params=policy_params_new,
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


__all__ = [
    "TDMPC2State",
    "make_update_step",
    "build_world_model_optimizer",
    "build_policy_optimizer",
]
