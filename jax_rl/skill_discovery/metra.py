"""METRA auxiliary module — Phi network, reward, phi loss, dual loss.

Park et al. 2024 (`seohongpark/METRA`). Faithful port of `metra.py:194-300`.

Phi network is a deterministic MLP `obs -> R^skill_dim`. The intrinsic reward
is the alignment of the phi-difference with the (continuous, unit-sphere) skill
vector z:

    r(s, s', z) = (phi(s') - phi(s)) · z

The phi network is trained to maximize expected alignment under a Lipschitz
constraint enforced by a Lagrangian dual variable lambda (stored as
`log_dual_lam`). Default constraint shape (`dual_dist="one"`) is
`mean((phi(s')-phi(s))**2) <= 1`. METRA reference uses `dual_slack=1e-3`.

Both `phi_loss` and `dual_lam_loss` follow the METRA reference verbatim, with
`stop_gradient` on the cross-network argument:
- phi_loss: stop_gradient on log_dual_lam (lambda is a constant during phi step)
- dual_lam_loss: stop_gradient on phi_params + on cst_penalty.mean()

See `.context/references/skill_discovery_source_extracts.md` §METRA for the
exact source-line references and `.superpowers/plans/2026-05-05-ant-metra.md`
for the sign-convention rationale (do NOT flip the dual_lam_loss sign).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax_rl.networks.activations import ACTIVATIONS


class Phi(nn.Module):
    """Deterministic state representation phi: obs -> R^skill_dim.

    METRA reference uses Gaussian module head but only consumes `.mean`, so we
    use a plain MLP. Hidden width 1024×1024 is the METRA reference default;
    D3 fork uses 256×256.
    """

    hidden_dim: tuple[int, ...]
    out_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = ACTIVATIONS[self.activation]
        for h in self.hidden_dim:
            x = act(nn.Dense(h)(x))
        return nn.Dense(self.out_dim)(x)


def metra_reward(phi_params, phi, obs_factor, next_obs_factor, z):
    """Sample-time METRA reward: (phi(s') - phi(s)) · z.

    Per `metra.py:194-229`. Uses the *current* phi params (no target net).
    """
    phi_x = phi.apply(phi_params, obs_factor)
    phi_y = phi.apply(phi_params, next_obs_factor)
    return jnp.sum((phi_y - phi_x) * z, axis=-1)


def _cst_dist(dual_dist: str, obs, next_obs, ref_shape):
    """Constraint RHS — `dual_dist="one"` returns 1, `"l2"` returns ||Δs||²/dim."""
    if dual_dist == "one":
        return jnp.ones(ref_shape)
    if dual_dist == "l2":
        return jnp.mean((next_obs - obs) ** 2, axis=-1)
    raise ValueError(f"unknown dual_dist: {dual_dist!r} (expected 'one' or 'l2')")


def phi_loss(
    phi_params,
    log_dual_lam,
    phi,
    obs,
    next_obs,
    z,
    dual_dist: str = "one",
    dual_slack: float = 1e-3,
):
    """Phi loss = -(alignment + lambda * cst_penalty).mean.

    Verbatim from METRA reference `metra.py:245-289`:
        cst_penalty = cst_dist - mean((phi_y - phi_x)**2)
        cst_penalty = min(cst_penalty, dual_slack)        # upper-clamp
        te_obj = alignment + stop_grad(exp(log_dual_lam)) * cst_penalty
        loss   = -te_obj.mean()

    Note `mean` (not `sum`) on `(phi_y-phi_x)**2`. METRA reference uses mean;
    D3 fork uses sqrt(sum) vector_norm. Different formulations — do not swap.
    """
    phi_x = phi.apply(phi_params, obs)
    phi_y = phi.apply(phi_params, next_obs)
    alignment = jnp.sum((phi_y - phi_x) * z, axis=-1)
    diff_norm_sq = jnp.mean((phi_y - phi_x) ** 2, axis=-1)
    cst_dist = _cst_dist(dual_dist, obs, next_obs, ref_shape=alignment.shape)
    cst_penalty = cst_dist - diff_norm_sq
    cst_penalty = jnp.minimum(cst_penalty, dual_slack)  # upper-clamp
    dual_lam = jnp.exp(jax.lax.stop_gradient(log_dual_lam))
    te_obj = alignment + dual_lam * cst_penalty
    loss = -te_obj.mean()
    metrics = {
        "phi_loss": loss,
        "phi_alignment": alignment.mean(),
        "phi_cst_penalty": cst_penalty.mean(),
        "phi_diff_norm_sq": diff_norm_sq.mean(),
        "dual_lam": dual_lam,
    }
    return loss, metrics


def dual_lam_loss(
    log_dual_lam,
    phi_params,
    phi,
    obs,
    next_obs,
    dual_dist: str = "one",
    dual_slack: float = 1e-3,
):
    """Dual-lambda loss = log_dual_lam * stop_grad(cst_penalty.mean()).

    Verbatim from METRA reference `metra.py:292-300`. Adam **descent** on this
    loss → ascent on lambda when cst_penalty < 0 (constraint violated):
        ∂L/∂log_λ = cst_penalty.mean()
        log_λ ← log_λ − lr · cst_penalty.mean()

    When constraint violated (mean(‖Δφ‖²) > 1 → cst_penalty < 0), grad < 0,
    log_λ INCREASES, lambda grows → constraint enforced harder. Correct sign.
    DO NOT flip — see plan risk register row 2.
    """
    sg_params = jax.lax.stop_gradient(phi_params)
    phi_x = phi.apply(sg_params, obs)
    phi_y = phi.apply(sg_params, next_obs)
    diff_norm_sq = jnp.mean((phi_y - phi_x) ** 2, axis=-1)
    cst_dist = _cst_dist(dual_dist, obs, next_obs, ref_shape=diff_norm_sq.shape)
    cst_penalty = cst_dist - diff_norm_sq
    cst_penalty = jnp.minimum(cst_penalty, dual_slack)
    loss = log_dual_lam * jax.lax.stop_gradient(cst_penalty.mean())
    metrics = {
        "log_dual_lam": log_dual_lam,
        "dual_loss": loss,
    }
    return loss, metrics
