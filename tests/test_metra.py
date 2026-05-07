"""Hermetic METRA aux module tests.

No env / sim. Plain JAX; runs on CPU lane (no GPU markers).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import optax
import pytest

from jax_rl.skill_discovery.metra import Phi, metra_reward, phi_loss, dual_lam_loss


def test_phi_init_apply_shape():
    """Phi network init + apply produces (B, out_dim)."""
    phi = Phi(hidden_dim=(64, 64), out_dim=4, activation="relu")
    key = jax.random.PRNGKey(0)
    params = phi.init(key, jnp.zeros((1, 27)))
    out = phi.apply(params, jnp.zeros((8, 27)))
    assert out.shape == (8, 4)


def test_metra_reward_alignment():
    """Aligned (phi_diff, z) → positive; opposite → negative."""
    phi = Phi(hidden_dim=(32,), out_dim=3, activation="relu")
    key = jax.random.PRNGKey(0)
    obs_dim = 6
    params = phi.init(key, jnp.zeros((1, obs_dim)))

    # Pick obs/next_obs deterministically; we don't need phi to be aligned
    # — we just need reward = (phi(s')-phi(s)) · z to behave with sign flips.
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, obs_dim))
    next_obs = jax.random.normal(jax.random.PRNGKey(2), (4, obs_dim))
    diff = phi.apply(params, next_obs) - phi.apply(params, obs)  # (4, 3)

    # Aligned: z = diff (unnormalized)
    r_pos = metra_reward(params, phi, obs, next_obs, diff)
    # Opposite: z = -diff
    r_neg = metra_reward(params, phi, obs, next_obs, -diff)

    assert jnp.all(r_pos > 0)
    assert jnp.all(r_neg < 0)
    assert jnp.allclose(r_pos, -r_neg)


def test_phi_loss_finite_at_init():
    """phi_loss + dual_lam_loss return finite values at random init (no NaN)."""
    phi = Phi(hidden_dim=(64, 64), out_dim=4, activation="relu")
    key = jax.random.PRNGKey(0)
    params = phi.init(key, jnp.zeros((1, 27)))
    log_dual_lam = jnp.log(jnp.array(30.0))

    obs = jax.random.normal(jax.random.PRNGKey(1), (16, 27))
    next_obs = obs + 0.01 * jax.random.normal(jax.random.PRNGKey(2), (16, 27))
    z = jax.random.normal(jax.random.PRNGKey(3), (16, 4))
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    loss_phi, mphi = phi_loss(params, log_dual_lam, phi, obs, next_obs, z)
    loss_dl, mdl = dual_lam_loss(log_dual_lam, params, phi, obs, next_obs)

    assert jnp.isfinite(loss_phi)
    assert jnp.isfinite(loss_dl)
    for v in mphi.values():
        assert jnp.all(jnp.isfinite(v))
    for v in mdl.values():
        assert jnp.all(jnp.isfinite(v))


def test_phi_loss_drops_with_aligned_data():
    """Synthetic data with known monotone phi-direction: phi_loss should
    decrease across a few Adam steps (alignment term grows)."""
    obs_dim = 8
    skill_dim = 3
    phi = Phi(hidden_dim=(64, 64), out_dim=skill_dim, activation="relu")
    key = jax.random.PRNGKey(0)
    params = phi.init(key, jnp.zeros((1, obs_dim)))
    log_dual_lam = jnp.log(jnp.array(30.0))

    # Construct a batch where (next_obs - obs) is aligned with z in the *obs*
    # space: pick z's, set obs random, next_obs = obs + z·padding (project to
    # obs_dim by zero-pad). Phi's gradient should pull phi(s')-phi(s) toward z.
    rng = jax.random.PRNGKey(7)
    z = jax.random.normal(jax.random.split(rng, 2)[0], (64, skill_dim))
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    obs = jax.random.normal(jax.random.split(rng, 2)[1], (64, obs_dim))
    # next_obs differs from obs in a direction encoding z: pad z with zeros.
    delta = jnp.concatenate([z * 0.5, jnp.zeros((64, obs_dim - skill_dim))], axis=-1)
    next_obs = obs + delta

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_only(p):
        loss, _ = phi_loss(p, log_dual_lam, phi, obs, next_obs, z)
        return loss

    loss_init = loss_only(params)
    for _ in range(50):
        grads = jax.grad(loss_only)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    loss_final = loss_only(params)

    assert loss_final < loss_init - 1e-3, (
        f"phi_loss did not decrease: init={loss_init}, final={loss_final}"
    )
