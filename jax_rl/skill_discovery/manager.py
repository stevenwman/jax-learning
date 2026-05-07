"""SkillManager — orchestrates skill lifecycle, aux networks, and intrinsic rewards.

Owns per-factor aux networks + optimizers (built once at construction), skill
sampling, obs augmentation, sample-time intrinsic reward, and aux gradient
updates. No env / buffer / training-loop knowledge.

Supported methods (per-factor):
- ``diayn`` — Discriminator q(z|s), softmax cross-entropy update,
  reward = log q(z|s) - log p(z) (Eysenbach 2018).
- ``metra`` — Phi(s) representation network, dual-Lagrangian Lipschitz
  constraint, reward = (phi(s')-phi(s)) · z (Park 2024).

DIAYN paths use a single optimizer per factor; METRA uses two (phi + dual).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from jax_rl.skill_discovery.config import SkillDiscoveryConfig
from jax_rl.skill_discovery.diayn import Discriminator, diayn_reward, diayn_update
from jax_rl.skill_discovery.factors import resolve_factor, resolve_factor_next
from jax_rl.skill_discovery.metra import (
    Phi, metra_reward, phi_loss, dual_lam_loss,
)
from jax_rl.skill_discovery.prior import sample_skill


# METRA reference defaults (`tests/main.py:98-136` via source_extracts §METRA).
_METRA_PHI_HIDDEN = (1024, 1024)
_METRA_PHI_ACTIVATION = "relu"
_METRA_PHI_LR = 1e-4
_METRA_DUAL_LR = 1e-4
_METRA_DUAL_LAM_INIT = 30.0
_METRA_DUAL_SLACK = 1e-3
_METRA_DUAL_DIST = "one"  # constraint = ||Δφ||² ≤ 1


class SkillManager:
    """Orchestrator for skill discovery aux modules."""

    def __init__(self, config: SkillDiscoveryConfig):
        self.config = config

        # Validate ALL factors before building anything.
        for factor in config.factors:
            if factor.method not in ("diayn", "metra"):
                raise ValueError(f"unknown factor method: {factor.method!r}")

        # Per-factor network + optimizer storage. DIAYN factors get one
        # network + one optimizer; METRA factors get a Phi network + two
        # optimizers (phi + dual).
        self._networks: dict[str, object] = {}
        self._optimizers: dict = {}                 # phi optimizer for METRA, disc opt for DIAYN
        self._dual_optimizers: dict = {}            # METRA only
        self._z_slices: list[tuple[int, int]] = []

        offset = 0
        for factor in config.factors:
            if factor.method == "diayn":
                self._networks[factor.name] = Discriminator(
                    hidden_dim=(256, 256),
                    num_skills=factor.skill_dim,
                    activation="relu",
                )
                self._optimizers[factor.name] = optax.adam(3e-4)
            elif factor.method == "metra":
                self._networks[factor.name] = Phi(
                    hidden_dim=_METRA_PHI_HIDDEN,
                    out_dim=factor.skill_dim,
                    activation=_METRA_PHI_ACTIVATION,
                )
                self._optimizers[factor.name] = optax.adam(_METRA_PHI_LR)
                self._dual_optimizers[factor.name] = optax.adam(_METRA_DUAL_LR)
            self._z_slices.append((offset, offset + factor.skill_dim))
            offset += factor.skill_dim

    @property
    def total_skill_dim(self) -> int:
        return self.config.total_skill_dim

    def _get_factor_z(self, z: jax.Array, idx: int) -> jax.Array:
        start, end = self._z_slices[idx]
        return z[:, start:end]

    def init(self, key) -> dict:
        """Initialize per-factor params + opt_state.

        DIAYN: ``{"params": ..., "opt_state": ...}``.
        METRA: ``{"phi_params": ..., "phi_opt_state": ...,
                  "log_dual_lam": ..., "dual_opt_state": ...}``.
        """
        aux: dict = {}
        keys = jax.random.split(key, max(len(self.config.factors), 1))
        for k, factor in zip(keys, self.config.factors):
            net = self._networks[factor.name]
            params = net.init(k, jnp.zeros((1, factor.dim)))
            if factor.method == "diayn":
                opt_state = self._optimizers[factor.name].init(params)
                aux[factor.name] = {"params": params, "opt_state": opt_state}
            elif factor.method == "metra":
                phi_opt_state = self._optimizers[factor.name].init(params)
                log_dual_lam = jnp.log(jnp.array(_METRA_DUAL_LAM_INIT))
                dual_opt_state = self._dual_optimizers[factor.name].init(log_dual_lam)
                aux[factor.name] = {
                    "phi_params": params,
                    "phi_opt_state": phi_opt_state,
                    "log_dual_lam": log_dual_lam,
                    "dual_opt_state": dual_opt_state,
                }
        return aux

    def sample_skills(self, key, num_envs: int) -> jax.Array:
        """Sample one skill vector per env by concatenating per-factor priors."""
        keys = jax.random.split(key, max(len(self.config.factors), 1))
        parts = []
        for k, factor in zip(keys, self.config.factors):
            parts.append(
                sample_skill(
                    k,
                    prior=self.config.prior,
                    num_envs=num_envs,
                    skill_dim=factor.skill_dim,
                )
            )
        return jnp.concatenate(parts, axis=-1)

    def resample_on_done(self, current_z: jax.Array, done: jax.Array, key) -> jax.Array:
        """Replace rows where done==1 with newly sampled skills."""
        new_z = self.sample_skills(key, current_z.shape[0])
        return jnp.where(done[:, None] > 0, new_z, current_z)

    def augment_actor_obs(self, obs: jax.Array, z: jax.Array) -> jax.Array:
        """Concat skill onto obs along the last axis."""
        assert z.shape[-1] == self.total_skill_dim, (
            f"z dim {z.shape[-1]} != total_skill_dim {self.total_skill_dim}"
        )
        return jnp.concatenate([obs, z], axis=-1)

    def compute_intrinsic_reward(self, aux_state: dict, batch: dict) -> jax.Array:
        """Sum of per-factor rewards, uniformly weighted (1/N)."""
        n = len(self.config.factors)
        if n == 0:
            return jnp.zeros((batch["obs"].shape[0],))
        weight = 1.0 / n
        total = None
        for idx, factor in enumerate(self.config.factors):
            obs_factor = resolve_factor(factor, batch)
            z_slice = self._get_factor_z(batch["skill_z"], idx)
            if factor.method == "diayn":
                r = diayn_reward(
                    aux_state[factor.name]["params"],
                    self._networks[factor.name],
                    obs_factor,
                    z_slice,
                    num_skills=factor.skill_dim,
                )
            elif factor.method == "metra":
                next_obs_factor = resolve_factor_next(factor, batch)
                r = metra_reward(
                    aux_state[factor.name]["phi_params"],
                    self._networks[factor.name],
                    obs_factor,
                    next_obs_factor,
                    z_slice,
                )
            contrib = weight * r
            total = contrib if total is None else total + contrib
        return total

    def update(self, aux_state: dict, batch: dict) -> tuple[dict, dict]:
        """Run one gradient step per factor, return (new_aux_state, flat_metrics).

        DIAYN: single softmax cross-entropy step on the discriminator.
        METRA: phi step (Adam on phi_loss) then dual step (Adam on dual_lam_loss),
        on the same batch. Phi-then-dual order matches D3 reference.
        """
        new_aux: dict = {}
        merged: dict = {}
        for idx, factor in enumerate(self.config.factors):
            obs_factor = resolve_factor(factor, batch)
            z_slice = self._get_factor_z(batch["skill_z"], idx)
            if factor.method == "diayn":
                z_indices = jnp.argmax(z_slice, axis=-1)
                new_params, new_opt_state, metrics = diayn_update(
                    aux_state[factor.name]["params"],
                    aux_state[factor.name]["opt_state"],
                    self._networks[factor.name],
                    self._optimizers[factor.name],
                    obs_factor,
                    z_indices,
                )
                new_aux[factor.name] = {"params": new_params, "opt_state": new_opt_state}
            elif factor.method == "metra":
                next_obs_factor = resolve_factor_next(factor, batch)
                phi = self._networks[factor.name]
                phi_opt = self._optimizers[factor.name]
                dual_opt = self._dual_optimizers[factor.name]

                phi_params = aux_state[factor.name]["phi_params"]
                phi_opt_state = aux_state[factor.name]["phi_opt_state"]
                log_dual_lam = aux_state[factor.name]["log_dual_lam"]
                dual_opt_state = aux_state[factor.name]["dual_opt_state"]

                # Phi step.
                def _phi_loss(p, ldl=log_dual_lam):
                    loss, m = phi_loss(
                        p, ldl, phi, obs_factor, next_obs_factor, z_slice,
                        dual_dist=_METRA_DUAL_DIST, dual_slack=_METRA_DUAL_SLACK,
                    )
                    return loss, m

                (phi_loss_val, phi_metrics), phi_grads = jax.value_and_grad(
                    _phi_loss, has_aux=True
                )(phi_params)
                phi_updates, new_phi_opt_state = phi_opt.update(
                    phi_grads, phi_opt_state, phi_params
                )
                new_phi_params = optax.apply_updates(phi_params, phi_updates)

                # Dual step (uses post-phi params; D3 reference does this).
                def _dual_loss(ldl, p=new_phi_params):
                    loss, m = dual_lam_loss(
                        ldl, p, phi, obs_factor, next_obs_factor,
                        dual_dist=_METRA_DUAL_DIST, dual_slack=_METRA_DUAL_SLACK,
                    )
                    return loss, m

                (dual_loss_val, dual_metrics), dual_grads = jax.value_and_grad(
                    _dual_loss, has_aux=True
                )(log_dual_lam)
                dual_updates, new_dual_opt_state = dual_opt.update(
                    dual_grads, dual_opt_state, log_dual_lam
                )
                new_log_dual_lam = optax.apply_updates(log_dual_lam, dual_updates)

                new_aux[factor.name] = {
                    "phi_params": new_phi_params,
                    "phi_opt_state": new_phi_opt_state,
                    "log_dual_lam": new_log_dual_lam,
                    "dual_opt_state": new_dual_opt_state,
                }
                metrics = dict(phi_metrics)
                metrics.update(dual_metrics)
            for k, v in metrics.items():
                merged[f"{factor.name}_{k}"] = v
        return new_aux, merged
