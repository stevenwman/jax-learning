"""SkillManager — orchestrates skill lifecycle, aux networks, and intrinsic rewards.

Owns per-factor aux networks + optimizers (built once at construction), skill
sampling, obs augmentation, sample-time intrinsic reward, and aux gradient
updates. No env / buffer / training-loop knowledge.

SD-A only ships DIAYN factors. METRA factors raise NotImplementedError; they
land in SD-E. AuxNetConfig plumb-through (custom hidden dims, optimizers) is
deferred to SD-C — SD-A hardcodes Discriminator and Adam defaults at the
construction site.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from jax_rl.skill_discovery.config import SkillDiscoveryConfig
from jax_rl.skill_discovery.diayn import Discriminator, diayn_reward, diayn_update
from jax_rl.skill_discovery.factors import resolve_factor
from jax_rl.skill_discovery.prior import sample_skill


class SkillManager:
    """Orchestrator for skill discovery aux modules."""

    def __init__(self, config: SkillDiscoveryConfig):
        self.config = config

        # Validate ALL factors before building anything.
        for factor in config.factors:
            if factor.method == "metra":
                raise NotImplementedError("METRA deferred to SD-E")
            if factor.method != "diayn":
                raise ValueError(f"unknown factor method: {factor.method!r}")

        # Build per-factor network + optimizer once. SD-A defaults are
        # hardcoded here; AuxNetConfig plumb-through is SD-C work.
        self._networks: dict[str, Discriminator] = {}
        self._optimizers: dict = {}
        self._z_slices: list[tuple[int, int]] = []

        offset = 0
        for factor in config.factors:
            self._networks[factor.name] = Discriminator(
                hidden_dim=(256, 256),
                num_skills=factor.skill_dim,
                activation="relu",
            )
            self._optimizers[factor.name] = optax.adam(3e-4)
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

        Returns {factor_name: {"params": ..., "opt_state": ...}}.
        """
        aux: dict = {}
        keys = jax.random.split(key, max(len(self.config.factors), 1))
        for k, factor in zip(keys, self.config.factors):
            net = self._networks[factor.name]
            params = net.init(k, jnp.zeros((1, factor.dim)))
            opt_state = self._optimizers[factor.name].init(params)
            aux[factor.name] = {"params": params, "opt_state": opt_state}
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
        """Sum of per-factor DIAYN rewards, uniformly weighted (1/N)."""
        n = len(self.config.factors)
        if n == 0:
            return jnp.zeros((batch["obs"].shape[0],))
        weight = 1.0 / n
        total = None
        for idx, factor in enumerate(self.config.factors):
            obs_factor = resolve_factor(factor, batch)
            z_slice = self._get_factor_z(batch["skill_z"], idx)
            r = diayn_reward(
                aux_state[factor.name]["params"],
                self._networks[factor.name],
                obs_factor,
                z_slice,
                num_skills=factor.skill_dim,
            )
            contrib = weight * r
            total = contrib if total is None else total + contrib
        return total

    def update(self, aux_state: dict, batch: dict) -> tuple[dict, dict]:
        """Run one gradient step per factor, return (new_aux_state, flat_metrics)."""
        new_aux: dict = {}
        merged: dict = {}
        for idx, factor in enumerate(self.config.factors):
            obs_factor = resolve_factor(factor, batch)
            z_slice = self._get_factor_z(batch["skill_z"], idx)
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
            for k, v in metrics.items():
                merged[f"{factor.name}_{k}"] = v
        return new_aux, merged
