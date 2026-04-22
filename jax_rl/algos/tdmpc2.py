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
