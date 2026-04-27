"""Networks: Encoder, Dynamics, Reward, QHead, QEnsemble, PolicyPrior.

Plus building blocks (NormedLinear, simnorm, mish) and policy helpers
(bound_log_std, squash_log_prob_correction, gaussian_log_prob, compute_scaled_entropy)."""

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


# TODO(vision): For pixel obs, swap with a CNN-bodied subclass.
# Source impl: nicklashansen/tdmpc2 common/world_model.py:enc_pixels.
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


# ------------------ Policy-loss helper ------------------

def compute_scaled_entropy(log_prob_pre: jax.Array, action_dim: int) -> jax.Array:
    """Single-task simplification of source's scaled_entropy formula.

    Source: /tmp/tdmpc2/tdmpc2/common/world_model.py:176-183. Pre-squash log_prob × action_dim.

    scaled_entropy = -log_prob_pre * action_dim
    """
    return -log_prob_pre * action_dim


__all__ = [
    "mish", "NormedLinear",
    "Encoder", "Dynamics", "Reward", "QHead", "QEnsemble", "PolicyPrior",
    "bound_log_std", "squash_log_prob_correction", "gaussian_log_prob",
    "compute_scaled_entropy",
]
