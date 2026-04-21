"""Two-hot categorical encoding + symlog/symexp for TD-MPC2 reward/value targets.

Matches source /tmp/tdmpc2/tdmpc2/common/math.py.
"""
import jax
import jax.numpy as jnp


def symlog(x: jax.Array) -> jax.Array:
    """Signed log: sign(x) * log(|x| + 1). Monotone, symmetric about 0."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Inverse of symlog."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


def two_hot(x: jax.Array, vmin: float, vmax: float, num_bins: int) -> jax.Array:
    """Encode scalar(s) as two-hot distribution over num_bins.

    Args:
        x: (..., 1) scalar values (already symlog'd if applicable). Out-of-range values clamp.
        vmin, vmax: value range covered by bins.
        num_bins: number of bins (source uses 101).

    Returns:
        (..., num_bins) probability distribution (sums to 1, non-zero on at most 2 adjacent bins).
    """
    x = jnp.clip(x, vmin, vmax)
    bin_size = (vmax - vmin) / (num_bins - 1)
    bin_pos = (x.squeeze(-1) - vmin) / bin_size
    lower_idx = jnp.floor(bin_pos).astype(jnp.int32)
    upper_idx = jnp.clip(lower_idx + 1, 0, num_bins - 1)
    lower_idx = jnp.clip(lower_idx, 0, num_bins - 1)
    upper_weight = bin_pos - lower_idx.astype(bin_pos.dtype)
    lower_weight = 1.0 - upper_weight

    one_hot_lower = jax.nn.one_hot(lower_idx, num_bins)
    one_hot_upper = jax.nn.one_hot(upper_idx, num_bins)
    return lower_weight[..., None] * one_hot_lower + upper_weight[..., None] * one_hot_upper


def two_hot_inv(
    probs: jax.Array,
    vmin: float,
    vmax: float,
    num_bins: int,
    apply_symexp: bool = True,
) -> jax.Array:
    """Decode probability distribution back to scalar value.

    Args:
        probs: (..., num_bins) — softmax over logits, or a two-hot-encoded distribution.
        apply_symexp: if True, applies symexp to the decoded bin-centered value.

    Returns:
        (..., 1) scalar values.
    """
    bin_centers = jnp.linspace(vmin, vmax, num_bins)
    value = (probs * bin_centers).sum(axis=-1, keepdims=True)
    if apply_symexp:
        value = symexp(value)
    return value


def two_hot_ce_loss(logits: jax.Array, target_value: jax.Array,
                    vmin: float, vmax: float, num_bins: int,
                    apply_symlog: bool = True) -> jax.Array:
    """Cross-entropy loss between logits and two-hot target.

    Args:
        logits: (..., num_bins)
        target_value: (..., 1) scalar values (raw; symlog will be applied if apply_symlog=True)

    Returns:
        (...,) per-sample CE loss.
    """
    if apply_symlog:
        target_value = symlog(target_value)
    target = two_hot(target_value, vmin, vmax, num_bins)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -(target * log_probs).sum(axis=-1)
