"""C51 distributional RL utilities.

Implements the categorical distributional Bellman update from:
  Bellemare et al., "A Distributional Perspective on RL" (2017)

Used by FastTD3/FastSAC for distributional critic training.
"""

import jax
import jax.numpy as jnp


def make_support(v_min: float, v_max: float, num_atoms: int) -> jax.Array:
    """Create evenly-spaced atom support: z_i = v_min + i * delta."""
    return jnp.linspace(v_min, v_max, num_atoms)


def logits_to_q(logits: jax.Array, support: jax.Array) -> jax.Array:
    """Expected Q-value from distributional logits.

    Args:
        logits: (batch, num_atoms) unnormalized log-probs
        support: (num_atoms,) atom values

    Returns:
        (batch,) expected Q values
    """
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * support, axis=-1)


def project_distribution(
    target_probs: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    gamma: float,
    support: jax.Array,
) -> jax.Array:
    """C51 categorical projection — redistribute shifted atoms onto fixed support.

    After the Bellman update, target atoms shift to r + γ * z_j. These shifted
    atoms generally don't align with the fixed support, so we redistribute
    probability mass via linear interpolation (the Φ operator from the paper).

    Args:
        target_probs: (batch, num_atoms) target distribution probabilities
        rewards: (batch,) rewards
        dones: (batch,) done flags (1.0 = terminal)
        gamma: discount factor
        support: (num_atoms,) atom values [v_min, ..., v_max]

    Returns:
        (batch, num_atoms) projected target distribution
    """
    num_atoms = support.shape[0]
    v_min = support[0]
    v_max = support[-1]
    delta_z = (v_max - v_min) / (num_atoms - 1)

    # Shift support by Bellman operator: ẑ_j = r + γ * (1 - done) * z_j
    # Shape: (batch, num_atoms)
    shifted = rewards[:, None] + gamma * (1.0 - dones[:, None]) * support[None, :]
    shifted = jnp.clip(shifted, v_min, v_max)

    # Find neighbor indices on the fixed support
    b = (shifted - v_min) / delta_z  # (batch, num_atoms), fractional index
    lo = jnp.floor(b).astype(jnp.int32)
    hi = jnp.clip(lo + 1, 0, num_atoms - 1)
    lo = jnp.clip(lo, 0, num_atoms - 1)

    # Linear interpolation weights
    hi_weight = b - lo.astype(jnp.float32)  # weight for upper neighbor
    lo_weight = 1.0 - hi_weight              # weight for lower neighbor

    # Scatter probability mass to neighbors
    # Use segment_sum via one_hot for JAX compatibility (no in-place scatter)
    batch_size = target_probs.shape[0]
    projected = jnp.zeros((batch_size, num_atoms))

    # For each atom j, add p_j * lo_weight to lo[j] and p_j * hi_weight to hi[j]
    lo_contrib = target_probs * lo_weight  # (batch, num_atoms)
    hi_contrib = target_probs * hi_weight  # (batch, num_atoms)

    # Scatter via one_hot: for each source atom, create a one_hot at the target index
    # and multiply by the contribution. Sum over source atoms.
    lo_onehot = jax.nn.one_hot(lo, num_atoms)  # (batch, num_atoms_src, num_atoms_dst)
    hi_onehot = jax.nn.one_hot(hi, num_atoms)  # (batch, num_atoms_src, num_atoms_dst)

    projected = jnp.einsum('bs,bsd->bd', lo_contrib, lo_onehot) + \
                jnp.einsum('bs,bsd->bd', hi_contrib, hi_onehot)

    return projected


def safe_log_softmax(logits: jax.Array, axis: int = -1, min_log: float = -30.0) -> jax.Array:
    """log_softmax clamped to a minimum to prevent NaN in cross-entropy.

    Why the clamp: ``log_softmax`` produces ``-inf`` for zero-probability atoms.
    In cross-entropy, ``target_prob * log_pred`` becomes ``0 * (-inf) = NaN``.
    Clamping log_probs to ``min_log`` (default -30 ≈ prob 1e-13) prevents NaN
    while preserving valid gradients on non-zero atoms. See LESSONS.md
    "C51 log_prob NaN" for the original debugging trail.
    """
    return jnp.maximum(jax.nn.log_softmax(logits, axis=axis), min_log)


def cross_entropy_categorical(
    target_probs: jax.Array,
    logits: jax.Array,
    axis: int = -1,
    min_log: float = -30.0,
) -> jax.Array:
    """Per-sample categorical cross-entropy with safe-log-softmax.

    Computes ``-sum(target_probs * log_softmax(logits))`` along ``axis``,
    with the log clamped at ``min_log`` to avoid 0 * -inf = NaN.

    Args:
        target_probs: (..., num_atoms) target distribution.
        logits: (..., num_atoms) unnormalized log-probs of the prediction.
        axis: reduction axis (default last).
        min_log: clamp floor on log_softmax output.

    Returns:
        (...) per-sample cross-entropy (same shape as inputs minus ``axis``).
    """
    log_probs = safe_log_softmax(logits, axis=axis, min_log=min_log)
    return -jnp.sum(target_probs * log_probs, axis=axis)
