"""Running observation normalization using online mean/variance estimation.

Normalizes observations to zero mean and unit variance using a running estimate
of E[X] and E[X²]. The variance is computed as Var(X) = E[X²] - E[X]².

This is the "sample-time" normalization approach: raw observations are stored in
the replay buffer, then normalized using current statistics when sampled for
training. This avoids the "stale normalization" bug where obs normalized with
old statistics become garbage as the statistics drift.

Usage::

    norm_state = init(obs_dim=17)

    # During collection: update statistics with raw obs
    norm_state = update(norm_state, raw_obs_batch)

    # During training: normalize sampled batch with current stats
    normalized_obs = normalize(norm_state, batch["obs"], eps=1e-2)

Note on eps:
    - eps=1e-8: standard, fine when obs have reasonable variance (e.g., dm_control)
    - eps=1e-2: recommended for off-policy RL, prevents near-zero variance dims
      from producing huge normalized values. The holosoma (FastSAC) paper uses 1e-2.
    See LESSONS.md "sample-time obs normalization" for the full story.
"""

import flax
import jax.numpy as jnp


@flax.struct.dataclass
class NormalizationState:
    """Running statistics for observation normalization.

    Tracks E[X] and E[X²] using a weighted running average. Variance is
    derived as Var(X) = E[X²] - E[X]², which is simpler than Welford's
    algorithm but slightly less numerically stable for very large counts.
    In practice, this is fine for RL observation normalization.

    Fields:
        mean: Running mean E[X], shape (obs_dim,).
        mean_of_squares: Running mean of squared values E[X²], shape (obs_dim,).
        count: Total number of samples seen (used for weighted averaging).
    """
    mean: jnp.ndarray
    mean_of_squares: jnp.ndarray
    count: int


def init(obs_dim: int) -> NormalizationState:
    """Create a fresh normalization state with zero mean and zero variance.

    Args:
        obs_dim: Dimensionality of the observation vector.

    Returns:
        A NormalizationState ready for updates. count=0 means the first
        update() call will set the statistics entirely from that batch.
    """
    return NormalizationState(
        mean=jnp.zeros(obs_dim),
        mean_of_squares=jnp.zeros(obs_dim),
        count=0,
    )


def update(state: NormalizationState, x: jnp.ndarray) -> NormalizationState:
    """Update running statistics with a new batch of observations.

    Uses a weighted running average:
        new_mean = (old_count * old_mean + batch_count * batch_mean) / total_count

    This is equivalent to computing the mean over all observations seen so far,
    but only requires storing the running mean (not all past observations).

    Args:
        state: Current normalization state.
        x: Batch of observations, shape (batch_size, obs_dim).

    Returns:
        Updated NormalizationState with new running statistics.
    """
    batch_count = x.shape[0]
    batch_mean = x.mean(axis=0)
    batch_mean_sq = (x ** 2).mean(axis=0)

    total_count = state.count + batch_count

    # Weighted average of old and new statistics
    new_mean = (state.count * state.mean + batch_count * batch_mean) / total_count
    new_mean_sq = (state.count * state.mean_of_squares + batch_count * batch_mean_sq) / total_count

    return state.replace(mean=new_mean, mean_of_squares=new_mean_sq, count=total_count)


def normalize(state: NormalizationState, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Normalize observations to approximately zero mean and unit variance.

    Formula: (x - mean) / (std + eps)
    where std = sqrt(max(E[X²] - E[X]², 0))

    The max(..., 0) prevents negative variance from floating-point errors.

    Args:
        state: Current normalization state (from update() calls).
        x: Observations to normalize, shape (batch_size, obs_dim).
        eps: Small constant added to std to prevent division by zero.
            Use 1e-2 for off-policy RL (see module docstring).

    Returns:
        Normalized observations, same shape as x.
    """
    variance = jnp.maximum(state.mean_of_squares - state.mean ** 2, 0.0)
    std = jnp.sqrt(variance) + eps
    return (x - state.mean) / std


def normalize_stacked(
    state: NormalizationState, x: jnp.ndarray, n_frames: int, eps: float = 1e-8,
) -> jnp.ndarray:
    """Normalize a frame-stacked obs using per-frame shared statistics.

    The norm state tracks stats for a single frame (raw_dim). Each frame slice
    in the stacked obs is normalized with the same stats, preserving relative
    differences between frames (which encode velocity/acceleration info).

    Args:
        state: Normalization state with shape (raw_dim,) statistics.
        x: Stacked observations, shape (batch_size, raw_dim * n_frames).
        n_frames: Number of stacked frames.
        eps: Small constant added to std to prevent division by zero.

    Returns:
        Normalized stacked observations, same shape as x.
    """
    variance = jnp.maximum(state.mean_of_squares - state.mean ** 2, 0.0)
    tiled_mean = jnp.tile(state.mean, n_frames)
    tiled_std = jnp.tile(jnp.sqrt(variance) + eps, n_frames)
    return (x - tiled_mean) / tiled_std


def unnormalize(state: NormalizationState, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Reverse the normalization: recover original-scale observations.

    Args:
        state: Current normalization state.
        x: Normalized observations, shape (batch_size, obs_dim).
        eps: Must match the eps used in normalize().

    Returns:
        Observations in original scale, same shape as x.
    """
    variance = jnp.maximum(state.mean_of_squares - state.mean ** 2, 0.0)
    std = jnp.sqrt(variance) + eps
    return x * std + state.mean
