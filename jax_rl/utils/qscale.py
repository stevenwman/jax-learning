"""Q-scale running percentile EMA tracker.

Tracks 5th/95th percentiles of Q outputs across updates, EMA'd with tau.
Scales Q values in the policy loss before adding the entropy bonus.
Source: /tmp/tdmpc2/tdmpc2/common/scale.py
"""
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class QScaleState:
    """Single scalar range (p95 - p5), EMA'd, clamped to min 1.0."""
    range_ema: jax.Array  # scalar


def qscale_init() -> QScaleState:
    return QScaleState(range_ema=jnp.array(1.0))


def qscale_update(state: QScaleState, qs: jax.Array, tau: float = 0.01) -> QScaleState:
    """Update range EMA from batch of Q values.

    Args:
        state: current QScaleState.
        qs: Q values, any shape. Percentiles computed across ALL elements.
        tau: EMA rate.
    """
    p5 = jnp.percentile(qs, 5.0)
    p95 = jnp.percentile(qs, 95.0)
    new_range = jnp.maximum(p95 - p5, 1.0)  # load-bearing clamp (source scale.py:41)
    range_ema = state.range_ema + tau * (new_range - state.range_ema)
    return QScaleState(range_ema=range_ema)


def qscale_apply(state: QScaleState, qs: jax.Array) -> jax.Array:
    """Divide Q values by current range EMA."""
    return qs / (state.range_ema + 1e-8)
