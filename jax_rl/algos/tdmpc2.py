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
