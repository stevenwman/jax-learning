"""Polyak-averaged target-network soft update.

target_new = tau * online + (1 - tau) * target_old

Extracted from sac/td3/fast_sac/fast_td3/flash_sac which all defined the
identical inline lambda. Single helper, one source of truth.
"""

from typing import Any

import jax


def soft_update(online: Any, target: Any, tau: float) -> Any:
    """Polyak-averaged update of target network params toward online.

    Args:
        online: pytree of online network params.
        target: pytree of target network params (same structure as online).
        tau: blending coefficient in [0, 1]. tau=1 → online, tau=0 → target.

    Returns:
        New target pytree: tau * online + (1 - tau) * target, leaf-wise.

    Note:
        Apply only to learned params. Do NOT include batch_stats / other
        non-learnable state in either tree (caller's responsibility).
    """
    return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)
