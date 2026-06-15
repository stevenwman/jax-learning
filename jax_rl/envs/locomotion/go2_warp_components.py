"""Back-compat shim — the Go2 Warp env components moved to the
``jax_rl.envs.locomotion.components`` package (one module per concern:
``actuation`` / ``terrain`` / ``controllers`` / ``force_fields``).

New code should import from ``jax_rl.envs.locomotion.components`` (or its
submodules) directly. This shim re-exports the full former public surface — plus
the private helpers external callers still import (``_make_heightfield`` in
``tests/test_terrain_component.py``) — so existing
``from ...go2_warp_components import X`` lines keep working unchanged.

See ``.superpowers/specs/2026-06-15-components-package-split.md``.
"""
from __future__ import annotations

from jax_rl.envs.locomotion.components import *  # noqa: F401,F403
# Private helpers not covered by `*` (underscore-prefixed) that external code or
# tests import by name.
from jax_rl.envs.locomotion.components import (  # noqa: F401
    _N_STIFFNESS, _norm01, _fractal_perlin_noise_2d, _make_heightfield,
)
