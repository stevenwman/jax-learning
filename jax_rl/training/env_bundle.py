"""EnvBundle — backend-agnostic wrapper around env creation output.

Currently MJX-only (via env_setup.make_env_bundle). Other backends will be
added under jax_rl.training.env_backends/ — gym (CPU), isaaclab (PyTorch GPU).

Each backend constructs an EnvBundle conforming to this dataclass; downstream
training loops (run_offpolicy_loop, run_onpolicy_pyloop) work against the
bundle interface rather than backend-specific entrypoints.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np


BackendKind = Literal["mjx", "gym", "isaaclab"]


@dataclass
class EnvBundle:
    """Env setup bundle for training scripts.

    Wraps env construction output with metadata so training scripts don't
    re-detect dict-obs / asymmetric-critic structure or backend-specific
    quirks.

    Backend kinds:
    - "mjx": MuJoCo Playground envs, vmap'd in JAX. env_step is JIT'd.
    - "gym": gymnasium envs (CPU / numpy). env_step is plain Python; arrays
      auto-converted at buffer ingest.
    - "isaaclab": IsaacLab envs (PyTorch GPU, dlpack bridge). Theoretical
      until a working install + smoke env land.
    """

    env: Any
    env_step: Callable
    env_state: Any
    eval_env: Any
    obs_dim: int
    action_dim: int
    critic_obs_dim: Optional[int]   # None if symmetric (critic obs = actor obs)
    has_privileged: bool
    dict_obs: bool
    key: Any                         # jax.Array (mjx) or seed int (gym/isaaclab)

    backend_kind: BackendKind = "mjx"
    num_envs: int = 1
    render_fn: Optional[Callable[[Any, int], Optional[np.ndarray]]] = None

    def render(self, state: Any, env_idx: int = 0) -> Optional[np.ndarray]:
        """Return last-frame RGB array (H, W, 3) for env_idx, or None.

        Backend-specific. MJX path uses Playground's renderer (wired in
        record_video.py for now); gym uses env.render(mode='rgb_array');
        isaaclab unsupported.

        Phase 0 ships render_fn=None for the existing MJX bundle — the
        record_video path bypasses bundle.render. Phase 5 wires this up.
        """
        if self.render_fn is None:
            return None
        return self.render_fn(state, env_idx)
