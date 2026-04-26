"""Env setup — thin shim that dispatches to the right backend.

The MJX backend lives in `jax_rl/training/env_backends/mjx_backend.py`. Other
backends (gym, isaaclab) are added in subsequent refactor phases.

`make_env_bundle(cfg, seed)` is the canonical entrypoint for new code; it
auto-dispatches via `env_backends.build_env_bundle`. `make_envs(cfg, seed)`
is re-exported for legacy PPO scripts that consume the 7-tuple directly.
"""

import jax.numpy as jnp

from jax_rl.configs.train_config import TrainConfig
from jax_rl.training.env_bundle import EnvBundle
from jax_rl.training.env_backends import build_env_bundle
# Re-export the MJX-specific make_envs so legacy callers (train_ppo*, train_flashsac)
# keep working unchanged.
from jax_rl.training.env_backends.mjx_backend import make_envs, _make_nan_safe_step  # noqa: F401
from jax_rl.utils.normalization import NormalizationState


def make_env_bundle(cfg: TrainConfig, seed: int) -> EnvBundle:
    """Build an EnvBundle for cfg.env_name (auto-detects backend)."""
    return build_env_bundle(cfg, seed)


def make_identity_norm_state(obs_dim: int) -> NormalizationState:
    """Identity norm state for off-policy algos (no obs normalization).

    Q-network LayerNorm handles input scaling instead.
    Kept as identity for checkpoint/inference compatibility with record_video.py.
    """
    return NormalizationState(
        mean=jnp.zeros(obs_dim),
        mean_of_squares=jnp.ones(obs_dim),
        count=1,
    )
