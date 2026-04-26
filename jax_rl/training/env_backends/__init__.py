"""Env-backend registry and dispatch.

Each backend (mjx, gym, isaaclab) implements a builder
`(TrainConfig, seed) -> EnvBundle` and registers itself at module import via
`register_backend(name, builder)`.

`build_env_bundle(cfg, seed)` looks up the backend implied by `cfg.env_name`
(via `detect_backend`) and dispatches.

Usage in a training script:
    from jax_rl.training.env_backends import build_env_bundle
    bundle = build_env_bundle(cfg, seed)

Or via the top-level shim that already re-exports under
`jax_rl.training.env_setup.make_env_bundle`.
"""

from typing import Callable

from jax_rl.training.env_bundle import EnvBundle, BackendKind


BACKEND_BUILDERS: dict[str, Callable] = {}


def register_backend(name: BackendKind, builder: Callable) -> None:
    """Register a backend builder. Called at module import by each backend."""
    BACKEND_BUILDERS[name] = builder


def detect_backend(env_name: str) -> BackendKind:
    """Pick backend from env name.

    Convention:
    - "IsaacLab/<name>"      → "isaaclab"
    - "Gym/<name>" or known gym env names ("PushT", ...)  → "gym"
    - Anything else          → "mjx" (default; mjx_backend will validate
      against `pg_registry`)

    Phase 1 ships with only "mjx" registered; gym/isaaclab branches added
    in subsequent phases. Detection logic is here (not in mjx_backend) so
    it stays backend-agnostic.
    """
    if env_name.startswith("IsaacLab/"):
        return "isaaclab"
    if env_name.startswith("Gym/") or env_name in _GYM_ENV_NAMES:
        return "gym"
    return "mjx"


# Filled in by gym_backend.py when it lands (Phase 2). Kept as a module-level
# set so detect_backend doesn't import the gym backend prematurely.
_GYM_ENV_NAMES: set[str] = set()


def register_gym_env_name(name: str) -> None:
    """Called by gym_backend.py to declare a name routes to the gym backend."""
    _GYM_ENV_NAMES.add(name)


def build_env_bundle(cfg, seed: int) -> EnvBundle:
    """Dispatch to the correct backend for cfg.env_name."""
    backend = detect_backend(cfg.env_name)
    if backend not in BACKEND_BUILDERS:
        registered = sorted(BACKEND_BUILDERS.keys())
        raise ValueError(
            f"Backend {backend!r} for env {cfg.env_name!r} is not registered. "
            f"Registered backends: {registered}. "
            f"Make sure the relevant backend module is imported."
        )
    return BACKEND_BUILDERS[backend](cfg, seed)


# Trigger backend registrations on import.
from jax_rl.training.env_backends import mjx_backend  # noqa: F401, E402  (side-effect import)
from jax_rl.training.env_backends import gym_backend  # noqa: F401, E402
