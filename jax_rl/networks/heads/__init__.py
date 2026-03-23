"""Head implementations."""

from jax_rl.networks.heads.gaussian import GaussianHead
from jax_rl.networks.heads.value import ValueHead
from jax_rl.networks.heads.deterministic import DeterministicHead

__all__ = ["GaussianHead", "ValueHead", "DeterministicHead"]
