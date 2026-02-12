"""JAX RL: A modular reinforcement learning library built on JAX and Flax NNX."""

__version__ = "0.1.0"

from jax_rl import configs
from jax_rl import networks
from jax_rl import algos
from jax_rl import buffers

__all__ = ["configs", "networks", "algos", "buffers"]
