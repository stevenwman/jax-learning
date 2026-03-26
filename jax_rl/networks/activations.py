"""Shared activation function registry.

Maps string names to JAX activation functions. Used by encoders and heads
so that activation type can be specified as a string in config files and
resolved to the actual function at network construction time.

Usage::

    from jax_rl.networks.activations import ACTIVATIONS

    act_fn = ACTIVATIONS[config.activation]  # e.g., "relu" -> nn.relu
"""

from flax import linen as nn

ACTIVATIONS = {
    "relu": nn.relu,
    "swish": nn.swish,
    "silu": nn.silu,      # same as swish
    "tanh": nn.tanh,
    "elu": nn.elu,
    "gelu": nn.gelu,
}
