"""FlashSAC network building blocks.

Implements the inverted residual architecture with BatchNorm from FlashSAC.
Key properties:
- All Dense layers are bias-free with orthogonal init
- BatchNorm uses running stats during eval (use_running_average=not train)
- Residual connections require hidden_dim to stay constant through blocks
- Actor/Critic output biases are free parameters (not part of Dense)

Reference: FlashSAC PyTorch implementation (flash_rl/agents/flashSAC/layer.py)
"""

from flax import linen as nn
import jax.numpy as jnp
import jax
from typing import Any


class FlashSACEmbedder(nn.Module):
    """Projects raw input to hidden_dim after BatchNorm normalization.

    BatchNorm first normalizes the raw input (handles varying observation
    scales). Dense then projects to the hidden dimension with no activation —
    the first FlashSACBlock provides the first nonlinearity.

    Attributes:
        hidden_dim: Output feature dimensionality.
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = nn.BatchNorm(
            momentum=0.01, epsilon=1e-5, use_running_average=not train,
        )(x)
        x = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(),
        )(x)
        return x


class FlashSACBlock(nn.Module):
    """Inverted residual block with BatchNorm and ReLU activations.

    Expands to hidden_dim * expansion internally, then projects back to
    hidden_dim. A residual skip connection is added after the second norm+relu.

    Attributes:
        hidden_dim: Input/output feature dimensionality.
        expansion: Internal expansion factor (default 4).
    """
    hidden_dim: int
    expansion: int = 4

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        residual = x
        x = nn.Dense(
            self.hidden_dim * self.expansion,
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(),
        )(x)
        x = nn.BatchNorm(
            momentum=0.01, epsilon=1e-5, use_running_average=not train,
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(),
        )(x)
        x = nn.BatchNorm(
            momentum=0.01, epsilon=1e-5, use_running_average=not train,
        )(x)
        x = nn.relu(x)
        x = x + residual
        return x


class UnitRMSNorm(nn.Module):
    """RMS normalization with a learnable per-feature scale.

    Normalizes each vector to unit RMS, then rescales by a learned parameter.
    No bias term — keeps output distribution centered.
    """

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * scale


class FlashSACActor(nn.Module):
    """FlashSAC actor: outputs (mean, log_std) for a Gaussian policy.

    Architecture: Embedder → N x Block → UnitRMSNorm → mean head + logstd head.
    Log-std is clamped to [-10, 2] via a tanh-scaled mapping.

    Attributes:
        hidden_dim: Feature dimensionality throughout the trunk.
        num_blocks: Number of FlashSACBlock residual blocks.
        expansion: Expansion factor for each block.
        action_dim: Dimensionality of the action space.
    """
    hidden_dim: int
    num_blocks: int
    expansion: int
    action_dim: int

    @nn.compact
    def __call__(
        self, obs: jax.Array, train: bool,
    ) -> tuple[jax.Array, jax.Array]:
        x = FlashSACEmbedder(self.hidden_dim)(obs, train)
        for _ in range(self.num_blocks):
            x = FlashSACBlock(self.hidden_dim, self.expansion)(x, train)
        x = UnitRMSNorm()(x)

        mean = (
            nn.Dense(
                self.action_dim,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
            )(x)
            + self.param('mean_bias', nn.initializers.zeros, (self.action_dim,))
        )

        raw_logstd = (
            nn.Dense(
                self.action_dim,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
            )(x)
            + self.param('logstd_bias', nn.initializers.zeros, (self.action_dim,))
        )
        # Maps raw_logstd → [-10, 2] via tanh squashing
        log_std = -10.0 + 12.0 * 0.5 * (1.0 + jnp.tanh(raw_logstd))

        return mean, log_std


class FlashSACCritic(nn.Module):
    """FlashSAC critic: distributional value head outputting atom logits.

    Concatenates obs and action, runs through Embedder → N x Block →
    UnitRMSNorm → linear projection to num_atoms logits.

    Attributes:
        hidden_dim: Feature dimensionality throughout the trunk.
        num_blocks: Number of FlashSACBlock residual blocks.
        expansion: Expansion factor for each block.
        num_atoms: Number of distributional value atoms.
    """
    hidden_dim: int
    num_blocks: int
    expansion: int
    num_atoms: int

    @nn.compact
    def __call__(
        self, obs: jax.Array, action: jax.Array, train: bool,
    ) -> jax.Array:
        x = jnp.concatenate([obs, action], axis=-1)
        x = FlashSACEmbedder(self.hidden_dim)(x, train)
        for _ in range(self.num_blocks):
            x = FlashSACBlock(self.hidden_dim, self.expansion)(x, train)
        x = UnitRMSNorm()(x)

        logits = (
            nn.Dense(
                self.num_atoms,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
            )(x)
            + self.param('value_bias', nn.initializers.zeros, (self.num_atoms,))
        )
        return logits  # (batch, num_atoms)


def normalize_weights(params: Any) -> Any:
    """Project network parameters to the FlashSAC constraint set.

    Three normalization rules applied in a single pass over the pytree:

    1. Dense kernels (shape (in, out)): each column normalized to unit L2 norm.
    2. BatchNorm scale+bias: normalized jointly so ||(scale, bias)||₂ = sqrt(D).
    3. UnitRMSNorm scale: normalized so ||scale||₂ = sqrt(D).
    4. Everything else (mean_bias, logstd_bias, value_bias): untouched.

    Pure function; JIT-compatible.
    """
    # --- Pass 1: collect BatchNorm (scale, bias) pairs by parent path string ---
    # We need both leaves simultaneously to compute the joint normalization factor.
    bn_sqsums: dict[str, jax.Array] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        path_str = '/'.join(str(p) for p in path)
        if 'BatchNorm' in path_str and ('scale' in path_str or 'bias' in path_str):
            parent = path_str.rsplit('/', 1)[0]
            if parent not in bn_sqsums:
                bn_sqsums[parent] = jnp.zeros(())
            bn_sqsums[parent] = bn_sqsums[parent] + jnp.sum(leaf ** 2)

    # --- Pass 2: apply normalization leaf-by-leaf ---
    def _normalize_leaf(path: tuple, leaf: jax.Array) -> jax.Array:
        path_str = '/'.join(str(p) for p in path)

        # Dense kernels: normalize each column (output neuron) to unit L2 norm.
        if 'kernel' in path_str and leaf.ndim == 2:
            col_norms = jnp.linalg.norm(leaf, axis=0, keepdims=True)  # (1, out)
            return leaf / jnp.maximum(col_norms, 1e-8)

        # BatchNorm scale or bias: joint normalization to sqrt(D).
        if 'BatchNorm' in path_str and ('scale' in path_str or 'bias' in path_str):
            parent = path_str.rsplit('/', 1)[0]
            d = leaf.shape[-1]
            sqsum = bn_sqsums[parent]
            factor = jnp.sqrt(d) / jnp.sqrt(sqsum + 1e-8)
            return leaf * factor

        # UnitRMSNorm scale: normalize to sqrt(D).
        if 'UnitRMSNorm' in path_str and 'scale' in path_str:
            d = leaf.shape[-1]
            sqsum = jnp.sum(leaf ** 2)
            factor = jnp.sqrt(d) / jnp.sqrt(sqsum + 1e-8)
            return leaf * factor

        # Free bias params (mean_bias, logstd_bias, value_bias): untouched.
        return leaf

    return jax.tree_util.tree_map_with_path(_normalize_leaf, params)
