"""Contraction metric network: MLP → lower-triangular L → SPD metric M = L Lᵀ.

Port of /tmp/cppo/contraction_metric.py (PyTorch). Spectral-norm Lipschitz
bounding deferred to follow-up; MVP uses orthogonal init only (see plan
Deviation 1).
"""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

_ACT = {"elu": nn.elu, "relu": nn.relu, "tanh": jnp.tanh, "swish": nn.swish}


class ContractionMetric(nn.Module):
    """Maps x ∈ ℝⁿ → M(x) ∈ ℝⁿˣⁿ symmetric positive-definite.

    Output dim packs n(n+1)/2 lower-triangular entries; diagonal passed through
    softplus + min_diagonal_value to guarantee positive diagonal of L, hence
    positive-definite M = L Lᵀ.
    """

    constraint_dim: int
    hidden_dims: Sequence[int] = (128, 128)
    activation: str = "elu"
    min_diagonal_value: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        act = _ACT[self.activation]
        n = self.constraint_dim
        out_dim = n * (n + 1) // 2

        h = x
        for d in self.hidden_dims:
            h = act(nn.Dense(d, kernel_init=nn.initializers.orthogonal())(h))
        flat = nn.Dense(out_dim, kernel_init=nn.initializers.orthogonal())(h)

        rows, cols = jnp.tril_indices(n)
        batch = flat.shape[0]
        L = jnp.zeros((batch, n, n), dtype=flat.dtype).at[:, rows, cols].set(flat)

        diag_idx = jnp.arange(n)
        raw_diag = L[:, diag_idx, diag_idx]
        pos_diag = jax.nn.softplus(raw_diag) + self.min_diagonal_value
        L = L.at[:, diag_idx, diag_idx].set(pos_diag)

        return L @ jnp.swapaxes(L, 1, 2)
