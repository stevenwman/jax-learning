"""Tests for ContractionMetric network.

Ref: /tmp/cppo/contraction_metric.py (PyTorch original).
"""

import jax
import jax.numpy as jnp
import pytest

from jax_rl.networks.contraction_metric import ContractionMetric


def test_output_shape_and_spd():
    """Network output M(x) must be symmetric positive-definite, shape (B, n, n)."""
    net = ContractionMetric(constraint_dim=4, hidden_dims=(32, 32))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 4)))
    x = jax.random.normal(jax.random.PRNGKey(1), (16, 4))
    M = net.apply(params, x)

    assert M.shape == (16, 4, 4)
    # symmetry
    assert jnp.allclose(M, jnp.swapaxes(M, 1, 2), atol=1e-5)
    # positive definite (all eigenvalues > 0)
    eigs = jnp.linalg.eigvalsh(M)
    assert (eigs > 0).all(), f"min eig = {eigs.min()}"


def test_min_diagonal_floor():
    """Diagonal of L is softplus+min_diag → M's diagonal ≥ min_diag²."""
    min_diag = 0.1
    net = ContractionMetric(constraint_dim=3, min_diagonal_value=min_diag)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
    # Many random inputs — all should respect floor
    xs = jax.random.normal(jax.random.PRNGKey(42), (128, 3)) * 5.0
    M = net.apply(params, xs)
    diag = jnp.diagonal(M, axis1=1, axis2=2)
    assert (diag >= min_diag**2 - 1e-6).all(), f"min diag = {diag.min()}"


def test_v_dot_chain_rule():
    """V(x) = xᵀM(x)x; ∂V/∂x computed via jax.grad; V̇ = ∇V · ẋ finite."""
    net = ContractionMetric(constraint_dim=4, hidden_dims=(16,))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 4)))

    def V_single(c):
        M = net.apply(params, c[None])[0]
        return c @ M @ c

    c = jnp.array([0.1, -0.2, 0.3, 0.05])
    c_dot = jnp.array([0.01, 0.02, -0.01, 0.0])
    grad_V = jax.grad(V_single)(c)
    V_dot = grad_V @ c_dot

    assert grad_V.shape == (4,)
    assert jnp.isfinite(V_dot)
    # V̇ known analytic lower bound: none here (M is learned), but not NaN
    assert jnp.isfinite(grad_V).all()


def test_v_dot_identity_metric_case():
    """Hand-computable sanity: at init, V is roughly quadratic. For a frozen
    L = I (achieved by zero-weight init + softplus(0)+min_diag=1.1 diag),
    V(x) = 1.21 xᵀx, ∇V = 2 · 1.21 · x, V̇ = 2.42 · x·ẋ.
    We don't force zero-weight init here; instead verify ∂V/∂x has the
    analytic form 2 M(x) x + (∂M/∂x) contribution, dominated by 2Mx at small x."""
    net = ContractionMetric(constraint_dim=3, hidden_dims=(8,))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))

    def V_single(c):
        return c @ net.apply(params, c[None])[0] @ c

    # Very small x → (∂M/∂x · xᵀx) term vanishes → ∇V ≈ 2 M(0) x
    c = jnp.array([1e-4, -1e-4, 5e-5])
    M0 = net.apply(params, jnp.zeros((1, 3)))[0]
    grad_V = jax.grad(V_single)(c)
    expected = 2.0 * M0 @ c

    assert jnp.allclose(grad_V, expected, atol=1e-6, rtol=1e-3), \
        f"∇V={grad_V} expected≈{expected}"


def test_outer_gradient_finite():
    """Double-backward: outer grad of mean(penalty) wrt params must be finite.
    This is the load-bearing path for metric training."""
    net = ContractionMetric(constraint_dim=3)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))

    c_batch = jax.random.normal(jax.random.PRNGKey(1), (8, 3)) * 0.2
    c_dot_batch = jax.random.normal(jax.random.PRNGKey(2), (8, 3)) * 0.2
    alpha, eps = 0.1, 1e-3

    def loss_fn(p):
        def V_single(c):
            return c @ net.apply(p, c[None])[0] @ c
        V = jax.vmap(V_single)(c_batch)
        grad_V = jax.vmap(jax.grad(V_single))(c_batch)
        V_dot = jnp.sum(grad_V * c_dot_batch, axis=-1)
        return jnp.mean(jax.nn.relu(V_dot + alpha * V + eps))

    grads = jax.grad(loss_fn)(params)
    flat, _ = jax.tree_util.tree_flatten(grads)
    for leaf in flat:
        assert jnp.isfinite(leaf).all(), "non-finite grad in outer backward"
