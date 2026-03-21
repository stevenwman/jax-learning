"""Tests for observation normalization utilities."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_rl.utils.normalization import init, update, normalize, unnormalize


def test_init():
    """Init should produce zeros with count=0."""
    state = init(obs_dim=4)
    assert state.mean.shape == (4,)
    assert state.mean_of_squares.shape == (4,)
    assert state.count == 0
    assert jnp.allclose(state.mean, 0.0)
    assert jnp.allclose(state.mean_of_squares, 0.0)
    print("  init: OK")


def test_single_batch_stats():
    """After one update, mean/var should match the batch exactly."""
    state = init(obs_dim=3)
    data = jnp.array([[1.0, 2.0, 3.0],
                       [3.0, 4.0, 5.0]])  # shape (2, 3)
    state = update(state, data)

    expected_mean = data.mean(axis=0)             # [2, 3, 4]
    expected_mos = (data ** 2).mean(axis=0)        # [5, 10, 17]

    assert jnp.allclose(state.mean, expected_mean), f"mean: {state.mean} != {expected_mean}"
    assert jnp.allclose(state.mean_of_squares, expected_mos), f"mos: {state.mean_of_squares} != {expected_mos}"
    assert state.count == 2
    print("  single batch stats: OK")


def test_multi_batch_converges():
    """Running stats should converge to true mean/var over many batches."""
    key = jax.random.PRNGKey(0)
    true_mean = jnp.array([5.0, -3.0])
    true_std = jnp.array([2.0, 0.5])

    state = init(obs_dim=2)
    for _ in range(200):
        key, subkey = jax.random.split(key)
        batch = true_mean + true_std * jax.random.normal(subkey, (64, 2))
        state = update(state, batch)

    # Derived variance: E[X^2] - E[X]^2
    var = state.mean_of_squares - state.mean ** 2

    assert jnp.allclose(state.mean, true_mean, atol=0.1), f"mean: {state.mean} vs {true_mean}"
    assert jnp.allclose(var, true_std ** 2, atol=0.2), f"var: {var} vs {true_std ** 2}"
    print("  multi batch convergence: OK")


def test_normalize_zero_mean_unit_var():
    """Normalized output should be ~N(0,1) for data matching the stats."""
    key = jax.random.PRNGKey(1)
    true_mean = jnp.array([10.0, -5.0, 0.0])
    true_std = jnp.array([3.0, 1.0, 7.0])

    state = init(obs_dim=3)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        batch = true_mean + true_std * jax.random.normal(subkey, (128, 3))
        state = update(state, batch)

    # Normalize a fresh batch drawn from the same distribution
    key, subkey = jax.random.split(key)
    test_data = true_mean + true_std * jax.random.normal(subkey, (1000, 3))
    normed = normalize(state, test_data)

    assert jnp.allclose(normed.mean(axis=0), 0.0, atol=0.15), f"normed mean: {normed.mean(axis=0)}"
    assert jnp.allclose(normed.std(axis=0), 1.0, atol=0.15), f"normed std: {normed.std(axis=0)}"
    print("  normalize zero-mean unit-var: OK")


def test_unnormalize_roundtrip():
    """unnormalize(normalize(x)) should recover x."""
    key = jax.random.PRNGKey(2)

    state = init(obs_dim=3)
    for _ in range(50):
        key, subkey = jax.random.split(key)
        state = update(state, jax.random.normal(subkey, (32, 3)) * 5 + 2)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (16, 3))
    recovered = unnormalize(state, normalize(state, x))

    assert jnp.allclose(recovered, x, atol=1e-5), f"max error: {jnp.abs(recovered - x).max()}"
    print("  unnormalize roundtrip: OK")


def test_zero_variance_safe():
    """Constant features should not produce NaN/Inf."""
    state = init(obs_dim=2)
    # All-constant data: variance = 0
    data = jnp.ones((10, 2)) * 3.0
    state = update(state, data)

    result = normalize(state, data)
    assert jnp.all(jnp.isfinite(result)), f"got non-finite values: {result}"
    print("  zero variance safety: OK")


def main():
    print("=" * 50)
    print("Normalization Tests")
    print("=" * 50)

    test_init()
    test_single_batch_stats()
    test_multi_batch_converges()
    test_normalize_zero_mean_unit_var()
    test_unnormalize_roundtrip()
    test_zero_variance_safe()

    print("\nAll normalization tests passed!")


if __name__ == "__main__":
    main()
