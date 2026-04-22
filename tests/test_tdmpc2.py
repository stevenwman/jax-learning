"""Tests for TD-MPC2 networks and loss components."""
import jax
import jax.numpy as jnp
from flax import linen as nn

from jax_rl.algos.tdmpc2 import NormedLinear


def test_normed_linear_shape_and_activation():
    layer = NormedLinear(features=32)
    params = layer.init(jax.random.PRNGKey(0), jnp.zeros((4, 16)))
    y = layer.apply(params, jnp.ones((4, 16)))
    assert y.shape == (4, 32)
    assert jnp.all(jnp.isfinite(y))


def test_normed_linear_truncnormal_init():
    """Kernel init should be trunc_normal(std=0.02); bias zero."""
    layer = NormedLinear(features=64)
    params = layer.init(jax.random.PRNGKey(0), jnp.zeros((1, 32)))
    # Kernel params exist and are small (std=0.02)
    kernel = params["params"]["Dense_0"]["kernel"]
    assert kernel.shape == (32, 64)
    assert abs(float(kernel.std())) < 0.1  # far under 1.0 stdlib default
    # Bias should be zero
    bias = params["params"]["Dense_0"]["bias"]
    assert jnp.allclose(bias, 0.0)


def test_normed_linear_dropout_off_by_default():
    """Default dropout=0.0, so output deterministic regardless of deterministic flag."""
    layer = NormedLinear(features=8)
    params = layer.init(jax.random.PRNGKey(0), jnp.zeros((2, 4)))
    y1 = layer.apply(params, jnp.ones((2, 4)))
    y2 = layer.apply(params, jnp.ones((2, 4)))
    assert jnp.allclose(y1, y2)


def test_encoder_output_shape_and_simnorm():
    from jax_rl.algos.tdmpc2 import Encoder
    enc = Encoder(enc_dim=256, num_layers=2, latent_dim=512, simnorm_dim=8)
    params = enc.init(jax.random.PRNGKey(0), jnp.zeros((4, 48)))
    z = enc.apply(params, jnp.ones((4, 48)))
    assert z.shape == (4, 512)
    # Latent respects SimNorm (chunks sum to 1)
    chunks = z.reshape(4, 512 // 8, 8)
    assert jnp.allclose(chunks.sum(-1), 1.0, atol=1e-5)


def test_encoder_gradient_flows():
    from jax_rl.algos.tdmpc2 import Encoder
    enc = Encoder(enc_dim=64, num_layers=2, latent_dim=32, simnorm_dim=4)
    params = enc.init(jax.random.PRNGKey(0), jnp.zeros((2, 10)))
    def loss(p, x):
        return enc.apply(p, x).sum()
    g = jax.grad(loss)(params, jnp.ones((2, 10)))
    # Gradient tree should be fully finite
    leaves = jax.tree_util.tree_leaves(g)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_dynamics_output_shape_and_simnorm():
    from jax_rl.algos.tdmpc2 import Dynamics
    dyn = Dynamics(mlp_dim=512, latent_dim=512, simnorm_dim=8)
    params = dyn.init(jax.random.PRNGKey(0), jnp.zeros((4, 512)), jnp.zeros((4, 6)))
    z_next = dyn.apply(params, jnp.ones((4, 512)), jnp.ones((4, 6)))
    assert z_next.shape == (4, 512)
    chunks = z_next.reshape(4, 512 // 8, 8)
    assert jnp.allclose(chunks.sum(-1), 1.0, atol=1e-5)


def test_dynamics_concatenates_z_and_action():
    """Changing action should change output (dynamics actually uses action)."""
    from jax_rl.algos.tdmpc2 import Dynamics
    dyn = Dynamics(mlp_dim=64, latent_dim=32, simnorm_dim=4)
    params = dyn.init(jax.random.PRNGKey(0), jnp.zeros((2, 32)), jnp.zeros((2, 3)))
    z = jnp.ones((2, 32))
    out1 = dyn.apply(params, z, jnp.zeros((2, 3)))
    out2 = dyn.apply(params, z, jnp.ones((2, 3)))
    assert not jnp.allclose(out1, out2)


def test_reward_output_shape():
    from jax_rl.algos.tdmpc2 import Reward
    r = Reward(mlp_dim=512, num_bins=101)
    params = r.init(jax.random.PRNGKey(0), jnp.zeros((4, 512)), jnp.zeros((4, 6)))
    out = r.apply(params, jnp.ones((4, 512)), jnp.ones((4, 6)))
    assert out.shape == (4, 101)


def test_reward_output_layer_zero_init():
    """Final Dense kernel must be zero at init (load-bearing, source world_model.py:31).

    Zero init ensures initial reward predictions center on the bin corresponding to
    symlog(0) = 0, preventing early-training bias.
    """
    from jax_rl.algos.tdmpc2 import Reward
    r = Reward(mlp_dim=64, num_bins=21)
    params = r.init(jax.random.PRNGKey(0), jnp.zeros((2, 32)), jnp.zeros((2, 3)))
    # The output layer is the last Dense; in Flax naming, nth module of its type.
    # Since there's only one bare `nn.Dense` (the output), Dense_0 in the apex scope.
    # Safer: check that at least one leaf in params tree is all zeros with shape ending in num_bins.
    leaves = jax.tree_util.tree_leaves_with_path(params)
    # Find the kernel with last dim == num_bins
    output_kernels = [leaf for path, leaf in leaves
                       if leaf.ndim == 2 and leaf.shape[-1] == 21 and "kernel" in str(path).lower()]
    assert len(output_kernels) >= 1, f"No output kernel found with trailing dim {21}"
    # At least one matching kernel should be all zeros
    assert any(jnp.allclose(k, 0.0) for k in output_kernels), \
        "Expected output layer kernel to be zero-init"


def test_q_ensemble_output_shape():
    from jax_rl.algos.tdmpc2 import QEnsemble
    q = QEnsemble(mlp_dim=512, num_bins=101, num_q=5, dropout=0.01)
    params = q.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        jnp.zeros((4, 512)),
        jnp.zeros((4, 6)),
        deterministic=True,
    )
    out = q.apply(
        params,
        jnp.ones((4, 512)),
        jnp.ones((4, 6)),
        deterministic=True,
    )
    assert out.shape == (5, 4, 101), f"Expected (5, 4, 101), got {out.shape}"


def test_q_ensemble_output_zero_at_init():
    """Each Q head has zero-init final Dense layer → output near 0 at init."""
    from jax_rl.algos.tdmpc2 import QEnsemble
    q = QEnsemble(mlp_dim=64, num_bins=21, num_q=3, dropout=0.0)
    params = q.init(
        {"params": jax.random.PRNGKey(0)},
        jnp.zeros((2, 32)),
        jnp.zeros((2, 4)),
        deterministic=True,
    )
    out = q.apply(params, jnp.ones((2, 32)), jnp.ones((2, 4)), deterministic=True)
    # Zero-init on final kernel → output is exactly 0 (since bias is also zero)
    assert jnp.allclose(out, 0.0, atol=1e-6)


def test_q_ensemble_heads_have_independent_params():
    """vmap over params means each head has distinct kernel values after init."""
    from jax_rl.algos.tdmpc2 import QEnsemble
    q = QEnsemble(mlp_dim=32, num_bins=11, num_q=4, dropout=0.0)
    params = q.init(
        {"params": jax.random.PRNGKey(0)},
        jnp.zeros((1, 16)),
        jnp.zeros((1, 2)),
        deterministic=True,
    )
    # Walk the tree, find a NormedLinear kernel; its leading dim should equal num_q=4
    # and values across the 4 heads should differ.
    leaves = jax.tree_util.tree_leaves(params)
    # Find a kernel with 3 dimensions (num_q stacked) — e.g. shape (4, in_dim, out_dim).
    # Exclude all-zero kernels (zero-init output Dense layer) — those are identical across
    # heads by design and would give a false "params shared" signal.
    stacked_kernels = [
        leaf for leaf in leaves
        if leaf.ndim == 3 and leaf.shape[0] == 4 and not jnp.allclose(leaf, 0.0)
    ]
    assert len(stacked_kernels) >= 1, "No non-zero stacked (num_q, ...) kernel found — vmap not wiring params"
    # Across heads, params should differ (not all identical)
    k = stacked_kernels[0]
    assert not jnp.allclose(k[0], k[1]), "Q heads share params — vmap should make them independent"


def test_bound_log_std_maps_to_range():
    """bound_log_std: raw=-inf → log_std_min; raw=+inf → log_std_max; raw=0 → midpoint."""
    from jax_rl.algos.tdmpc2 import bound_log_std
    raw = jnp.array([-10.0, 0.0, 10.0])
    bounded = bound_log_std(raw, log_std_min=-10.0, log_std_max=2.0)
    # At raw=-10, bounded ≈ log_std_min; at raw=+10, ≈ log_std_max
    assert float(bounded[0]) < -9.5
    assert float(bounded[2]) > 1.5
    # Midpoint at raw=0
    mid = (-10.0 + 2.0) / 2
    assert abs(float(bounded[1]) - mid) < 0.5


def test_squash_log_prob_correction_saturation_safe():
    """At |tanh| → 1, naive 1-tanh² is 0 → log(0) = -inf. Source uses relu(1-a²)+1e-6."""
    from jax_rl.algos.tdmpc2 import squash_log_prob_correction
    # Actions at near-saturation
    pre = jnp.array([[20.0, -20.0], [0.1, -0.1]])
    a = jnp.tanh(pre)
    corr = squash_log_prob_correction(a)  # (batch,)
    assert corr.shape == (2,)
    assert jnp.all(jnp.isfinite(corr)), f"Non-finite at saturation: {corr}"


def test_gaussian_log_prob_matches_scipy():
    """Sanity: gaussian_log_prob on independent dims matches sum of per-dim log N(x; μ, σ)."""
    from jax_rl.algos.tdmpc2 import gaussian_log_prob
    import math
    x = jnp.array([[0.5, -1.0, 2.0]])
    mean = jnp.array([[0.0, 0.0, 0.0]])
    log_std = jnp.array([[0.0, 0.0, 0.0]])  # std=1
    lp = gaussian_log_prob(x, mean, log_std)
    # Sum of log N(x; 0, 1) = -0.5·Σx² - 0.5·D·log(2π)
    expected = -0.5 * (0.25 + 1.0 + 4.0) - 0.5 * 3 * math.log(2 * math.pi)
    assert jnp.allclose(lp, expected, atol=1e-5)


def test_policy_prior_output_shapes_and_bounds():
    from jax_rl.algos.tdmpc2 import PolicyPrior
    pol = PolicyPrior(mlp_dim=512, action_dim=6, log_std_min=-10.0, log_std_max=2.0)
    params = pol.init(jax.random.PRNGKey(0), jnp.zeros((4, 512)), jax.random.PRNGKey(1))
    action, extras = pol.apply(params, jnp.ones((4, 512)), jax.random.PRNGKey(2))
    assert action.shape == (4, 6)
    # Tanh-squashed
    assert jnp.all(jnp.abs(action) <= 1.0)
    # Extras contain both log-probs
    assert "log_prob_pre" in extras
    assert "log_prob_post" in extras
    assert "mean" in extras and "log_std" in extras and "pre" in extras
    assert extras["log_prob_pre"].shape == (4,)
    assert extras["log_prob_post"].shape == (4,)
    # log_std bounded in [log_std_min, log_std_max]
    assert jnp.all(extras["log_std"] >= -10.0 - 1e-5)
    assert jnp.all(extras["log_std"] <= 2.0 + 1e-5)


def test_policy_prior_log_prob_post_equals_pre_minus_correction():
    """log_prob_post = log_prob_pre - squash_log_prob_correction(action)."""
    from jax_rl.algos.tdmpc2 import PolicyPrior, squash_log_prob_correction
    pol = PolicyPrior(mlp_dim=64, action_dim=3, log_std_min=-10.0, log_std_max=2.0)
    params = pol.init(jax.random.PRNGKey(0), jnp.zeros((2, 32)), jax.random.PRNGKey(1))
    action, extras = pol.apply(params, jnp.ones((2, 32)), jax.random.PRNGKey(2))
    expected_post = extras["log_prob_pre"] - squash_log_prob_correction(action)
    assert jnp.allclose(extras["log_prob_post"], expected_post, atol=1e-5)
