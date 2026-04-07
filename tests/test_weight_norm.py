import jax
import jax.numpy as jnp
import pytest
from jax_rl.networks.flash_blocks import (
    FlashSACActor, FlashSACCritic, normalize_weights,
)

KEY = jax.random.PRNGKey(0)
BATCH = 8
OBS_DIM = 12
ACTION_DIM = 4
HIDDEN_DIM = 32

def test_kernel_columns_become_unit_norm():
    model = FlashSACActor(hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4, action_dim=ACTION_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    normed_params = normalize_weights(variables['params'])
    leaves = jax.tree_util.tree_leaves_with_path(normed_params)
    for path, leaf in leaves:
        path_str = '/'.join(str(p) for p in path)
        if 'kernel' in path_str and leaf.ndim == 2:
            col_norms = jnp.linalg.norm(leaf, axis=0)
            assert jnp.allclose(col_norms, 1.0, atol=1e-6), f"Kernel at {path_str} not unit norm: {col_norms}"

def test_batchnorm_scale_bias_joint_norm_sqrt_d():
    model = FlashSACActor(hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4, action_dim=ACTION_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    normed_params = normalize_weights(variables['params'])
    flat = jax.tree_util.tree_leaves_with_path(normed_params)
    bn_groups = {}
    for path, leaf in flat:
        path_str = '/'.join(str(p) for p in path)
        if 'BatchNorm' in path_str and ('scale' in path_str or 'bias' in path_str):
            parent = path_str.rsplit('/', 1)[0]
            if parent not in bn_groups:
                bn_groups[parent] = {}
            key_name = 'scale' if 'scale' in path_str else 'bias'
            bn_groups[parent][key_name] = leaf
    assert len(bn_groups) > 0, "No BatchNorm modules found"
    for parent, params in bn_groups.items():
        if 'scale' in params and 'bias' in params:
            scale, bias = params['scale'], params['bias']
            d = scale.shape[-1]
            joint_norm = jnp.sqrt(jnp.sum(scale**2 + bias**2))
            assert jnp.allclose(joint_norm, jnp.sqrt(d), atol=1e-5), \
                f"BN at {parent}: joint norm {joint_norm:.4f} != sqrt({d})={jnp.sqrt(d):.4f}"

def test_normalize_is_idempotent():
    model = FlashSACActor(hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4, action_dim=ACTION_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    normed_once = normalize_weights(variables['params'])
    normed_twice = normalize_weights(normed_once)
    leaves_once = jax.tree_util.tree_leaves(normed_once)
    leaves_twice = jax.tree_util.tree_leaves(normed_twice)
    for l1, l2 in zip(leaves_once, leaves_twice):
        assert jnp.allclose(l1, l2, atol=1e-6), "Weight norm is not idempotent"

def test_free_bias_params_unchanged():
    model = FlashSACActor(hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4, action_dim=ACTION_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    original_params = variables['params']
    normed_params = normalize_weights(original_params)
    leaves_orig = jax.tree_util.tree_leaves_with_path(original_params)
    leaves_norm = jax.tree_util.tree_leaves_with_path(normed_params)
    for (path_o, leaf_o), (path_n, leaf_n) in zip(leaves_orig, leaves_norm):
        path_str = '/'.join(str(p) for p in path_o)
        if 'mean_bias' in path_str or 'logstd_bias' in path_str or 'value_bias' in path_str:
            assert jnp.allclose(leaf_o, leaf_n), f"Free bias at {path_str} was modified"
