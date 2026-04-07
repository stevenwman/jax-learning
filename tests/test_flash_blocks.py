import jax
import jax.numpy as jnp
import pytest
from jax_rl.networks.flash_blocks import (
    FlashSACEmbedder, FlashSACBlock, UnitRMSNorm,
    FlashSACActor, FlashSACCritic,
)

HIDDEN_DIM = 32
OBS_DIM = 12
ACTION_DIM = 4
BATCH = 8
KEY = jax.random.PRNGKey(0)

def test_embedder_shapes():
    model = FlashSACEmbedder(hidden_dim=HIDDEN_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    assert 'params' in variables
    assert 'batch_stats' in variables
    out = model.apply(variables, jnp.ones((BATCH, OBS_DIM)), train=False)
    assert out.shape == (BATCH, HIDDEN_DIM)

def test_embedder_bn_updates_on_train():
    model = FlashSACEmbedder(hidden_dim=HIDDEN_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    x = jax.random.normal(KEY, (BATCH, OBS_DIM))
    out, updates = model.apply(variables, x, train=True, mutable=['batch_stats'])
    assert 'batch_stats' in updates
    old_mean = variables['batch_stats']['BatchNorm_0']['mean']
    new_mean = updates['batch_stats']['BatchNorm_0']['mean']
    assert not jnp.allclose(old_mean, new_mean)

def test_block_residual_connection():
    model = FlashSACBlock(hidden_dim=HIDDEN_DIM, expansion=4)
    x = jnp.ones((BATCH, HIDDEN_DIM))
    variables = model.init(KEY, x, train=False)
    out = model.apply(variables, x, train=False)
    assert out.shape == (BATCH, HIDDEN_DIM)
    assert not jnp.allclose(out, x)
    assert not jnp.allclose(out, jnp.zeros_like(out))

def test_rms_norm_output_shape():
    model = UnitRMSNorm()
    variables = model.init(KEY, jnp.zeros((BATCH, HIDDEN_DIM)))
    out = model.apply(variables, jnp.ones((BATCH, HIDDEN_DIM)))
    assert out.shape == (BATCH, HIDDEN_DIM)

def test_actor_output_shapes():
    model = FlashSACActor(
        hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4, action_dim=ACTION_DIM,
    )
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    (mean, log_std) = model.apply(variables, jnp.ones((BATCH, OBS_DIM)), train=False)
    assert mean.shape == (BATCH, ACTION_DIM)
    assert log_std.shape == (BATCH, ACTION_DIM)

def test_critic_output_shapes():
    model = FlashSACCritic(
        hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4, num_atoms=51,
    )
    obs = jnp.zeros((BATCH, OBS_DIM))
    act = jnp.zeros((BATCH, ACTION_DIM))
    variables = model.init(KEY, obs, act, train=False)
    logits = model.apply(variables, obs, act, train=False)
    assert logits.shape == (BATCH, 51)

def test_actor_bn_mutable_forward():
    model = FlashSACActor(
        hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4, action_dim=ACTION_DIM,
    )
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    x = jax.random.normal(KEY, (BATCH, OBS_DIM))
    (mean, log_std), updates = model.apply(
        variables, x, train=True, mutable=['batch_stats'])
    assert 'batch_stats' in updates
    assert mean.shape == (BATCH, ACTION_DIM)
