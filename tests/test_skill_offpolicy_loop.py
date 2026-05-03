"""Skill off-policy loop unit tests — gradient-step contract.

Full integration smoke runs separately at SD-B Task 5; these tests verify
the loop's per-step contract on synthetic batches.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp

from jax_rl.skill_discovery.config import SkillDiscoveryConfig, FactorConfig
from jax_rl.skill_discovery.factors import register_extractor
from jax_rl.skill_discovery.manager import SkillManager
from jax_rl.training.skill_offpolicy_loop import (
    compose_skill_batch, intrinsic_reward_then_update,
)

KEY = jax.random.PRNGKey(0)
BATCH = 32


@register_extractor(name="loop_test_obs", source="actor_obs", dim=5)
def _loop_test_extract(batch):
    return batch["obs"]


def _make_cfg(num_skills=4):
    return SkillDiscoveryConfig(
        mode="diayn",
        total_skill_dim=num_skills,
        factors=(FactorConfig(name="full", method="diayn", skill_dim=num_skills,
                              source="actor_obs", extractor="loop_test_obs", dim=5),),
    )


def test_compose_skill_batch_concat_shapes():
    """compose_skill_batch should append skill_z to obs/next_obs and return
    a batch with augmented obs shapes for the algo step."""
    raw_batch = {
        "obs": jax.random.normal(KEY, (BATCH, 5)),
        "next_obs": jax.random.normal(KEY, (BATCH, 5)),
        "action": jax.random.uniform(KEY, (BATCH, 2), minval=-1, maxval=1),
        "reward": jnp.zeros((BATCH, 1)),
        "done": jnp.zeros((BATCH, 1)),
        "truncation": jnp.zeros((BATCH, 1)),
        "skill_z": jax.nn.one_hot(jnp.arange(BATCH) % 4, 4),
        "next_skill_z": jax.nn.one_hot(jnp.arange(BATCH) % 4, 4),
    }
    composed = compose_skill_batch(raw_batch, raw_obs_dim=5, skill_dim=4)
    assert composed["obs"].shape == (BATCH, 9)
    assert composed["next_obs"].shape == (BATCH, 9)
    # Action / reward / done unchanged
    assert composed["action"].shape == (BATCH, 2)


def test_intrinsic_reward_replaces_env_reward():
    """intrinsic_reward_then_update should overwrite batch['reward'] with the
    skill-manager-computed intrinsic reward (modulo intrinsic_weight scaling).

    Note: _make_cfg() defaults already give intrinsic_weight=1.0 + task_reward_weight=0.0
    (SD-B pure DIAYN). No explicit override needed — defaults are correct.
    """
    cfg = _make_cfg(4)
    mgr = SkillManager(cfg)
    aux = mgr.init(KEY)

    raw_batch = {
        "obs": jax.random.normal(KEY, (BATCH, 5)),
        "next_obs": jax.random.normal(KEY, (BATCH, 5)),
        "action": jax.random.uniform(KEY, (BATCH, 2), minval=-1, maxval=1),
        "reward": jnp.full((BATCH, 1), 100.0),  # large env reward — should be overwritten
        "done": jnp.zeros((BATCH, 1)),
        "truncation": jnp.zeros((BATCH, 1)),
        "skill_z": jax.nn.one_hot(jnp.arange(BATCH) % 4, 4),
        "next_skill_z": jax.nn.one_hot(jnp.arange(BATCH) % 4, 4),
    }
    composed = compose_skill_batch(raw_batch, raw_obs_dim=5, skill_dim=4)
    final, _ = intrinsic_reward_then_update(
        composed, raw_batch, mgr, aux, cfg, reward_scaling=1.0,
    )
    # Reward should NOT be 100.0 anymore
    assert not jnp.allclose(final["reward"].flatten(), 100.0)


def test_aux_state_changes_under_update():
    """The loop's aux update step should mutate aux params."""
    mgr = SkillManager(_make_cfg(4))
    aux = mgr.init(KEY)
    raw_batch = {
        "obs": jax.random.normal(KEY, (BATCH, 5)),
        "next_obs": jax.random.normal(KEY, (BATCH, 5)),
        "skill_z": jax.nn.one_hot(jax.random.randint(KEY, (BATCH,), 0, 4), 4),
    }
    new_aux, metrics = mgr.update(aux, raw_batch)
    leaves_old = jax.tree_util.tree_leaves(aux["full"]["params"])
    leaves_new = jax.tree_util.tree_leaves(new_aux["full"]["params"])
    assert any(not jnp.array_equal(a, b) for a, b in zip(leaves_old, leaves_new))
    assert "full_disc_loss" in metrics
