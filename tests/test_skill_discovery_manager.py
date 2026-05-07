"""Tests for SkillManager orchestrator."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import pytest

from jax_rl.skill_discovery.config import SkillDiscoveryConfig, FactorConfig
from jax_rl.skill_discovery.manager import SkillManager
from jax_rl.skill_discovery.factors import register_extractor

KEY = jax.random.PRNGKey(0)
BATCH = 32


# Register a test extractor used by manager tests
@register_extractor(name="manager_test_obs", source="actor_obs", dim=5)
def _manager_test_extract(batch):
    return batch["obs"]


# Paired next-obs extractor — required by METRA factors that read phi(s').
@register_extractor(name="manager_test_obs_next", source="actor_obs", dim=5)
def _manager_test_extract_next(batch):
    return batch["next_obs"]


def _make_diayn_cfg(num_skills=4):
    return SkillDiscoveryConfig(
        mode="diayn",
        total_skill_dim=num_skills,
        prior="one_hot",
        factors=(FactorConfig(name="full", method="diayn", skill_dim=num_skills,
                              source="actor_obs", extractor="manager_test_obs", dim=5),),
    )


def test_manager_init_creates_aux_state():
    mgr = SkillManager(_make_diayn_cfg(num_skills=4))
    aux = mgr.init(KEY)
    assert "full" in aux
    assert "params" in aux["full"]
    assert "opt_state" in aux["full"]


def test_manager_sample_skills_one_hot():
    mgr = SkillManager(_make_diayn_cfg(num_skills=4))
    z = mgr.sample_skills(KEY, num_envs=16)
    assert z.shape == (16, 4)
    assert jnp.allclose(z.sum(axis=-1), 1.0)


def test_manager_resample_on_done():
    mgr = SkillManager(_make_diayn_cfg(num_skills=4))
    z_old = mgr.sample_skills(KEY, num_envs=4)
    done = jnp.array([0.0, 1.0, 0.0, 1.0])
    z_new = mgr.resample_on_done(z_old, done, jax.random.PRNGKey(99))
    assert jnp.array_equal(z_new[0], z_old[0])
    assert jnp.array_equal(z_new[2], z_old[2])
    # rows 1, 3 may equal old (~25% chance with 4 skills) — don't assert difference


def test_manager_augment_actor_obs():
    mgr = SkillManager(_make_diayn_cfg(num_skills=4))
    obs = jnp.ones((8, 17))
    z = jnp.zeros((8, 4))
    aug = mgr.augment_actor_obs(obs, z)
    assert aug.shape == (8, 21)


def test_manager_compute_intrinsic_reward_deterministic():
    """Same aux_state + batch → same reward. SD-A acceptance."""
    mgr = SkillManager(_make_diayn_cfg(num_skills=4))
    aux = mgr.init(KEY)
    batch = {
        "obs": jax.random.normal(jax.random.PRNGKey(1), (BATCH, 5)),
        "skill_z": jax.nn.one_hot(jnp.arange(BATCH) % 4, 4),
    }
    r1 = mgr.compute_intrinsic_reward(aux, batch)
    r2 = mgr.compute_intrinsic_reward(aux, batch)
    assert r1.shape == (BATCH,)
    assert jnp.array_equal(r1, r2)


def test_manager_update_changes_params_and_returns_metrics():
    mgr = SkillManager(_make_diayn_cfg(num_skills=4))
    aux = mgr.init(KEY)
    batch = {
        "obs": jax.random.normal(jax.random.PRNGKey(1), (BATCH, 5)),
        "skill_z": jax.nn.one_hot(jax.random.randint(KEY, (BATCH,), 0, 4), 4),
    }
    new_aux, metrics = mgr.update(aux, batch)
    assert "full_disc_loss" in metrics
    # Params should have changed (gradient step taken)
    leaves_old = jax.tree_util.tree_leaves(aux["full"]["params"])
    leaves_new = jax.tree_util.tree_leaves(new_aux["full"]["params"])
    any_changed = any(not jnp.array_equal(a, b) for a, b in zip(leaves_old, leaves_new))
    assert any_changed


def test_manager_compute_intrinsic_reward_changes_with_aux_state():
    """SD-B contract: aux update changes the sample-time reward."""
    mgr = SkillManager(_make_diayn_cfg(num_skills=4))
    aux = mgr.init(KEY)

    # Train aux on synthetic separable data so disc params shift meaningfully
    train_batch = {
        "obs": jnp.concatenate([jnp.ones((16, 5)) * i for i in range(4)]),
        "skill_z": jnp.concatenate([
            jax.nn.one_hot(jnp.full(16, i, dtype=jnp.int32), 4) for i in range(4)
        ]),
    }
    aux_pre = aux
    for _ in range(20):
        aux, _ = mgr.update(aux, train_batch)
    aux_post = aux

    eval_batch = {
        "obs": jax.random.normal(jax.random.PRNGKey(7), (BATCH, 5)),
        "skill_z": jax.nn.one_hot(jnp.arange(BATCH) % 4, 4),
    }
    r_pre = mgr.compute_intrinsic_reward(aux_pre, eval_batch)
    r_post = mgr.compute_intrinsic_reward(aux_post, eval_batch)
    assert not jnp.array_equal(r_pre, r_post)


def test_manager_total_skill_dim_property():
    mgr = SkillManager(_make_diayn_cfg(num_skills=8))
    assert mgr.total_skill_dim == 8


# ── METRA branch tests ────────────────────────────────────────────────


def _make_metra_cfg(skill_dim=4):
    return SkillDiscoveryConfig(
        mode="metra",
        total_skill_dim=skill_dim,
        prior="unit_sphere",
        factors=(FactorConfig(name="m", method="metra", skill_dim=skill_dim,
                              source="actor_obs", extractor="manager_test_obs", dim=5),),
    )


def test_manager_metra_init_returns_phi_and_dual_state():
    """METRA factor aux state: 4 keys (phi_params, phi_opt_state, log_dual_lam, dual_opt_state)."""
    mgr = SkillManager(_make_metra_cfg(skill_dim=4))
    aux = mgr.init(KEY)
    assert "m" in aux
    for k in ("phi_params", "phi_opt_state", "log_dual_lam", "dual_opt_state"):
        assert k in aux["m"], f"missing METRA aux key: {k}"
    # log_dual_lam initialized to log(30.0)
    assert jnp.isclose(aux["m"]["log_dual_lam"], jnp.log(jnp.array(30.0)))


def test_manager_metra_intrinsic_reward_finite():
    """METRA reward = (phi(s')-phi(s))·z. Finite at random init."""
    mgr = SkillManager(_make_metra_cfg(skill_dim=3))
    aux = mgr.init(KEY)
    rng = jax.random.PRNGKey(1)
    obs = jax.random.normal(rng, (BATCH, 5))
    next_obs = obs + 0.01 * jax.random.normal(jax.random.PRNGKey(2), (BATCH, 5))
    z = jax.random.normal(jax.random.PRNGKey(3), (BATCH, 3))
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    batch = {"obs": obs, "next_obs": next_obs, "skill_z": z}
    r = mgr.compute_intrinsic_reward(aux, batch)
    assert r.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(r))


def test_manager_metra_update_returns_finite_metrics():
    """METRA update: phi step then dual step. All metrics finite, params change."""
    mgr = SkillManager(_make_metra_cfg(skill_dim=3))
    aux = mgr.init(KEY)
    rng = jax.random.PRNGKey(1)
    obs = jax.random.normal(rng, (BATCH, 5))
    next_obs = obs + 0.05 * jax.random.normal(jax.random.PRNGKey(2), (BATCH, 5))
    z = jax.random.normal(jax.random.PRNGKey(3), (BATCH, 3))
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    batch = {"obs": obs, "next_obs": next_obs, "skill_z": z}
    new_aux, metrics = mgr.update(aux, batch)
    # Required keys present + finite
    expected_keys = (
        "m_phi_loss", "m_phi_alignment", "m_phi_cst_penalty",
        "m_phi_diff_norm_sq", "m_dual_lam", "m_log_dual_lam",
    )
    for k in expected_keys:
        assert k in metrics, f"missing metric: {k}"
        assert jnp.all(jnp.isfinite(metrics[k])), f"non-finite metric: {k}"
    # Phi params should have changed (phi step ran).
    leaves_old = jax.tree_util.tree_leaves(aux["m"]["phi_params"])
    leaves_new = jax.tree_util.tree_leaves(new_aux["m"]["phi_params"])
    any_changed = any(not jnp.array_equal(a, b) for a, b in zip(leaves_old, leaves_new))
    assert any_changed
