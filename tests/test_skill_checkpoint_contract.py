"""Skill checkpoint contract tests."""
import os, sys, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp

from jax_rl.skill_discovery.config import (
    SkillDiscoveryConfig, FactorConfig, SkillDeployConfig,
)
from jax_rl.skill_discovery.factors import register_extractor
from jax_rl.skill_discovery.manager import SkillManager
from jax_rl.skill_discovery.checkpointing import (
    save_skill_aux_state, load_skill_aux_state, write_skill_meta_block,
)

KEY = jax.random.PRNGKey(0)


@register_extractor(name="ckpt_test_obs", source="actor_obs", dim=5)
def _ckpt_test_extract(batch):
    return batch["obs"]


def _make_cfg():
    return SkillDiscoveryConfig(
        mode="diayn",
        total_skill_dim=4,
        factors=(FactorConfig(name="full", method="diayn", skill_dim=4,
                              source="actor_obs", extractor="ckpt_test_obs", dim=5),),
    )


def test_save_load_round_trip():
    """Save aux state, load it back into a fresh manager, params should match."""
    mgr = SkillManager(_make_cfg())
    aux_orig = mgr.init(KEY)

    with tempfile.TemporaryDirectory() as tmp:
        save_skill_aux_state(aux_orig, tmp)
        aux_loaded = load_skill_aux_state(tmp, template=aux_orig)

    # Compare params via tree equality
    leaves_orig = jax.tree_util.tree_leaves(aux_orig["full"]["params"])
    leaves_loaded = jax.tree_util.tree_leaves(aux_loaded["full"]["params"])
    for a, b in zip(leaves_orig, leaves_loaded):
        assert jnp.array_equal(a, b)


def test_meta_block_round_trip():
    """write_skill_meta_block produces JSON-serializable dict matching spec contract."""
    cfg = _make_cfg()
    block = write_skill_meta_block(cfg)
    # Required fields per spec §"Checkpoint metadata block":
    assert block["enabled"] is True
    assert block["mode"] == "diayn"
    assert block["prior"] == "one_hot"
    assert block["total_skill_dim"] == 4
    assert block["resample"] == "episode"
    assert block["hardware_ready"] is False
    assert block["schema_version"] == 1
    assert "factors" in block
    # Round-trip through JSON
    s = json.dumps(block)
    block2 = json.loads(s)
    assert block2 == block


def test_load_with_missing_aux_dir_raises():
    """Loading from a directory without skill_aux/ should raise informatively."""
    import pytest
    mgr = SkillManager(_make_cfg())
    aux_template = mgr.init(KEY)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="skill_aux"):
            load_skill_aux_state(tmp, template=aux_template)


def test_save_creates_skill_aux_subdir():
    """Save creates a 'skill_aux/' subdirectory (per spec design)."""
    mgr = SkillManager(_make_cfg())
    aux = mgr.init(KEY)
    with tempfile.TemporaryDirectory() as tmp:
        save_skill_aux_state(aux, tmp)
        assert os.path.isdir(os.path.join(tmp, "skill_aux"))
