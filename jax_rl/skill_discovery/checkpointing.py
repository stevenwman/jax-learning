"""Skill discovery checkpoint helpers — save/restore aux state + meta block."""
from __future__ import annotations
import os
from typing import Any

import jax
import numpy as np

from jax_rl.skill_discovery.config import SkillDiscoveryConfig, config_to_dict


SCHEMA_VERSION = 1


def save_skill_aux_state(aux_state: dict, ckpt_dir: str) -> None:
    """Write aux_state to <ckpt_dir>/skill_aux/ as numpy arrays.

    Layout: skill_aux/<factor_name>/{params.npz, opt_state.npz}
    """
    skill_dir = os.path.join(ckpt_dir, "skill_aux")
    os.makedirs(skill_dir, exist_ok=True)
    for factor_name, state in aux_state.items():
        factor_dir = os.path.join(skill_dir, factor_name)
        os.makedirs(factor_dir, exist_ok=True)
        # Flatten params PyTree to numpy arrays for portable storage
        params_flat = _flatten_pytree(state["params"])
        np.savez(os.path.join(factor_dir, "params.npz"), **params_flat)
        opt_flat = _flatten_pytree(state["opt_state"])
        np.savez(os.path.join(factor_dir, "opt_state.npz"), **opt_flat)


def load_skill_aux_state(ckpt_dir: str, template: dict) -> dict:
    """Restore aux_state from <ckpt_dir>/skill_aux/.

    Args:
        template: a freshly-initialized aux_state with the right PyTree
            structure (used to unflatten the loaded numpy arrays).
    """
    skill_dir = os.path.join(ckpt_dir, "skill_aux")
    if not os.path.isdir(skill_dir):
        raise FileNotFoundError(f"skill_aux/ subdirectory not found at {ckpt_dir}")

    loaded = {}
    for factor_name, tpl in template.items():
        factor_dir = os.path.join(skill_dir, factor_name)
        params_flat = dict(np.load(os.path.join(factor_dir, "params.npz")))
        opt_flat = dict(np.load(os.path.join(factor_dir, "opt_state.npz")))
        loaded[factor_name] = {
            "params": _unflatten_pytree(params_flat, tpl["params"]),
            "opt_state": _unflatten_pytree(opt_flat, tpl["opt_state"]),
        }
    return loaded


def write_skill_meta_block(config: SkillDiscoveryConfig) -> dict:
    """Build the meta.json 'skill_discovery' block per spec design."""
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": config.enabled,
        "mode": config.mode,
        "prior": config.prior,
        "total_skill_dim": config.total_skill_dim,
        "obs_injection": "concat_after_obs_pipeline",
        "resample": config.resample,
        "hardware_ready": False,
        "factors": [
            {"name": f.name, "method": f.method, "skill_dim": f.skill_dim,
             "source": f.source, "extractor": f.extractor, "dim": f.dim}
            for f in config.factors
        ],
        "deploy": {
            "skill_input_mode": config.deploy.skill_input_mode,
            "default_skill": config.deploy.default_skill,
        },
    }


def _flatten_pytree(tree) -> dict[str, Any]:
    """Flatten a PyTree of arrays into a dict of numpy arrays for npz storage."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}


def _unflatten_pytree(flat: dict[str, Any], template):
    """Reverse of _flatten_pytree using a template PyTree for structure."""
    _, treedef = jax.tree_util.tree_flatten(template)
    leaves = [flat[f"leaf_{i}"] for i in range(len(jax.tree_util.tree_leaves(template)))]
    return jax.tree_util.tree_unflatten(treedef, leaves)
