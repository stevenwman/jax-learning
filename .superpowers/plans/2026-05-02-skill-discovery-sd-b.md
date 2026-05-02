# Skill Discovery SD-B — DIAYN Training Loop on CheetahRun

**Date:** 2026-05-02
**Spec:** `.superpowers/specs/2026-04-28-skill-discovery.md` §SD-B
**Validation methodology:** `.context/references/skill_discovery_validation.md` Part 4 SD-B
**Source extracts:** `.context/references/skill_discovery_source_extracts.md`
**SD-A landed:** commits 84883a7 / 0317db7 / 593ddf4 / bd0bcf8 / 725eced — config, prior, factor registry, DIAYN aux module, SkillManager. 31/31 SD-A tests pass; zero regressions.
**Phase:** SD-B (second of SD-A → SD-E)
**Status:** ready

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Strict TDD: test → fail → impl → pass → commit per step. Tickbox each step as you go.

## Goal

Stand up a working DIAYN training loop on CheetahRun (MJX) using vanilla SAC + the SD-A `SkillManager`. Replicate DIAYN App. D.4-style behavior: rising discriminator accuracy + per-skill task-return diversity + qualitatively distinct rendered behaviors per fixed skill.

**Out of scope for SD-B:**
- METRA (deferred to SD-E)
- Go2 wiring (SD-C)
- Frame-stack support (SD-C/E)
- Deploy contract changes (SD-D)
- D3-style factorization, style/safety rewards (SD-E)
- FastSAC C51 distributional critic (deferred — see "Algo choice" below)

## Algo choice — vanilla SAC for SD-B

Spec §SD-B "Initial algorithm scope" reads `algo: FastSAC only`. **We override here** to vanilla SAC for SD-B because:

1. FastSAC's C51 categorical critic uses fixed `V_min`/`V_max` bins. DIAYN intrinsic reward `log q(z|s) − log p(z)` is unbounded as the discriminator sharpens; with `gamma=0.99` discounted return can pin the upper bin and lose resolution. Empirical bin tuning is a separate optimization problem that would slow SD-B's plumbing-validation goal.
2. Vanilla SAC's scalar Q dodges the V_min/V_max problem entirely. ~30-line script delta vs FastSAC.
3. SAC + skill-conditioned policy is the published DIAYN reference (`ben-eysenbach/sac:diayn.py`) — direct paper-replication.

**FastSAC switch lands in SD-C/E** once we have empirical reward magnitude data from SD-B to set bins.

## Smoke + acceptance env

**CheetahRun (MJX, mujoco_playground.dm_control_suite)**:
- Already wired in `jax_rl/training/env_backends/mjx_backend.py`
- `EnvBundle` actor obs dim = 18d state, action dim = 6
- Matches DIAYN App. D.4 protocol (HalfCheetah is the primary DIAYN reward-histogram fig env)

`num_skills = 8` for SD-B (legible visualization, 8 colors per plot). **If 8 fails to converge** (discriminator accuracy stalls < `1/num_skills + 0.1` after 100K steps), escalate per DIAYN reference (`ben-eysenbach/sac:mujoco_all_diayn.py:38` defaults to 50; paper text uses both 20 and 50 across figures — re-grep paper before tuning, **don't hallucinate a specific recommended value**).

## Acceptance for SD-B as a whole

From spec §SD-B "Acceptance" + validation doc Part 4 SD-B:

**Plumbing gates:**
- [ ] 10K-step CPU smoke produces finite aux losses, no NaN, buffer fills.
- [ ] Existing asymmetric-critic buffer tests still pass after generic extras land.
- [ ] Skill replay storage round-trips: store/sample `skill_z`, `next_skill_z`, `factor_obs`, `next_factor_obs` alongside critic extras (when applicable).
- [ ] Skill checkpoint resume restores aux params + optimizer state.
- [ ] Existing FastSAC tests pass unchanged.
- [ ] No changes to `train_fast_sac.py` behavior (we add a new script, not edit the existing one).

**Behavioral gates (paper-grounded):**
- [ ] Discriminator accuracy curve rises above chance (`1/num_skills = 0.125`) within first 100K steps.
- [ ] At 1M steps: per-skill task-return histogram (M=10 episodes per fixed z). Spread (max − min across 8 skills) > 50% of any single skill's return. Replicates DIAYN App. D.4 protocol.
- [ ] At 1M steps: render 1 video per skill via `record_video.py`. Skills should show qualitatively diverse gaits (subjective check, replicates DIAYN HalfCheetah figure intent).
- [ ] **Seeds:** 3 seeds minimum, mean ± std reporting. Upgrade to 5 if results contentious.

## File structure

```
New files:
  jax_rl/training/skill_offpolicy_loop.py   — skill-aware off-policy training loop
  jax_rl/skill_discovery/checkpointing.py   — wrapper around base CheckpointManager
  scripts/train_skill_discovery.py          — CLI entry point (mirrors train_sac.py shape)
  tests/test_skill_offpolicy_loop.py
  tests/test_skill_checkpoint_contract.py
  tests/test_obs_pipeline_extras.py         — non-regression on critic_obs + new generic extras

Modified files:
  jax_rl/training/obs_pipeline.py           — make_buffer accepts generic extra_obs_dims
```

## Independence map (waves)

```
Wave A (parallel):
  ├── Task 1: ObsPipeline.make_buffer generic extras
  └── Task 2: Skill checkpoint wrapper

Wave B (after Wave A):
  └── Task 3: skill_offpolicy_loop

Wave C (after Wave B):
  └── Task 4: train_skill_discovery.py CLI

Wave D (after Wave C):
  └── Task 5: smoke + 100K + 1M acceptance runs
```

Wave A = 2 parallel subagents. Waves B, C, D = 1 agent each (serial). Total: 5 tasks.

---

## Task 1: ObsPipeline.make_buffer generic extras

**Files:**
- Modify: `jax_rl/training/obs_pipeline.py`
- Create: `tests/test_obs_pipeline_extras.py`

**Goal:** extend `ObsPipeline.make_buffer` to accept generic `extra_obs_dims: dict[str, int] | None`, merge with existing `critic_obs` machinery. `JaxReplayBuffer` already supports the underlying `extra_obs_dims` constructor arg (`jax_rl/buffers/jax_replay_buffer.py:36`); only the pipeline wrapper needs to be widened.

**Non-regression goal:** existing asymmetric-critic flow (FastSAC on Go2) must produce byte-identical buffer behavior.

### Step 1.1: Write tests (RED)

`tests/test_obs_pipeline_extras.py`:

```python
"""Non-regression + generic extras tests for ObsPipeline.make_buffer."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from jax_rl.training.obs_pipeline import ObsPipeline


def _make_pipeline(has_privileged=False, n_frame_stack=1):
    return ObsPipeline(
        obs_norm=False,
        has_privileged=has_privileged,
        n_frame_stack=n_frame_stack,
    )


def test_make_buffer_no_extras():
    """Default case — no critic, no extras. extra_obs_dims should be None."""
    pipe = _make_pipeline(has_privileged=False)
    buf = pipe.make_buffer(obs_dim=18, action_dim=6, buffer_size=1000)
    # Buffer should expose no extras
    assert buf._extra_obs_dims == {}


def test_make_buffer_critic_only_unchanged():
    """Asymmetric critic path — same behavior as before SD-B."""
    pipe = _make_pipeline(has_privileged=True)
    buf = pipe.make_buffer(
        obs_dim=18, action_dim=6, buffer_size=1000, critic_obs_dim=64
    )
    assert buf._extra_obs_dims == {"critic_obs": 64}


def test_make_buffer_extras_only():
    """New generic extras — no critic. Only skill_z + factor_obs."""
    pipe = _make_pipeline(has_privileged=False)
    buf = pipe.make_buffer(
        obs_dim=18, action_dim=6, buffer_size=1000,
        extra_obs_dims={"skill_z": 8, "factor_obs": 18},
    )
    assert buf._extra_obs_dims == {"skill_z": 8, "factor_obs": 18}


def test_make_buffer_critic_plus_extras_merged():
    """Both critic_obs and generic extras — should merge into one dict."""
    pipe = _make_pipeline(has_privileged=True)
    buf = pipe.make_buffer(
        obs_dim=18, action_dim=6, buffer_size=1000,
        critic_obs_dim=64,
        extra_obs_dims={"skill_z": 8, "factor_obs": 18},
    )
    assert buf._extra_obs_dims == {"critic_obs": 64, "skill_z": 8, "factor_obs": 18}


def test_make_buffer_critic_obs_in_extras_raises():
    """Caller should not pass critic_obs in extra_obs_dims when has_privileged=True
    — that would conflict with the auto-injected one. Raise to prevent silent overwrite."""
    pipe = _make_pipeline(has_privileged=True)
    with pytest.raises(ValueError, match="critic_obs"):
        pipe.make_buffer(
            obs_dim=18, action_dim=6, buffer_size=1000,
            critic_obs_dim=64,
            extra_obs_dims={"critic_obs": 999},  # would conflict
        )


def test_make_buffer_critic_required_when_privileged():
    """Existing constraint preserved — critic_obs_dim required when has_privileged."""
    pipe = _make_pipeline(has_privileged=True)
    with pytest.raises(ValueError, match="critic_obs_dim"):
        pipe.make_buffer(obs_dim=18, action_dim=6, buffer_size=1000)
```

- [ ] Run: `uv run python -m pytest tests/test_obs_pipeline_extras.py -v` → expect failures (`extra_obs_dims` kwarg not yet supported).

### Step 1.2: Implement

In `jax_rl/training/obs_pipeline.py::ObsPipeline.make_buffer`:

```python
def make_buffer(self, obs_dim, action_dim, buffer_size,
                critic_obs_dim=None, num_envs=None,
                extra_obs_dims=None):
    """Create JaxReplayBuffer with correct frame_stack + extra_obs_dims.

    Args:
        obs_dim, action_dim, buffer_size: as before.
        critic_obs_dim: Privileged critic obs dim. Required when has_privileged.
        num_envs: required when n_frame_stack > 1.
        extra_obs_dims: dict[str, int] | None — additional named extras (e.g.
            {"skill_z": 8, "factor_obs": 18}). Merged with auto-injected
            critic_obs when has_privileged. Caller must not put 'critic_obs'
            in extra_obs_dims; that key is reserved for the privileged path.
    """
    merged_extras = dict(extra_obs_dims) if extra_obs_dims else {}

    if self.has_privileged:
        if critic_obs_dim is None:
            raise ValueError("critic_obs_dim required when has_privileged=True")
        if "critic_obs" in merged_extras:
            raise ValueError(
                "extra_obs_dims must not contain 'critic_obs' when has_privileged=True; "
                "use critic_obs_dim arg instead"
            )
        merged_extras["critic_obs"] = critic_obs_dim

    # frame_stack path (unchanged)
    frame_stack_config = None
    if self.n_frame_stack > 1:
        if num_envs is None:
            raise ValueError("num_envs required when n_frame_stack > 1")
        raw_dim = obs_dim // self.n_frame_stack
        frame_stack_config = FrameStackConfig(
            n_frames=self.n_frame_stack, raw_dim=raw_dim, num_envs=num_envs
        )
        return JaxReplayBuffer(
            raw_dim, action_dim, max_size=buffer_size,
            frame_stack_config=frame_stack_config,
            extra_obs_dims=merged_extras or None,
        )

    return JaxReplayBuffer(
        obs_dim, action_dim, max_size=buffer_size,
        extra_obs_dims=merged_extras or None,
    )
```

- [ ] Run: `uv run python -m pytest tests/test_obs_pipeline_extras.py -v` → expect 6 passed.
- [ ] Run: `uv run python -m pytest tests/test_jax_replay_buffer.py tests/test_obs_pipeline.py -v` → existing critic_obs tests still green (non-regression).

### Step 1.3: Commit

```bash
git add jax_rl/training/obs_pipeline.py tests/test_obs_pipeline_extras.py
git commit -m "feat(skill): ObsPipeline.make_buffer accepts generic extra_obs_dims"
```

No `Co-Authored-By` line.

---

## Task 2: Skill checkpoint wrapper

**Files:**
- Create: `jax_rl/skill_discovery/checkpointing.py`
- Create: `tests/test_skill_checkpoint_contract.py`

**Goal:** Wrap the base `CheckpointManager` to additionally save/restore the skill auxiliary state (per-factor params + opt_state) and patch `meta.json` with a `skill_discovery` block. Resume must restore both base training state and skill aux state.

### Step 2.1: Write tests (RED)

```python
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
    # Required fields per spec lines 333-352:
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
```

- [ ] Run: `uv run python -m pytest tests/test_skill_checkpoint_contract.py -v` → expect ImportError.

### Step 2.2: Implement

`jax_rl/skill_discovery/checkpointing.py`:

```python
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
```

- [ ] Run: `uv run python -m pytest tests/test_skill_checkpoint_contract.py -v` → expect 4 passed.

### Step 2.3: Commit

```bash
git add jax_rl/skill_discovery/checkpointing.py tests/test_skill_checkpoint_contract.py
git commit -m "feat(skill): add skill aux checkpoint save/load + meta.json block"
```

---

## Task 3: skill_offpolicy_loop

**Files:**
- Create: `jax_rl/training/skill_offpolicy_loop.py`
- Create: `tests/test_skill_offpolicy_loop.py`

**Depends on Tasks 1 + 2.** Run after Wave A commits land.

**Goal:** new function `run_skill_offpolicy_loop(...)` that mirrors `run_offpolicy_loop` but adds skill discovery hooks. Per spec §"V2 implementation shape" — "Create a dedicated loop first instead of generalizing `run_offpolicy_loop` prematurely."

**Spec gradient-step order** (spec lines 432-442):
1. Sample replay batch.
2. Normalize raw obs through ObsPipeline.
3. Append `skill_z` to actor and critic obs.
4. Compute intrinsic reward using current aux state.
5. Compose final batch reward (sample-time replacement).
6. Update actor/critic with the composed reward.
7. Update aux params + opt state on the same batch.

Actor/critic gradients must NOT flow through aux networks.

### Step 3.1: Write tests (RED)

`tests/test_skill_offpolicy_loop.py`:

```python
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
    skill-manager-computed intrinsic reward (modulo intrinsic_weight scaling)."""
    cfg = _make_cfg(4)
    cfg.intrinsic_weight = 1.0
    cfg.task_reward_weight = 0.0
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
```

- [ ] Run: `uv run python -m pytest tests/test_skill_offpolicy_loop.py -v` → expect ImportError.

### Step 3.2: Implement

`jax_rl/training/skill_offpolicy_loop.py`:

The full loop is large (~250 lines). Follow `jax_rl/training/offpolicy_loop.py::run_offpolicy_loop` structure, with these additions:

1. **Manager init at startup**: take `skill_cfg: SkillDiscoveryConfig` arg + `skill_manager: SkillManager` (constructed at script level). Call `mgr.init(key)` → `aux_state`. Init per-env `current_z = mgr.sample_skills(key, num_envs)`.

2. **Buffer**: include `skill_z`, `next_skill_z`, `factor_obs`, `next_factor_obs` as named extras in `pipeline.make_buffer(..., extra_obs_dims={"skill_z": skill_dim, "next_skill_z": skill_dim, "factor_obs": ..., "next_factor_obs": ...})`. **For SD-B with single `actor_obs_full` factor**, `factor_obs == obs` (same data) — could omit the extra storage to save buffer memory. Decision: store them anyway for forward-compat with multi-factor configs in SD-E.

3. **Collection step**: after env step, store `(obs, action, reward_env_unscaled, next_obs, done, truncation, skill_z=current_z, next_skill_z=current_z)` per the spec lifecycle rule (next_skill_z=skill_z under `resample="episode"`).

4. **Resample on done**: `current_z = mgr.resample_on_done(current_z, env_state.done, key)`.

5. **Gradient-step inner loop**: factor out two helpers:
   - `compose_skill_batch(raw_batch, raw_obs_dim, skill_dim)`: returns batch with `obs = concat(obs_normed, skill_z)` and `next_obs = concat(next_obs_normed, next_skill_z)`. Critic obs analogously if asymmetric.
   - `intrinsic_reward_then_update(composed_batch, raw_batch, mgr, aux_state, skill_cfg, reward_scaling)`: computes intrinsic reward via `mgr.compute_intrinsic_reward(aux_state, raw_batch)`, composes final reward `intrinsic_weight * intrinsic + task_reward_weight * batch["reward"] * reward_scaling` (style/safety zero in SD-B), overwrites `batch["reward"]`. Returns `(composed_batch, intrinsic_reward)` for logging.

6. **Algo update**: `training_state, sac_metrics = algo.update(training_state, composed_batch)` — algo receives augmented obs, sees the composed reward.

7. **Aux update** (after algo update, on the same sampled batch): `aux_state, aux_metrics = mgr.update(aux_state, raw_batch)`. **Important**: actor/critic gradients must not flow through aux nets — but since the algo update happened first with `aux_state` fixed, and aux update uses `raw_batch` (not the composed one), this is naturally enforced.

8. **Checkpoint**: at save points, call existing `CheckpointManager.save(...)` then `save_skill_aux_state(aux_state, ckpt_dir)` and patch `meta.json` with `write_skill_meta_block(skill_cfg)`.

9. **Resume**: at startup if `--resume`, load base state via `load_checkpoint`, then `load_skill_aux_state(ckpt_dir, template=aux_state)` to restore aux.

10. **Logging**: extend metrics with per-factor `f"{name}_disc_loss"`, `f"{name}_disc_accuracy"`, `intrinsic_reward_mean`, `intrinsic_reward_std`, env-reward (informational).

- [ ] Run: `uv run python -m pytest tests/test_skill_offpolicy_loop.py -v` → expect 3 passed.

### Step 3.3: Commit

```bash
git add jax_rl/training/skill_offpolicy_loop.py tests/test_skill_offpolicy_loop.py
git commit -m "feat(skill): add skill_offpolicy_loop with sample-time intrinsic reward + aux update"
```

---

## Task 4: train_skill_discovery.py CLI

**Files:**
- Create: `scripts/train_skill_discovery.py`

**Depends on Task 3.**

**Goal:** CLI entry point that mirrors `scripts/train_sac.py` shape (124 lines), adds skill discovery surface (skill_cfg construction, manager init, hand-off to `run_skill_offpolicy_loop`).

### Step 4.1: Implement

Mirror `scripts/train_sac.py`:

```python
"""DIAYN skill discovery training script (vanilla SAC + SkillManager).

Usage:
    uv run python scripts/train_skill_discovery.py --env CheetahRun --num-skills 8
    uv run python scripts/train_skill_discovery.py --env CheetahRun --num-skills 8 --total-timesteps 1000000 --wandb
"""
import argparse
import dataclasses
import optax

from jax_rl.algos.sac import SAC
from jax_rl.configs.env_presets import get_sac_preset
from jax_rl.training.env_setup import make_env_bundle
from jax_rl.training.skill_offpolicy_loop import run_skill_offpolicy_loop
from jax_rl.skill_discovery.config import (
    SkillDiscoveryConfig, FactorConfig, SkillDeployConfig,
)
from jax_rl.skill_discovery.factors import register_extractor
from jax_rl.skill_discovery.manager import SkillManager


# Register the SD-B default factor extractor: full actor obs.
# The built-in `actor_obs_full` from factors.py uses dim=-1 sentinel; SD-B
# specifies the env-specific dim explicitly via FactorConfig.dim at runtime.


def _build_skill_cfg(num_skills: int, obs_dim: int) -> SkillDiscoveryConfig:
    """SD-B default: single DIAYN factor over full actor obs."""
    return SkillDiscoveryConfig(
        mode="diayn",
        total_skill_dim=num_skills,
        prior="one_hot",
        resample="episode",
        intrinsic_weight=1.0,
        task_reward_weight=0.0,  # pure DIAYN for SD-B; SD-C adds task mix
        factors=(FactorConfig(
            name="full_state",
            method="diayn",
            skill_dim=num_skills,
            source="actor_obs",
            extractor="actor_obs_full",
            dim=obs_dim,
        ),),
        deploy=SkillDeployConfig(
            skill_input_mode="fixed",
            default_skill=None,  # set at deploy time
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="DIAYN skill discovery training")
    parser.add_argument("--env", type=str, default="CheetahRun")
    parser.add_argument("--num-skills", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-norm", action="store_true")
    parser.add_argument("--reset-mode", type=str, default=None,
                        choices=["per_step", "per_episode", None])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="jax-rl")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    cfg, algo_cfg = get_sac_preset(args.env)
    if args.total_timesteps:
        cfg = dataclasses.replace(cfg, total_timesteps=args.total_timesteps)
    if args.obs_norm:
        cfg = dataclasses.replace(cfg, obs_norm=True)
    if args.reset_mode:
        cfg = dataclasses.replace(cfg, reset_mode=args.reset_mode)

    # Build env bundle to learn obs/action dims
    env_bundle = make_env_bundle(cfg, args.seed)
    obs_dim = env_bundle.obs_dim
    action_dim = env_bundle.action_dim

    # Skill discovery setup
    skill_cfg = _build_skill_cfg(args.num_skills, obs_dim)
    skill_manager = SkillManager(skill_cfg)

    # SAC sees augmented obs (raw obs + skill_z)
    augmented_obs_dim = obs_dim + skill_cfg.total_skill_dim

    # Optimizer: vanilla SAC w/ optional grad clipping
    if algo_cfg.grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(algo_cfg.grad_clip_norm),
            optax.adam(cfg.lr),
        )
    else:
        optimizer = optax.adam(cfg.lr)

    algo = SAC(
        algo_cfg=algo_cfg,
        obs_dim=augmented_obs_dim,  # actor sees obs + skill_z
        action_dim=action_dim,
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
        gamma=cfg.gamma,
    )

    def explore_fn(actor_params, obs, key):
        return algo.actor.apply(actor_params, obs, key)["action"]

    run_skill_offpolicy_loop(
        cfg=cfg,
        algo_cfg=algo_cfg,
        algo=algo,
        algo_name="sac_skill",
        env_bundle=env_bundle,
        explore_fn=explore_fn,
        log_extra_fields=["intrinsic_reward_mean", "full_state_disc_loss",
                          "full_state_disc_accuracy"],
        log_extra_keys=["intrinsic_reward_mean", "full_state_disc_loss",
                        "full_state_disc_accuracy"],
        skill_cfg=skill_cfg,
        skill_manager=skill_manager,
        seed=args.seed,
        resume=args.resume,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
```

- [ ] Manual smoke: `uv run python scripts/train_skill_discovery.py --env CheetahRun --num-skills 8 --total-timesteps 10000` → expect to start, run 10K steps, no NaN, finite aux losses.

### Step 4.2: Commit

```bash
git add scripts/train_skill_discovery.py
git commit -m "feat(skill): add train_skill_discovery.py — vanilla SAC + DIAYN on CheetahRun"
```

---

## Task 5: Smoke + 100K + 1M acceptance runs

**Files:** none (runs only, plus `.context/journals/2026-05-XX.md` entry).

**Depends on Tasks 1-4.**

**Goal:** validate SD-B against acceptance gates. Three runs at increasing scale.

### Step 5.1: 10K CPU smoke

- [ ] `uv run python scripts/train_skill_discovery.py --env CheetahRun --num-skills 8 --total-timesteps 10000 --seed 0`
- [ ] Verify: no NaN, buffer fills, aux losses finite, checkpoint writes.
- [ ] Runtime: minutes on CPU; seconds on GPU.

### Step 5.2: 100K validation

- [ ] `uv run python scripts/train_skill_discovery.py --env CheetahRun --num-skills 8 --total-timesteps 100000 --seed 0` (background recommended; monitor via `grep "EVAL\|disc_acc" <output> | tail -10`).
- [ ] Acceptance: discriminator accuracy curve shows mean accuracy > `1/num_skills + 0.1 = 0.225` by step 100K. Plot accuracy vs steps from `metrics.csv`.
- [ ] If accuracy plateaus at chance: try `--num-skills 20` per DIAYN reference; if still stuck, try `--num-skills 50`. **Verify the exact section/figure of the DIAYN paper before tuning higher** — paper text uses both 20 and 50 across figures (don't pick 50 just because it's the source default; re-read paper for the env-specific recommendation).

### Step 5.3: 1M acceptance run + figure generation

- [ ] **3 seeds:** seed 0, 1, 2. Run all three in background.
- [ ] `uv run python scripts/train_skill_discovery.py --env CheetahRun --num-skills 8 --total-timesteps 1000000 --seed <s>`
- [ ] Total runtime: ~30 min/seed on GPU.
- [ ] **Per-skill task-return histogram (DIAYN App. D.4 protocol):** load best checkpoint per seed; for each fixed `z = one_hot(i)` for `i in range(8)`, run M=10 episodes; record env reward (forward velocity for CheetahRun). Compute mean ± std per skill. Spread (max − min across skills) > 50% of any single skill's mean return → PASS.
- [ ] **Per-skill rendered videos:** `MUJOCO_GL=egl uv run python scripts/record_video.py --checkpoint <best_ckpt> --skill-index <0..7>` → 8 videos. Visually inspect for behavioral diversity. Subjective check; "running fwd / running back / hopping / standing / etc." level of distinctness.
- [ ] **Cross-seed agreement:** verify all 3 seeds show > chance discriminator accuracy and qualitatively distinct skills (the *which* skills emerge varies by seed; that's expected).

### Step 5.4: Journal entry + acceptance commit

- [ ] Write `.context/journals/2026-05-XX-skill-discovery-sd-b-validation.md`:
  - 10K smoke results (NaN/no-NaN, runtime)
  - 100K accuracy curve summary
  - 1M per-skill return spread per seed
  - Qualitative video notes per seed (which skills emerged)
  - Comparisons against DIAYN App. D.4 if applicable
- [ ] No code commit needed for runs themselves — figures + journal land in `.context/journals/`.

```bash
git add .context/journals/2026-05-XX-skill-discovery-sd-b-validation.md
git commit -m "docs(skill): SD-B validation — DIAYN on CheetahRun, 3 seeds, 1M steps"
```

---

## SD-B Wrap-up

After Task 5 lands:

- [ ] Update `.context/TODO.md` SD-B section → mark complete, point next-action at SD-C.
- [ ] If we want the canonical DIAYN Ant xy-coverage figure: run CPU `Ant-v5` via existing `gym_backend.py:200`, generate the 8-color xy plot once. Defer to "post-SD-B before SD-C" if compute/time constrained.
- [ ] Lessons file additions: anything surprising (DIAYN reward magnitude on CheetahRun, FastSAC bin estimates if we measured, SAC reward-scale interactions with intrinsic reward).

## Pointers for SD-C (next plan, do not write yet)

- Switch env to `Go2WarpJoystickUnitree` (45d hardware-conservative obs, action_scale=0.25)
- Replace single `actor_obs_full` factor with command-conditioned behavior class factor (named extractor, sim_data source)
- Add `meta["skill_discovery"]` deploy contract block to checkpoint
- Per-skill rollout protocol: M=10 episodes per fixed z, report mean ± std return + episode length + command-tracking error + per-leg torque RMS
- Switch to FastSAC (with V_min/V_max tuned from SD-B intrinsic reward range observations)

## Open implementation questions to resolve before each task starts

| Question | Resolve by | Resolution path |
|---|---|---|
| `JaxReplayBuffer.extra_obs_dims` already supports the constructor pattern? | Task 1 start | Verified yes per source extracts (`jax_replay_buffer.py:36`); pipeline-side wrapper is the only edit needed. |
| How are aux state PyTrees flattened/unflattened across save→load? | Task 2 start | `jax.tree_util.tree_flatten` + `tree_unflatten` with template pattern (resume requires a freshly-init'd manager state for structure). |
| Does the existing `run_offpolicy_loop` expose hooks we can call into, or do we copy code? | Task 3 start | Spec says "create dedicated loop first, don't generalize prematurely" → copy + extend. Refactor to shared hooks lands at SD-E if duplication becomes painful. |
| Does CheetahRun (MJX playground) have `EnvBundle.has_privileged`? | Task 4 start | Verify via `make_env_bundle("CheetahRun", 0)`; expect `has_privileged=False` (DM Control envs are flat-obs). If True, extra_obs_dims merge logic from Task 1 covers it. |
| Vanilla SAC config: which preset works for CheetahRun? | Task 4 start | `get_sac_preset("CheetahRun")` — verify exists. If not, copy WalkerWalk preset and rename. |
| Discriminator accuracy threshold for SD-B PASS: `1/num_skills + 0.1` or higher? | Task 5.2 start | DIAYN paper doesn't fix this; our convention is "above chance" (see validation doc Part 4 SD-B). If 100K stalls at chance, escalate num_skills before declaring failure. |

## Skipped from spec SD-B (deferred to later phases)

- **Frame-stack support**: spec line 444 says "allow only `resample='episode'` when `n_frame_stack > 1`. Fixed-step resampling with frame stack needs skill history in replay and deploy, so defer it until after the episode-skill path works." SD-B uses `n_frame_stack=1` (CheetahRun default); frame-stack interaction is SD-C/E concern.
- **Style / safety / task reward mix**: SD-B is pure DIAYN (`intrinsic_weight=1.0, task_reward_weight=0.0`). SD-C adds nonzero task/style/safety mix per spec.
- **METRA factor type**: SD-A's `SkillManager.__init__` raises `NotImplementedError("METRA deferred to SD-E")` for any METRA factor. SD-B configs only use DIAYN factors.
- **FastSAC**: deferred (see "Algo choice" above).
- **Asymmetric critic for skill discovery**: CheetahRun is symmetric obs (no privileged); SD-C wires the asymmetric path on Go2.
