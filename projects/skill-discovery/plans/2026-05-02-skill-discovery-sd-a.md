# Skill Discovery SD-A — Contract and Scaffolding

**Date:** 2026-05-02
**Spec:** `specs/2026-04-28-skill-discovery.md`
**Validation methodology:** `references/skill_discovery_validation.md` ← paper-grounded eval contracts and method tradeoffs
**Source extracts:** `references/skill_discovery_source_extracts.md` ← ground-truth values from DIAYN, METRA, D3, DADS, DUSDi, SkiLD official repos (file:line cites)
**Phase:** SD-A (first of SD-A → SD-E)
**Status:** ready

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Strict TDD: test → fail → impl → pass → commit per step. Tickbox each step as you go.

## Goal

Build the skill discovery scaffolding — config, priors, factor extractor registry, DIAYN aux module, and `SkillManager` — as pure, env-agnostic, training-loop-agnostic code with full unit-test coverage.

**Out of scope for SD-A:**
- METRA (deferred to SD-E)
- env wiring or training script (SD-B)
- replay buffer changes (SD-B)
- checkpoint/deploy contract (SD-D)
- Dirichlet/hypersphere priors (SD-E — SD-A only ships `one_hot`)

## Acceptance for SD-A as a whole

From spec SD-A "Acceptance" block:

- [ ] Config round-trips through JSON.
- [ ] Skill priors sample expected shapes (one_hot only in SD-A).
- [ ] `SkillManager.augment_actor_obs(obs, z)` is pure and shape-checked.
- [ ] `SkillManager.compute_intrinsic_reward(...)` is deterministic for a fixed aux state and batch.
- [ ] Aux network update is a pure function and reduces loss on synthetic data.
- [ ] All new tests pass: `uv run python -m pytest tests/test_skill_discovery_*.py -v`.
- [ ] `uv run python -m pytest tests/ -v` still green (no regressions).

## File structure

```
New files:
  jax_rl/skill_discovery/__init__.py        — package init, public API
  jax_rl/skill_discovery/config.py          — SkillDiscoveryConfig, FactorConfig, SkillDeployConfig
  jax_rl/skill_discovery/prior.py           — sample_one_hot (Dirichlet/hypersphere stubs in SD-E)
  jax_rl/skill_discovery/factors.py         — FactorExtractor registry, source/extractor resolution
  jax_rl/skill_discovery/diayn.py           — Discriminator + diayn_reward + diayn_update
  jax_rl/skill_discovery/manager.py         — SkillManager (init/sample/resample/augment/reward/update)
  tests/test_skill_discovery_config.py
  tests/test_skill_discovery_prior.py
  tests/test_skill_discovery_factors.py
  tests/test_skill_discovery_diayn.py
  tests/test_skill_discovery_manager.py
```

## Independence map (for parallel subagent dispatch)

```
Wave 1 (parallel — see package-init note below):
  ├── Task 1: config.py + test_config        — pure dataclasses, JSON round-trip
  ├── Task 2: prior.py + test_prior          — depends only on JAX (no internal deps)
  ├── Task 3: factors.py + test_factors      — uses FactorConfig from Task 1*
  └── Task 4: diayn.py + test_diayn          — depends on networks/activations only

Wave 2 (after Wave 1):
  └── Task 5: manager.py + test_manager      — composes all of Wave 1
                                              + integrates priors + diayn + factors
```

**Package-init note:** every Wave 1 task must defensively `Write` the
`jax_rl/skill_discovery/__init__.py` file containing a single one-line
docstring (`"""Skill discovery framework — DIAYN (SD-A) + METRA (SD-E) as pluggable aux modules."""`).
All four tasks write the same content; last-write-wins is safe (idempotent).
This lets the package import work for each task's tests independently without
requiring a "Task 0" or serial ordering. The first task's commit creates the
file; subsequent tasks' identical writes produce no diff.

***Task 3 imports `from jax_rl.skill_discovery.config import FactorConfig`** —
this requires Task 1's `config.py` to exist on disk by the time Task 3's tests
run. If Tasks 1 and 3 dispatch truly concurrently, Task 3's RED step will fail
with `ModuleNotFoundError` until Task 1 has at least written `config.py` (no
need to wait for Task 1's full commit). In practice, dispatching all 4 in
parallel is fine — pytest will retry once Task 1's file lands. Alternatively,
serialize: dispatch Task 1, then dispatch Tasks 2/3/4 in parallel after
Task 1's commit.

Wave 1 = 4 subagents (parallel with above caveat). Wave 2 = 1 agent after Wave 1 commits land.

---

## Task 1: Config dataclasses

**Files:** `jax_rl/skill_discovery/__init__.py`, `jax_rl/skill_discovery/config.py`, `tests/test_skill_discovery_config.py`

Pure dataclasses. No JAX. No I/O beyond JSON round-trip helpers.

### Step 1.1: Write tests (RED)

`tests/test_skill_discovery_config.py`:

```python
"""Tests for skill discovery config dataclasses."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.skill_discovery.config import (
    SkillDiscoveryConfig, FactorConfig, SkillDeployConfig,
    config_to_dict, config_from_dict,
)


def test_factor_config_minimal():
    fc = FactorConfig(
        name="full_state", method="diayn", skill_dim=8,
        source="actor_obs", extractor="full", dim=48,
    )
    assert fc.name == "full_state"
    assert fc.method == "diayn"


def test_skill_discovery_config_defaults():
    cfg = SkillDiscoveryConfig()
    assert cfg.enabled is True
    assert cfg.mode == "diayn"
    assert cfg.prior == "one_hot"
    assert cfg.resample == "episode"
    assert cfg.reward_mode == "sample_time"
    assert cfg.intrinsic_weight == 1.0
    assert cfg.task_reward_weight == 0.0
    assert cfg.style_reward_weight == 0.0
    assert cfg.safety_penalty_weight == 0.0
    assert cfg.factors == ()
    assert isinstance(cfg.deploy, SkillDeployConfig)


def test_skill_deploy_config_defaults():
    d = SkillDeployConfig()
    assert d.skill_input_mode == "fixed"
    assert d.default_skill is None  # set at deploy time


def test_config_json_round_trip():
    cfg = SkillDiscoveryConfig(
        mode="diayn",
        total_skill_dim=8,
        factors=(
            FactorConfig(name="full_state", method="diayn", skill_dim=8,
                         source="actor_obs", extractor="full", dim=48),
        ),
        deploy=SkillDeployConfig(
            skill_input_mode="fixed",
            default_skill=[1, 0, 0, 0, 0, 0, 0, 0],
        ),
    )
    d = config_to_dict(cfg)
    s = json.dumps(d)
    cfg2 = config_from_dict(json.loads(s))
    assert cfg2 == cfg


def test_config_validation_total_skill_dim_matches_factors():
    """If factors are set, total_skill_dim must match sum(skill_dim)."""
    import pytest
    with pytest.raises(ValueError, match="total_skill_dim"):
        SkillDiscoveryConfig(
            total_skill_dim=4,  # wrong — factors sum to 8
            factors=(
                FactorConfig(name="a", method="diayn", skill_dim=8,
                             source="actor_obs", extractor="full", dim=48),
            ),
        )


def test_config_resample_steps_required_when_fixed():
    import pytest
    with pytest.raises(ValueError, match="resample_steps"):
        SkillDiscoveryConfig(resample="fixed_steps", resample_steps=None)
```

- [ ] Run: `uv run python -m pytest tests/test_skill_discovery_config.py -v` → expect ImportError.

### Step 1.2: Implement (GREEN)

`jax_rl/skill_discovery/__init__.py`:
```python
"""Skill discovery framework — DIAYN (SD-A) + METRA (SD-E) as pluggable aux modules."""
```

`jax_rl/skill_discovery/config.py`:
- `SkillDeployConfig` — `skill_input_mode: Literal["fixed", "operator", "external"]`, `default_skill: list[float] | None`
- `FactorConfig` — `name`, `method: Literal["diayn", "metra"]`, `skill_dim: int`, `source: Literal["actor_obs","critic_obs","sim_data","info"]`, `extractor: str`, `dim: int`
- `SkillDiscoveryConfig` — fields per spec §SD-A "Core config" block + `__post_init__` validation:
  - if `factors`: assert `total_skill_dim == sum(f.skill_dim for f in factors)`
  - if `resample == "fixed_steps"`: assert `resample_steps is not None`
- **DO NOT include `AuxNetConfig` in SD-A.** It's referenced in the spec for forward-compat but is deferred to SD-C plumb-through; SD-A hardcodes Discriminator/optimizer defaults at the manager site (see Task 5).
- **`__post_init__` empty-factors guard:** wrap the `total_skill_dim == sum(factor.skill_dim)` check in `if self.factors:` so a default `SkillDiscoveryConfig()` (factors=()) is valid. Without the guard, `0 == sum([])` happens to pass anyway, but explicit gating documents intent.
- **`SkillDiscoveryConfig.deploy` must use `field(default_factory=SkillDeployConfig)`**, NOT `= SkillDeployConfig()`. Bare instances as defaults raise `ValueError: mutable default ... not allowed` at class definition. (Same applies to any nested dataclass default.)
- `config_to_dict(cfg) -> dict` — recursive dataclass → dict (use `dataclasses.asdict` + tuple→list conversion for the `factors` field)
- `config_from_dict(d) -> SkillDiscoveryConfig` — inverse: convert `factors` list → tuple of `FactorConfig(**f)`, rebuild `deploy` as `SkillDeployConfig(**d['deploy'])`. A bare `SkillDiscoveryConfig(**d)` will not work because nested dataclasses don't auto-rehydrate.

- [ ] Run: `uv run python -m pytest tests/test_skill_discovery_config.py -v` → expect 6 passed.

### Step 1.3: Commit

```bash
git add jax_rl/skill_discovery/__init__.py jax_rl/skill_discovery/config.py tests/test_skill_discovery_config.py
git commit -m "feat(skill): add SkillDiscoveryConfig with factor/deploy types and JSON round-trip"
```

---

## Task 2: Skill priors

**Files:** `jax_rl/skill_discovery/prior.py`, `tests/test_skill_discovery_prior.py`

SD-A ships `one_hot` only. Dirichlet/hypersphere stub raises `NotImplementedError` until SD-E.

### Step 2.1: Write tests (RED)

```python
"""Tests for skill priors."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import pytest

from jax_rl.skill_discovery.prior import sample_skill, validate_skill

KEY = jax.random.PRNGKey(0)


def test_one_hot_shape_and_values():
    z = sample_skill(KEY, prior="one_hot", num_envs=16, skill_dim=8)
    assert z.shape == (16, 8)
    # one-hot: each row sums to 1, all entries 0 or 1
    assert jnp.allclose(z.sum(axis=-1), 1.0)
    assert jnp.all((z == 0) | (z == 1))


def test_one_hot_uniform_distribution_over_skills():
    """Sampling many envs should approximate uniform over skill indices."""
    z = sample_skill(KEY, prior="one_hot", num_envs=10_000, skill_dim=4)
    counts = z.sum(axis=0)
    # each skill should get ~2500 ± 200 (3-sigma loose bound)
    assert jnp.all(counts > 2000)
    assert jnp.all(counts < 3000)


def test_one_hot_deterministic_under_same_key():
    z1 = sample_skill(KEY, prior="one_hot", num_envs=8, skill_dim=4)
    z2 = sample_skill(KEY, prior="one_hot", num_envs=8, skill_dim=4)
    assert jnp.array_equal(z1, z2)


def test_dirichlet_not_implemented_in_sd_a():
    with pytest.raises(NotImplementedError, match="SD-E"):
        sample_skill(KEY, prior="dirichlet", num_envs=4, skill_dim=4)


def test_hypersphere_not_implemented_in_sd_a():
    with pytest.raises(NotImplementedError, match="SD-E"):
        sample_skill(KEY, prior="hypersphere", num_envs=4, skill_dim=2)


def test_validate_skill_one_hot():
    z = jax.nn.one_hot(jnp.arange(4), 4)
    validate_skill(z, prior="one_hot", skill_dim=4)  # no raise

    bad = jnp.ones((4, 4)) * 0.5  # not one-hot
    with pytest.raises(ValueError, match="one-hot"):
        validate_skill(bad, prior="one_hot", skill_dim=4)
```

- [ ] Run → expect ImportError.

### Step 2.2: Implement

```python
def sample_skill(key, prior, num_envs, skill_dim):
    if prior == "one_hot":
        idx = jax.random.randint(key, (num_envs,), 0, skill_dim)
        return jax.nn.one_hot(idx, skill_dim)
    if prior == "dirichlet":
        raise NotImplementedError("Dirichlet prior deferred to SD-E")
    if prior == "hypersphere":
        raise NotImplementedError("Hypersphere prior deferred to SD-E")
    raise ValueError(f"unknown prior: {prior}")


def validate_skill(z, prior, skill_dim):
    if z.shape[-1] != skill_dim:
        raise ValueError(f"skill_dim mismatch: got {z.shape[-1]}, expected {skill_dim}")
    if prior == "one_hot":
        sums = z.sum(axis=-1)
        if not jnp.allclose(sums, 1.0):
            raise ValueError("one-hot prior requires rows to sum to 1")
        if not jnp.all((z == 0) | (z == 1)):
            raise ValueError("one-hot prior requires 0/1 entries")
```

- [ ] Run → expect 6 passed.

### Step 2.3: Commit

```bash
git add jax_rl/skill_discovery/prior.py tests/test_skill_discovery_prior.py
git commit -m "feat(skill): add one_hot prior; Dirichlet/hypersphere deferred to SD-E"
```

---

## Task 3: Factor extractor registry

**Files:** `jax_rl/skill_discovery/factors.py`, `tests/test_skill_discovery_factors.py`

A factor extractor is a named function `(batch) -> jax.Array` that pulls factor inputs out of a replay batch. Each `FactorConfig` references one by name + source.

**Why a registry, not magic indices:** spec §"Design corrections from the old plan" #1. Decouples policy obs layout from auxiliary reward inputs. Extractors can read `sim_data` / `info` that aren't in deploy actor obs.

(Batch dict keys + actor_obs_full built-in spec are inlined in Step 3.2 below.)

### Step 3.1: Write tests (RED)

```python
"""Tests for factor extractor registry."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import pytest

from jax_rl.skill_discovery.factors import (
    FactorExtractor, register_extractor, get_extractor, resolve_factor,
)
from jax_rl.skill_discovery.config import FactorConfig

KEY = jax.random.PRNGKey(0)


def test_register_and_get_extractor():
    @register_extractor(name="test_full_actor", source="actor_obs", dim=48)
    def _extract(batch):
        return batch["obs"]

    ext = get_extractor("test_full_actor")
    assert ext.name == "test_full_actor"
    assert ext.source == "actor_obs"
    assert ext.dim == 48


def test_get_unknown_extractor_raises():
    with pytest.raises(KeyError, match="unknown extractor"):
        get_extractor("definitely_not_registered_xyz")


def test_resolve_factor_pulls_named_extractor():
    @register_extractor(name="test_resolve_xy", source="sim_data", dim=2)
    def _extract(batch):
        return batch["sim_data"][:, :2]

    fc = FactorConfig(name="pos", method="metra", skill_dim=2,
                      source="sim_data", extractor="test_resolve_xy", dim=2)
    batch = {"sim_data": jax.random.normal(KEY, (4, 6))}
    out = resolve_factor(fc, batch)
    assert out.shape == (4, 2)
    assert jnp.array_equal(out, batch["sim_data"][:, :2])


def test_resolve_factor_dim_mismatch_raises():
    @register_extractor(name="test_dim_mismatch", source="sim_data", dim=2)
    def _extract(batch):
        return batch["sim_data"][:, :3]  # returns 3, declared 2

    fc = FactorConfig(name="bad", method="metra", skill_dim=2,
                      source="sim_data", extractor="test_dim_mismatch", dim=2)
    batch = {"sim_data": jax.random.normal(KEY, (4, 6))}
    with pytest.raises(ValueError, match="dim mismatch"):
        resolve_factor(fc, batch)


def test_builtin_extractors_actor_obs_full():
    """Built-in: full actor_obs passthrough."""
    fc = FactorConfig(name="all", method="diayn", skill_dim=4,
                      source="actor_obs", extractor="actor_obs_full", dim=48)
    batch = {"obs": jax.random.normal(KEY, (4, 48))}
    out = resolve_factor(fc, batch)
    assert jnp.array_equal(out, batch["obs"])
```

- [ ] Run → expect ImportError.

### Step 3.2: Implement

- `FactorExtractor` dataclass: `name`, `source`, `dim` (use `-1` as sentinel meaning "any dim, skip extractor-side check"), `fn: Callable[[dict], jax.Array]`
- Module-level `_REGISTRY: dict[str, FactorExtractor]`
- `register_extractor(name, source, dim)` — decorator. **Duplicate-registration policy: last-write-wins (silent overwrite).** This makes registration idempotent across `pytest-xdist` re-imports and avoids module-level errors during test collection. If hard error-on-duplicate is preferred later, flip in SD-B.
- `get_extractor(name)` — lookup, raise `KeyError("unknown extractor: ...")` if missing
- `resolve_factor(factor: FactorConfig, batch: dict) -> jax.Array` — get extractor, run, **validate against `factor.dim` only** (not `ext.dim`). The config is the source of truth at validation time; `ext.dim == -1` sentinel skips the extractor-level check entirely. Raise `ValueError("dim mismatch: ...")` if `out.shape[-1] != factor.dim`.
- Built-in extractor: `@register_extractor(name="actor_obs_full", source="actor_obs", dim=-1)` returning `batch["obs"]`. The `dim=-1` sentinel lets this single built-in serve any actor obs width (48d Flat / 45d Unitree / future variants) without re-registration.

**Batch dict keys** that factor extractors may consume now or in future phases:

- `obs` (raw actor obs) — DIAYN, METRA factors
- `next_obs` (raw actor next obs) — METRA, future DADS factors
- `action` — already stored by SAC replay buffer for Q update; available
  for free if/when DADS-style factors land. No replay schema migration needed.
- `skill_z`, `next_skill_z` — for skill-conditioned features
- `sim_data` — privileged simulator state (base xy, height, contacts) — D3 factor pattern
- `info` — env info dict (per-step metadata)

SD-A only wires `obs` / `next_obs` / `sim_data` / `info` in the test extractors. Future extractors that need `action` or `skill_z` work without registry refactor.

- [ ] Run → expect 5 passed.

### Step 3.3: Commit

```bash
git add jax_rl/skill_discovery/factors.py tests/test_skill_discovery_factors.py
git commit -m "feat(skill): add factor extractor registry with named source/dim resolution"
```

---

## Task 4: DIAYN aux module

**Files:** `jax_rl/skill_discovery/diayn.py`, `tests/test_skill_discovery_diayn.py`

Pure functions + Flax `Discriminator` module. No `SkillManager` knowledge.

### Step 4.1: Write tests (RED)

```python
"""Tests for DIAYN auxiliary module."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import optax

from jax_rl.skill_discovery.diayn import Discriminator, diayn_reward, diayn_update

KEY = jax.random.PRNGKey(42)
BATCH = 32


def test_discriminator_init_and_forward():
    disc = Discriminator(hidden_dim=(64, 64), num_skills=10)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    logits = disc.apply(params, jnp.zeros((BATCH, 5)))
    assert logits.shape == (BATCH, 10)


def test_diayn_reward_finite_and_correct_shape():
    disc = Discriminator(hidden_dim=(64, 64), num_skills=10)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    obs = jax.random.normal(KEY, (BATCH, 5))
    z = jax.nn.one_hot(jnp.zeros(BATCH, dtype=jnp.int32), 10)
    r = diayn_reward(params, disc, obs, z, num_skills=10)
    assert r.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(r))


def test_diayn_reward_deterministic_for_fixed_params_and_batch():
    """Sample-time reward must be deterministic — required by SD-A acceptance."""
    disc = Discriminator(hidden_dim=(64, 64), num_skills=4)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, 5))
    z = jax.nn.one_hot(jnp.arange(BATCH) % 4, 4)
    r1 = diayn_reward(params, disc, obs, z, num_skills=4)
    r2 = diayn_reward(params, disc, obs, z, num_skills=4)
    assert jnp.array_equal(r1, r2)


def test_diayn_update_reduces_loss():
    disc = Discriminator(hidden_dim=(64, 64), num_skills=4)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    # Synthetic separable data: skill i has obs = i * ones(5)
    obs = jnp.concatenate([jnp.ones((8, 5)) * i for i in range(4)])
    z_idx = jnp.concatenate([jnp.full(8, i, dtype=jnp.int32) for i in range(4)])

    for _ in range(50):
        params, opt_state, m = diayn_update(params, opt_state, disc, opt, obs, z_idx)

    logits = disc.apply(params, obs)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == z_idx)
    assert acc > 0.5, f"discriminator acc {acc:.2f} should exceed 0.5 after 50 updates"


def test_diayn_update_metrics_keys():
    disc = Discriminator(hidden_dim=(32,), num_skills=4)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    obs = jax.random.normal(KEY, (16, 5))
    z_idx = jax.random.randint(KEY, (16,), 0, 4)
    _, _, metrics = diayn_update(params, opt_state, disc, opt, obs, z_idx)
    assert "disc_loss" in metrics
    assert "disc_accuracy" in metrics
```

- [ ] Run → expect ImportError.

### Step 4.2: Implement

DIAYN reward (Eysenbach 2018, **current state s**, NOT next state — `ben-eysenbach/sac:diayn.py:175-180`):

```python
EPS = 1e-6  # matches DIAYN reference (diayn.py:21, 185)
log_q = jax.nn.log_softmax(disc.apply(params, obs_factor), axis=-1)
log_q_z = jnp.sum(log_q * z_onehot, axis=-1)   # gather chosen-skill log-prob
log_p_z = -jnp.log(num_skills)                 # uniform prior
reward = log_q_z - log_p_z + EPS               # (batch,)
```

DIAYN update: integer-label softmax cross-entropy on `disc(obs_factor)` vs `z_indices`, Adam.

API:

- `Discriminator(nn.Module)` — **plain MLP** using `ACTIVATIONS` from `jax_rl.networks.activations`. Fields: `hidden_dim: tuple[int, ...]`, `num_skills: int`, `activation: str = "relu"`. **DO NOT use ELU or SimBa here** — D3's reference uses ELU+SimBa residual MLP, but those are SD-C/E upgrades. SD-A ships plain ReLU MLP.
- `diayn_reward(disc_params, disc, obs_factor, z_onehot, num_skills) -> (batch,)`
- `diayn_update(disc_params, opt_state, disc, optimizer, obs_factor, z_indices) -> (new_params, new_opt_state, metrics)`
  - `optax.softmax_cross_entropy_with_integer_labels(logits, z_indices).mean()`
  - `disc_accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == z_indices)`
  - returns dict with `disc_loss` and `disc_accuracy`

- [ ] Run → expect 5 passed.

### Step 4.3: Commit

```bash
git add jax_rl/skill_discovery/diayn.py tests/test_skill_discovery_diayn.py
git commit -m "feat(skill): add DIAYN discriminator, sample-time reward, and gradient update"
```

---

## Task 5: SkillManager

**Files:** `jax_rl/skill_discovery/manager.py`, `tests/test_skill_discovery_manager.py`

**Depends on Tasks 1-4.** Run after Wave 1 commits land.

`SkillManager` is the orchestrator. **No env knowledge, no buffer knowledge, no training-loop knowledge.** It owns:
- aux network params + optimizers (built once at `init`, reused)
- skill sampling via `prior.sample_skill`
- `augment_actor_obs(obs, z) -> concat(obs, z)`
- `compute_intrinsic_reward(aux_state, batch) -> (batch,)` — uses factor extractors
- `update(aux_state, batch) -> (new_aux_state, metrics)` — runs gradient steps on all factors

**Important fix vs v1:** optimizers built once in `__init__`, stored on `self`. Do NOT recreate inside `update()`.

**DIAYN reward uses current state `s`, NOT next state `s'`** (per Eysenbach 2018 — discriminator `q(z|s)` is on current state). Common confusion: DADS uses `q(s'|s,z)` which is forward-direction; DIAYN reverses to predict z from s. v1 retired plan had this wrong — used `next_obs` for the discriminator. SD-A's `compute_intrinsic_reward` and `update` both feed `batch["obs"]` (the factor extractor pulls from current state).

When SD-E adds METRA, METRA's reward is `(φ(s')−φ(s))ᵀz` and DOES need both `s` and `s'`. The manager dispatches by `factor.method`:
- `method="diayn"` → extract from `batch["obs"]` only
- `method="metra"` → extract from `batch["obs"]` and `batch["next_obs"]`

SD-A only ships DIAYN, so tests only put `"obs"` in the batch.

### Step 5.1: Write tests (RED)

```python
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


def test_manager_metra_factor_raises_in_sd_a():
    """METRA factors should raise NotImplementedError until SD-E."""
    cfg = SkillDiscoveryConfig(
        mode="metra",
        total_skill_dim=2,
        factors=(FactorConfig(name="m", method="metra", skill_dim=2,
                              source="actor_obs", extractor="manager_test_obs", dim=5),),
    )
    with pytest.raises(NotImplementedError, match="SD-E"):
        SkillManager(cfg)
```

- [ ] Run: `uv run python -m pytest tests/test_skill_discovery_manager.py -v` → expect ImportError.

### Step 5.2: Implement

`SkillManager` API:

```python
class SkillManager:
    def __init__(self, config: SkillDiscoveryConfig):
        # validate ALL factors (not just first); any factor.method == "metra" → NotImplementedError("METRA deferred to SD-E")
        # build per-factor network + optimizer with hardcoded SD-A defaults:
        #   Discriminator(hidden_dim=(256, 256), num_skills=factor.skill_dim, activation="relu")
        #   optax.adam(3e-4)
        # store on self._networks[name], self._optimizers[name]
        # AuxNetConfig plumb-through deferred to SD-C.

    @property
    def total_skill_dim(self) -> int: ...

    def init(self, key) -> dict:
        """Returns {factor_name: {"params": ..., "opt_state": ...}}."""

    def sample_skills(self, key, num_envs) -> jax.Array:
        """Calls prior.sample_skill once per factor, concats."""

    def resample_on_done(self, current_z, done, key) -> jax.Array:
        """jnp.where(done[:, None], new_z, current_z)."""

    def augment_actor_obs(self, obs, z) -> jax.Array:
        """concat([obs, z], axis=-1). Shape-checked."""

    def compute_intrinsic_reward(self, aux_state, batch) -> jax.Array:
        """For each factor: extract obs via factors.resolve_factor(factor, batch),
        slice z (one-hot for DIAYN), run diayn_reward.
        Sum across factors (default: uniform weights 1/len(factors)).
        Per-factor weighting is wired in SD-E.
        """

    def update(self, aux_state, batch) -> tuple[dict, dict]:
        """For each factor: extract obs, slice z, convert one-hot z → int via
        jnp.argmax(z_slice, axis=-1) for diayn_update's z_indices arg, run
        diayn_update with self._optimizers[name].
        Merge per-factor metrics into a flat dict keyed f'{factor.name}_{key}',
        e.g. 'full_disc_loss', 'full_disc_accuracy'. Tests assert this naming.
        """
```

**Z slicing helper** (private):
- `_z_slices: list[tuple[int,int]]` precomputed from `factors` in `__init__`
- `_get_factor_z(z, idx) -> z[:, start:end]`

**Factor weights:** default uniform = `1/len(factors)`. SD-A only ships single-factor configs in tests, but support N for forward compat (actual D3 mixing arrives in SD-E).

**One-hot ↔ integer skill index:**
- `compute_intrinsic_reward` and `diayn_reward` consume one-hot `z_onehot`.
- `diayn_update` consumes integer `z_indices`. Manager converts via `jnp.argmax(z_slice, axis=-1)` before calling.

- [ ] Run: `uv run python -m pytest tests/test_skill_discovery_manager.py -v` → expect 9 passed.

### Step 5.3: Run full SD-A suite

- [ ] `uv run python -m pytest tests/test_skill_discovery_*.py -v` → expect all passed.
- [ ] `uv run python -m pytest tests/ -v` → expect no regressions.

### Step 5.4: Commit

```bash
git add jax_rl/skill_discovery/manager.py tests/test_skill_discovery_manager.py
git commit -m "feat(skill): add SkillManager — z lifecycle, aux init/update, intrinsic reward composition"
```

---

## SD-A Wrap-up

After Task 5 lands:

- [ ] Update `.context/TODO.md` SD-A section → mark scaffolding complete, point next-action at SD-B.
- [ ] Add a journal entry `.context/journals/2026-05-02.md` covering: spec retirement, v1 banner, SD-A delivery.
- [ ] No new lessons unless something surprised us during implementation.

## Pointers for SD-B (next plan, do not write yet)

- Extend `ObsPipeline.make_buffer(...)` with generic `extra_obs_dims: dict[str, int] | None`
- Add `scripts/train_skill_discovery.py` + `jax_rl/training/skill_offpolicy_loop.py`
- Wire SkillManager into the loop per spec §SD-B (collect skill_z → buffer → sample-time reward replacement → algo.update → aux update)
- 10k-step CPU smoke + 100k-step DIAYN smoke on CheetahRun or WalkerWalk
- Ensure existing `test_jax_replay_buffer.py` asymmetric critic tests stay green after generic extras land

## Open questions surfaced during SD-A (resolve before SD-B)

- **Factor extractor side effects:** SD-A registers a global `_REGISTRY`. If two test files both register `actor_obs_full`, do we want last-write-wins, error-on-duplicate, or namespacing? Default: error-on-duplicate. Tests register unique names per test (already do).
- **Where do Go2-specific extractors live?** Either `jax_rl/skill_discovery/factors_go2.py` (clean) or env file imports + registers (couples env to skill_discovery package). Defer decision to SD-C.
