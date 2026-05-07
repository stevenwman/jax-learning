# Skill Discovery Framework Implementation Plan

> ## ⛔ SUPERSEDED — DO NOT IMPLEMENT FROM THIS DOC
>
> Superseded by `projects/skill-discovery/specs/2026-04-28-skill-discovery.md` (V2 audit + design)
> and `projects/skill-discovery/plans/2026-05-02-skill-discovery-sd-a.md` (SD-A execution plan).
>
> Why: this v1 plan predates RewardSpec/ObsSpec, the env-backend refactor, the
> EnvBundle entrypoint, ObsPipeline sample-time normalization, schema-stamped
> deploy contracts, and the off-policy script split. Its file targets
> (`go2_joystick.py`, `train_offpolicy.py`), magic obs indices (`[48:50]
> base_xy`, `[50] base_z`), and "store normalized augmented obs in replay"
> design are all wrong against current repo.
>
> Kept for historical intent (DIAYN/METRA module sketches, factorized USD
> rationale). All implementation guidance is stale.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Drift note (2026-04-20):** This plan predates the 2026-04-12 off-policy script split. References to `train_offpolicy.py` as "the starting point" should be read as "the per-algo script closest to your target (`train_sac.py`, `train_fast_sac.py`, etc.) plus the shared loop at `jax_rl/training/offpolicy_loop.py::run_offpolicy_loop`". The legacy dispatcher lives at `archive/train_offpolicy.py` for reference only.

> **Drift note (2026-04-26):** This plan also predates the env-backend refactor. References to `make_envs(cfg, seed)` returning a 7-tuple should now use `make_env_bundle(cfg, seed) -> EnvBundle` (the 7-tuple `make_envs` is still re-exported for legacy callers but new code should use the bundle). Env registration moves from `env_setup.py` (now a thin shim) to the appropriate `jax_rl/training/env_backends/{name}_backend.py`.

**Goal:** Factorized unsupervised skill discovery (DIAYN + METRA) as pluggable reward modules, algo-agnostic, validated on Go2 and DM Control envs.

**Architecture:** DIAYN and METRA are reward modules that sit between env and algo. A SkillManager handles z lifecycle (sample/resample), state factorization (which obs dims → which method), and reward composition. The base RL algo (SAC) sees augmented obs (concat with z) and intrinsic rewards — it doesn't know skill discovery exists. The training script orchestrates everything.

**Tech Stack:** JAX, Flax, optax, MuJoCo Playground (MJX)

**Key references:**
- DIAYN: "Diversity Is All You Need" (Eysenbach et al., 2018)
- METRA: "Scalable Unsupervised RL with Metric-Aware Abstraction" (Park et al., 2024)
- D3: "Divide, Discover, Deploy" (Cathomen et al., 2025) — factorized USD framework

**Design decisions:**
- Start with vanilla SAC (QHead, not C51). C51/FastSAC swap is a future upgrade avenue when scaling.
- **Buffer-agnostic:** z is concatenated into augmented obs, intrinsic reward stored as regular reward. No buffer modifications — works with both replay buffer (SAC) and rollout buffer (PPO). For auxiliary updates, split augmented obs by known offset.
- **Off-policy reward staleness (accepted tradeoff):** Intrinsic rewards are computed at collection time and stored in the buffer. As the discriminator/φ evolves during training, stored rewards become stale. This isn't an issue for PPO (on-policy, data consumed immediately). For SAC, it's acceptable for v1 — revisit if skill learning stalls.
- Style rewards: investigate further, D3's penalty-based style factor as starting reference.
- Eval metrics: design separately before calling implementation done.

---

## File Structure

```
New files:
  jax_rl/skill_discovery/__init__.py          — package init, public API
  jax_rl/skill_discovery/config.py            — FactorConfig, SkillDiscoveryConfig
  jax_rl/skill_discovery/diayn.py             — Discriminator network + reward fn + update fn
  jax_rl/skill_discovery/metra.py             — Representation phi + Lagrangian + reward fn + update fn
  jax_rl/skill_discovery/manager.py           — SkillManager: z lifecycle, reward composition
  train_skill_discovery.py                    — Training script with skill discovery loop
  tests/test_skill_discovery.py               — Tests for all skill discovery components

Modified files:
  jax_rl/envs/locomotion/go2_joystick.py      — add include_base_position/height config flags
  jax_rl/training/env_setup.py                — register Go2SkillDiscovery preset
  jax_rl/configs/env_presets.py               — add skill discovery presets
```

**Module responsibilities:**
- `config.py` — all dataclasses, no logic
- `diayn.py` — Discriminator Flax module, `diayn_reward()`, `diayn_update()` (all pure functions)
- `metra.py` — RepresentationPhi Flax module, `metra_reward()`, `metra_update()`, `metra_dual_update()` (all pure functions)
- `manager.py` — SkillManager class: composes factors, manages z, delegates to diayn/metra
- `train_skill_discovery.py` — training loop, CLI, env setup (follows `train_offpolicy.py` structure)

---

### Task 1: Skill Discovery Config

**Files:**
- Create: `jax_rl/skill_discovery/__init__.py`
- Create: `jax_rl/skill_discovery/config.py`
- Test: `tests/test_skill_discovery.py`

- [ ] **Step 1: Write config tests**

```python
# tests/test_skill_discovery.py
"""Tests for skill discovery framework."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_rl.skill_discovery.config import FactorConfig, SkillDiscoveryConfig


def test_factor_config_creation():
    fc = FactorConfig(
        name="position", obs_indices=(0, 1), method="metra", skill_dim=2,
    )
    assert fc.name == "position"
    assert fc.method == "metra"
    assert fc.skill_dim == 2


def test_skill_discovery_config_total_skill_dim():
    cfg = SkillDiscoveryConfig(factors=[
        FactorConfig("pos", (0, 1), "metra", skill_dim=2),
        FactorConfig("vel", (2, 3, 4), "diayn", skill_dim=4),
    ])
    assert cfg.total_skill_dim == 6


def test_skill_discovery_config_defaults():
    cfg = SkillDiscoveryConfig(factors=[
        FactorConfig("full", tuple(range(17)), "diayn", skill_dim=10),
    ])
    assert cfg.discriminator_lr == 1e-4
    assert cfg.phi_lr == 1e-4
    assert cfg.dual_lam_init == 30.0
    assert cfg.style_weight == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_skill_discovery.py -v`
Expected: ImportError (module doesn't exist yet)

- [ ] **Step 3: Implement config dataclasses**

```python
# jax_rl/skill_discovery/__init__.py
"""Skill discovery framework — DIAYN and METRA as pluggable reward modules."""

# jax_rl/skill_discovery/config.py
"""Configuration for skill discovery (DIAYN, METRA, factorized USD)."""
from dataclasses import dataclass, field


@dataclass
class FactorConfig:
    """Configuration for a single state factor in factorized USD."""
    name: str                        # human-readable name ("position", "heading", etc.)
    obs_indices: tuple[int, ...]     # which dims of the obs vector this factor reads
    method: str                      # "diayn" or "metra"
    skill_dim: int                   # for DIAYN: num discrete skills (one-hot size)
                                     # for METRA: continuous latent dim


@dataclass
class SkillDiscoveryConfig:
    """Top-level config for factorized skill discovery."""
    factors: list[FactorConfig] = field(default_factory=list)

    # DIAYN settings
    discriminator_lr: float = 1e-4
    discriminator_hidden: tuple[int, ...] = (256, 256)
    discriminator_activation: str = "relu"

    # METRA settings
    phi_lr: float = 1e-4
    phi_hidden: tuple[int, ...] = (256, 256)
    phi_activation: str = "relu"
    dual_lam_init: float = 30.0
    dual_lam_lr: float = 1e-4
    dual_slack: float = 1e-5

    # Reward composition
    style_weight: float = 0.0       # fraction of env reward to keep (0 = pure intrinsic)
    factor_weights: list[float] | None = None  # per-factor weights (default: equal)

    @property
    def total_skill_dim(self) -> int:
        return sum(f.skill_dim for f in self.factors)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_skill_discovery.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add jax_rl/skill_discovery/__init__.py jax_rl/skill_discovery/config.py tests/test_skill_discovery.py
git commit -m "feat: skill discovery config — FactorConfig and SkillDiscoveryConfig dataclasses"
```

---

### Task 2: DIAYN Reward Module

**Files:**
- Create: `jax_rl/skill_discovery/diayn.py`
- Test: `tests/test_skill_discovery.py` (append)

The DIAYN module provides three things:
1. `Discriminator` — Flax MLP module: state_factor → logits over skills
2. `diayn_reward()` — pure function: log q(z|s) - log p(z)
3. `diayn_update()` — pure function: one gradient step on cross-entropy loss

- [ ] **Step 1: Write DIAYN tests**

Append to `tests/test_skill_discovery.py`:

```python
import jax
import jax.numpy as jnp
import optax

KEY = jax.random.PRNGKey(42)
BATCH = 32


def test_discriminator_init_and_forward():
    from jax_rl.skill_discovery.diayn import Discriminator
    disc = Discriminator(hidden_dim=(64, 64), num_skills=10)
    params = disc.init(KEY, jnp.zeros((1, 5)))  # 5-dim state factor
    logits = disc.apply(params, jnp.zeros((BATCH, 5)))
    assert logits.shape == (BATCH, 10)


def test_diayn_reward_shape_and_range():
    from jax_rl.skill_discovery.diayn import Discriminator, diayn_reward
    disc = Discriminator(hidden_dim=(64, 64), num_skills=10)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    obs_factor = jax.random.normal(KEY, (BATCH, 5))
    z_onehot = jax.nn.one_hot(jnp.zeros(BATCH, dtype=jnp.int32), 10)
    reward = diayn_reward(params, disc, obs_factor, z_onehot, num_skills=10)
    assert reward.shape == (BATCH,)
    assert not jnp.any(jnp.isnan(reward))
    # Reward should be log q(z|s) - log(1/K) = log q(z|s) + log(K)
    # log q(z|s) <= 0, log(K) = log(10) ≈ 2.3, so reward can be positive
    # Just verify finite
    assert jnp.all(jnp.isfinite(reward))


def test_diayn_update_reduces_loss():
    from jax_rl.skill_discovery.diayn import Discriminator, diayn_update
    disc = Discriminator(hidden_dim=(64, 64), num_skills=4)
    params = disc.init(KEY, jnp.zeros((1, 5)))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    # Create data where skill 0 always comes with obs=+1, skill 1 with obs=-1, etc.
    obs_factor = jnp.concatenate([jnp.ones((8, 5)) * i for i in range(4)])
    z_indices = jnp.concatenate([jnp.full(8, i, dtype=jnp.int32) for i in range(4)])

    # Run 50 updates
    for _ in range(50):
        params, opt_state, metrics = diayn_update(params, opt_state, disc, opt, obs_factor, z_indices)

    # Discriminator should get better — accuracy > chance (25%)
    logits = disc.apply(params, obs_factor)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == z_indices)
    assert accuracy > 0.5, f"Discriminator accuracy {accuracy:.2f} should be > 0.5 after 50 steps"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_skill_discovery.py::test_discriminator_init_and_forward -v`
Expected: ImportError

- [ ] **Step 3: Implement DIAYN module**

```python
# jax_rl/skill_discovery/diayn.py
"""DIAYN reward module — discriminator-based skill discovery.

Provides:
  Discriminator: Flax MLP that classifies state factors into skill indices.
  diayn_reward: log q(z|s) - log p(z) intrinsic reward.
  diayn_update: one gradient step on cross-entropy classification loss.

Reference: Eysenbach et al., "Diversity Is All You Need" (2018)
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from jax_rl.networks.activations import ACTIVATIONS


class Discriminator(nn.Module):
    """MLP classifier: state_factor -> logits over discrete skills."""
    hidden_dim: tuple[int, ...]
    num_skills: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, state_factor: jax.Array) -> jax.Array:
        x = state_factor
        act_fn = ACTIVATIONS[self.activation]
        for d in self.hidden_dim:
            x = nn.Dense(d)(x)
            x = act_fn(x)
        return nn.Dense(self.num_skills)(x)


def diayn_reward(
    disc_params, discriminator: Discriminator,
    obs_factor: jax.Array, z_onehot: jax.Array, num_skills: int,
) -> jax.Array:
    """Intrinsic reward: log q(z|s) - log p(z).

    Args:
        obs_factor: (batch, factor_dim) — state factor for this DIAYN factor
        z_onehot: (batch, num_skills) — one-hot skill vector
        num_skills: number of discrete skills (for uniform prior)
    Returns:
        reward: (batch,) — per-transition intrinsic reward
    """
    logits = discriminator.apply(disc_params, obs_factor)
    log_q = jax.nn.log_softmax(logits, axis=-1)
    log_q_z = jnp.sum(log_q * z_onehot, axis=-1)
    log_p_z = -jnp.log(num_skills)  # uniform prior
    return log_q_z - log_p_z


def diayn_update(
    disc_params, opt_state, discriminator: Discriminator,
    optimizer: optax.GradientTransformation,
    obs_factor: jax.Array, z_indices: jax.Array,
) -> tuple:
    """One gradient step on discriminator cross-entropy loss.

    Args:
        obs_factor: (batch, factor_dim)
        z_indices: (batch,) integer skill indices
    Returns:
        (new_params, new_opt_state, metrics_dict)
    """
    def loss_fn(params):
        logits = discriminator.apply(params, obs_factor)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, z_indices).mean()
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(preds == z_indices)
        return loss, {"disc_loss": loss, "disc_accuracy": accuracy}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(disc_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params=disc_params)
    new_params = optax.apply_updates(disc_params, updates)
    return new_params, new_opt_state, metrics
```

- [ ] **Step 4: Run DIAYN tests**

Run: `uv run python -m pytest tests/test_skill_discovery.py -k diayn -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add jax_rl/skill_discovery/diayn.py tests/test_skill_discovery.py
git commit -m "feat: DIAYN reward module — discriminator, intrinsic reward, gradient update"
```

---

### Task 3: METRA Reward Module

**Files:**
- Create: `jax_rl/skill_discovery/metra.py`
- Test: `tests/test_skill_discovery.py` (append)

The METRA module provides:
1. `RepresentationPhi` — Flax MLP module: state_factor → latent embedding
2. `metra_reward()` — pure function: (phi(s') - phi(s))^T z
3. `metra_update()` — pure function: one gradient step on phi (maximize inner product + constraint)
4. `metra_dual_update()` — pure function: one gradient step on dual variable lambda

- [ ] **Step 1: Write METRA tests**

Append to `tests/test_skill_discovery.py`:

```python
def test_representation_phi_init_and_forward():
    from jax_rl.skill_discovery.metra import RepresentationPhi
    phi = RepresentationPhi(hidden_dim=(64, 64), latent_dim=2)
    params = phi.init(KEY, jnp.zeros((1, 5)))
    out = phi.apply(params, jnp.zeros((BATCH, 5)))
    assert out.shape == (BATCH, 2)


def test_metra_reward_shape():
    from jax_rl.skill_discovery.metra import RepresentationPhi, metra_reward
    phi = RepresentationPhi(hidden_dim=(64, 64), latent_dim=2)
    params = phi.init(KEY, jnp.zeros((1, 5)))
    obs = jax.random.normal(KEY, (BATCH, 5))
    next_obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, 5))
    z = jax.random.normal(jax.random.PRNGKey(2), (BATCH, 2))
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)  # unit sphere
    reward = metra_reward(params, phi, obs, next_obs, z)
    assert reward.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(reward))


def test_metra_update_runs():
    from jax_rl.skill_discovery.metra import RepresentationPhi, metra_update, metra_dual_update
    phi = RepresentationPhi(hidden_dim=(64, 64), latent_dim=2)
    params = phi.init(KEY, jnp.zeros((1, 5)))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    log_dual_lam = jnp.log(jnp.array(30.0))
    dual_opt = optax.adam(1e-4)
    dual_opt_state = dual_opt.init(log_dual_lam)

    obs = jax.random.normal(KEY, (BATCH, 5))
    next_obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, 5))
    z = jax.random.normal(jax.random.PRNGKey(2), (BATCH, 2))
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    # Phi update
    new_params, new_opt, phi_metrics = metra_update(
        params, opt_state, phi, opt, obs, next_obs, z, log_dual_lam, dual_slack=1e-5,
    )
    assert "metra_inner" in phi_metrics
    assert "metra_norm" in phi_metrics

    # Dual update
    new_lam, new_dual_opt, dual_metrics = metra_dual_update(
        log_dual_lam, dual_opt_state, dual_opt, new_params, phi, obs, next_obs, dual_slack=1e-5,
    )
    assert "dual_lam" in dual_metrics
    assert jnp.isfinite(new_lam)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_skill_discovery.py -k metra -v`
Expected: ImportError

- [ ] **Step 3: Implement METRA module**

```python
# jax_rl/skill_discovery/metra.py
"""METRA reward module — metric-aware representation for skill discovery.

Provides:
  RepresentationPhi: Flax MLP that maps state factors to a latent space.
  metra_reward: (phi(s') - phi(s))^T z inner-product reward.
  metra_update: gradient step on phi (maximize alignment + Lipschitz constraint).
  metra_dual_update: gradient step on Lagrange multiplier lambda.

Reference: Park et al., "METRA: Scalable Unsupervised RL with Metric-Aware Abstraction" (2024)
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from jax_rl.networks.activations import ACTIVATIONS


class RepresentationPhi(nn.Module):
    """MLP representation: state_factor -> latent embedding."""
    hidden_dim: tuple[int, ...]
    latent_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, state_factor: jax.Array) -> jax.Array:
        x = state_factor
        act_fn = ACTIVATIONS[self.activation]
        for d in self.hidden_dim:
            x = nn.Dense(d)(x)
            x = act_fn(x)
        return nn.Dense(self.latent_dim)(x)


def metra_reward(
    phi_params, phi: RepresentationPhi,
    obs_factor: jax.Array, next_obs_factor: jax.Array, z: jax.Array,
) -> jax.Array:
    """Intrinsic reward: (phi(s') - phi(s))^T z.

    Args:
        obs_factor: (batch, factor_dim)
        next_obs_factor: (batch, factor_dim)
        z: (batch, latent_dim) — continuous skill vector
    Returns:
        reward: (batch,)
    """
    phi_s = phi.apply(phi_params, obs_factor)
    phi_s_next = phi.apply(phi_params, next_obs_factor)
    delta_phi = phi_s_next - phi_s
    return jnp.sum(delta_phi * z, axis=-1)


def metra_update(
    phi_params, opt_state, phi: RepresentationPhi,
    optimizer: optax.GradientTransformation,
    obs_factor: jax.Array, next_obs_factor: jax.Array, z: jax.Array,
    log_dual_lam: jax.Array, dual_slack: float = 1e-5,
) -> tuple:
    """One gradient step on phi: maximize inner product + lambda * constraint.

    The Lipschitz constraint ||phi(s')-phi(s)||_2 <= 1 is enforced via
    Lagrangian relaxation with dual variable lambda (passed as log for positivity).
    Lambda is detached (stop_gradient) during phi update.
    """
    def loss_fn(params):
        phi_s = phi.apply(params, obs_factor)
        phi_s_next = phi.apply(params, next_obs_factor)
        delta_phi = phi_s_next - phi_s

        inner = jnp.sum(delta_phi * z, axis=-1)
        norm_sq = jnp.sum(delta_phi ** 2, axis=-1)
        constraint = jnp.minimum(dual_slack, 1.0 - norm_sq)

        dual_lam = jnp.exp(jax.lax.stop_gradient(log_dual_lam))
        obj = inner + dual_lam * constraint
        loss = -obj.mean()  # maximize -> negate for minimization
        return loss, {
            "metra_inner": inner.mean(),
            "metra_norm": jnp.sqrt(norm_sq).mean(),
            "metra_constraint": constraint.mean(),
        }

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(phi_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params=phi_params)
    new_params = optax.apply_updates(phi_params, updates)
    return new_params, new_opt_state, metrics


def metra_dual_update(
    log_dual_lam: jax.Array, opt_state, optimizer: optax.GradientTransformation,
    phi_params, phi: RepresentationPhi,
    obs_factor: jax.Array, next_obs_factor: jax.Array, dual_slack: float = 1e-5,
) -> tuple:
    """One gradient step on dual variable lambda.

    Phi is detached (stop_gradient) during lambda update.
    Lambda stored in log-space to ensure positivity.
    """
    def loss_fn(log_lam):
        phi_s = jax.lax.stop_gradient(phi.apply(phi_params, obs_factor))
        phi_s_next = jax.lax.stop_gradient(phi.apply(phi_params, next_obs_factor))
        delta_phi = phi_s_next - phi_s

        norm_sq = jnp.sum(delta_phi ** 2, axis=-1)
        constraint = jnp.minimum(dual_slack, 1.0 - norm_sq)

        dual_lam = jnp.exp(log_lam)
        loss = (dual_lam * constraint).mean()
        return loss, {"dual_lam": dual_lam}

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(log_dual_lam)
    updates, new_opt_state = optimizer.update(grads, opt_state, params=log_dual_lam)
    new_log_lam = optax.apply_updates(log_dual_lam, updates)
    return new_log_lam, new_opt_state, metrics
```

- [ ] **Step 4: Run METRA tests**

Run: `uv run python -m pytest tests/test_skill_discovery.py -k metra -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add jax_rl/skill_discovery/metra.py tests/test_skill_discovery.py
git commit -m "feat: METRA reward module — representation phi, inner-product reward, Lagrangian update"
```

---

### Task 4: Skill Manager

**Files:**
- Create: `jax_rl/skill_discovery/manager.py`
- Test: `tests/test_skill_discovery.py` (append)

The SkillManager is the orchestrator. It:
- Knows the factor layout (which obs indices → which method → which z slice)
- Samples/resamples z vectors (discrete one-hot for DIAYN, unit sphere for METRA)
- Initializes all auxiliary network params and optimizers
- Computes composed intrinsic reward from all factors
- Updates all auxiliary networks

- [ ] **Step 1: Write SkillManager tests**

Append to `tests/test_skill_discovery.py`:

```python
def test_skill_manager_sample_skills():
    from jax_rl.skill_discovery.config import FactorConfig, SkillDiscoveryConfig
    from jax_rl.skill_discovery.manager import SkillManager

    cfg = SkillDiscoveryConfig(factors=[
        FactorConfig("f1", (0, 1), "diayn", skill_dim=4),
        FactorConfig("f2", (2, 3), "metra", skill_dim=2),
    ])
    mgr = SkillManager(cfg)

    z = mgr.sample_skills(KEY, num_envs=8)
    assert z.shape == (8, 6)  # 4 + 2

    # DIAYN part should be one-hot
    z_diayn = z[:, :4]
    assert jnp.allclose(z_diayn.sum(axis=-1), 1.0)
    assert jnp.all((z_diayn == 0) | (z_diayn == 1))

    # METRA part should be unit-norm
    z_metra = z[:, 4:]
    norms = jnp.linalg.norm(z_metra, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)


def test_skill_manager_resample_on_done():
    from jax_rl.skill_discovery.config import FactorConfig, SkillDiscoveryConfig
    from jax_rl.skill_discovery.manager import SkillManager

    cfg = SkillDiscoveryConfig(factors=[
        FactorConfig("f1", (0, 1), "diayn", skill_dim=4),
    ])
    mgr = SkillManager(cfg)

    z_old = mgr.sample_skills(KEY, num_envs=4)
    done = jnp.array([0.0, 1.0, 0.0, 1.0])
    z_new = mgr.resample_on_done(z_old, done, jax.random.PRNGKey(99))

    # Env 0, 2 should keep old z
    assert jnp.allclose(z_new[0], z_old[0])
    assert jnp.allclose(z_new[2], z_old[2])
    # Env 1, 3 should have new z (almost certainly different)
    # Not guaranteed to differ, but extremely unlikely with different key


def test_skill_manager_init_aux_state():
    from jax_rl.skill_discovery.config import FactorConfig, SkillDiscoveryConfig
    from jax_rl.skill_discovery.manager import SkillManager

    cfg = SkillDiscoveryConfig(factors=[
        FactorConfig("pos", (0, 1), "metra", skill_dim=2),
        FactorConfig("vel", (2, 3, 4), "diayn", skill_dim=4),
    ])
    mgr = SkillManager(cfg)
    aux_state = mgr.init(KEY)

    assert "pos" in aux_state
    assert "vel" in aux_state
    assert "params" in aux_state["pos"]
    assert "params" in aux_state["vel"]


def test_skill_manager_compute_reward():
    from jax_rl.skill_discovery.config import FactorConfig, SkillDiscoveryConfig
    from jax_rl.skill_discovery.manager import SkillManager

    cfg = SkillDiscoveryConfig(factors=[
        FactorConfig("full", tuple(range(5)), "diayn", skill_dim=4),
    ])
    mgr = SkillManager(cfg)
    aux_state = mgr.init(KEY)

    obs = jax.random.normal(KEY, (BATCH, 5))
    next_obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, 5))
    z = jax.nn.one_hot(jnp.zeros(BATCH, dtype=jnp.int32), 4)

    reward = mgr.compute_reward(aux_state, obs, next_obs, z)
    assert reward.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(reward))


def test_skill_manager_augment_obs():
    from jax_rl.skill_discovery.config import FactorConfig, SkillDiscoveryConfig
    from jax_rl.skill_discovery.manager import SkillManager

    cfg = SkillDiscoveryConfig(factors=[
        FactorConfig("full", tuple(range(5)), "diayn", skill_dim=4),
    ])
    mgr = SkillManager(cfg)
    obs = jnp.ones((8, 5))
    z = jnp.zeros((8, 4))
    aug = mgr.augment_obs(obs, z)
    assert aug.shape == (8, 9)  # 5 + 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_skill_discovery.py -k "skill_manager" -v`
Expected: ImportError

- [ ] **Step 3: Implement SkillManager**

```python
# jax_rl/skill_discovery/manager.py
"""SkillManager — orchestrates factorized skill discovery.

Handles z lifecycle (sample/resample), obs augmentation, intrinsic reward
composition, and auxiliary network management. Delegates actual reward
computation and gradient steps to diayn.py and metra.py.
"""
import jax
import jax.numpy as jnp
import optax

from jax_rl.skill_discovery.config import SkillDiscoveryConfig, FactorConfig
from jax_rl.skill_discovery.diayn import Discriminator, diayn_reward, diayn_update
from jax_rl.skill_discovery.metra import (
    RepresentationPhi, metra_reward, metra_update, metra_dual_update,
)


class SkillManager:
    """Manages factorized skill discovery across multiple state factors."""

    def __init__(self, config: SkillDiscoveryConfig):
        self.config = config
        self.factors = config.factors

        # Precompute z slicing info: each factor's offset into the concatenated z vector
        self._z_slices = []
        offset = 0
        for f in self.factors:
            self._z_slices.append((offset, offset + f.skill_dim))
            offset += f.skill_dim
        self._total_skill_dim = offset

        # Build networks per factor (lightweight — no params yet)
        self._networks = {}
        for f in self.factors:
            if f.method == "diayn":
                self._networks[f.name] = Discriminator(
                    hidden_dim=config.discriminator_hidden,
                    num_skills=f.skill_dim,
                    activation=config.discriminator_activation,
                )
            elif f.method == "metra":
                self._networks[f.name] = RepresentationPhi(
                    hidden_dim=config.phi_hidden,
                    latent_dim=f.skill_dim,
                    activation=config.phi_activation,
                )

        # Factor weights (default: equal)
        if config.factor_weights is not None:
            self._weights = jnp.array(config.factor_weights)
        else:
            self._weights = jnp.ones(len(self.factors)) / len(self.factors)

    @property
    def total_skill_dim(self) -> int:
        return self._total_skill_dim

    def init(self, key: jax.Array) -> dict:
        """Initialize all auxiliary network params and optimizer states."""
        aux_state = {}
        for i, f in enumerate(self.factors):
            key, k = jax.random.split(key)
            net = self._networks[f.name]
            factor_dim = len(f.obs_indices)
            dummy = jnp.zeros((1, factor_dim))
            params = net.init(k, dummy)

            if f.method == "diayn":
                opt = optax.adam(self.config.discriminator_lr)
                aux_state[f.name] = {
                    "params": params,
                    "opt_state": opt.init(params),
                }
            elif f.method == "metra":
                opt = optax.adam(self.config.phi_lr)
                log_dual_lam = jnp.log(jnp.array(self.config.dual_lam_init))
                dual_opt = optax.adam(self.config.dual_lam_lr)
                aux_state[f.name] = {
                    "params": params,
                    "opt_state": opt.init(params),
                    "log_dual_lam": log_dual_lam,
                    "dual_opt_state": dual_opt.init(log_dual_lam),
                }
        return aux_state

    def sample_skills(self, key: jax.Array, num_envs: int) -> jax.Array:
        """Sample z for all envs. Returns (num_envs, total_skill_dim)."""
        z_parts = []
        for f in self.factors:
            key, k = jax.random.split(key)
            if f.method == "diayn":
                indices = jax.random.randint(k, (num_envs,), 0, f.skill_dim)
                z_parts.append(jax.nn.one_hot(indices, f.skill_dim))
            elif f.method == "metra":
                raw = jax.random.normal(k, (num_envs, f.skill_dim))
                z_parts.append(raw / jnp.linalg.norm(raw, axis=-1, keepdims=True))
        return jnp.concatenate(z_parts, axis=-1)

    def resample_on_done(
        self, current_z: jax.Array, done: jax.Array, key: jax.Array,
    ) -> jax.Array:
        """Resample z for envs where done=1. Returns updated z array."""
        new_z = self.sample_skills(key, current_z.shape[0])
        mask = done[:, None]
        return jnp.where(mask, new_z, current_z)

    def augment_obs(self, obs: jax.Array, z: jax.Array) -> jax.Array:
        """Concatenate skill vector z to observation."""
        return jnp.concatenate([obs, z], axis=-1)

    def _get_factor_obs(self, obs: jax.Array, factor: FactorConfig) -> jax.Array:
        """Extract factor's obs dimensions."""
        return obs[:, jnp.array(factor.obs_indices)]

    def _get_factor_z(self, z: jax.Array, factor_idx: int) -> jax.Array:
        """Extract factor's z slice from concatenated z vector."""
        start, end = self._z_slices[factor_idx]
        return z[:, start:end]

    def compute_reward(
        self, aux_state: dict, obs: jax.Array, next_obs: jax.Array, z: jax.Array,
    ) -> jax.Array:
        """Compute composed intrinsic reward from all factors.

        Args:
            aux_state: dict of per-factor auxiliary state
            obs: (batch, obs_dim) raw obs (without z)
            next_obs: (batch, obs_dim) raw next obs
            z: (batch, total_skill_dim) concatenated skill vector
        Returns:
            reward: (batch,) weighted sum of per-factor intrinsic rewards
        """
        total_reward = jnp.zeros(obs.shape[0])
        for i, f in enumerate(self.factors):
            z_i = self._get_factor_z(z, i)
            params = aux_state[f.name]["params"]
            net = self._networks[f.name]

            if f.method == "diayn":
                # DIAYN: discriminator operates on next_obs factor
                obs_factor = self._get_factor_obs(next_obs, f)
                r_i = diayn_reward(params, net, obs_factor, z_i, num_skills=f.skill_dim)
            elif f.method == "metra":
                # METRA: phi operates on both obs and next_obs factors
                obs_factor = self._get_factor_obs(obs, f)
                next_obs_factor = self._get_factor_obs(next_obs, f)
                r_i = metra_reward(params, net, obs_factor, next_obs_factor, z_i)

            total_reward = total_reward + self._weights[i] * r_i
        return total_reward

    def update(
        self, aux_state: dict, obs: jax.Array, next_obs: jax.Array, z: jax.Array,
    ) -> tuple[dict, dict]:
        """Update all auxiliary networks. Returns (new_aux_state, metrics)."""
        new_aux_state = {}
        all_metrics = {}

        for i, f in enumerate(self.factors):
            z_i = self._get_factor_z(z, i)
            state = aux_state[f.name]
            net = self._networks[f.name]

            if f.method == "diayn":
                # Train discriminator on next_obs (matching reward which uses next_obs)
                obs_factor = self._get_factor_obs(next_obs, f)
                z_indices = jnp.argmax(z_i, axis=-1)
                opt = optax.adam(self.config.discriminator_lr)  # BUG: store at init, see note below
                new_params, new_opt, metrics = diayn_update(
                    state["params"], state["opt_state"], net, opt, obs_factor, z_indices,
                )
                new_aux_state[f.name] = {"params": new_params, "opt_state": new_opt}
                all_metrics.update({f"{f.name}_{k}": v for k, v in metrics.items()})

            elif f.method == "metra":
                obs_factor = self._get_factor_obs(obs, f)
                next_obs_factor = self._get_factor_obs(next_obs, f)
                phi_opt = optax.adam(self.config.phi_lr)      # BUG: store at init, see note below
                dual_opt = optax.adam(self.config.dual_lam_lr)  # BUG: store at init, see note below

                new_params, new_opt, phi_metrics = metra_update(
                    state["params"], state["opt_state"], net, phi_opt,
                    obs_factor, next_obs_factor, z_i,
                    state["log_dual_lam"], self.config.dual_slack,
                )
                new_lam, new_dual_opt, dual_metrics = metra_dual_update(
                    state["log_dual_lam"], state["dual_opt_state"], dual_opt,
                    new_params, net, obs_factor, next_obs_factor, self.config.dual_slack,
                )
                new_aux_state[f.name] = {
                    "params": new_params,
                    "opt_state": new_opt,
                    "log_dual_lam": new_lam,
                    "dual_opt_state": new_dual_opt,
                }
                all_metrics.update({f"{f.name}_{k}": v for k, v in {**phi_metrics, **dual_metrics}.items()})

        return new_aux_state, all_metrics
```

**Important:** The `update()` method shown above recreates optimizers on each call. This is incorrect — store the optimizer objects in `self._optimizers` dict at `__init__` time instead. Each factor gets its own optimizer entry (and METRA factors get a `dual_optimizer` entry too). Then `update()` references `self._optimizers[f.name]` instead of calling `optax.adam(...)`. This is mandatory, not optional — optimizer recreation is wasteful and will break if anyone uses stateful transforms like `optax.chain` with schedules.

- [ ] **Step 4: Run SkillManager tests**

Run: `uv run python -m pytest tests/test_skill_discovery.py -k "skill_manager" -v`
Expected: 5 passed

- [ ] **Step 5: Run all skill discovery tests together**

Run: `uv run python -m pytest tests/test_skill_discovery.py -v`
Expected: All passed (3 config + 3 DIAYN + 3 METRA + 5 manager = 14 total)

- [ ] **Step 6: Commit**

```bash
git add jax_rl/skill_discovery/manager.py tests/test_skill_discovery.py
git commit -m "feat: SkillManager — z lifecycle, reward composition, auxiliary network orchestration"
```

---

### Task 5: Go2 Env Extension

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_joystick.py`
- Modify: `jax_rl/training/env_setup.py`
- Modify: `jax_rl/configs/env_presets.py`
- Test: `tests/test_go2_env.py` (append)

Add `include_base_position` and `include_base_height` config flags to Go2 Joystick. Register `Go2SkillDiscovery` preset.

- [ ] **Step 1: Write Go2 obs extension tests**

Append to `tests/test_go2_env.py`:

```python
class TestGo2SkillDiscoveryObs:
    """Tests for skill discovery obs extensions (base position/height)."""

    def test_default_obs_unchanged(self):
        """Default config should still produce 48d state obs."""
        env = Joystick(task="flat_terrain")
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs["state"].shape == (48,)

    def test_include_base_position(self):
        """include_base_position adds 2d (xy) to state obs."""
        cfg = default_config()
        cfg.include_base_position = True
        env = Joystick(task="flat_terrain", config=cfg)
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs["state"].shape == (50,)

    def test_include_base_height(self):
        """include_base_height adds 1d (z) to state obs."""
        cfg = default_config()
        cfg.include_base_height = True
        env = Joystick(task="flat_terrain", config=cfg)
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs["state"].shape == (49,)

    def test_include_both(self):
        """Both flags produce 51d state obs."""
        cfg = default_config()
        cfg.include_base_position = True
        cfg.include_base_height = True
        env = Joystick(task="flat_terrain", config=cfg)
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs["state"].shape == (51,)
        # privileged_state should also grow by 3
        priv_dim = state.obs["privileged_state"].shape[0]
        assert priv_dim > 51  # privileged includes state + extras

    def test_position_values_are_from_qpos(self):
        """Appended position values should match qpos[:2]."""
        cfg = default_config()
        cfg.include_base_position = True
        env = Joystick(task="flat_terrain", config=cfg)
        state = env.reset(jax.random.PRNGKey(0))
        # qpos[0:2] is base xy. With noise=0, should match exactly.
        # With noise_level=1.0, values won't match exactly, but should be finite.
        assert jnp.all(jnp.isfinite(state.obs["state"][48:50]))
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `uv run python -m pytest tests/test_go2_env.py -k "skill_discovery or default_obs" -v`
Expected: Fail (config field doesn't exist)

- [ ] **Step 3: Implement env extension**

In `go2_joystick.py`:
1. Add to `default_config()`:
   ```python
   include_base_position=False,  # xy-position for skill discovery
   include_base_height=False,    # z-position for skill discovery
   ```

2. In `_get_obs()`, after building the 48d `state`, conditionally append:
   ```python
   if self._config.include_base_position:
       base_xy = data.qpos[0:2]
       state = jp.hstack([state, base_xy])
   if self._config.include_base_height:
       base_z = data.qpos[2:3]
       state = jp.hstack([state, base_z])
   ```

3. In `env_setup.py`, register `Go2SkillDiscovery`:
   ```python
   if "Go2SkillDiscovery" not in pg_locomotion._envs:
       sd_config = default_config()
       sd_config.include_base_position = True
       sd_config.include_base_height = True
       pg_locomotion.register_environment(
           "Go2SkillDiscovery",
           functools.partial(Joystick, task="flat_terrain", config=sd_config),
           lambda: sd_config,
       )
   ```

4. In `env_presets.py`, add a skill discovery preset (same as Go2JoystickFlat but with env_name changed):
   ```python
   "Go2SkillDiscovery": TrainConfig(
       env_name="Go2SkillDiscovery",
       ...  # same training config as Go2JoystickFlat
   ),
   ```

- [ ] **Step 4: Run Go2 tests**

Run: `uv run python -m pytest tests/test_go2_env.py -v`
Expected: All passed (existing + new)

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/locomotion/go2_joystick.py jax_rl/training/env_setup.py jax_rl/configs/env_presets.py tests/test_go2_env.py
git commit -m "feat: Go2SkillDiscovery env — base position/height in obs for skill discovery"
```

---

### Task 6: Training Script

**Files:**
- Create: `train_skill_discovery.py`

This is the main training loop. It follows `train_offpolicy.py` closely but adds:
1. SkillManager creation from config
2. z sampling at start, resampling on done
3. Obs augmentation (concat z before passing to algo)
4. Intrinsic reward replaces env reward at collection time
5. Buffer stores augmented obs and intrinsic reward as regular fields (buffer-agnostic)
6. Auxiliary network updates alongside algo.update() (z split from augmented obs by offset)
7. Skill-specific logging (discriminator accuracy, METRA norm, etc.)

**Buffer-agnostic design:** The buffer has no knowledge of skill discovery. It stores `augmented_obs` (raw obs + z) as its regular `obs` field, and intrinsic reward as its regular `reward` field. This means:
- No buffer modifications needed — works with both replay buffer (SAC) and rollout buffer (PPO)
- The training script computes intrinsic reward at collection time and stores it directly
- For auxiliary updates, the training script splits augmented obs by known offset: `raw_obs = batch["obs"][:, :obs_dim]`, `z = batch["obs"][:, obs_dim:]`

**Off-policy reward staleness tradeoff:** When using SAC (off-policy), intrinsic rewards computed at collection time become stale as the discriminator/φ evolves during training. A transition stored with reward `r_t` based on discriminator at time `t` may have a very different reward under the discriminator at time `t+1000`. This is an accepted tradeoff for v1 — the D3 paper uses PPO (on-policy) where data is consumed immediately and this isn't an issue. For off-policy, two mitigations exist but are deferred: (a) recompute intrinsic rewards at sample time (doubles forward passes), (b) high UTD ratio to keep discriminator/buffer drift small. Worth revisiting if off-policy skill learning stalls.

The script should support both pure single-factor mode (vanilla DIAYN/METRA) and multi-factor mode (D3-style factorized USD).

- [ ] **Step 1: Define CLI interface**

```bash
# Pure DIAYN on CheetahRun (validation)
uv run python train_skill_discovery.py --env CheetahRun --skill-mode diayn --num-skills 10

# Pure METRA on WalkerWalk (validation)
uv run python train_skill_discovery.py --env WalkerWalk --skill-mode metra --skill-dim 2

# Factorized on Go2 (full D3-style)
uv run python train_skill_discovery.py --env Go2SkillDiscovery --skill-mode factorized

# Base algo selection (default: sac)
uv run python train_skill_discovery.py --env CheetahRun --skill-mode diayn --algo sac
```

- [ ] **Step 2: Implement training script**

The training script follows `train_offpolicy.py` closely. Copy it as the starting point, then modify. Below are the key sections that differ.

**Dict obs handling (critical for Go2):**
Go2 returns `{"state": 48-51d, "privileged_state": ~122d}`. For skill discovery, both actor and critic see the "state" key (same as existing off-policy Go2 setup — no asymmetric critic). The `_get_obs` helper from `train_offpolicy.py` extracts the flat state. The skill manager then augments this flat state with z.

```python
def _get_obs(obs):
    """Extract flat obs from dict or flat. Same as train_offpolicy.py."""
    return obs["state"] if dict_obs else obs

# In the training loop:
raw_obs = _get_obs(env_state.obs)           # (num_envs, obs_dim) — flat, no z
augmented_obs = skill_manager.augment_obs(raw_obs, current_z)  # (num_envs, obs_dim + skill_dim)
action = explore(training_state.actor_params, augmented_obs, key)
```

**Convenience configs for pure modes:**
```python
def _make_pure_diayn_config(obs_dim, num_skills):
    return SkillDiscoveryConfig(factors=[
        FactorConfig("full_state", tuple(range(obs_dim)), "diayn", skill_dim=num_skills),
    ])

def _make_pure_metra_config(obs_dim, skill_dim):
    return SkillDiscoveryConfig(factors=[
        FactorConfig("full_state", tuple(range(obs_dim)), "metra", skill_dim=skill_dim),
    ])
```

**Go2 factorized config (obs indices reference 51d Go2SkillDiscovery state):**
```python
def _make_go2_factorized_config():
    # Indices reference Go2SkillDiscovery 51d state:
    #   [0:3] linvel, [3:6] gyro, [6:9] gravity, [9:21] joint_pos,
    #   [21:33] joint_vel, [33:45] last_act, [45:48] command,
    #   [48:50] base_xy (new), [50] base_z (new)
    return SkillDiscoveryConfig(factors=[
        FactorConfig("position",      obs_indices=(48, 49),     method="metra", skill_dim=2),
        FactorConfig("lin_velocity",   obs_indices=(0, 1, 2),    method="diayn", skill_dim=4),
        FactorConfig("heading_rate",   obs_indices=(5,),         method="diayn", skill_dim=4),
        FactorConfig("base_height",    obs_indices=(50,),        method="diayn", skill_dim=4),
        FactorConfig("orientation",    obs_indices=(6, 7, 8),    method="diayn", skill_dim=4),
    ])
```

**Algo + skill manager initialization:**
```python
# Env setup — reuse make_envs() from train_offpolicy.py
env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)
dict_obs = isinstance(env_state.obs, dict)
if dict_obs:
    obs_dim = env_state.obs["state"].shape[-1]

# Skill manager
if args.skill_mode == "diayn":
    skill_cfg = _make_pure_diayn_config(obs_dim, args.num_skills)
elif args.skill_mode == "metra":
    skill_cfg = _make_pure_metra_config(obs_dim, args.skill_dim)
elif args.skill_mode == "factorized":
    skill_cfg = _make_go2_factorized_config()

skill_manager = SkillManager(skill_cfg)
key, aux_key, z_key = jax.random.split(key, 3)
aux_state = skill_manager.init(aux_key)
current_z = skill_manager.sample_skills(z_key, cfg.num_envs)

# Algo sees augmented obs dim (raw obs + z)
augmented_obs_dim = obs_dim + skill_manager.total_skill_dim
algo = _make_algo(algo_name, algo_cfg, augmented_obs_dim, action_dim, cfg)
training_state = algo.init(init_key)

# Buffer stores augmented obs (raw obs + z) as regular obs — no special fields needed.
# This keeps the buffer completely agnostic to skill discovery and works with
# both replay buffer (off-policy) and rollout buffer (on-policy/PPO).
buffer = JaxReplayBuffer(augmented_obs_dim, action_dim, max_size=algo_cfg.buffer_size)
```

**Training loop core (key differences from train_offpolicy.py):**
```python
for outer_step in range(...):
    raw_obs = _get_obs(env_state.obs)  # flat obs, no z

    # Obs norm (if enabled): normalize raw obs, then augment with z
    obs_normed = norm_normalize(norm_state, raw_obs) if use_obs_norm else raw_obs
    augmented_obs = skill_manager.augment_obs(obs_normed, current_z)

    # Action selection
    if len(buffer) < algo_cfg.min_buffer_size:
        action = jax.random.uniform(ak, (cfg.num_envs, action_dim), minval=-1, maxval=1)
    else:
        action = explore(training_state.actor_params, augmented_obs, ak)

    # Env step
    env_state = env_step(env_state, action)
    next_raw_obs = _get_obs(env_state.obs)

    # Compute intrinsic reward at collection time
    intrinsic_reward = skill_manager.compute_reward(
        aux_state, obs_normed,
        norm_normalize(norm_state, next_raw_obs) if use_obs_norm else next_raw_obs,
        current_z)
    if skill_cfg.style_weight > 0:
        intrinsic_reward += skill_cfg.style_weight * env_state.reward * cfg.reward_scaling

    # Buffer: store augmented obs (raw+z) and intrinsic reward as regular fields.
    # The buffer is completely unaware of skill discovery.
    next_obs_normed = norm_normalize(norm_state, next_raw_obs) if use_obs_norm else next_raw_obs
    augmented_next = skill_manager.augment_obs(next_obs_normed, current_z)
    buffer.add_batch(obs=augmented_obs, action=action,
                     reward=intrinsic_reward,
                     next_obs=augmented_next, done=env_state.done,
                     truncation=truncation)

    # Resample z for reset envs
    key, resample_key = jax.random.split(key)
    current_z = skill_manager.resample_on_done(current_z, env_state.done, resample_key)

    # Gradient updates
    if len(buffer) >= algo_cfg.min_buffer_size:
        for _ in range(algo_cfg.grad_updates_per_step):
            batch = buffer.sample(algo_cfg.batch_size, key=sample_key)

            # SAC update — batch already has augmented obs and intrinsic reward
            training_state, sac_metrics = algo.update(training_state, batch)

            # Split augmented obs back into raw obs + z for auxiliary updates
            sampled_raw_obs = batch["obs"][:, :obs_dim]
            sampled_raw_next = batch["next_obs"][:, :obs_dim]
            sampled_z = batch["obs"][:, obs_dim:]

            # Update auxiliary networks (discriminator / phi)
            aux_state, aux_metrics = skill_manager.update(
                aux_state, sampled_raw_obs, sampled_raw_next, sampled_z)
```

**Note on obs normalization with buffer-agnostic design:** When `--obs-norm` is enabled, raw obs are normalized *before* augmentation with z, so the buffer stores `[normalized_obs, z]`. This means normalization is applied at collection time, which technically violates the "normalize at sample time" rule for off-policy (see `.context/lessons/offpolicy.md`). However, since z is fixed per transition and the normalized obs + z are consumed together, this is acceptable. The obs norm statistics are frozen after the warmup period anyway. If staleness becomes an issue, we can store raw obs in the buffer and normalize at sample time — but this requires splitting augmented obs, normalizing just the obs portion, and re-concatenating, which adds complexity for minimal benefit.

**Eval structure:**
For v1, eval is simpler than `train_offpolicy.py` because there's no single env reward to track. The eval loop:
1. For each of `num_eval_skills` discrete skills (DIAYN) or sampled z vectors (METRA):
   - Run `num_eval_episodes` episodes with that fixed z
   - Record env reward (for reference) and intrinsic reward
2. Log: mean intrinsic reward per skill, discriminator accuracy (DIAYN), phi norm (METRA)
3. Checkpointing: save training_state + aux_state + norm_state

```python
def eval_skills(algo, training_state, skill_manager, aux_state, eval_env,
                cfg, norm_state, key, num_skills_to_eval=5, episodes_per_skill=3):
    """Run eval rollouts per skill. Returns metrics dict."""
    metrics = {}
    for skill_idx in range(num_skills_to_eval):
        key, z_key = jax.random.split(key)
        # Sample a deterministic z for this skill
        z = skill_manager.sample_skills(z_key, 1)  # (1, skill_dim)
        z_batch = jnp.broadcast_to(z, (cfg.num_envs, z.shape[-1]))  # broadcast for parallel eval

        # Run episodes and collect returns
        # (reuse eval pattern from train_offpolicy.py — rollout with select_action deterministic=True)
        ...
    return metrics
```

Full eval protocol is deferred to a follow-up task (see "Future Avenues: Eval Protocol").

**Logging additions beyond train_offpolicy.py:**
- Per-factor discriminator accuracy (DIAYN factors)
- Per-factor METRA inner product and norm
- Dual lambda value (METRA factors)
- Intrinsic reward mean/std
- Env reward mean (for reference, not optimized)

**Preset wiring:**
The `_make_algo()` function from `train_offpolicy.py` is reused directly. It takes `(algo_name, algo_cfg, obs_dim, action_dim, cfg)` — we pass `augmented_obs_dim` as `obs_dim`. Presets from `env_presets.py` are loaded the same way via `get_sac_preset(args.env)`. No changes to the preset system needed.

- [ ] **Step 3: Smoke test — pure DIAYN on CheetahRun (200k steps)**

```bash
uv run python train_skill_discovery.py --env CheetahRun --skill-mode diayn --num-skills 10 --total-timesteps 200000
```

Expected: No crashes, discriminator accuracy improves from ~10% (random) toward >50%, intrinsic reward increases.

- [ ] **Step 4: Smoke test — pure METRA on WalkerWalk (200k steps)**

```bash
uv run python train_skill_discovery.py --env WalkerWalk --skill-mode metra --skill-dim 2 --total-timesteps 200000
```

Expected: No crashes, METRA inner product and norm metrics are finite and non-zero.

- [ ] **Step 5: Commit**

```bash
git add train_skill_discovery.py
git commit -m "feat: skill discovery training script — DIAYN, METRA, and factorized USD modes"
```

---

### Task 7: Integration Validation

**Files:**
- Test: `tests/test_skill_discovery.py` (append integration test)

- [ ] **Step 1: Write integration test (end-to-end without env)**

Append to `tests/test_skill_discovery.py`:

```python
def test_full_loop_diayn():
    """End-to-end: SAC + DIAYN skill manager, 5 gradient steps."""
    from jax_rl.skill_discovery.config import FactorConfig, SkillDiscoveryConfig
    from jax_rl.skill_discovery.manager import SkillManager
    from jax_rl.algos.sac import SAC
    from jax_rl.configs.sac_config import SACConfig

    obs_dim = 17
    action_dim = 6
    num_skills = 4

    # Skill setup
    skill_cfg = SkillDiscoveryConfig(factors=[
        FactorConfig("full", tuple(range(obs_dim)), "diayn", skill_dim=num_skills),
    ])
    mgr = SkillManager(skill_cfg)
    aux_state = mgr.init(KEY)

    # SAC sees augmented obs
    aug_obs_dim = obs_dim + num_skills
    sac_cfg = SACConfig(hidden_dim=(64, 64), batch_size=BATCH)
    sac = SAC(sac_cfg, aug_obs_dim, action_dim,
              optax.adam(3e-4), optax.adam(3e-4), gamma=0.99)
    sac_state = sac.init(KEY)

    # Fake data
    key = KEY
    for step in range(5):
        key, k1, k2, k3 = jax.random.split(key, 4)
        obs = jax.random.normal(k1, (BATCH, obs_dim))
        next_obs = jax.random.normal(k2, (BATCH, obs_dim))
        z = mgr.sample_skills(k3, BATCH)
        action = jax.random.uniform(k1, (BATCH, action_dim), minval=-1, maxval=1)

        # Intrinsic reward
        reward = mgr.compute_reward(aux_state, obs, next_obs, z)

        # SAC update with augmented obs
        sac_batch = {
            "obs": mgr.augment_obs(obs, z),
            "action": action,
            "reward": reward.reshape(-1, 1),
            "next_obs": mgr.augment_obs(next_obs, z),
            "done": jnp.zeros((BATCH, 1)),
            "truncation": jnp.zeros((BATCH, 1)),
        }
        sac_state, sac_metrics = sac.update(sac_state, sac_batch)

        # Auxiliary update
        aux_state, aux_metrics = mgr.update(aux_state, obs, next_obs, z)

    assert "actor_loss" in sac_metrics
    assert "full_disc_loss" in aux_metrics
    assert jnp.all(jnp.isfinite(jnp.array(list(sac_metrics.values()))))


def test_full_loop_metra():
    """End-to-end: SAC + METRA skill manager, 5 gradient steps."""
    from jax_rl.skill_discovery.config import FactorConfig, SkillDiscoveryConfig
    from jax_rl.skill_discovery.manager import SkillManager
    from jax_rl.algos.sac import SAC
    from jax_rl.configs.sac_config import SACConfig

    obs_dim = 17
    action_dim = 6
    skill_dim = 2

    skill_cfg = SkillDiscoveryConfig(factors=[
        FactorConfig("full", tuple(range(obs_dim)), "metra", skill_dim=skill_dim),
    ])
    mgr = SkillManager(skill_cfg)
    aux_state = mgr.init(KEY)

    aug_obs_dim = obs_dim + skill_dim
    sac_cfg = SACConfig(hidden_dim=(64, 64), batch_size=BATCH)
    sac = SAC(sac_cfg, aug_obs_dim, action_dim,
              optax.adam(3e-4), optax.adam(3e-4), gamma=0.99)
    sac_state = sac.init(KEY)

    key = KEY
    for step in range(5):
        key, k1, k2, k3 = jax.random.split(key, 4)
        obs = jax.random.normal(k1, (BATCH, obs_dim))
        next_obs = jax.random.normal(k2, (BATCH, obs_dim))
        z = mgr.sample_skills(k3, BATCH)
        action = jax.random.uniform(k1, (BATCH, action_dim), minval=-1, maxval=1)

        reward = mgr.compute_reward(aux_state, obs, next_obs, z)
        sac_batch = {
            "obs": mgr.augment_obs(obs, z),
            "action": action,
            "reward": reward.reshape(-1, 1),
            "next_obs": mgr.augment_obs(next_obs, z),
            "done": jnp.zeros((BATCH, 1)),
            "truncation": jnp.zeros((BATCH, 1)),
        }
        sac_state, sac_metrics = sac.update(sac_state, sac_batch)
        aux_state, aux_metrics = mgr.update(aux_state, obs, next_obs, z)

    assert "actor_loss" in sac_metrics
    assert "full_metra_inner" in aux_metrics
```

- [ ] **Step 2: Run integration tests**

Run: `uv run python -m pytest tests/test_skill_discovery.py -k "full_loop" -v`
Expected: 2 passed

- [ ] **Step 3: Run full test suite to verify nothing is broken**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests pass (existing + new)

- [ ] **Step 4: Commit**

```bash
git add tests/test_skill_discovery.py
git commit -m "test: skill discovery integration tests — SAC + DIAYN and SAC + METRA end-to-end"
```

---

## Future Avenues (Not in v1 Scope)

### Algo Upgrades
- **FastSAC (C51):** Swap SAC for FastSAC as base algo. Requires tuning V_min/V_max for intrinsic reward magnitude. The training script already supports `--algo fast_sac` via `_make_algo()`.
- **PPO support:** Add on-policy variant of the training loop. D3 uses PPO with per-factor value functions and advantage aggregation. Would require a separate `train_skill_discovery_ppo.py` or a `--on-policy` flag.

### D3 Paper Features
- **Symmetry augmentation:** Mirror transitions via morphological symmetries (K=4 for quadrupeds). Requires defining `M_s`, `M_a`, `M_z` permutation matrices for Go2.
- **Per-factor value functions:** Separate critic per reward component with UCB advantage aggregation. PPO-specific.
- **Lambda conditioning:** Policy conditioned on factor weights `lambda` in addition to `(s, z)`. Enables runtime weight tuning.
- **Dirichlet priors:** Replace discrete uniform with symmetric Dirichlet for DIAYN skills. Curriculum from sparse to uniform.
- **METRA norm-matching reward:** Adaptive curriculum switching between inner-product and norm-matching objectives.

### Style Rewards
- Investigate D3's style factor approach: keep Go2's penalty terms (orientation, torques, contacts) as a weighted style component alongside intrinsic rewards. The `style_weight` config field is already in place.

### Eval Protocol
- Design skill diversity metrics: state coverage per skill, discriminator accuracy, pairwise skill distance
- Per-skill video rollouts: record one episode per skill z, save as grid video
- Downstream task evaluation: use learned skills for goal-reaching (zero-shot z selection via discriminator/phi)
