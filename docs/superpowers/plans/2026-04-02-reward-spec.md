# RewardSpec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract reward computation from inline env methods into a composable `RewardSpec` — a list of `(name, fn)` tuples that the env iterates. DIAYN swaps reward by replacing the spec, not forking the env.

**Architecture:** `RewardSpec` is a list of `RewardTerm` dataclasses, each holding a name and callable. `compute_rewards(spec, **kwargs)` iterates terms and returns an unweighted `dict[str, scalar]`. The existing `step()` code applies weights from `reward_config.scales` as before — zero behavior change. Reward functions stay as instance methods on the env class, called by lambdas in the spec.

**Tech Stack:** Python dataclasses, JAX

**Key constraint:** Zero behavior change. Same reward values, same metrics, same training curves. This is a pure refactor.

---

## Design

**`compute_rewards()` returns UNWEIGHTED values.** Weights stay in `step()` where they already are (line 258-261 in go2_warp_joystick.py). This means:
- `compute_rewards` is simple: iterate terms, call fns, return dict
- `step()` is unchanged after the `_get_reward()` call — same weighting loop
- No risk of double-weighting

**DIAYN future note:** When DIAYN replaces the reward spec, it will also need to supply its own weights (or embed them in the term fns). This is the Option A→B upgrade — defer to DIAYN implementation.

---

## Important Env Differences

- **Go2 Warp Joystick:** 17 reward terms, all as `self._reward_*` / `self._cost_*` methods
- **Go2 MJX Joystick:** 16 reward terms (no `base_height`). Otherwise same method names. **Do NOT copy Warp spec verbatim — read the MJX env's `_get_reward()` and build its spec from what's actually there.**
- **Bongo Board Handstand:** ~9 reward terms, computed **inline in `_get_reward()`** — no helper methods. The spec lambdas must contain the inline expressions, not call missing methods.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `jax_rl/envs/reward_spec.py` | Create | `RewardTerm`, `compute_rewards()` |
| `jax_rl/envs/locomotion/go2_warp_joystick.py` | Modify | Build spec in `_post_init()`, use in `_get_reward()` |
| `jax_rl/envs/locomotion/go2_joystick.py` | Modify | Same (16 terms, no base_height) |
| `jax_rl/envs/locomotion/go2_bongo_handstand.py` | Modify | Same (9 terms, inline expressions) |
| `tests/test_reward_spec.py` | Create | Unit tests for compute_rewards |

---

### Task 1: Create RewardSpec module

**Files:**
- Create: `jax_rl/envs/reward_spec.py`
- Create: `tests/test_reward_spec.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_reward_spec.py
"""Tests for RewardSpec — composable reward terms."""
import jax.numpy as jnp
import pytest

from jax_rl.envs.reward_spec import RewardTerm, compute_rewards


def test_compute_rewards_returns_dict():
    """compute_rewards should return a dict of {name: unweighted scalar}."""
    terms = [
        RewardTerm("a", lambda **kw: jnp.float32(1.0)),
        RewardTerm("b", lambda **kw: jnp.float32(2.0)),
    ]
    result = compute_rewards(terms)
    assert "a" in result
    assert "b" in result
    assert jnp.allclose(result["a"], 1.0)
    assert jnp.allclose(result["b"], 2.0)


def test_compute_rewards_passes_kwargs():
    """Reward functions should receive kwargs (data, action, info, etc.)."""
    def my_reward(data=None, **kw):
        return data * 2.0

    terms = [RewardTerm("x", my_reward)]
    result = compute_rewards(terms, data=jnp.float32(5.0))
    assert jnp.allclose(result["x"], 10.0)


def test_empty_spec():
    """Empty term list should return empty dict."""
    result = compute_rewards([])
    assert result == {}


def test_reward_term_name():
    """RewardTerm should store name and fn."""
    fn = lambda **kw: jnp.float32(0.0)
    t = RewardTerm("test", fn)
    assert t.name == "test"
    assert t.fn is fn


def test_multiple_kwargs():
    """Terms should receive all kwargs passed to compute_rewards."""
    def needs_both(action=None, info=None, **kw):
        return action + info

    terms = [RewardTerm("r", needs_both)]
    result = compute_rewards(terms, action=jnp.float32(3.0), info=jnp.float32(4.0))
    assert jnp.allclose(result["r"], 7.0)
```

- [ ] **Step 2: Implement RewardSpec**

```python
# jax_rl/envs/reward_spec.py
"""Composable reward specification.

Reward terms are standalone functions that receive keyword args (data, action,
info, done, etc.) and return a scalar. The env builds a list of RewardTerms
in _post_init() and calls compute_rewards() in step().

compute_rewards() returns UNWEIGHTED values. The env's step() applies weights
from reward_config.scales as before. This keeps the refactor zero-behavior-change.

For DIAYN: swap the env's _reward_spec list. The step() weighting loop uses
whatever term names are in the spec — DIAYN terms just need matching weights.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RewardTerm:
    """A single reward component."""
    name: str
    fn: Callable[..., Any]  # (**kwargs) -> scalar


def compute_rewards(terms: list[RewardTerm], **kwargs) -> dict[str, Any]:
    """Compute all reward terms, return unweighted dict.

    Args:
        terms: List of RewardTerms to evaluate.
        **kwargs: Passed to each reward function (data, action, info, etc.)

    Returns:
        Dict of {term_name: unweighted_scalar}.
    """
    return {term.name: term.fn(**kwargs) for term in terms}
```

- [ ] **Step 3: Run tests**

Run: `JAX_PLATFORMS=cpu uv run python -m pytest tests/test_reward_spec.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/envs/reward_spec.py tests/test_reward_spec.py
git commit -m "feat: RewardSpec — composable reward terms for env reward computation"
```

---

### Task 2: Refactor Go2 Warp Joystick to use RewardSpec

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_warp_joystick.py`

The 17 reward methods stay on the class. We build a spec in `_post_init()` and replace `_get_reward()` body.

- [ ] **Step 1: Add import and build spec in `_post_init()`**

At the end of `_post_init()`, add:

```python
from jax_rl.envs.reward_spec import RewardTerm, compute_rewards

self._reward_spec = [
    RewardTerm("tracking_lin_vel", lambda data, info, **kw:
        self._reward_tracking_lin_vel(info["command"], self.get_local_linvel(data))),
    RewardTerm("tracking_ang_vel", lambda data, info, **kw:
        self._reward_tracking_ang_vel(info["command"], self.get_gyro(data))),
    RewardTerm("lin_vel_z", lambda data, **kw:
        self._cost_lin_vel_z(self.get_global_linvel(data))),
    RewardTerm("ang_vel_xy", lambda data, **kw:
        self._cost_ang_vel_xy(self.get_global_angvel(data))),
    RewardTerm("orientation", lambda data, **kw:
        self._cost_orientation(self.get_upvector(data))),
    RewardTerm("torques", lambda data, **kw:
        self._cost_torques(data.actuator_force)),
    RewardTerm("action_rate", lambda action, info, **kw:
        self._cost_action_rate(action, info["last_act"], info["last_last_act"])),
    RewardTerm("energy", lambda data, **kw:
        self._cost_energy(data.qvel[6:], data.actuator_force)),
    RewardTerm("dof_pos_limits", lambda data, **kw:
        self._cost_joint_pos_limits(data.qpos[7:])),
    RewardTerm("feet_air_time", lambda info, first_contact, **kw:
        self._reward_feet_air_time(info["feet_air_time"], first_contact, info["command"])),
    RewardTerm("feet_slip", lambda data, contact, info, **kw:
        self._cost_feet_slip(data, contact, info)),
    RewardTerm("feet_clearance", lambda data, **kw:
        self._cost_feet_clearance(data)),
    RewardTerm("feet_height", lambda info, first_contact, **kw:
        self._cost_feet_height(info["swing_peak"], first_contact, info)),
    RewardTerm("termination", lambda done, **kw:
        self._cost_termination(done)),
    RewardTerm("stand_still", lambda data, info, **kw:
        self._cost_stand_still(info["command"], data.qpos[7:])),
    RewardTerm("pose", lambda data, **kw:
        self._reward_pose(data.qpos[7:])),
    RewardTerm("base_height", lambda data, **kw:
        self._cost_base_height(data)),
]
```

- [ ] **Step 2: Replace `_get_reward()` body**

Change from the inline dict to:

```python
def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
    del metrics
    return compute_rewards(
        self._reward_spec,
        data=data, action=action, info=info,
        done=done, first_contact=first_contact, contact=contact,
    )
```

**IMPORTANT:** The `step()` code that applies weights (lines 258-261) stays UNCHANGED:
```python
rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
```
This applies weights to the unweighted dict from `compute_rewards`. Zero behavior change.

- [ ] **Step 3: Run existing Go2 Warp tests**

Run: `uv run python -m pytest tests/test_go2_warp_env.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/envs/locomotion/go2_warp_joystick.py
git commit -m "refactor: Go2 Warp joystick uses RewardSpec"
```

---

### Task 3: Refactor Go2 MJX Joystick to use RewardSpec

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_joystick.py`

**IMPORTANT:** The MJX env has 16 reward terms — it does NOT have `base_height`. Read the actual `_get_reward()` in this file and build the spec from what's there. Do NOT copy the Warp spec verbatim.

- [ ] **Step 1: Read `go2_joystick.py` and build spec from its actual reward terms**
- [ ] **Step 2: Replace `_get_reward()` body with `compute_rewards()`**
- [ ] **Step 3: Run tests**

Run: `uv run python -m pytest tests/test_go2_env.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/envs/locomotion/go2_joystick.py
git commit -m "refactor: Go2 MJX joystick uses RewardSpec"
```

---

### Task 4: Refactor Bongo Board Handstand to use RewardSpec

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_bongo_handstand.py`

**IMPORTANT:** The bongo env computes rewards INLINE in `_get_reward()` — there are no helper methods like `_reward_*` or `_cost_*`. The spec lambdas must contain the inline expressions directly. Read the actual `_get_reward()` and wrap each expression in a RewardTerm lambda.

- [ ] **Step 1: Read `go2_bongo_handstand.py` `_get_reward()` and build spec from inline expressions**
- [ ] **Step 2: Replace `_get_reward()` body with `compute_rewards()`**
- [ ] **Step 3: Run tests**

Run: `uv run python -m pytest tests/test_go2_bongo_env.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/envs/locomotion/go2_bongo_handstand.py
git commit -m "refactor: Bongo handstand uses RewardSpec"
```

---

### Task 5: Full test suite + update docs

- [ ] **Step 1: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Update docs**

Update `.context/TODO.md` — mark RewardSpec as done.
Update `.context/journals/` — add notes.

- [ ] **Step 3: Commit**

```bash
git add .context/
git commit -m "docs: update for RewardSpec refactor"
```

---

## Future: DIAYN Reward Swap (NOT implemented now)

When DIAYN arrives, the reward swap looks like:

```python
# DIAYN overrides the env's reward spec
env._reward_spec = [
    RewardTerm("discriminator", lambda obs, skill_z, **kw:
        discriminator(obs, skill_z)),
]
# Also needs to provide matching weights in reward_config.scales,
# OR upgrade to Option B where weights are embedded in RewardTerm.
```

The env's `step()` doesn't change — `compute_rewards()` iterates whatever spec it has. The weighting loop in `step()` uses whatever keys are in the dict.
