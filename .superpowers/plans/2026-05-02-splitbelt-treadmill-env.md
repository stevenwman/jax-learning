# Splitbelt Treadmill Env Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `Go2WarpSplitbelt` env (mjx/Warp backend), schedule samplers, offline gait analyzer, and the test surface that locks the contract — per the design at [`.superpowers/specs/2026-05-02-splitbelt-treadmill-env-design.md`](../specs/2026-05-02-splitbelt-treadmill-env-design.md).

**Architecture:** Two long belt slabs on slide joints with velocity actuators (M1 mechanics). Robot subclasses `Go2WarpBase`, reads belt commands from a per-episode `(T, 2)` schedule table sampled at reset. Reward = port from joystick env + new `treadmill_drift`. Termination = contact-based primary + threshold backstop. Four `obs_mode` configs gate what the policy sees; privileged_state always full info. Offline analyzer (pure numpy) computes adaptation metrics from logged time-series.

**Tech Stack:** JAX/Flax, MuJoCo Warp via `mujoco_playground`, `obs_spec` library, `DomainRandWrapper`. Tests use `pytest` with marker discipline from `.context/lessons/testing_new_envs.md`.

**Spec sections referenced:** S§N below = section N of the spec doc.

**Hard rules (do not violate):**
- `uv run python <cmd>` always.
- No module-level `jax.random.PRNGKey(...)`, `jnp.zeros(...)`, env construction in tests. Push to fixtures.
- Mark GPU tests with the full set `[gpu, warp, go2]` (or `[gpu, warp, go2, deploy]`); markers do NOT inherit.
- Hermetic CPU is the default lane.
- No hardcoded `/home/stevenman/...` paths.
- No symmetry reward in env (S§7.2 — load-bearing for the science).
- No Co-Authored-By in commits (per `MEMORY.md`).
- `cmd_x = 0` is the canonical regime; do NOT add walking-cmd variant in this plan (S§11.2).

---

## Stage 0: Pre-flight

### Task 0.1: Verify default test lane is green before changes

**Files:** none

- [ ] **Step 1: Run baseline pytest**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q
```

Expected: green. Record the pass count — your changes must not lower it.

- [ ] **Step 2: Record collection baseline for marker selectors**

```bash
uv run python -m pytest --collect-only -q -m gpu | tail -20
uv run python -m pytest --collect-only -q -m "warp and go2" | tail -20
```

Expected: each prints a list of currently-marked tests. You'll re-grep against these later to confirm new tests are picked up by their markers.

- [ ] **Step 3: Confirm spec is committed**

```bash
git log --oneline .superpowers/specs/2026-05-02-splitbelt-treadmill-env-design.md
```

Expected: shows commits `4027412` + `5f54eb8` (or later spec edits).

---

## Stage 1: Pure-python modules (hermetic, no env)

These come first because they're isolated, fast to test, and don't depend on the XML or env class.

### Task 1.1: Schedule samplers — module skeleton + `tied`

**Files:**
- Create: `jax_rl/envs/locomotion/splitbelt_schedules.py`
- Test: `tests/test_splitbelt_schedules.py`

Schedule samplers are pure functions returning `(T, 2)` jax arrays. Episode reset calls one to fill the belt schedule table. See S§5.4.

- [ ] **Step 1: Write the failing test**

`tests/test_splitbelt_schedules.py`:

```python
"""Hermetic tests for splitbelt schedule samplers (S§10.1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.locomotion import splitbelt_schedules as sched


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


def test_tied_shape_and_value(rng):
    table = sched.tied(rng, T=100, v=0.7)
    assert table.shape == (100, 2)
    assert jnp.allclose(table, 0.7)
```

- [ ] **Step 2: Run test, confirm failure**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_schedules.py::test_tied_shape_and_value
```

Expected: FAIL with `ModuleNotFoundError: jax_rl.envs.locomotion.splitbelt_schedules`.

- [ ] **Step 3: Create module with `tied` only**

`jax_rl/envs/locomotion/splitbelt_schedules.py`:

```python
"""Per-episode belt-speed schedule samplers for SplitbeltTreadmill envs.

Each factory: (rng, T, **cfg) -> jax.Array of shape (T, 2) with columns (vL, vR).
Pure functions; no env knowledge. See spec S§5.4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def tied(rng: jax.Array, T: int, *, v: float = 0.5) -> jax.Array:
    """Both belts at constant speed v for the whole episode."""
    del rng  # deterministic
    return jnp.full((T, 2), v, dtype=jnp.float32)
```

- [ ] **Step 4: Run test, confirm pass**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_schedules.py::test_tied_shape_and_value
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/locomotion/splitbelt_schedules.py tests/test_splitbelt_schedules.py
git commit -m "feat(splitbelt): schedule samplers skeleton + tied factory"
```

---

### Task 1.2: `split_constant` sampler

**Files:**
- Modify: `jax_rl/envs/locomotion/splitbelt_schedules.py`
- Modify: `tests/test_splitbelt_schedules.py`

- [ ] **Step 1: Add failing test**

```python
def test_split_constant(rng):
    table = sched.split_constant(rng, T=50, vL=0.5, vR=1.0)
    assert table.shape == (50, 2)
    assert jnp.allclose(table[:, 0], 0.5)
    assert jnp.allclose(table[:, 1], 1.0)
```

- [ ] **Step 2: Run, expect fail**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_schedules.py::test_split_constant
```

Expected: FAIL `AttributeError: module ... has no attribute 'split_constant'`.

- [ ] **Step 3: Implement**

Append to `splitbelt_schedules.py`:

```python
def split_constant(
    rng: jax.Array, T: int, *, vL: float = 0.5, vR: float = 1.0
) -> jax.Array:
    """Asymmetric belts at constant (vL, vR) for the whole episode (A2 fixed-context)."""
    del rng
    row = jnp.array([vL, vR], dtype=jnp.float32)
    return jnp.tile(row, (T, 1))
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(splitbelt): split_constant schedule sampler"
```

---

### Task 1.3: `tied_split_tied` (A1 protocol schedule)

**Files:**
- Modify: `splitbelt_schedules.py`, `test_splitbelt_schedules.py`

A1 protocol: warmup tied → split → return-tied. Phase boundaries at `t1` and `t1+t2`. See S§3 and S§5.4.

- [ ] **Step 1: Write tests for shape, phase values, phase boundaries**

```python
def test_tied_split_tied_phase_values(rng):
    T, t1, t2 = 1000, 200, 600
    table = sched.tied_split_tied(
        rng, T=T,
        v_warm=0.5, vL_split=0.5, vR_split=1.0, t1=t1, t2=t2,
    )
    assert table.shape == (T, 2)
    # Warmup phase: tied at 0.5
    assert jnp.allclose(table[:t1], 0.5)
    # Split phase: vL=0.5, vR=1.0
    assert jnp.allclose(table[t1:t1+t2, 0], 0.5)
    assert jnp.allclose(table[t1:t1+t2, 1], 1.0)
    # Return phase: tied at 0.5
    assert jnp.allclose(table[t1+t2:], 0.5)


def test_tied_split_tied_phase_boundaries(rng):
    T, t1, t2 = 100, 30, 40
    table = sched.tied_split_tied(
        rng, T=T,
        v_warm=0.5, vL_split=0.3, vR_split=0.9, t1=t1, t2=t2,
    )
    # Step before split: tied
    assert table[t1 - 1, 0] == table[t1 - 1, 1]
    # First step of split: asymmetric
    assert table[t1, 0] != table[t1, 1]
    # Last step of split: asymmetric
    assert table[t1 + t2 - 1, 0] != table[t1 + t2 - 1, 1]
    # First step of return: tied
    assert table[t1 + t2, 0] == table[t1 + t2, 1]
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement**

```python
def tied_split_tied(
    rng: jax.Array,
    T: int,
    *,
    v_warm: float = 0.5,
    vL_split: float = 0.5,
    vR_split: float = 1.0,
    t1: int = 200,
    t2: int = 600,
) -> jax.Array:
    """Three-phase schedule for A1 (within-episode adaptation) protocol.

    Phase 1: 0 <= t < t1, tied at v_warm.
    Phase 2: t1 <= t < t1+t2, split at (vL_split, vR_split).
    Phase 3: t1+t2 <= t < T, tied at v_warm.
    """
    del rng
    idx = jnp.arange(T)
    in_split = (idx >= t1) & (idx < t1 + t2)
    vL = jnp.where(in_split, vL_split, v_warm).astype(jnp.float32)
    vR = jnp.where(in_split, vR_split, v_warm).astype(jnp.float32)
    return jnp.stack([vL, vR], axis=-1)
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(splitbelt): tied_split_tied schedule (A1 protocol)"
```

---

### Task 1.4: `random_per_episode` (A2/A3 training schedule)

**Files:** as above.

- [ ] **Step 1: Write tests (range + determinism)**

```python
def test_random_per_episode_range(rng):
    table = sched.random_per_episode(
        rng, T=10, v_range=(0.3, 1.5), ratio_range=(0.5, 2.0),
    )
    assert table.shape == (10, 2)
    # Constant within episode
    assert jnp.allclose(table[0], table[-1])
    vL, vR = table[0]
    assert 0.3 <= vL <= 1.5
    assert 0.3 * 0.5 <= vR <= 1.5 * 2.0


def test_random_per_episode_determinism(rng):
    a = sched.random_per_episode(rng, T=20)
    b = sched.random_per_episode(rng, T=20)
    assert jnp.allclose(a, b)
    different = sched.random_per_episode(jax.random.PRNGKey(1), T=20)
    assert not jnp.allclose(a, different)
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement**

```python
def random_per_episode(
    rng: jax.Array,
    T: int,
    *,
    v_range: tuple[float, float] = (0.3, 1.5),
    ratio_range: tuple[float, float] = (0.5, 2.0),
) -> jax.Array:
    """Sample (vL, vR) once at episode start; constant for whole episode (A2/A3 training)."""
    k1, k2 = jax.random.split(rng)
    v = jax.random.uniform(k1, (), minval=v_range[0], maxval=v_range[1])
    ratio = jax.random.uniform(k2, (), minval=ratio_range[0], maxval=ratio_range[1])
    vL = v
    vR = v * ratio
    row = jnp.stack([vL, vR]).astype(jnp.float32)
    return jnp.tile(row, (T, 1))
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(splitbelt): random_per_episode schedule sampler"
```

---

### Task 1.5: `continual_phase` (A4 protocol schedule)

**Files:** as above.

A4 = pretrain on tied, then perturb to split. The sampler picks based on `phase_id` (0 = tied warmup, 1 = split fine-tune).

- [ ] **Step 1: Write test**

```python
def test_continual_phase_warmup_is_tied(rng):
    table = sched.continual_phase(
        rng, T=50, phase_id=0, v_warm=0.5, vL_split=0.5, vR_split=1.0,
    )
    assert jnp.allclose(table, 0.5)


def test_continual_phase_split_is_split(rng):
    table = sched.continual_phase(
        rng, T=50, phase_id=1, v_warm=0.5, vL_split=0.5, vR_split=1.0,
    )
    assert jnp.allclose(table[:, 0], 0.5)
    assert jnp.allclose(table[:, 1], 1.0)
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement**

```python
def continual_phase(
    rng: jax.Array,
    T: int,
    *,
    phase_id: int,
    v_warm: float = 0.5,
    vL_split: float = 0.5,
    vR_split: float = 1.0,
) -> jax.Array:
    """A4 continual-learning schedule: phase 0 = tied warmup, phase 1 = split fine-tune."""
    if phase_id == 0:
        return tied(rng, T, v=v_warm)
    if phase_id == 1:
        return split_constant(rng, T, vL=vL_split, vR=vR_split)
    raise ValueError(f"continual_phase: phase_id must be 0 or 1, got {phase_id}")
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(splitbelt): continual_phase schedule (A4 protocol)"
```

---

### Task 1.6: Sampler registry + dispatch helper

**Files:** as above.

Env config will specify `schedule_kind: str` + `schedule_params: dict`. Add a single dispatcher that routes string kind to factory. Avoids env knowing about each factory.

- [ ] **Step 1: Write test**

```python
def test_sample_schedule_dispatch(rng):
    table = sched.sample_schedule(rng, T=50, kind="tied", params={"v": 0.4})
    assert jnp.allclose(table, 0.4)
    table2 = sched.sample_schedule(
        rng, T=20, kind="split_constant", params={"vL": 0.3, "vR": 0.6}
    )
    assert jnp.allclose(table2[:, 0], 0.3)


def test_sample_schedule_unknown_kind(rng):
    with pytest.raises(ValueError, match="unknown schedule kind"):
        sched.sample_schedule(rng, T=10, kind="not_a_real_kind", params={})
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement dispatcher**

```python
_FACTORIES = {
    "tied": tied,
    "split_constant": split_constant,
    "tied_split_tied": tied_split_tied,
    "random_per_episode": random_per_episode,
    "continual_phase": continual_phase,
}


def sample_schedule(rng: jax.Array, T: int, *, kind: str, params: dict) -> jax.Array:
    """Dispatch to the appropriate sampler by string kind."""
    if kind not in _FACTORIES:
        raise ValueError(
            f"unknown schedule kind {kind!r}; valid: {sorted(_FACTORIES)}"
        )
    return _FACTORIES[kind](rng, T, **params)
```

- [ ] **Step 4: Run, expect pass + run full file**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_schedules.py
```

Expected: all schedule tests pass.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(splitbelt): schedule sampler dispatch by kind"
```

---

### Task 1.7: Belt-assignment helper (foot_belt_id) — pure function + hermetic test

**Files:**
- Create: `jax_rl/envs/locomotion/splitbelt_geom.py`
- Test: `tests/test_splitbelt_belt_assignment.py`

This is the **§5 risk-callout** test from the spec. Belt assignment from foot xy positions, hermetic, runs every default-lane execution.

The geometry: belts run along world-x. Left belt covers `y ∈ [y_left_min, y_left_max]`, right belt covers `y ∈ [y_right_min, y_right_max]`. Gap is between them.

- [ ] **Step 1: Write test**

`tests/test_splitbelt_belt_assignment.py`:

```python
"""Hermetic test for foot-to-belt id mapping (S§5 risk callout)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jax_rl.envs.locomotion import splitbelt_geom as geom


@pytest.fixture
def belt_layout():
    return geom.BeltLayout(
        left_y_min=-0.30, left_y_max=-0.025,
        right_y_min=0.025, right_y_max=0.30,
    )


def test_foot_in_left_belt(belt_layout):
    foot_xy = jnp.array([[0.0, -0.15]])
    assert int(geom.foot_belt_id(foot_xy, belt_layout)[0]) == 0


def test_foot_in_right_belt(belt_layout):
    foot_xy = jnp.array([[0.0, 0.15]])
    assert int(geom.foot_belt_id(foot_xy, belt_layout)[0]) == 1


def test_foot_in_gap(belt_layout):
    foot_xy = jnp.array([[0.0, 0.0]])
    assert int(geom.foot_belt_id(foot_xy, belt_layout)[0]) == -1


def test_foot_off_belt_y(belt_layout):
    foot_xy = jnp.array([[0.0, 0.5]])
    assert int(geom.foot_belt_id(foot_xy, belt_layout)[0]) == -1


def test_vectorized_over_4_feet(belt_layout):
    foot_xy = jnp.array([
        [0.0, -0.15],   # left
        [0.0, 0.15],    # right
        [0.0, 0.0],     # gap
        [0.0, 0.5],     # off
    ])
    ids = geom.foot_belt_id(foot_xy, belt_layout)
    assert tuple(int(i) for i in ids) == (0, 1, -1, -1)
```

- [ ] **Step 2: Run, expect fail**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_belt_assignment.py
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`jax_rl/envs/locomotion/splitbelt_geom.py`:

```python
"""Robot-agnostic geometry helpers for splitbelt env (foot-to-belt assignment)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BeltLayout(NamedTuple):
    """Y-range of left and right belts. X-range is unbounded (long slabs)."""
    left_y_min: float
    left_y_max: float
    right_y_min: float
    right_y_max: float


def foot_belt_id(foot_xy: jax.Array, layout: BeltLayout) -> jax.Array:
    """Map foot (x, y) positions to belt id.

    Returns int32 array of shape foot_xy.shape[:-1]: 0 = left, 1 = right, -1 = neither.
    """
    y = foot_xy[..., 1]
    in_left = (y >= layout.left_y_min) & (y <= layout.left_y_max)
    in_right = (y >= layout.right_y_min) & (y <= layout.right_y_max)
    # Mutually exclusive by construction (gap > 0). Use int casts.
    return (in_right.astype(jnp.int32) - in_left.astype(jnp.int32) * 2 + (~(in_left | in_right)).astype(jnp.int32) * (-1) - 0).astype(jnp.int32)  # placeholder
```

Wait — the placeholder formula is wrong. Replace with a clean dispatch:

```python
def foot_belt_id(foot_xy: jax.Array, layout: BeltLayout) -> jax.Array:
    """Map foot (x, y) positions to belt id.

    Returns int32 array of shape foot_xy.shape[:-1]: 0 = left, 1 = right, -1 = neither.
    """
    y = foot_xy[..., 1]
    in_left = (y >= layout.left_y_min) & (y <= layout.left_y_max)
    in_right = (y >= layout.right_y_min) & (y <= layout.right_y_max)
    # left=0, right=1, neither=-1; gap layout enforces mutual exclusion.
    return jnp.where(in_left, 0, jnp.where(in_right, 1, -1)).astype(jnp.int32)
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/locomotion/splitbelt_geom.py tests/test_splitbelt_belt_assignment.py
git commit -m "feat(splitbelt): foot_belt_id geometry helper + hermetic test"
```

---

### Task 1.8: Offline analyzer skeleton + step-detection test

**Files:**
- Create: `jax_rl/envs/locomotion/splitbelt_analysis.py`
- Test: `tests/test_splitbelt_metrics.py`

Pure numpy library that the eventual `scripts/analyze_splitbelt.py` will wrap. Building the library first (testable) and the script later (CLI thin shim).

Step events from contact time-series: detect rising edge (touchdown) and falling edge (lift-off) per foot.

- [ ] **Step 1: Write test**

```python
"""Hermetic tests for splitbelt offline analysis (S§9.3, S§10.1)."""

from __future__ import annotations

import numpy as np
import pytest

from jax_rl.envs.locomotion import splitbelt_analysis as sba


def test_detect_step_events_simple():
    # Foot 0: touches down at t=2, lifts at t=5, touches down again at t=8, lifts at t=11.
    contact = np.zeros((15, 1), dtype=bool)
    contact[2:5, 0] = True
    contact[8:11, 0] = True
    events = sba.detect_step_events(contact)
    assert events["touchdown_steps"][0] == [2, 8]
    assert events["liftoff_steps"][0] == [5, 11]


def test_detect_step_events_starts_in_contact():
    # Already in contact at t=0; first touchdown should not register at t=0.
    contact = np.zeros((10, 1), dtype=bool)
    contact[0:3, 0] = True
    contact[6:9, 0] = True
    events = sba.detect_step_events(contact)
    # First detected touchdown is the t=6 rising edge, not t=0.
    assert events["touchdown_steps"][0] == [6]
    assert events["liftoff_steps"][0] == [3, 9]
```

- [ ] **Step 2: Run, expect fail**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_metrics.py::test_detect_step_events_simple
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`jax_rl/envs/locomotion/splitbelt_analysis.py`:

```python
"""Offline gait analysis for splitbelt rollouts.

Pure numpy. Loads `splitbelt_traj.npz` (per-step primitives) + schedule_table,
returns adaptation metrics per spec S§9.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def detect_step_events(contact: np.ndarray) -> dict[str, list[list[int]]]:
    """Detect touchdown (rising-edge) and liftoff (falling-edge) per foot.

    Args:
        contact: bool array (T, F) where F is number of feet.

    Returns:
        dict with keys "touchdown_steps", "liftoff_steps", each a list of length F,
        each element a list of step indices.
    """
    T, F = contact.shape
    diffs = np.diff(contact.astype(np.int8), axis=0)  # (T-1, F)
    touchdown = [[int(t + 1) for t in np.flatnonzero(diffs[:, f] > 0)] for f in range(F)]
    liftoff = [[int(t + 1) for t in np.flatnonzero(diffs[:, f] < 0)] for f in range(F)]
    return {"touchdown_steps": touchdown, "liftoff_steps": liftoff}
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/locomotion/splitbelt_analysis.py tests/test_splitbelt_metrics.py
git commit -m "feat(splitbelt): offline analysis - step event detection"
```

---

### Task 1.9: Offline analyzer — step length + asymmetry

Step length = forward distance between consecutive ipsilateral touchdowns. Asymmetry = `(SL_R - SL_L) / (SL_R + SL_L)` per spec S§9.4.

- [ ] **Step 1: Write tests**

Append to `tests/test_splitbelt_metrics.py`:

```python
def test_step_length_symmetric():
    # 4 feet (FL, FR, RL, RR). Stride 0.4 m, all feet step at same length.
    T = 200
    contact = np.zeros((T, 4), dtype=bool)
    foot_xy = np.zeros((T, 4, 2))
    # Synthesize 5 strides per foot, each 0.4 m forward, alternating contact phases.
    stride_len = 0.4
    for f in range(4):
        for k in range(5):
            t_td = 20 + k * 30 + (10 if f in (1, 2) else 0)  # stagger diagonal feet
            t_lo = t_td + 15
            if t_lo < T:
                contact[t_td:t_lo, f] = True
                foot_xy[t_td, f, 0] = k * stride_len  # x at touchdown
    events = sba.detect_step_events(contact)
    sl = sba.step_lengths(events, foot_xy)
    # All 4 feet have stride_len == 0.4 (mostly; first stride may differ depending on stagger).
    for f in range(4):
        if len(sl[f]) > 0:
            assert all(abs(s - stride_len) < 1e-6 for s in sl[f])


def test_step_length_asymmetry_nonzero():
    # Right legs (FR=1, RR=3) walk twice as fast as left legs (FL=0, RL=2).
    T = 400
    contact = np.zeros((T, 4), dtype=bool)
    foot_xy = np.zeros((T, 4, 2))
    # Left feet: 0.3 m strides
    for f in (0, 2):
        for k in range(4):
            t_td = 20 + k * 50
            if t_td + 25 < T:
                contact[t_td:t_td + 25, f] = True
                foot_xy[t_td, f, 0] = k * 0.3
    # Right feet: 0.6 m strides (matching belt 2x faster)
    for f in (1, 3):
        for k in range(4):
            t_td = 20 + k * 50
            if t_td + 25 < T:
                contact[t_td:t_td + 25, f] = True
                foot_xy[t_td, f, 0] = k * 0.6
    events = sba.detect_step_events(contact)
    asym = sba.step_length_asymmetry(events, foot_xy)
    # Right > left → asym > 0
    assert asym > 0.2  # well outside zero
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement**

Append to `splitbelt_analysis.py`:

```python
# Foot index convention: FL=0, FR=1, RL=2, RR=3 (matches go2_constants.FEET_GEOMS order).
_LEFT_FEET = (0, 2)
_RIGHT_FEET = (1, 3)


def step_lengths(events: dict, foot_xy: np.ndarray) -> list[list[float]]:
    """Per-foot list of forward stride distances (touchdown_x[k+1] - touchdown_x[k])."""
    out: list[list[float]] = []
    for f, td_steps in enumerate(events["touchdown_steps"]):
        if len(td_steps) < 2:
            out.append([])
            continue
        xs = foot_xy[td_steps, f, 0]
        out.append([float(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)])
    return out


def step_length_asymmetry(events: dict, foot_xy: np.ndarray) -> float:
    """(SL_R - SL_L) / (SL_R + SL_L), averaged across all strides per side."""
    sl = step_lengths(events, foot_xy)
    left = [s for f in _LEFT_FEET for s in sl[f]]
    right = [s for f in _RIGHT_FEET for s in sl[f]]
    if not left or not right:
        return float("nan")
    sl_l = float(np.mean(left))
    sl_r = float(np.mean(right))
    if sl_l + sl_r == 0:
        return 0.0
    return (sl_r - sl_l) / (sl_r + sl_l)
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(splitbelt): offline step length + asymmetry"
```

---

### Task 1.10: Stage 1 sanity gate

- [ ] **Step 1: Run all hermetic splitbelt tests**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_schedules.py tests/test_splitbelt_belt_assignment.py tests/test_splitbelt_metrics.py
```

Expected: all pass.

- [ ] **Step 2: Run full default lane**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q
```

Expected: pass count ≥ baseline from Task 0.1.

---

## Stage 2: XML assets

The treadmill mesh is robot-agnostic; the scene combines it with go2.xml.

### Task 2.1: Treadmill XML asset (robot-agnostic)

**Files:**
- Create: `jax_rl/envs/locomotion/xmls/treadmill_splitbelt.xml`

Per S§5.1: two long box geoms on slide joints with velocity actuators, plus a static `fallback_floor` between/under them. Friction mu=1.0. Gap ~5 cm. Slabs ~50 m long.

- [ ] **Step 1: Sketch geometry**

Belt slab dims: `size="25 0.15 0.005"` (half-extents: 25 m × 0.15 m × 0.005 m). Centered at (0, ±0.175, 0), so y-extent is `±(0.175 + 0.15)` = `0.025..0.325`. Gap is `[-0.025, 0.025]` (5 cm).

Fallback floor: `size="50 1.0 0.001"` (50 m × 1 m × 1 mm slab) at (0, 0, -0.001) — sits just below belt level so feet that wander off into the gap or off-belt fall onto it.

- [ ] **Step 2: Write XML**

`jax_rl/envs/locomotion/xmls/treadmill_splitbelt.xml`:

```xml
<mujoco model="treadmill_splitbelt">
  <!--
    Robot-agnostic split-belt treadmill apparatus.
    Two long belt slabs on slide joints driven by velocity actuators.
    Use as <include> from a robot-specific scene XML. See spec S§5.1.
  -->

  <option timestep="0.004"/>

  <default>
    <default class="treadmill_belt">
      <geom type="box" rgba="0.3 0.3 0.35 1" friction="1.0 0.005 0.0001"/>
      <joint type="slide" axis="1 0 0" damping="0" frictionloss="0"/>
    </default>
  </default>

  <worldbody>
    <!-- Static fallback floor catches off-belt feet + falling base.
         Sits 1 mm below belt surface so belt always wins contact when foot is on a belt. -->
    <geom name="fallback_floor"
          type="plane"
          size="50 1.0 0.1"
          pos="0 0 -0.005"
          rgba="0.6 0.55 0.5 1"
          friction="1.0 0.005 0.0001"/>

    <!-- Left belt: y in [-0.325, -0.025], slides along world x. -->
    <body name="left_belt" pos="0 -0.175 0">
      <joint name="left_belt_joint" class="treadmill_belt"/>
      <geom name="left_belt_geom" class="treadmill_belt" size="25 0.15 0.005"/>
    </body>

    <!-- Right belt: y in [0.025, 0.325]. -->
    <body name="right_belt" pos="0 0.175 0">
      <joint name="right_belt_joint" class="treadmill_belt"/>
      <geom name="right_belt_geom" class="treadmill_belt" size="25 0.15 0.005"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Velocity actuators: ctrl = target velocity (m/s). High kv for tight tracking. -->
    <velocity name="left_belt_vel" joint="left_belt_joint" kv="200" ctrlrange="-3.0 3.0"/>
    <velocity name="right_belt_vel" joint="right_belt_joint" kv="200" ctrlrange="-3.0 3.0"/>
  </actuator>
</mujoco>
```

- [ ] **Step 3: Verify the XML loads in standalone MuJoCo (CPU, hermetic)**

Smoke-load check (one-liner from shell):

```bash
JAX_PLATFORMS=cpu uv run python -c "
import mujoco
from pathlib import Path
xml = Path('jax_rl/envs/locomotion/xmls/treadmill_splitbelt.xml').read_text()
m = mujoco.MjModel.from_xml_string(xml)
print('joints:', [m.jnt(i).name for i in range(m.njnt)])
print('actuators:', [m.actuator(i).name for i in range(m.nu)])
print('geoms:', [m.geom(i).name for i in range(m.ngeom)])
"
```

Expected: prints `['left_belt_joint', 'right_belt_joint']`, `['left_belt_vel', 'right_belt_vel']`, geoms list including `fallback_floor`, `left_belt_geom`, `right_belt_geom`.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/envs/locomotion/xmls/treadmill_splitbelt.xml
git commit -m "feat(splitbelt): treadmill XML asset (robot-agnostic)"
```

---

### Task 2.2: Go2 splitbelt scene XML

**Files:**
- Create: `jax_rl/envs/locomotion/xmls/go2_warp_splitbelt_scene.xml`

Includes go2 model + treadmill, defines spawn keyframe + contact filter.

Reference: existing `go2_warp_scene_flat.xml` (or similar) for the `<include>` + keyframe pattern.

- [ ] **Step 1: Read existing flat scene as template**

```bash
cat jax_rl/envs/locomotion/xmls/go2_warp_scene_flat.xml
```

Note its `<include>` of go2.xml, the keyframe block, and its contact pair declarations.

- [ ] **Step 2: Write splitbelt scene**

`jax_rl/envs/locomotion/xmls/go2_warp_splitbelt_scene.xml`:

```xml
<mujoco model="go2_warp_splitbelt_scene">
  <!--
    Go2 on a split-belt treadmill. See spec S§5.2.
    Spawn keyframe places the robot centered between belts, FL+RL on left belt, FR+RR on right.
  -->

  <include file="treadmill_splitbelt.xml"/>
  <include file="unitree_go2/go2.xml"/>

  <!-- Contact filter (S§5.2 + bongo lesson 2026-04-02 — exact contact sensors, not heuristics):
       - feet × {left_belt, right_belt, fallback_floor}
       - base/torso × {left_belt, right_belt, fallback_floor} -->
  <contact>
    <pair geom1="FL" geom2="left_belt_geom"/>
    <pair geom1="RL" geom2="left_belt_geom"/>
    <pair geom1="FR" geom2="right_belt_geom"/>
    <pair geom1="RR" geom2="right_belt_geom"/>
    <pair geom1="FL" geom2="fallback_floor"/>
    <pair geom1="FR" geom2="fallback_floor"/>
    <pair geom1="RL" geom2="fallback_floor"/>
    <pair geom1="RR" geom2="fallback_floor"/>
    <!-- base/torso × belts (catches collapse onto a belt; primary fall detector) -->
    <pair geom1="torso_box" geom2="left_belt_geom"/>
    <pair geom1="torso_box" geom2="right_belt_geom"/>
    <pair geom1="torso_box" geom2="fallback_floor"/>
    <pair geom1="torso_cyl" geom2="left_belt_geom"/>
    <pair geom1="torso_cyl" geom2="right_belt_geom"/>
    <pair geom1="torso_cyl" geom2="fallback_floor"/>
    <pair geom1="torso_nose" geom2="left_belt_geom"/>
    <pair geom1="torso_nose" geom2="right_belt_geom"/>
    <pair geom1="torso_nose" geom2="fallback_floor"/>
  </contact>

  <sensor>
    <!-- Boolean contact-pair sensors per bongo lesson 2026-04-02: exact, threshold-free.
         Drives both `foot_belt_id` (S§9.1) and termination cause (S§7.3). -->
    <contact name="FL_left_belt"  geom1="FL" geom2="left_belt_geom"/>
    <contact name="FL_right_belt" geom1="FL" geom2="right_belt_geom"/>
    <contact name="FR_left_belt"  geom1="FR" geom2="left_belt_geom"/>
    <contact name="FR_right_belt" geom1="FR" geom2="right_belt_geom"/>
    <contact name="RL_left_belt"  geom1="RL" geom2="left_belt_geom"/>
    <contact name="RL_right_belt" geom1="RL" geom2="right_belt_geom"/>
    <contact name="RR_left_belt"  geom1="RR" geom2="left_belt_geom"/>
    <contact name="RR_right_belt" geom1="RR" geom2="right_belt_geom"/>
    <contact name="FL_floor" geom1="FL" geom2="fallback_floor"/>
    <contact name="FR_floor" geom1="FR" geom2="fallback_floor"/>
    <contact name="RL_floor" geom1="RL" geom2="fallback_floor"/>
    <contact name="RR_floor" geom1="RR" geom2="fallback_floor"/>
    <!-- Fall-cause sensors: torso geoms × any belt or floor. -->
    <contact name="torso_left_belt"  geom1="torso_box" geom2="left_belt_geom"/>
    <contact name="torso_right_belt" geom1="torso_box" geom2="right_belt_geom"/>
    <contact name="torso_floor"      geom1="torso_box" geom2="fallback_floor"/>
  </sensor>

  <keyframe>
    <!-- Spawn pose: standard Go2 stance, base centered between belts.
         Copy qpos from existing go2 home keyframe; adjust base z so feet sit on belt surface. -->
    <key name="splitbelt_spawn"
         qpos="0 0 0.275  1 0 0 0
               0.0 0.9 -1.8   0.0 0.9 -1.8   0.0 0.9 -1.8   0.0 0.9 -1.8
               0 0"
         ctrl="0 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"/>
    <!-- last 2 qpos slots are belt slide joints; ctrl includes belt actuators tail. -->
  </keyframe>
</mujoco>
```

> **Note for implementer:** the `qpos` ordering in the keyframe depends on the order MuJoCo assigns joints across the included models. Run `mujoco.MjModel.from_xml_path(...)` in a one-liner to verify; if the belt joints come BEFORE go2 joints, swap the qpos/ctrl ordering accordingly. The existing flat scene's keyframe is the source of truth for go2 joint values; copy those and append/prepend belt slots as needed.

- [ ] **Step 3: Smoke-load + assert geometry**

```bash
JAX_PLATFORMS=cpu uv run python -c "
import mujoco
from pathlib import Path
import os
os.chdir('jax_rl/envs/locomotion/xmls')
m = mujoco.MjModel.from_xml_path('go2_warp_splitbelt_scene.xml')
print('njnt:', m.njnt, 'nu:', m.nu, 'nkey:', m.nkey)
print('contact pairs:', m.npair)
print('sensors:', m.nsensor)
"
```

Expected: njnt > 19 (12 go2 joints + 1 free + 2 belt slides + ...); nu > 12 (12 go2 motors + 2 belt vel actuators); nkey ≥ 1; sensors > 12.

If keyframe qpos/ctrl ordering is wrong, MuJoCo will print a warning at load. Fix by re-ordering per the implementer note above.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/envs/locomotion/xmls/go2_warp_splitbelt_scene.xml
git commit -m "feat(splitbelt): Go2 splitbelt scene XML (treadmill + go2 + contact filter)"
```

---

## Stage 3: Env class — TDD-driven

### Task 3.1: Obs schema test (hermetic — name-layout only)

**Files:**
- Create: `tests/test_splitbelt_obs_schema.py`

Per S§10.1 and lesson §7.1. We need a hermetic test that checks which obs term *names* end up in `state` vs `privileged_state` per `obs_mode`, without building the env (no fn-bound lambdas, no real Warp graph).

Since `ObsTerm` (per [`jax_rl/envs/obs_spec.py:21-26`](../../jax_rl/envs/obs_spec.py#L21-L26)) requires `(name: str, fn: Callable, noise_scale: float)` — we cannot construct real ObsTerms hermetically without binding lambdas to env methods. So we split obs-group construction into two halves:

- `obs_term_names(obs_mode: str) -> dict[str, list[str]]` — pure, name-only, no env. Hermetic-test target.
- `build_obs_groups(env) -> dict[str, list[ObsTerm | IncludeGroup]]` — real, env-method-bound. Used in `_post_init`.

The contract: `[t.name for t in build_obs_groups(env)[group]] == obs_term_names(env._config.obs_mode)[group]`.

- [ ] **Step 1: Write hermetic test (RED until module exists)**

```python
"""Hermetic obs name-layout tests for SplitbeltTreadmill env (S§10.1, lesson §7.1)."""

from __future__ import annotations

import pytest

from jax_rl.envs.locomotion.go2_warp_splitbelt import obs_term_names


_VALID_MODES = ("blind", "informed", "error", "history")


@pytest.mark.parametrize("obs_mode", _VALID_MODES)
def test_obs_term_names_has_state_and_privileged(obs_mode):
    layout = obs_term_names(obs_mode)
    assert "state" in layout
    assert "privileged_state" in layout
    assert isinstance(layout["state"], list)
    assert isinstance(layout["privileged_state"], list)


def test_blind_state_excludes_belt_speeds():
    layout = obs_term_names("blind")
    assert "belt_vel" not in layout["state"]


def test_informed_state_includes_belt_speeds():
    layout = obs_term_names("informed")
    assert "belt_vel" in layout["state"]


def test_error_state_includes_tracking_error_not_belt():
    layout = obs_term_names("error")
    assert "cmd_track_error" in layout["state"]
    assert "drift_xy" in layout["state"]
    assert "belt_vel" not in layout["state"]


def test_privileged_always_full_info():
    for mode in _VALID_MODES:
        priv = obs_term_names(mode)["privileged_state"]
        for required in ("belt_vel", "cmd_track_error", "drift_xy"):
            assert required in priv, f"mode={mode}: privileged missing {required}"


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="obs_mode"):
        obs_term_names("not_a_real_mode")
```

- [ ] **Step 2: Run, expect fail**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_obs_schema.py
```

Expected: FAIL `ModuleNotFoundError: jax_rl.envs.locomotion.go2_warp_splitbelt`. (Env class doesn't exist yet — that's the next task.)

- [ ] **Step 3: Commit (test first, RED before GREEN)**

```bash
git add tests/test_splitbelt_obs_schema.py
git commit -m "test(splitbelt): obs schema tests (RED, env not yet created)"
```

---

### Task 3.2: Env class skeleton + obs_groups (GREEN the schema test)

**Files:**
- Create: `jax_rl/envs/locomotion/go2_warp_splitbelt.py`

Subclass `Go2WarpEnv` (per `go2_warp_base.py`). Stub `_step`/`_reset` with `NotImplementedError`-flavored placeholders; only `build_obs_groups` is real for this task. Reference: `go2_bongo_handstand.py` for class layout, `go2_warp_joystick.py` for obs term examples.

- [ ] **Step 1: Read references**

```bash
cat jax_rl/envs/locomotion/go2_bongo_handstand.py | head -120
cat jax_rl/envs/locomotion/go2_warp_joystick.py | head -120
```

Note `_post_init`, `_obs_groups` definition pattern, ObsTerm + IncludeGroup usage.

- [ ] **Step 2: Implement skeleton**

`jax_rl/envs/locomotion/go2_warp_splitbelt.py`:

```python
"""Go2 SplitbeltTreadmill env (Warp backend). See spec at
.superpowers/specs/2026-05-02-splitbelt-treadmill-env-design.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import numpy as np

from jax_rl.envs.locomotion import go2_warp_base
from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.locomotion import splitbelt_geom as geom
from jax_rl.envs.locomotion import splitbelt_schedules as sched
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup


_VALID_OBS_MODES = ("blind", "informed", "error", "history")

# Shared proprio name list — kept in sync with build_obs_groups bindings below.
_PROPRIO_NAMES = ("joint_pos", "joint_vel", "last_action", "gravity", "gyro", "cmd")


def obs_term_names(obs_mode: str) -> Dict[str, list[str]]:
    """Pure name-layout (S§5 obs modes). Hermetic — no env, no fn binding.

    Returns dict of {"state": [name, ...], "privileged_state": [name, ...]}.
    Mirrors the ObsTerm structure that `build_obs_groups` produces; used by tests
    and as the source-of-truth name list.
    """
    if obs_mode not in _VALID_OBS_MODES:
        raise ValueError(
            f"obs_mode must be one of {_VALID_OBS_MODES}, got {obs_mode!r}"
        )
    proprio = list(_PROPRIO_NAMES)
    if obs_mode == "blind" or obs_mode == "history":
        # history applies frame-stacking via wrapper; same name layout as blind.
        state_names = list(proprio)
    elif obs_mode == "informed":
        state_names = list(proprio) + ["belt_vel"]
    elif obs_mode == "error":
        state_names = list(proprio) + ["cmd_track_error", "drift_xy"]
    else:
        raise AssertionError("unreachable")
    privileged_names = list(proprio) + [
        "belt_vel", "cmd_track_error", "drift_xy",
        "base_lin_vel", "base_ang_vel",
    ]
    return {"state": state_names, "privileged_state": privileged_names}


def default_config() -> config_dict.ConfigDict:
    """Default config: tied belts at 0.5 m/s, blind obs, cmd=0 (smoke baseline)."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1250,  # 25 s
        Kp=20.0,
        Kd=0.5,
        torque_speed_model=False,
        action_repeat=1,
        action_scale=1.0,
        soft_joint_pos_limit_factor=0.95,
        impl="warp",
        contact_mode="training",
        # Splitbelt-specific
        obs_mode="blind",
        history_len=4,
        belt_layout=config_dict.create(
            left_y_min=-0.325, left_y_max=-0.025,
            right_y_min=0.025, right_y_max=0.325,
        ),
        schedule_kind="tied",
        schedule_params=config_dict.create(v=0.5),
        cmd_zero=True,
        treadmill_drift_lateral_weight=2.0,
        treadmill_drift_forward_weight=0.5,
        # Reuse joystick reward scales — will be filled in Task 3.5.
        reward_config=config_dict.create(scales=config_dict.create()),
        # MJX/Warp tuning
        naconmax=4 * 8192,
        naccdmax=4000,
        njmax=100,
    )


def build_obs_groups(env: Any) -> Dict[str, list]:
    """Build real ObsTerm dispatch with env-method-bound lambdas (used in _post_init).

    Names must match `obs_term_names(env._config.obs_mode)` exactly — that is the
    contract validated in tests. Pattern follows `go2_warp_joystick.py:131`.
    """
    cfg = env._config
    layout = obs_term_names(cfg.obs_mode)

    # Map term name → (fn, noise_scale). Fns close over `env`; per joystick precedent,
    # they accept **kw to absorb compute_obs's keyword args.
    term_factory = {
        "joint_pos": (lambda data, **kw: env.get_joint_pos(data), 0.03),
        "joint_vel": (lambda data, **kw: env.get_joint_vel(data), 1.5),
        "last_action": (lambda info, **kw: info["last_action"], 0.0),
        "gravity": (lambda data, **kw: env.get_gravity(data), 0.05),
        "gyro": (lambda data, **kw: env.get_gyro(data), 0.2),
        "cmd": (lambda info, **kw: info["cmd"], 0.0),
        "belt_vel": (lambda info, **kw: info["splitbelt"]["belt_vel"], 0.0),
        "cmd_track_error": (lambda info, **kw: info["splitbelt"]["cmd_track_error"], 0.0),
        "drift_xy": (lambda info, **kw: info["splitbelt"]["drift_xy"], 0.0),
        "base_lin_vel": (lambda data, **kw: data.qvel[:3], 0.0),
        "base_ang_vel": (lambda data, **kw: data.qvel[3:6], 0.0),
    }

    def _build(names: list[str]) -> list[ObsTerm]:
        return [ObsTerm(name=n, fn=term_factory[n][0], noise_scale=term_factory[n][1])
                for n in names]

    return {
        "state": _build(layout["state"]),
        "privileged_state": _build(layout["privileged_state"]),
    }


class Go2WarpSplitbeltEnv(go2_warp_base.Go2WarpEnv):
    """Go2 on a split-belt treadmill (S§5.3)."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ) -> None:
        cfg = config if config is not None else default_config()
        xml_path = (
            Path(__file__).parent / "xmls" / "go2_warp_splitbelt_scene.xml"
        ).as_posix()
        super().__init__(xml_path=xml_path, config=cfg, config_overrides=config_overrides)
        # _post_init is invoked from base; the heavy lifting lives there.

    def _post_init(self) -> None:
        # TODO(Task 3.3): schedule_table buffer, contact sensor IDs, belt geometry, default_pose.
        super()._post_init()
        self._obs_groups = build_obs_groups(self)
        self._action_dim = 12

    def _reset(self, rng: jax.Array):
        raise NotImplementedError("Implemented in Task 3.4")

    def _step(self, state, action: jax.Array):
        raise NotImplementedError("Implemented in Task 3.5")
```

- [ ] **Step 3: Run obs-schema test, expect pass**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_obs_schema.py
```

Expected: all parametrized + privileged tests pass.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/envs/locomotion/go2_warp_splitbelt.py
git commit -m "feat(splitbelt): env skeleton + obs_groups dispatch (GREEN obs schema)"
```

---

### Task 3.3: `_post_init` — schedule_table, contact IDs, belt geometry

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_warp_splitbelt.py`

This populates env attributes used by `_step`/`_reset`. No new tests at this stage — `_post_init` is exercised by Task 3.6 smoke test [gpu, warp, go2].

- [ ] **Step 1: Implement**

Replace the stub `_post_init` with:

```python
def _post_init(self) -> None:
    super()._post_init()
    self._action_dim = 12
    self._obs_groups = build_obs_groups(self)

    cfg = self._config
    self._belt_layout = geom.BeltLayout(
        left_y_min=cfg.belt_layout.left_y_min,
        left_y_max=cfg.belt_layout.left_y_max,
        right_y_min=cfg.belt_layout.right_y_min,
        right_y_max=cfg.belt_layout.right_y_max,
    )

    # Belt actuator IDs (for writing belt-vel commands per step).
    self._left_belt_act_id = self._mj_model.actuator("left_belt_vel").id
    self._right_belt_act_id = self._mj_model.actuator("right_belt_vel").id
    self._left_belt_jnt_id = self._mj_model.joint("left_belt_joint").id
    self._right_belt_jnt_id = self._mj_model.joint("right_belt_joint").id

    # Foot contact-pair sensor IDs for foot_belt_id (S§9.1).
    feet_order = ("FL", "FR", "RL", "RR")
    self._foot_left_belt_sensors = jp.array(
        [self._mj_model.sensor(f"{f}_left_belt").adr[0] for f in feet_order]
    )
    self._foot_right_belt_sensors = jp.array(
        [self._mj_model.sensor(f"{f}_right_belt").adr[0] for f in feet_order]
    )
    self._foot_floor_sensors = jp.array(
        [self._mj_model.sensor(f"{f}_floor").adr[0] for f in feet_order]
    )

    # Torso fall sensors.
    self._torso_left_belt_sensor = self._mj_model.sensor("torso_left_belt").adr[0]
    self._torso_right_belt_sensor = self._mj_model.sensor("torso_right_belt").adr[0]
    self._torso_floor_sensor = self._mj_model.sensor("torso_floor").adr[0]

    # Default pose from spawn keyframe.
    spawn_id = self._mj_model.keyframe("splitbelt_spawn").id
    self._default_pose = jp.array(self._mj_model.key_qpos[spawn_id])

    # Episode-length cap for schedule_table allocation (T = episode_length).
    self._schedule_T = int(cfg.episode_length)
```

- [ ] **Step 2: Smoke-import (the env still raises NotImplementedError on _step but should at least construct)**

```bash
JAX_PLATFORMS=cpu uv run python -c "
from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
env = Go2WarpSplitbeltEnv()
print('action_dim', env._action_dim)
print('belt_layout', env._belt_layout)
print('obs_groups keys', list(env._obs_groups.keys()))
"
```

Expected: prints without crashing. (CPU MuJoCo path; Warp not invoked because `_reset`/`_step` are not called.) If this fails on Warp-only setup, this manual check is `[gpu, warp]`-only — skip and continue; the smoke test in Task 3.6 covers it.

- [ ] **Step 3: Commit**

```bash
git commit -am "feat(splitbelt): _post_init wires sensors + belt geometry + default pose"
```

---

### Task 3.4: `_reset` — schedule sample, state init, term_cause zeroing

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_warp_splitbelt.py`

Per S§6.2 and S§9.1: sample schedule table, set belt qvel, sample cmd, zero `term_cause`.

- [ ] **Step 1: Implement `_reset`**

Reference `go2_bongo_handstand.py::_reset` for the qpos/qvel sampling pattern + DR + obs computation. Adapt:

```python
def _reset(self, rng: jax.Array):
    rng, k_qpos, k_qvel, k_sched, k_cmd = jax.random.split(rng, 5)

    # qpos/qvel: spawn keyframe + small noise (existing pattern).
    qpos = self._default_pose + jax.random.uniform(
        k_qpos, self._default_pose.shape, minval=-0.01, maxval=0.01
    )
    qvel = jp.zeros(self._mj_model.nv)

    # Belt slabs back to origin; initial belt qvel = first row of schedule_table.
    schedule_table = sched.sample_schedule(
        k_sched,
        T=self._schedule_T,
        kind=self._config.schedule_kind,
        params=dict(self._config.schedule_params),
    )
    # Index belt slide-joint qvel by joint id; the qvel layout matches qpos for slides.
    qvel = qvel.at[self._left_belt_jnt_id].set(schedule_table[0, 0])
    qvel = qvel.at[self._right_belt_jnt_id].set(schedule_table[0, 1])
    # Belt qpos at zero (slabs back to origin).
    qpos = qpos.at[self._left_belt_jnt_id].set(0.0)
    qpos = qpos.at[self._right_belt_jnt_id].set(0.0)

    # cmd: always-zero by config default (S§7.4).
    cmd = jp.zeros(3) if self._config.cmd_zero else jax.random.uniform(
        k_cmd, (3,), minval=-1.0, maxval=1.0
    )

    # Build initial pipeline state via base helper.
    data = self._init_pipeline(qpos, qvel)

    # info dict carries schedule + step_idx + splitbelt primitives (S§9.1).
    info = {
        "rng": rng,
        "step_idx": jp.int32(0),
        "belt_schedule": schedule_table,
        "cmd": cmd,
        "last_action": jp.zeros(self._action_dim),
        "splitbelt": {
            "foot_in_contact": jp.zeros(4, dtype=jp.bool_),
            "foot_pos_world": jp.zeros((4, 3)),
            "foot_belt_id": jp.full((4,), -1, dtype=jp.int32),
            "base_pos_world": jp.zeros(3),
            "base_vel_world": jp.zeros(3),
            "base_yaw": jp.float32(0.0),
            "belt_vel": schedule_table[0],
            "cmd_track_error": jp.zeros(3),
            "drift_xy": jp.zeros(2),
            "term_cause": jp.int32(0),  # S§9.1: 0 = none/truncated, 1=fall, 2=off-belt, 3=tilt
            "step_idx": jp.int32(0),
        },
    }

    obs = self._compute_obs(data, info)
    reward, done = jp.zeros(()), jp.zeros((), dtype=jp.bool_)
    metrics = {}
    return self._make_state(data, obs, reward, done, info, metrics)
```

> **Implementer note:** `_init_pipeline` and `_make_state` are conventions from `go2_warp_base` / `mjx_env.MjxEnv`. If the actual API differs (e.g., `mjx_env.init` + `State`), adapt — the precedent is `go2_bongo_handstand.py::_reset`. Don't invent new helpers; reuse base ones.

- [ ] **Step 2: Stub `_compute_obs` so reset doesn't crash**

Add a temporary stub that returns the right structure but with zeros — the real obs comp is part of step:

```python
def _compute_obs(self, data, info) -> Dict[str, jp.ndarray]:
    """TODO(Task 3.5): real obs from data + info. Stub returns zero-shaped placeholder."""
    # Compute obs term widths from groups.
    state_dim = sum(t.dim for t in self._obs_groups["state"])
    priv_dim = sum(t.dim for t in self._obs_groups["privileged_state"])
    return {
        "state": jp.zeros(state_dim),
        "privileged_state": jp.zeros(priv_dim),
    }
```

- [ ] **Step 3: Commit**

```bash
git commit -am "feat(splitbelt): _reset samples schedule + zeroes term_cause + cmd"
```

---

### Task 3.5: `_step` — belt actuation, gait primitives, obs, reward, termination

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_warp_splitbelt.py`

This is the largest sub-task; break into sub-steps.

#### Step A: belt-vel write + physics step

- [ ] **Implement `_step` skeleton with belt actuation only**

```python
def _step(self, state, action: jax.Array):
    info = state.info
    step_idx = info["step_idx"]
    belt_vel_target = info["belt_schedule"][step_idx]  # (2,)

    # Joint-space PD: target = action_scale * action + default_pose[joint_slice]
    # (Pattern: see go2_warp_base / go2_warp_joystick.)
    ctrl = self._compute_ctrl(action)  # 12-dim joint torques/positions per base impl

    # Append belt-vel commands. The full ctrl vector is (12 motor + 2 belt vel).
    full_ctrl = jp.concatenate([ctrl, belt_vel_target])

    data = self._physics_step(state.pipeline_state, full_ctrl)
    # ... (rest below)
```

> **Implementer note:** `_compute_ctrl` and `_physics_step` are again base-class conventions. Look at `go2_warp_base` for the exact helper names; they may be `_pre_step` + `_post_step` or `_apply_action` + `mjx.step` directly. Match the precedent.

#### Step B: gait primitive computation

```python
    # --- Gait primitives (S§9.1) ---
    sensordata = data.sensordata
    foot_in_left = sensordata[self._foot_left_belt_sensors] > 0.0
    foot_in_right = sensordata[self._foot_right_belt_sensors] > 0.0
    foot_in_floor = sensordata[self._foot_floor_sensors] > 0.0

    foot_in_contact = foot_in_left | foot_in_right | foot_in_floor
    foot_belt_id = jp.where(
        foot_in_left, jp.int32(0),
        jp.where(foot_in_right, jp.int32(1), jp.int32(-1)),
    )

    feet_geom_ids = jp.array([self._mj_model.geom(g).id for g in consts.FEET_GEOMS])
    foot_pos_world = data.geom_xpos[feet_geom_ids]
    base_pos_world = data.qpos[:3]
    base_vel_world = data.qvel[:3]
    base_yaw = self._extract_yaw(data)

    # Tracking error in body frame (cmd[:2] − body-frame v_xy, cmd[2] − body yaw rate).
    cmd = info["cmd"]
    body_vel = self._world_to_body(base_vel_world, base_yaw)
    cmd_track_error = jp.concatenate([
        cmd[:2] - body_vel[:2],
        cmd[2:3] - data.qvel[5:6],  # yaw rate from base ang vel
    ])
    drift_xy = base_pos_world[:2]  # treadmill_center = origin
```

#### Step C: termination with term_cause

```python
    # --- Termination (S§7.3) ---
    # Contact-based primary detectors:
    fall_torso_left = sensordata[self._torso_left_belt_sensor] > 0.0
    fall_torso_right = sensordata[self._torso_right_belt_sensor] > 0.0
    fall_torso_floor = sensordata[self._torso_floor_sensor] > 0.0
    is_fall = fall_torso_left | fall_torso_right | fall_torso_floor

    is_off_belt = jp.any(foot_in_floor)

    # Threshold backstop. Use the existing helper from go2_warp_base
    # (`go2_warp_base.py:189`): self.get_gravity(data) returns gravity in body frame.
    # The body-z component gives "uprightness": 1.0 = perfectly upright, 0 = sideways.
    gravity_body = self.get_gravity(data)
    is_tilt = (gravity_body[2] < 0.5) | (base_pos_world[2] < 0.18)

    done = is_fall | is_off_belt | is_tilt

    # term_cause priority: fall > off_belt > tilt.
    term_cause = jp.where(
        is_fall, jp.int32(1),
        jp.where(is_off_belt, jp.int32(2),
                 jp.where(is_tilt, jp.int32(3), jp.int32(0))),
    )
```

#### Step D: reward (S§7.1)

```python
    # --- Reward (S§7.1) ---
    scales = self._config.reward_config.scales
    # Tracking
    r_track_lin = jp.exp(-jp.sum(jp.square(cmd[:2] - body_vel[:2])) / 0.25)
    r_track_ang = jp.exp(-jp.square(cmd[2] - data.qvel[5]) / 0.25)
    # Stay-on-treadmill
    r_drift = -(
        self._config.treadmill_drift_lateral_weight * jp.square(drift_xy[1])
        + self._config.treadmill_drift_forward_weight * jp.square(drift_xy[0])
    )
    # Survival + termination
    r_survival = 1.0
    r_term = jp.where(done, -1.0, 0.0)
    # Smoothness — port a subset; see go2_warp_joystick.
    r_action_rate = -jp.sum(jp.square(action - info["last_action"]))

    reward = (
        scales.get("tracking_lin_vel_xy", 1.0) * r_track_lin
        + scales.get("tracking_ang_vel_z", 0.5) * r_track_ang
        + r_drift
        + scales.get("survival", 1.0) * r_survival
        + scales.get("termination", 1.0) * r_term
        + scales.get("action_rate", 0.01) * r_action_rate
    )
```

#### Step E: info update + obs + return

```python
    new_splitbelt = {
        "foot_in_contact": foot_in_contact,
        "foot_pos_world": foot_pos_world,
        "foot_belt_id": foot_belt_id,
        "base_pos_world": base_pos_world,
        "base_vel_world": base_vel_world,
        "base_yaw": base_yaw,
        "belt_vel": belt_vel_target,
        "cmd_track_error": cmd_track_error,
        "drift_xy": drift_xy,
        "term_cause": term_cause,  # nonzero only when done; else 0
        "step_idx": step_idx,
    }

    new_info = dict(info)
    new_info["splitbelt"] = new_splitbelt
    new_info["step_idx"] = step_idx + 1
    new_info["last_action"] = action
    # Truncation flag (lesson env_backends §2): timeout is truncation, not termination.
    new_info["truncation"] = jp.where(
        new_info["step_idx"] >= self._schedule_T, jp.int32(1), jp.int32(0)
    )

    obs = self._compute_obs(data, new_info)
    metrics = {
        "track_lin": r_track_lin,
        "track_ang": r_track_ang,
        "drift": r_drift,
    }
    return self._make_state(data, obs, reward, done, new_info, metrics)
```

#### Step F: real `_compute_obs`

Replace the stub:

```python
def _compute_obs(self, data, info) -> Dict[str, jp.ndarray]:
    sb = info["splitbelt"]
    cmd = info["cmd"]

    # Build proprio dict; ObsTerm names must match build_obs_groups.
    proprio = {
        "joint_pos": data.qpos[7:7 + 12],  # 12 actuated joints (skip 7 base + slides at end)
        "joint_vel": data.qvel[6:6 + 12],
        "last_action": info["last_action"],
        "gravity": self._gravity_body(data),
        "gyro": data.qvel[3:6],
        "cmd": cmd,
    }
    extras_state = {}
    extras_priv = {
        "belt_vel": sb["belt_vel"],
        "cmd_track_error": sb["cmd_track_error"],
        "drift_xy": sb["drift_xy"],
        "base_lin_vel": sb["base_vel_world"],
        "base_ang_vel": data.qvel[3:6],
    }

    mode = self._config.obs_mode
    if mode == "informed":
        extras_state["belt_vel"] = sb["belt_vel"]
    elif mode == "error":
        extras_state["cmd_track_error"] = sb["cmd_track_error"]
        extras_state["drift_xy"] = sb["drift_xy"]
    # blind / history: no extras for state group.

    state_terms = {**proprio, **extras_state}
    priv_terms = {**proprio, **extras_priv}

    return {
        "state": jp.concatenate([state_terms[t.name] for t in self._obs_groups["state"]]),
        "privileged_state": jp.concatenate(
            [priv_terms[t.name] for t in self._obs_groups["privileged_state"]]
        ),
    }
```

> **Implementer note:** the helpers `_extract_yaw`, `_world_to_body`, `_gravity_body`, `_compute_ctrl`, `_physics_step`, `_init_pipeline`, `_make_state` are go2_warp_base conventions — names approximate. Match what the base class actually exposes; if a name doesn't exist, look at `go2_bongo_handstand.py::_step` for the equivalent and reuse exactly that helper. Don't add new helpers to the base class.

- [ ] **Step 1: Wire all sub-steps into `_step` and verify the file imports + the env constructs cleanly** (no Warp invocation needed yet)

```bash
JAX_PLATFORMS=cpu uv run python -c "
from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv, default_config
cfg = default_config()
env = Go2WarpSplitbeltEnv(cfg)
print('OK env constructs; obs_dim placeholder', env._obs_groups)
"
```

Expected: prints. If `mujoco_playground` requires Warp to import, this CPU smoke might error — switch to checking via the GPU smoke test in Task 3.6.

- [ ] **Step 2: Commit**

```bash
git commit -am "feat(splitbelt): _step belt actuation + gait primitives + reward + termination"
```

---

### Task 3.6: GPU smoke test [gpu, warp, go2]

**Files:**
- Create: `tests/test_splitbelt_env_smoke.py`

Per S§10.2 and lesson §7.3. Real env, marked `[gpu, warp, go2]`, fixtures only.

- [ ] **Step 1: Write test**

```python
"""GPU/Warp smoke tests for SplitbeltTreadmill env (S§10.2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]


@pytest.fixture
def env():
    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    return Go2WarpSplitbeltEnv()


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


def test_reset_returns_finite_state(env, rng):
    state = env.reset(rng)
    assert jnp.all(jnp.isfinite(state.obs["state"]))
    assert jnp.all(jnp.isfinite(state.obs["privileged_state"]))
    # term_cause starts zeroed (S§6.2 step 7).
    assert int(state.info["splitbelt"]["term_cause"]) == 0


def test_single_step_no_nan(env, rng):
    state = env.reset(rng)
    action = jnp.zeros((env._action_dim,))
    state2 = env.step(state, action)
    assert jnp.all(jnp.isfinite(state2.obs["state"]))
    assert jnp.all(jnp.isfinite(state2.reward))


def test_belt_qvel_matches_schedule(env, rng):
    """Schedule plumbing: belt joint qvel after one step matches schedule_table[0]."""
    state = env.reset(rng)
    action = jnp.zeros((env._action_dim,))
    state2 = env.step(state, action)
    # The belt is velocity-actuated, so within ctrl_dt the joint qvel should be near schedule.
    schedule_step0 = state.info["belt_schedule"][0]
    actual_left = state2.pipeline_state.qvel[env._left_belt_jnt_id]
    actual_right = state2.pipeline_state.qvel[env._right_belt_jnt_id]
    # 10% tolerance for one-step transient (kv=200 should achieve this).
    assert jnp.abs(actual_left - schedule_step0[0]) < 0.1
    assert jnp.abs(actual_right - schedule_step0[1]) < 0.1


def test_off_belt_termination(env, rng):
    """Force a foot off-belt by shifting robot in y; assert term_cause = 2."""
    state = env.reset(rng)
    # Cheat: replace base y in qpos to put robot fully off the right side of belts.
    qpos = state.pipeline_state.qpos.at[1].set(2.0)  # y = 2 m, well outside belts
    new_pipeline = state.pipeline_state.replace(qpos=qpos)
    state = state.replace(pipeline_state=new_pipeline)
    action = jnp.zeros((env._action_dim,))
    s2 = env.step(state, action)
    # On the next step, foot×fallback_floor should fire.
    assert bool(s2.done)
    assert int(s2.info["splitbelt"]["term_cause"]) == 2  # off-belt
```

- [ ] **Step 2: Run + confirm gpu marker selects the file**

```bash
uv run python -m pytest --collect-only -q -m gpu | grep splitbelt_env_smoke
```

Expected: file lists in output (else marker typo — fix before running).

- [ ] **Step 3: Run on a GPU box**

```bash
uv run python -m pytest -q tests/test_splitbelt_env_smoke.py
```

Expected: all four tests pass. If `test_belt_qvel_matches_schedule` fails with a transient larger than 0.1, **either** raise tolerance to 0.3 (one-step transient is acceptable per S§6.2 implementer note about belt-qvel settling) **or** raise actuator `kv` from 200 to 400 in `treadmill_splitbelt.xml` and re-run. Don't bury the issue — document which mitigation in the commit.

- [ ] **Step 4: Default-lane sanity**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q
```

Expected: pass count unchanged (smoke file is gpu-marked, excluded from default lane).

- [ ] **Step 5: Commit**

```bash
git add tests/test_splitbelt_env_smoke.py
git commit -m "test(splitbelt): GPU/Warp env smoke (reset, step, belt qvel, off-belt term)"
```

---

## Stage 4: Backend wiring

### Task 4.1: Register env in mjx_backend

**Files:**
- Modify: `jax_rl/training/env_backends/mjx_backend.py`

Per S§5.5: register `Go2WarpSplitbelt` with default_config factory. Pattern: existing Go2 env registrations.

- [ ] **Step 1: Read existing registration pattern**

```bash
grep -n "pg_locomotion.register_environment\|Go2WarpJoystickFlat" jax_rl/training/env_backends/mjx_backend.py | head -20
```

Confirms the API is `pg_locomotion.register_environment(name, env_class=..., cfg_class=...)` (or similar — read the existing call args). Each registration is guarded with `if "<Name>" not in pg_locomotion._envs:`.

- [ ] **Step 2: Add registration**

Match the existing block exactly. Approximately (verify against the read above):

```python
# In mjx_backend.py, near the other Go2 registrations:
from jax_rl.envs.locomotion.go2_warp_splitbelt import (
    Go2WarpSplitbeltEnv,
    default_config as splitbelt_default_config,
)

if "Go2WarpSplitbelt" not in pg_locomotion._envs:
    pg_locomotion.register_environment(
        "Go2WarpSplitbelt",
        Go2WarpSplitbeltEnv,
        splitbelt_default_config,
    )
```

> **Implementer note:** Mirror the joystick registration's exact arg shape (positional vs kwarg) — do NOT invent new arg names.

- [ ] **Step 3: Smoke import**

```bash
JAX_PLATFORMS=cpu uv run python -c "
from jax_rl.training.env_backends import mjx_backend, detect_backend
print(detect_backend('Go2WarpSplitbelt'))
"
```

Expected: prints `'mjx'` (or whatever the backend literal is).

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(splitbelt): register Go2WarpSplitbelt in mjx_backend"
```

---

### Task 4.2: Bundle test [gpu]

**Files:**
- Create: `tests/test_splitbelt_bundle.py`

- [ ] **Step 1: Write test**

```python
"""Bundle test for SplitbeltTreadmill env (lesson §7.5)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu]


@pytest.fixture
def cfg():
    from jax_rl.training.train_config import TrainConfig  # actual class — adjust path if different
    return TrainConfig(env_name="Go2WarpSplitbelt", num_envs=4, total_timesteps=100)


def test_make_env_bundle_returns_populated(cfg):
    from jax_rl.training import make_env_bundle
    bundle = make_env_bundle(cfg, seed=0)
    assert bundle.backend_kind == "mjx"
    assert bundle.has_privileged is True
    assert bundle.dict_obs is True
    assert bundle.action_dim == 12
    assert bundle.obs_dim > 0
    assert bundle.critic_obs_dim is not None
```

- [ ] **Step 2: Run on GPU box**

```bash
uv run python -m pytest -q tests/test_splitbelt_bundle.py
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_splitbelt_bundle.py
git commit -m "test(splitbelt): bundle test [gpu]"
```

---

### Task 4.3: Control metadata test [gpu, warp, go2, deploy]

**Files:**
- Create: `tests/test_splitbelt_control_metadata.py`

Per lesson §7.2. Verify `env.get_control_metadata()` matches `deploy/go2_constants.py`.

- [ ] **Step 1: Write test**

```python
"""Control metadata test for SplitbeltTreadmill env (lesson §7.2)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2, pytest.mark.deploy]


@pytest.fixture
def env():
    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    return Go2WarpSplitbeltEnv()


def test_control_metadata_matches_deploy_constants(env):
    from deploy import go2_constants as deploy_consts
    md = env.get_control_metadata()
    assert md["Kp"] == deploy_consts.KP
    assert md["Kd"] == deploy_consts.KD
    assert md["action_scale"] == deploy_consts.ACTION_SCALE
    assert md["joint_order"] == deploy_consts.JOINT_ORDER
```

> **Implementer note:** the exact field names in `get_control_metadata()` and `go2_constants.py` may differ; mirror what `tests/test_deploy_e2e.py::test_env_metadata_matches_deploy_constants` asserts.

- [ ] **Step 2: Run on GPU box**

```bash
uv run python -m pytest -q tests/test_splitbelt_control_metadata.py
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_splitbelt_control_metadata.py
git commit -m "test(splitbelt): control metadata matches deploy constants"
```

---

## Stage 5: Validation + docs

### Task 5.1: Calibration smoke train (manual, post-implementation)

This is not a pytest test — it's a sanity training run that proves the env behaves like its joystick analog when belts are tied.

**Files:** none (logs only)

- [ ] **Step 1: Run tied-belt smoke**

```bash
uv run python scripts/train_fast_sac.py \
    --env Go2WarpSplitbelt \
    --num-envs 1024 \
    --total-timesteps 1000000 \
    --reset-mode per_step \
    > /tmp/splitbelt_tied_smoke.log 2>&1 &
```

Note: 1M-step run. Expect ~30-60 min on a 16 GB GPU.

- [ ] **Step 2: Periodic check**

```bash
grep "EVAL" /tmp/splitbelt_tied_smoke.log | tail -10
```

Expected end-of-run eval: same order of magnitude as `Go2WarpJoystickFlat` at the same belt speed (eval ~150-250 at 1M steps; full convergence ~280-290 at 5M).

If wildly off (e.g., < 50 or NaN-ing), env wiring is wrong — debug before continuing.

- [ ] **Step 3: Visual verification (HARD RULE — bongo lesson)**

```bash
MUJOCO_GL=egl uv run python scripts/record_video.py \
    --checkpoint checkpoints/<latest_splitbelt_ckpt>/best
```

Watch the rollout. Confirm:
- [ ] Robot stays on belts (does not slide off front/back).
- [ ] Stepping looks like stepping (not joint-locking + sliding).
- [ ] Belt slabs visibly translate.
- [ ] On a tied schedule, gait is symmetric to eyeball.

If any item fails, do NOT proceed to Task 5.2. The metric numbers are not to be trusted until visual passes.

- [ ] **Step 4: Document smoke result**

Open `.context/journals/2026-05-02.md` (create if absent). Append a section under "splitbelt env smoke":

```markdown
### Splitbelt env tied-belt smoke
- Run: 1M FastSAC, num_envs=1024, schedule=tied(0.5)
- Eval: <score> at <step>
- Visual: PASS / FAIL (notes)
- Wandb: <id if logged>
- Compare: Go2WarpJoystickFlat at same step gives ~<score>
```

- [ ] **Step 5: Commit doc**

```bash
git add .context/journals/2026-05-02.md
git commit -m "doc(splitbelt): journal entry for tied-belt calibration smoke"
```

---

### Task 5.2: Update repo docs (CLAUDE.md doc-sync checkpoint)

Per project CLAUDE.md "Doc Sync Checkpoint" — after a logical chunk of work, sweep docs.

- [ ] **Step 1: Update `.context/AGENT_HANDOFF.md`**

In the "Available Go2 envs" table, add a row:

```markdown
| `Go2WarpSplitbelt` | Two parallel belts (slab-on-slide + vel actuator) | Ideal PD | Adaptation benchmark substrate (A1/A2/A3/A4 protocols, see splitbelt_schedules.py) |
```

In "Available algos" or "Roadmap", add a one-liner under short-term: "Splitbelt benchmark protocol presets (deferred from env-build PR)."

- [ ] **Step 2: Update `.context/TODO.md`**

Add an entry under appropriate priority section:

```markdown
- [ ] Splitbelt: per-protocol algo presets (A1, A2, A3, A4) — deferred per spec §11.3.
- [ ] Splitbelt: humanoid asset selection + port — spec §11.1.
```

Mark splitbelt env build itself as completed:

```markdown
- [x] Splitbelt env (M1 mechanics, R3 robot-agnostic) — landed 2026-05-02.
```

- [ ] **Step 3: Update `docs/scripts/gen_env_presets.py` if presets land**

If you skipped the preset task per S§11.3 (which this plan does), no change needed. Otherwise:

```bash
uv run python docs/scripts/gen_env_presets.py
```

- [ ] **Step 4: Run docs-drift canaries**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_docs_drift.py tests/test_docs_code_blocks.py
```

Expected: pass (docs are in sync).

- [ ] **Step 5: Commit**

```bash
git add .context/AGENT_HANDOFF.md .context/TODO.md
git commit -m "doc(splitbelt): AGENT_HANDOFF + TODO updated for splitbelt env landing"
```

---

### Task 5.3: Final verification gate (lesson §8)

- [ ] **Step 1: Default lane**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q
```

Expected: pass count ≥ Task 0.1 baseline + (number of new hermetic tests).

- [ ] **Step 2: Drift canaries**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_docs_drift.py tests/test_docs_code_blocks.py
```

Expected: pass.

- [ ] **Step 3: All splitbelt tests**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_*.py
uv run python -m pytest --collect-only -q -m gpu | grep splitbelt
```

Expected: hermetic tests pass; collection lists the 3 gpu-marked files (env_smoke, bundle, control_metadata).

- [ ] **Step 4: Docs in sync**

```bash
git diff --name-only HEAD~10 HEAD | grep -E '\.(py|md|xml)$'
```

Inspect: every code change has an accompanying doc update where the cross-reference table in CLAUDE.md says it should.

- [ ] **Step 5: Open question for the user — proceed to A1 protocol benchmark or pause here?**

Splitbelt env build is complete. Next milestone (out of scope for this plan): per-protocol algo presets (PPO + FastSAC × A1/A2/A3/A4) and the offline analyzer CLI script (`scripts/analyze_splitbelt.py`) wrapping the analysis library. Either is a fresh brainstorm/spec/plan cycle.

---

## Out-of-scope items (cut intentionally per spec)

These were explicitly deferred per spec §11; do **not** add to this plan:

- Per-protocol algo presets (`Go2WarpSplitbelt` × {A1, A2, A3, A4} × {PPO, FastSAC}).
- Humanoid asset port (R3 step 2).
- Walking-cmd variant (`cmd_x > 0`).
- Composite limp index online (offline only).
- Belt-friction DR spec (off by default).
- `scripts/analyze_splitbelt.py` CLI shim (library exists; CLI when first protocol report is written).

---

## File summary

**Created:**
- `jax_rl/envs/locomotion/splitbelt_schedules.py` — schedule samplers + dispatch
- `jax_rl/envs/locomotion/splitbelt_geom.py` — BeltLayout + foot_belt_id helper
- `jax_rl/envs/locomotion/splitbelt_analysis.py` — offline gait analysis (numpy lib)
- `jax_rl/envs/locomotion/go2_warp_splitbelt.py` — env class
- `jax_rl/envs/locomotion/xmls/treadmill_splitbelt.xml` — robot-agnostic apparatus
- `jax_rl/envs/locomotion/xmls/go2_warp_splitbelt_scene.xml` — Go2 scene
- `tests/test_splitbelt_schedules.py` — hermetic
- `tests/test_splitbelt_belt_assignment.py` — hermetic
- `tests/test_splitbelt_metrics.py` — hermetic
- `tests/test_splitbelt_obs_schema.py` — hermetic
- `tests/test_splitbelt_env_smoke.py` — `[gpu, warp, go2]`
- `tests/test_splitbelt_bundle.py` — `[gpu]`
- `tests/test_splitbelt_control_metadata.py` — `[gpu, warp, go2, deploy]`

**Modified:**
- `jax_rl/training/env_backends/mjx_backend.py` — register Go2WarpSplitbelt
- `.context/AGENT_HANDOFF.md` — env list
- `.context/TODO.md` — completion + new entries
- `.context/journals/2026-05-02.md` — calibration smoke result

**Total commits:** ~25 (one per task step), each 2-5 minute units.
