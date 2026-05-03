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

## Stage 2: XML assets + supporting constants

The treadmill mesh is robot-agnostic; the scene combines it with go2.xml. We also add a `LEG_ACTUATOR_NAMES` constant that the env class needs for filtering `_act_to_joint` to leg-only actuators (Task 3.3).

### Task 2.0: Add `LEG_ACTUATOR_NAMES` constant to `go2_constants.py`

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_constants.py`

The splitbelt env's `_post_init` filters `_act_to_joint` to leg-only via this constant. Joystick env never needed it (it has no extra actuators). Names must match `unitree_go2/go2.xml:227-238` actuator declarations exactly, in MJX actuator order.

- [ ] **Step 1: Verify the actuator names + order**

```bash
grep "<motor " jax_rl/envs/locomotion/xmls/unitree_go2/go2.xml | grep -v ctrlrange
```

Expected: 12 motors named `FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf` (SDK leg order).

- [ ] **Step 2: Add the constant**

Append to `go2_constants.py`:

```python
LEG_ACTUATOR_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)
```

- [ ] **Step 3: Smoke test it imports + length is 12**

```bash
JAX_PLATFORMS=cpu uv run python -c "
from jax_rl.envs.locomotion import go2_constants as c
assert len(c.LEG_ACTUATOR_NAMES) == 12
print(c.LEG_ACTUATOR_NAMES)
"
```

- [ ] **Step 4: Default lane sanity (no test broken by the new constant)**

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(go2): add LEG_ACTUATOR_NAMES constant for splitbelt actuator filtering"
```

---

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
    <!-- Velocity actuators: ctrl = target velocity (m/s). High kv for tight tracking.
         forcerange MUST be explicit — Go2WarpEnv.__init__ at line 64-67 clobbers
         `actuator_forcerange[i] = actuator_ctrlrange[i]` for ALL nu actuators,
         so without an explicit forcerange the belt actuator would be capped at
         ±3 N (matching ctrlrange m/s magnitude), which is not enough force to
         accelerate the slab against foot drag. Pick ±200 N — enough for full
         schedule tracking under typical loading. The implementer must verify
         post-init that actuator_forcerange[belt_idx] == [-200, 200] (a smoke
         assertion in Task 3.6 catches drift). -->
    <velocity name="left_belt_vel"  joint="left_belt_joint"  kv="200" ctrlrange="-3.0 3.0" forcerange="-200 200"/>
    <velocity name="right_belt_vel" joint="right_belt_joint" kv="200" ctrlrange="-3.0 3.0" forcerange="-200 200"/>
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

  <!-- INCLUDE ORDER MATTERS. go2.xml MUST come first so its freejoint occupies
       qpos[0:7] and 12 leg joints occupy qpos[7:19]. Belt slide joints then
       occupy qpos[19:21]. All hardcoded slices in go2_warp_splitbelt.py
       (`qpos[7:7+12]`, `qvel[6:6+12]`, `actuator_force[:12]`, `data.ctrl[:12]`)
       depend on this. Same convention as `xmls/go2_bongo_scene.xml:3-4`. -->
  <include file="unitree_go2/go2.xml"/>
  <include file="treadmill_splitbelt.xml"/>

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
    <!-- IMU sensors — copied verbatim from go2_warp_scene_flat.xml:27-49.
         These are NOT in unitree's go2.xml; they're scene-level aliases that
         go2_warp_base / go2_constants reference by name. WITHOUT these the env
         crashes on `self.get_gyro(data)` / `self.get_accelerometer(data)` /
         `_foot_linvel_sensor_adr` lookup at construction. -->
    <gyro site="imu" name="gyro"/>
    <accelerometer site="imu" name="accelerometer"/>
    <framelinvel objtype="site" objname="imu" name="global_linvel"/>
    <frameangvel objtype="site" objname="imu" name="global_angvel"/>
    <velocimeter site="imu" name="local_linvel"/>
    <framezaxis objtype="site" objname="imu" name="upvector"/>
    <framexaxis objtype="site" objname="imu" name="forwardvector"/>

    <!-- Per-foot position sensors (relative to IMU) — needed for feet_clearance reward -->
    <framepos objtype="site" objname="FL_foot" name="FL_pos" reftype="site" refname="imu"/>
    <framepos objtype="site" objname="FR_foot" name="FR_pos" reftype="site" refname="imu"/>
    <framepos objtype="site" objname="RL_foot" name="RL_pos" reftype="site" refname="imu"/>
    <framepos objtype="site" objname="RR_foot" name="RR_pos" reftype="site" refname="imu"/>

    <!-- Per-foot linear velocity sensors — needed for feet_slip reward + foot_linvel_sensor_adr -->
    <framelinvel objtype="site" objname="FL_foot" name="FL_global_linvel"/>
    <framelinvel objtype="site" objname="FR_foot" name="FR_global_linvel"/>
    <framelinvel objtype="site" objname="RL_foot" name="RL_global_linvel"/>
    <framelinvel objtype="site" objname="RR_foot" name="RR_global_linvel"/>

    <!-- Required by Go2WarpEnv.__init__ (go2_warp_base.py:103-106). The base class
         looks up `{FL,FR,RL,RR}_floor_found` unconditionally. On splitbelt these fire
         ONLY when a foot touches `fallback_floor` (off-belt). The splitbelt env's
         step computes its own air-time grounding from the belt + floor contact-pair
         sensors below; do NOT use these for feet_air_time. -->
    <contact name="FL_floor_found" geom1="FL" geom2="fallback_floor" reduce="mindist" num="1" data="found"/>
    <contact name="FR_floor_found" geom1="FR" geom2="fallback_floor" reduce="mindist" num="1" data="found"/>
    <contact name="RL_floor_found" geom1="RL" geom2="fallback_floor" reduce="mindist" num="1" data="found"/>
    <contact name="RR_floor_found" geom1="RR" geom2="fallback_floor" reduce="mindist" num="1" data="found"/>

    <!-- Boolean contact-pair sensors per bongo lesson 2026-04-02: exact, threshold-free.
         Drives `foot_belt_id` (S§9.1), feet_air_time recompute, and termination (S§7.3). -->
    <contact name="FL_left_belt"  geom1="FL" geom2="left_belt_geom"/>
    <contact name="FL_right_belt" geom1="FL" geom2="right_belt_geom"/>
    <contact name="FR_left_belt"  geom1="FR" geom2="left_belt_geom"/>
    <contact name="FR_right_belt" geom1="FR" geom2="right_belt_geom"/>
    <contact name="RL_left_belt"  geom1="RL" geom2="left_belt_geom"/>
    <contact name="RL_right_belt" geom1="RL" geom2="right_belt_geom"/>
    <contact name="RR_left_belt"  geom1="RR" geom2="left_belt_geom"/>
    <contact name="RR_right_belt" geom1="RR" geom2="right_belt_geom"/>
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


@pytest.mark.parametrize("obs_mode", _VALID_MODES)
def test_build_obs_groups_matches_obs_term_names(obs_mode, monkeypatch):
    """Structural contract: schema-expanded names in build_obs_groups match
    obs_term_names. Hermetic — uses a fake env with the minimal interface
    build_obs_groups touches (just `_config.obs_mode` + `_config.noise_config.scales`).
    Catches IncludeGroup expansion drift in the default lane (no GPU needed).
    """
    from types import SimpleNamespace
    from jax_rl.envs.locomotion.go2_warp_splitbelt import build_obs_groups
    from jax_rl.envs.obs_spec import schema_from_obs_groups

    fake_noise = SimpleNamespace(
        joint_pos=0.0, joint_vel=0.0, gyro=0.0, gravity=0.0,
        linvel=0.0, accelerometer=0.0,
    )
    fake_env = SimpleNamespace(
        _config=SimpleNamespace(
            obs_mode=obs_mode,
            noise_config=SimpleNamespace(scales=fake_noise),
        ),
        # Methods that term_factory lambdas might capture — never invoked
        # by schema_from_obs_groups (it only walks names), so any callable suffices.
        get_gravity=lambda data: None,
        get_gyro=lambda data: None,
        get_local_linvel=lambda data: None,
        get_global_angvel=lambda data: None,
        _default_pose=None,
    )
    groups = build_obs_groups(fake_env)
    schema = schema_from_obs_groups(groups)
    layout = obs_term_names(obs_mode)
    assert schema["state"] == layout["state"]
    state_set = set(layout["state"])
    expected_priv = layout["state"] + [n for n in layout["privileged_state"] if n not in state_set]
    assert schema["privileged_state"] == expected_priv
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

Subclass `go2_warp_base.Go2WarpEnv` (real class name in [`go2_warp_base.py:34`](../../jax_rl/envs/locomotion/go2_warp_base.py#L34)). Match the joystick env's API EXACTLY:
- override `reset(rng)` and `step(state, action)` — **no leading underscore**. The framework calls these directly.
- inline-build `mjx_env.State(data, obs, reward, done, metrics, info)` — no `_init_pipeline`/`_make_state` helpers (those don't exist).
- `_get_obs(self, data, info)` — not `_compute_obs`.
- Use real base helpers only: `self.get_gravity(data)`, `self.get_gyro(data)`, `self.get_local_linvel(data)`, `self.get_global_linvel(data)`, `self.get_global_angvel(data)`, `self.get_upvector(data)`, `self.get_accelerometer(data)`. There is NO `get_joint_pos` / `get_joint_vel` / `_extract_yaw` / `_world_to_body`. Use `data.qpos[7:] - self._default_pose` and `data.qvel[6:]` directly.
- `__init__` MUST accept `task: str` kwarg (Playground's registry passes it via `functools.partial`; ignore it inside if not used). Joystick precedent: [`go2_warp_joystick.py:85-95`](../../jax_rl/envs/locomotion/go2_warp_joystick.py#L85-L95).
- The base class `Go2WarpEnv.__init__` looks up `{FL,FR,RL,RR}_floor_found` sensors at [`go2_warp_base.py:103-106`](../../jax_rl/envs/locomotion/go2_warp_base.py#L103-L106). The scene XML (Task 2.2) declares them. Don't override the lookup.

- [ ] **Step 1: Read precedent**

```bash
sed -n '40,200p' jax_rl/envs/locomotion/go2_warp_joystick.py    # default_config + _post_init + _obs_groups + _reward_spec
sed -n '236,395p' jax_rl/envs/locomotion/go2_warp_joystick.py   # reset + step + _get_obs
```

Note: real call signatures, real helper names, real `mjx_env.State` construction, real reward computation pattern (named lambdas → weighted sum × dt → clip).

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
_PROPRIO_NAMES = ("joint_pos", "joint_vel", "last_act", "gravity", "gyro", "command")


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
    """Default config: tied belts at 0.5 m/s, blind obs, cmd=0 (smoke baseline).

    Reward scales ported VERBATIM from go2_warp_joystick.default_config (S§7.1).
    Empty / `.get(name, default)` patterns are forbidden — they silently produce
    ~10× weaker tracking reward and lose the calibration target (S§10.5).
    """
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
        # Obs noise (consumed by compute_obs at step time; matches joystick env shape).
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05,
                linvel=0.1, accelerometer=0.1,
            ),
        ),
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
        # Replay-bloat gate (spec §11.X caveat). True = write full splitbelt
        # primitives every step (~200 MB extra per 1M-step replay buffer for
        # offline analysis). False = skip the writes during training.
        # Eval/recording presets should set True; train presets should set False
        # if the empirical bloat is a problem.
        log_splitbelt=True,
        # Reward scales — ported verbatim from go2_warp_joystick:47-69.
        # Splitbelt-specific term `treadmill_drift` added; `stand_still` dropped
        # because cmd is always zero (the term collapses with treadmill_drift).
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=10.0,
                tracking_ang_vel=5.0,
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
                dof_pos_limits=-1.0,
                feet_air_time=0.1,
                feet_slip=-0.1,
                feet_clearance=-2.0,
                feet_height=-0.2,
                termination=-1.0,
                pose=0.5,
                base_height=-5.0,
                # New for splitbelt:
                treadmill_drift=1.0,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        # MJX/Warp tuning
        naconmax=4 * 8192,
        naccdmax=4000,
        njmax=100,
    )


def build_obs_groups(env: Any) -> Dict[str, list]:
    """Build real ObsTerm dispatch with env-method-bound lambdas (used in _post_init).

    Names must match `obs_term_names(env._config.obs_mode)` exactly — that is the
    contract validated in tests. Pattern follows go2_warp_joystick.py:128-159.
    Uses IncludeGroup("state") in privileged so the actor's terms are inherited
    automatically (saves duplication, idiomatic per obs_spec.py:46-49).
    """
    cfg = env._config
    layout = obs_term_names(cfg.obs_mode)
    noise = cfg.noise_config.scales

    # `data.qpos[7:]` and `data.qvel[6:]` are the real way to read joint pos/vel
    # for Go2 (post-base, post-freejoint). Joystick precedent: lines 137-140.
    term_factory = {
        "joint_pos": (lambda data, **kw: data.qpos[7:7+12] - env._default_pose, noise.joint_pos),
        "joint_vel": (lambda data, **kw: data.qvel[6:6+12], noise.joint_vel),
        "last_act": (lambda info, **kw: info["last_act"], 0.0),
        "gravity": (lambda data, **kw: env.get_gravity(data), noise.gravity),
        "gyro": (lambda data, **kw: env.get_gyro(data), noise.gyro),
        "command": (lambda info, **kw: info["command"], 0.0),
        "belt_vel": (lambda info, **kw: info["splitbelt"]["belt_vel"], 0.0),
        "cmd_track_error": (lambda info, **kw: info["splitbelt"]["cmd_track_error"], 0.0),
        "drift_xy": (lambda info, **kw: info["splitbelt"]["drift_xy"], 0.0),
        # Privileged-only terms read TRUE base velocities (no noise).
        "base_lin_vel": (lambda data, **kw: env.get_local_linvel(data), 0.0),
        "base_ang_vel": (lambda data, **kw: env.get_global_angvel(data), 0.0),
    }

    def _build(names):
        return [ObsTerm(name=n, fn=term_factory[n][0], noise_scale=term_factory[n][1])
                for n in names]

    # State group: real ObsTerms.
    state_terms = _build(layout["state"])

    # Privileged: IncludeGroup("state") + privileged extras (the names in `layout`
    # that are NOT in `state`). Order preserved.
    state_set = set(layout["state"])
    priv_extras = [n for n in layout["privileged_state"] if n not in state_set]
    privileged_terms = [IncludeGroup("state")] + _build(priv_extras)

    return {"state": state_terms, "privileged_state": privileged_terms}


class Go2WarpSplitbeltEnv(go2_warp_base.Go2WarpEnv):
    """Go2 on a split-belt treadmill (S§5.3)."""

    def __init__(
        self,
        task: str = "splitbelt",
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ) -> None:
        # `task` accepted for Playground registry compatibility; not used internally.
        del task
        cfg = config if config is not None else default_config()
        xml_path = (
            Path(__file__).parent / "xmls" / "go2_warp_splitbelt_scene.xml"
        ).as_posix()
        super().__init__(
            xml_path=xml_path, config=cfg, config_overrides=config_overrides
        )
        self._post_init()

    def _post_init(self) -> None:
        # NOTE: go2_warp_base.Go2WarpEnv.__init__ does NOT call _post_init itself.
        # Joystick env calls it explicitly from its own __init__ (line 96).
        # Heavy lifting (sensor lookup, default_pose, belt IDs) lives here.
        # See Task 3.3 for the full implementation.
        self._action_dim = 12
        self._default_pose = jp.array(self._mj_model.keyframe("splitbelt_spawn").qpos[7:7+12])
        self._init_q = jp.array(self._mj_model.keyframe("splitbelt_spawn").qpos)
        self._obs_groups = build_obs_groups(self)
        # Task 3.3 fills in: belt_layout, belt actuator/joint IDs, contact-pair sensor IDs,
        # torso fall sensors, schedule_T.

    @property
    def action_size(self) -> int:
        # Override Go2WarpEnv.action_size (which returns mjx_model.nu = 14).
        # Belt actuators are env-internal (driven by schedule_table); the policy
        # only controls the 12 leg actuators. Without this override, the algo
        # would sample 14-d actions and shape-mismatch on every step.
        return 12

    def reset(self, rng: jax.Array):
        raise NotImplementedError("Implemented in Task 3.4")

    def step(self, state, action: jax.Array):
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

This populates env attributes used by `step`/`reset`. No new tests here — exercised by Task 3.6 smoke `[gpu, warp, go2]`.

- [ ] **Step 1: Replace the stub `_post_init` with the full version**

```python
def _post_init(self) -> None:
    self._action_dim = 12
    self._init_q = jp.array(self._mj_model.keyframe("splitbelt_spawn").qpos)
    self._default_pose = jp.array(
        self._mj_model.keyframe("splitbelt_spawn").qpos[7:7+12]
    )

    # Soft joint limits — first joint is freejoint; next 12 are leg joints.
    # Belt slide joints come AFTER the legs in the keyframe; only legs are limited.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:1+12].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.WARP_ROOT_BODY).id
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )

    cfg = self._config
    self._belt_layout = geom.BeltLayout(
        left_y_min=cfg.belt_layout.left_y_min,
        left_y_max=cfg.belt_layout.left_y_max,
        right_y_min=cfg.belt_layout.right_y_min,
        right_y_max=cfg.belt_layout.right_y_max,
    )

    # Belt actuator + joint IDs.
    self._left_belt_act_id = self._mj_model.actuator("left_belt_vel").id
    self._right_belt_act_id = self._mj_model.actuator("right_belt_vel").id
    # Belt slide joints sit AFTER the leg joints in the model.
    # Use joint().qposadr/dofadr for indexing into qpos/qvel — joint id alone is
    # not the qpos index. Joystick precedent reads qpos[7:7+12] directly because
    # leg joints are positions 7..18; for belts we look up by name.
    self._left_belt_qposadr = self._mj_model.joint("left_belt_joint").qposadr[0]
    self._right_belt_qposadr = self._mj_model.joint("right_belt_joint").qposadr[0]
    self._left_belt_dofadr = self._mj_model.joint("left_belt_joint").dofadr[0]
    self._right_belt_dofadr = self._mj_model.joint("right_belt_joint").dofadr[0]

    # Belt actuator addresses in `data.ctrl` (for writing belt-vel commands).
    # Leg actuators are 0..11; belt actuators come after.
    self._left_belt_ctrl_idx = self._mj_model.actuator("left_belt_vel").id
    self._right_belt_ctrl_idx = self._mj_model.actuator("right_belt_vel").id

    # Contact-pair sensor IDs for foot_belt_id + termination cause (S§9.1).
    feet_order = ("FL", "FR", "RL", "RR")
    def _adr(name):
        sid = self._mj_model.sensor(name).id
        return self._mj_model.sensor_adr[sid]
    self._foot_left_belt_adr = jp.array(
        [_adr(f"{f}_left_belt") for f in feet_order]
    )
    self._foot_right_belt_adr = jp.array(
        [_adr(f"{f}_right_belt") for f in feet_order]
    )
    # Floor-found sensors (declared in scene XML; Go2WarpEnv.__init__ also reads them).
    self._foot_floor_adr = jp.array(
        [_adr(f"{f}_floor_found") for f in feet_order]
    )
    self._torso_left_belt_adr = _adr("torso_left_belt")
    self._torso_right_belt_adr = _adr("torso_right_belt")
    self._torso_floor_adr = _adr("torso_floor")

    # Foot global linvel sensor adresses — needed by joystick's _cost_feet_slip
    # / _cost_feet_clearance helpers we copy verbatim. Joystick precedent:
    # go2_warp_joystick.py:113-123. The sensors are declared in go2.xml (per
    # foot: `{FL,FR,RL,RR}_global_linvel`). DO NOT skip this — the helpers
    # silently break otherwise.
    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
        name = site.replace("_foot", "") + "_global_linvel"
        sensor_id = self._mj_model.sensor(name).id
        sensor_adr = self._mj_model.sensor_adr[sensor_id]
        sensor_dim = self._mj_model.sensor_dim[sensor_id]
        foot_linvel_sensor_adr.append(
            list(range(sensor_adr, sensor_adr + sensor_dim))
        )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    # Leg-only act_to_joint slice. The base class builds _act_to_joint with
    # length nu=14 (12 legs + 2 belts). For the PD inner loop we want only
    # the 12 leg entries, plus the inverse permutation tau_joint[12] -> ctrl[12].
    leg_act_ids = jp.array([
        self._mj_model.actuator(name).id for name in consts.LEG_ACTUATOR_NAMES
    ])  # length 12, in actuator-order (FR,FL,RR,RL per consts)
    self._leg_act_ids = leg_act_ids
    # Filter the base-class _act_to_joint to only the leg entries:
    self._leg_act_to_joint = self._act_to_joint[leg_act_ids]

    # Restore belt actuator forcerange + RE-PUT model into MJX.
    # Go2WarpEnv.__init__ at line 64-67 clobbers actuator_forcerange[i] = ctrlrange[i]
    # for ALL nu actuators BEFORE calling mjx.put_model. The XML's forcerange
    # declaration AND any post-init mutation of self._mj_model do NOT propagate
    # to self._mjx_model (it's a frozen snapshot). The only working fix is to:
    #   1. Mutate self._mj_model.actuator_forcerange[belt_idx] back to a sane range.
    #   2. Re-snapshot via mjx.put_model so MJX/Warp picks up the change.
    self._mj_model.actuator_forcerange[self._left_belt_act_id] = np.array([-200.0, 200.0])
    self._mj_model.actuator_forcerange[self._right_belt_act_id] = np.array([-200.0, 200.0])
    # Re-put the model so the MJX snapshot reflects the corrected forcerange.
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    # Smoke assertion (caught immediately if the above pattern ever breaks):
    assert float(self._mjx_model.actuator_forcerange[self._left_belt_act_id, 1]) > 100.0, (
        "belt actuator forcerange clobbered; mjx.put_model re-call failed"
    )

    self._obs_groups = build_obs_groups(self)
    self._schedule_T = int(cfg.episode_length)
```

> **Implementer notes:**
> - `consts.WARP_ROOT_BODY`, `consts.FEET_SITES`, `consts.LEG_ACTUATOR_NAMES` are already used by joystick — reuse the constants. If `LEG_ACTUATOR_NAMES` does NOT exist in `go2_constants.py`, add it (12-element list of leg actuator names in actuator-order, derivable from joystick's `_kp` ordering).
> - We do NOT call `super()._post_init()`. The base class doesn't define one; joystick env calls `_post_init` itself from its own `__init__` (line 96), which is what `Go2WarpSplitbeltEnv.__init__` already does (Task 3.2 Step 2).
> - Soft joint limits use `[1:1+12]` (12 leg joints), not `[1:]` — joystick uses `[1:]` because it has no extra slide joints. We must skip the belt slide joints in the soft-limit slice.
> - **Belt actuator forcerange:** Empirically verified in round-3 audit that `Go2WarpEnv.__init__:64-67` clobbers forcerange BEFORE `mjx.put_model`. XML declaration alone is insufficient. The `_post_init` block above writes back `self._mj_model.actuator_forcerange[belt_idx]` and then re-snapshots via `mjx.put_model`. The smoke assertion catches drift if `mjx.put_model` semantics change.

- [ ] **Step 1.5: Add three required overrides on `Go2WarpSplitbeltEnv`**

These were called out as iteration-1 gaps. They MUST be on the env class for the PR to ship — not "left as implementer notes."

```python
def get_domain_randomization_spec(self):
    """Splitbelt reuses joystick's DR specs verbatim — port from
    go2_warp_joystick.py:200-232. The base class does NOT define this method,
    so the splitbelt env must declare it explicitly or DR is silently no-op.
    """
    from jax_rl.envs.wrappers.domain_rand import DRSpec
    return [
        DRSpec(name="friction", type="model", field="geom_friction",
               column=0, min=0.3, max=1.5, per_element=False, operation="set",
               description="Uniform friction across all geoms"),
        DRSpec(name="dof_damping", type="model", field="dof_damping",
               indices=(6, 18), min=0.7, max=2.0, per_element=True,
               description="Joint damping variation"),
        DRSpec(name="dof_armature", type="model", field="dof_armature",
               indices=(6, 18), min=0.9, max=1.3, per_element=True,
               description="Joint armature variation"),
        DRSpec(name="dof_frictionloss", type="model", field="dof_frictionloss",
               indices=(6, 18), min=0.7, max=1.5, per_element=True,
               description="Joint friction loss variation"),
        DRSpec(name="body_mass", type="model", field="body_mass",
               min=0.8, max=1.2, per_element=True,
               description="Per-link mass variation"),
        DRSpec(name="motor_strength", type="model", field="actuator_gainprm",
               column=0, min=0.9, max=1.1, per_element=True,
               description="Per-actuator motor heterogeneity"),
        DRSpec(name="torso_com_jitter", type="model", field="body_ipos",
               indices=(1, 2), min=-0.03, max=0.03,
               per_element=True, operation="add",
               description="Torso COM offset (x, y)"),
        DRSpec(name="body_inertia", type="model", field="body_inertia",
               min=0.85, max=1.15, per_element=True,
               description="Per-link inertia tensor variation"),
    ]


def get_control_metadata(self) -> dict:
    """Override base class get_control_metadata. Base shape-checks
    `_default_pose.shape == (mjx_model.nu,)` (go2_warp_base.py:141), but
    splitbelt has nu=14 (12 leg + 2 belt) while _default_pose is len-12.
    Slice to leg-only and call the base-class logic locally.
    """
    import numpy as np
    from deploy.go2_constants import POLICY_TO_SDK
    default_pose_policy = np.asarray(self._default_pose, dtype=np.float32)
    assert default_pose_policy.shape == (12,), default_pose_policy.shape
    default_pose_sdk = default_pose_policy[np.array(POLICY_TO_SDK)]
    return {
        "default_pose_policy": default_pose_policy.tolist(),
        "default_pose_sdk": default_pose_sdk.tolist(),
        "policy_to_sdk": list(POLICY_TO_SDK),
        "sdk_to_policy": list(np.argsort(POLICY_TO_SDK)),
        "action_scale": float(self._config.action_scale),
        "Kp": float(self._config.Kp),
        "Kd": float(self._config.Kd),
        "ctrl_dt": float(self._config.ctrl_dt),
        "sim_dt": float(self._config.sim_dt),
        "contact_mode": self._config.contact_mode,
    }


```

> **Implementer note — air-time grounding:** Task 3.5 Step B inlines `contact = foot_in_left | foot_in_right | foot_in_floor`. That OR is the splitbelt-specific grounding signal — joystick's reward helpers (`_reward_feet_air_time`, `_cost_feet_slip`, `_cost_feet_clearance`, `_cost_feet_height`) receive `contact` positionally, so we just pass the OR. **Do NOT** use `floor_found` sensors for grounding on splitbelt — they fire ONLY for off-belt landings, inverting the air/ground signal.

- [ ] **Step 2: Smoke-import (CPU path; only constructs — `step`/`reset` still raise NotImplementedError)**

```bash
JAX_PLATFORMS=cpu uv run python -c "
from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
env = Go2WarpSplitbeltEnv()
print('action_dim', env._action_dim)
print('belt_layout', env._belt_layout)
print('default_pose len', len(env._default_pose))
print('obs_groups keys', list(env._obs_groups.keys()))
"
```

Expected: prints without crashing. If this fails on a Warp-only setup, this manual check moves to `[gpu, warp]` — skip and let Task 3.6's smoke test catch issues.

- [ ] **Step 3: Commit**

```bash
git commit -am "feat(splitbelt): _post_init wires sensors + belt geometry + default pose"
```

---

### Task 3.4: `reset` — schedule sample, state init, term_cause zeroing

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_warp_splitbelt.py`

Per S§6.2 and S§9.1: sample schedule table, set belt qvel, sample cmd, zero `term_cause`. Mirror joystick's `reset` exactly ([`go2_warp_joystick.py:236-300`](../../jax_rl/envs/locomotion/go2_warp_joystick.py#L236-L300)).

- [ ] **Step 1: Add imports at top of file**

```python
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from jax_rl.envs.obs_spec import compute_obs
```

- [ ] **Step 2: Implement `reset` (no leading underscore — framework calls this directly)**

```python
def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k_qpos, k_qvel, k_sched = jax.random.split(rng, 4)

    # qpos: spawn keyframe + small per-joint noise on legs only.
    qpos = self._init_q
    qpos_noise = jax.random.uniform(
        k_qpos, (12,), minval=-0.01, maxval=0.01
    )
    qpos = qpos.at[7:7+12].set(qpos[7:7+12] + qpos_noise)

    # Belt slabs back to origin (override whatever was in keyframe).
    qpos = qpos.at[self._left_belt_qposadr].set(0.0)
    qpos = qpos.at[self._right_belt_qposadr].set(0.0)

    # Belt schedule sampled per episode (S§5.4).
    schedule_table = sched.sample_schedule(
        k_sched,
        T=self._schedule_T,
        kind=self._config.schedule_kind,
        params=dict(self._config.schedule_params),
    )

    # qvel: zeros for legs + base; belts pre-seeded to schedule[0]
    # (S§6.3 — accept first-step transient for non-tied schedules).
    qvel = jp.zeros(self.mjx_model.nv)
    qvel = qvel.at[self._left_belt_dofadr].set(schedule_table[0, 0])
    qvel = qvel.at[self._right_belt_dofadr].set(schedule_table[0, 1])

    # Build initial mjx data — same call as joystick:256-265.
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=jp.zeros(self.mjx_model.nu),
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # cmd: always-zero by default (S§7.4). Plumbed through obs/reward.
    cmd = jp.zeros(3)

    info = {
        "rng": rng,
        "step_idx": jp.int32(0),
        "belt_schedule": schedule_table,
        "command": cmd,  # joystick-aligned key (round-3 rename: was "cmd")
        "last_act": jp.zeros(self._action_dim),
        "last_last_act": jp.zeros(self._action_dim),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
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
            "term_cause": jp.int32(0),  # S§9.1: 0=alive, 1=fall, 2=off-belt, 3=tilt
            "step_idx": jp.int32(0),
        },
        "reward_components": {
            k: jp.zeros(()) for k in self._config.reward_config.scales.keys()
        },
    }

    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}

    obs = self._get_obs(data, info)

    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)
```

- [ ] **Step 3: Implement `_get_obs` (the canonical wrapper around `compute_obs`)**

```python
def _get_obs(self, data, info):
    obs, info["rng"] = compute_obs(
        self._obs_groups,
        noise_level=self._config.noise_config.level,
        rng=info["rng"],
        data=data, info=info,
    )
    return obs
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(splitbelt): reset + _get_obs (mjx_env.State pattern from joystick)"
```

---

### Task 3.5: `step` — belt actuation, gait primitives, obs, reward, termination

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_warp_splitbelt.py`

This is the largest sub-task. Mirror joystick's `step` ([`go2_warp_joystick.py:302-391`](../../jax_rl/envs/locomotion/go2_warp_joystick.py#L302-L391)) — same PD-substep `lax.scan`, same reward computation pattern, same `state.replace` return.

**Hard rules to remember:**
- DO NOT write `info["truncation"]` here — that's the wrapper layer's job (`EpisodeWrapper` / `DomainRandWrapper`).
- DO NOT use `state.pipeline_state` — it's `state.data` per `mjx_env.State`.
- DO NOT call `self._compute_ctrl` / `_physics_step` / `_init_pipeline` / `_make_state` / `_extract_yaw` / `_world_to_body` — none exist. Use the precedent below.
- Reward weights live in `self._config.reward_config.scales[name]` — direct dict lookup, NO `.get(name, default)`.

#### Step A: belt-vel write + physics substep loop

```python
def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = self._default_pose + action * self._config.action_scale

    step_idx = state.info["step_idx"]
    # Clamp index defense-in-depth (spec §6.3 invariant): step_idx is incremented
    # AFTER the step that just executed, so reads here are pre-increment and live
    # in [0, episode_length-1]. Wrapper composition could in principle let one
    # extra step fire before reset propagates; clamp prevents OOB indexing.
    safe_idx = jp.minimum(step_idx, self._schedule_T - 1)
    belt_vel_target = state.info["belt_schedule"][safe_idx]  # (2,)

    kp = self._kp
    kd = self._kd
    model = self.mjx_model
    leg_a2j = self._leg_act_to_joint  # length-12, leg-only act_to_joint slice (Task 3.3)
    leg_act_ids = self._leg_act_ids   # length-12, indices of leg actuators in data.ctrl

    def substep(data, _):
        current_q = data.qpos[7:7+12]      # legs (XML include order: go2 first, belts last)
        current_dq = data.qvel[6:6+12]
        tau_joint = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
        tau_joint = self._apply_torque_speed_limit(tau_joint, current_dq)
        # Write leg torques and belt velocities into ctrl by NAMED index (not slice).
        leg_ctrl = tau_joint[leg_a2j]
        full_ctrl = data.ctrl.at[leg_act_ids].set(leg_ctrl)
        full_ctrl = full_ctrl.at[self._left_belt_ctrl_idx].set(belt_vel_target[0])
        full_ctrl = full_ctrl.at[self._right_belt_ctrl_idx].set(belt_vel_target[1])
        data = data.replace(ctrl=full_ctrl)
        return mjx.step(model, data), None

    data = jax.lax.scan(substep, state.data, (), self.n_substeps)[0]
```

#### Step B: gait primitives (S§9.1)

```python
    sensordata = data.sensordata
    foot_in_left = sensordata[self._foot_left_belt_adr] > 0.0
    foot_in_right = sensordata[self._foot_right_belt_adr] > 0.0
    foot_in_floor = sensordata[self._foot_floor_adr] > 0.0  # off-belt landings

    contact = foot_in_left | foot_in_right | foot_in_floor
    foot_belt_id = jp.where(
        foot_in_left, jp.int32(0),
        jp.where(foot_in_right, jp.int32(1), jp.int32(-1)),
    )

    foot_pos_world = data.site_xpos[self._feet_site_id]   # joystick:344 pattern

    base_pos_world = data.qpos[:3]
    body_lin_vel = self.get_local_linvel(data)             # body-frame, joystick:163
    body_ang_vel = self.get_gyro(data)                     # body yaw rate at index 2

    # Tracking error in body frame (cmd is body-frame; S§3.4).
    cmd = state.info["command"]
    cmd_track_error = jp.concatenate([
        cmd[:2] - body_lin_vel[:2],
        cmd[2:3] - body_ang_vel[2:3],
    ])
    drift_xy = base_pos_world[:2]   # treadmill_center = (0, 0); S§7.1
```

#### Step C: termination with term_cause (S§7.3)

```python
    # Contact-based primary detectors.
    fall_torso = (
        (sensordata[self._torso_left_belt_adr] > 0.0)
        | (sensordata[self._torso_right_belt_adr] > 0.0)
        | (sensordata[self._torso_floor_adr] > 0.0)
    )
    is_off_belt = jp.any(foot_in_floor)

    # Threshold backstop. self.get_gravity(data) returns body-frame gravity;
    # the z component is "uprightness" (1 = upright, 0 = sideways).
    gravity_body = self.get_gravity(data)
    is_tilt = (gravity_body[2] < 0.5) | (base_pos_world[2] < 0.18)

    done = fall_torso | is_off_belt | is_tilt

    # term_cause priority: fall > off-belt > tilt.
    term_cause = jp.where(
        fall_torso, jp.int32(1),
        jp.where(is_off_belt, jp.int32(2),
                 jp.where(is_tilt, jp.int32(3), jp.int32(0))),
    )
```

#### Step D: reward — port full term set from joystick

Port [`_get_reward`](../../jax_rl/envs/locomotion/go2_warp_joystick.py#L416) verbatim with two changes:
- Replace `tracking_lin_vel`/`tracking_ang_vel` to use `cmd = jp.zeros(3)` substituted into joystick's existing kernels (the kernels work fine with cmd=0 — they reduce to "minimize body velocity" and "minimize yaw rate"). No code change to the kernels themselves.
- Replace `stand_still` with new `treadmill_drift` term:

```python
    # Use the existing reward helper methods from joystick verbatim — copy them
    # into go2_warp_splitbelt.py: _reward_tracking_lin_vel, _reward_tracking_ang_vel,
    # _cost_lin_vel_z, _cost_ang_vel_xy, _cost_orientation, _cost_torques,
    # _cost_action_rate, _cost_energy, _cost_joint_pos_limits, _reward_feet_air_time,
    # _cost_feet_slip, _cost_feet_clearance, _cost_feet_height, _cost_termination,
    # _reward_pose, _cost_base_height. Sources: go2_warp_joystick.py:416-560 (approx).

    def _reward_treadmill_drift(self, drift_xy):
        wL = self._config.treadmill_drift_lateral_weight
        wF = self._config.treadmill_drift_forward_weight
        return -(wL * jp.square(drift_xy[1]) + wF * jp.square(drift_xy[0]))

    # In step:
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_fz = foot_pos_world[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    rewards = {
        "tracking_lin_vel": self._reward_tracking_lin_vel(cmd, body_lin_vel),
        "tracking_ang_vel": self._reward_tracking_ang_vel(cmd, body_ang_vel),
        "lin_vel_z":        self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy":       self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation":      self._cost_orientation(self.get_upvector(data)),
        "torques":          self._cost_torques(data.actuator_force[:12]),
        "action_rate":      self._cost_action_rate(action, state.info["last_act"], state.info["last_last_act"]),
        "energy":           self._cost_energy(data.qvel[6:6+12], data.actuator_force[:12]),
        "dof_pos_limits":   self._cost_joint_pos_limits(data.qpos[7:7+12]),
        "feet_air_time":    self._reward_feet_air_time(state.info["feet_air_time"], first_contact, cmd),
        "feet_slip":        self._cost_feet_slip(data, contact, state.info),
        "feet_clearance":   self._cost_feet_clearance(data),
        "feet_height":      self._cost_feet_height(state.info["swing_peak"], first_contact, state.info),
        "termination":      self._cost_termination(done),
        "pose":             self._reward_pose(data.qpos[7:7+12]),
        "base_height":      self._cost_base_height(data),
        "treadmill_drift":  self._reward_treadmill_drift(drift_xy),
    }
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    state.info["reward_components"] = rewards
```

> **Implementer note:** the leg actuator force is at indices `[:12]` of `data.actuator_force`; belt actuators come after. Joystick uses bare `data.actuator_force` because all 12 entries are legs. We must slice.
> Helpers `_apply_torque_speed_limit`, `_act_to_joint`, `_kp`, `_kd`, `n_substeps`, `dt`, `mjx_model`, `mj_model` are inherited from `go2_warp_base.Go2WarpEnv` — used by joystick verbatim, no overrides needed.

#### Step E: info update + obs + return

```python
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step_idx"] = step_idx + 1
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact

    if self._config.log_splitbelt:
        state.info["splitbelt"] = {
            "foot_in_contact": contact,
            "foot_pos_world": foot_pos_world,
            "foot_belt_id": foot_belt_id,
            "base_pos_world": base_pos_world,
            "base_vel_world": data.qvel[:3],   # true world-frame; logged for analysis
            "base_yaw": jp.float32(0.0),       # placeholder; not used in reward — derive offline if needed
            "belt_vel": belt_vel_target,
            "cmd_track_error": cmd_track_error,
            "drift_xy": drift_xy,
            "term_cause": term_cause,          # nonzero only on the done step
            "step_idx": step_idx,
        }
    # else: leave the existing (zeroed) splitbelt dict from reset in place; only
    # eval/recording rollouts log full primitives. Training replay sees zeros.

    obs = self._get_obs(data, state.info)
    for k, v in rewards.items():
        state.metrics[f"reward/{k}"] = v
    state.metrics["splitbelt/term_cause"] = term_cause.astype(reward.dtype)

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)
```

> **Hard rule (env-backend lesson §2):** `info["truncation"]` is set by the `EpisodeWrapper`/`DomainRandWrapper`, not by inner `step`. Joystick env does not write it; we don't either. The wrapper pops/repopulates it.

#### Step F: copy reward helper methods from joystick

Copy the full set of `_reward_*` and `_cost_*` methods from `go2_warp_joystick.py` (the methods invoked above) verbatim into `go2_warp_splitbelt.py`. Per spec §5.3, we duplicate rather than factor — premature abstraction is YAGNI.

The new method `_reward_treadmill_drift` (defined above) is the only bespoke reward helper.

> **Implementer note — naming alignment:** Splitbelt info-dict keys match joystick exactly (`info["command"]`, `info["last_act"]`, `info["last_last_act"]`). Copy joystick reward helpers verbatim; keys line up by construction. (Earlier plan iterations used `info["cmd"]` / `info["last_action"]`; renamed in iteration 3 to eliminate the helper copy hazard.)

- [ ] **Step 1: Add all of A–F to `go2_warp_splitbelt.py`. Verify file imports.**

```bash
JAX_PLATFORMS=cpu uv run python -c "
from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
env = Go2WarpSplitbeltEnv()
print('OK env constructs')
print('reward terms:', list(env._config.reward_config.scales.keys()))
"
```

Expected: prints. Reward terms list matches `default_config()` exactly (17 entries: joystick's 17 minus `stand_still` plus splitbelt's `treadmill_drift`).

- [ ] **Step 2: Commit**

```bash
git commit -am "feat(splitbelt): step + full reward set + term_cause"
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
    schedule_step0 = state.info["belt_schedule"][0]
    actual_left = state2.data.qvel[env._left_belt_dofadr]
    actual_right = state2.data.qvel[env._right_belt_dofadr]
    # 10% tolerance for one-step transient (kv=200 should achieve this).
    assert jnp.abs(actual_left - schedule_step0[0]) < 0.1
    assert jnp.abs(actual_right - schedule_step0[1]) < 0.1


    # NOTE: structural drift assertion (env._obs_groups names match obs_term_names)
    # has moved to tests/test_splitbelt_obs_schema.py — it's a code-only contract
    # check that should run in the default (hermetic CPU) lane, not behind gpu/warp.


def test_off_belt_termination(env, rng):
    """Force a foot off-belt by shifting robot in y; assert term_cause = 2."""
    state = env.reset(rng)
    # Replace base y in qpos to put robot fully off the right side of belts.
    qpos = state.data.qpos.at[1].set(2.0)  # y = 2 m, well outside belts
    new_data = state.data.replace(qpos=qpos)
    state = state.replace(data=new_data)
    action = jnp.zeros((env._action_dim,))
    s2 = env.step(state, action)
    # foot×fallback_floor should fire.
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

- [ ] **Step 1: Read existing registration block**

```bash
sed -n '25,115p' jax_rl/training/env_backends/mjx_backend.py
```

Confirms: every registration lives inside `_register_custom_envs()` (line 26), uses `pg_locomotion.register_environment(name, functools.partial(EnvClass, task=...), default_config_factory)`, and is guarded with `if "<Name>" not in pg_locomotion._envs:`.

- [ ] **Step 2: Add the splitbelt block inside `_register_custom_envs`**

Insert just after the existing Bongo block (around line 111, after the `Go2BongoHandstandContraction` registration):

```python
    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    from jax_rl.envs.locomotion.go2_warp_splitbelt import default_config as splitbelt_default_config
    if "Go2WarpSplitbelt" not in pg_locomotion._envs:
        pg_locomotion.register_environment(
            "Go2WarpSplitbelt",
            functools.partial(Go2WarpSplitbeltEnv, task="splitbelt"),
            splitbelt_default_config,
        )
```

The `task="splitbelt"` is required for symmetry with the existing pattern — Playground's loader passes `config=` and `config_overrides=` kwargs to the partial, and the partial pre-binds `task=`. The env's `__init__` signature (Task 3.2) accepts `task` and discards it.

- [ ] **Step 3: Verify imports + the function still type-checks**

```bash
JAX_PLATFORMS=cpu uv run python -c "
from jax_rl.training.env_backends import mjx_backend
print('Go2WarpSplitbelt' in __import__('mujoco_playground._src.locomotion', fromlist=['_envs'])._envs)
"
```

Expected: prints `True`. (Triggering `_register_custom_envs` may require importing the env_backend module — `mjx_backend` calls it at module load.)

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

### Task 4.2: Bundle test [gpu, warp, go2]

**Files:**
- Create: `tests/test_splitbelt_bundle.py`

Per lesson §7.5 + §1: `make_env_bundle("Go2WarpSplitbelt", ...)` constructs the real Warp env, so the file needs the FULL marker set `[gpu, warp, go2]`. (The existing `tests/test_env_bundle.py` is `[gpu]`-only — that's an existing contract violation in the codebase; we do not copy it.)

- [ ] **Step 1: Write test**

```python
"""Bundle test for SplitbeltTreadmill env (lesson §7.5)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]


@pytest.fixture
def cfg():
    # TrainConfig lives in jax_rl.configs.train_config (jax_rl/configs/train_config.py:8),
    # NOT jax_rl.training.train_config — verify by `grep -n 'class TrainConfig' jax_rl/`.
    from jax_rl.configs.train_config import TrainConfig
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
    assert bundle.num_envs == 4   # mjx doesn't cap; should equal cfg.num_envs


def test_make_env_bundle_obs_schema_matches_name_layout(cfg):
    """Locks in deploy-side contract: schema_from_obs_groups output names ==
    obs_term_names() output names. Catches IncludeGroup expansion drift."""
    from jax_rl.training import make_env_bundle
    from jax_rl.envs.obs_spec import schema_from_obs_groups
    from jax_rl.envs.locomotion.go2_warp_splitbelt import obs_term_names
    bundle = make_env_bundle(cfg, seed=0)
    # bundle.env exposes the inner env's _obs_groups via the wrapper chain.
    inner = bundle.env
    while hasattr(inner, "env"):
        inner = inner.env
    schema = schema_from_obs_groups(inner._obs_groups)
    layout = obs_term_names(inner._config.obs_mode)
    assert schema["state"] == layout["state"]
    # IncludeGroup expansion: privileged should be state names + privileged-only.
    state_set = set(layout["state"])
    expected_priv = layout["state"] + [n for n in layout["privileged_state"] if n not in state_set]
    assert schema["privileged_state"] == expected_priv
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

Per lesson §7.2. Mirrors `tests/test_deploy_e2e.py::test_env_metadata_matches_deploy_constants` ([test_deploy_e2e.py:130-165](../../tests/test_deploy_e2e.py#L130-L165)) — same assertions, env swapped.

Real `deploy/go2_constants.py` exports `DEFAULT_POSE_POLICY`, `DEFAULT_POSE_SDK`, `POLICY_TO_SDK`, `SDK_TO_POLICY`, `ACTION_SCALE` (and `KP_WARP`/`KD_WARP`/etc.). There is **no** `KP`/`KD`/`JOINT_ORDER` symbol — those were guesses in the v1 plan.

- [ ] **Step 1: Write test**

```python
"""Control metadata test for SplitbeltTreadmill env (lesson §7.2).

Catches drift between env XML keyframe / scale and deploy/go2_constants.py —
the same drift class that bricked the deploy stack 2026-04-10 → 2026-04-24.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2, pytest.mark.deploy]


def test_splitbelt_metadata_matches_deploy_constants():
    pytest.importorskip("mujoco")
    pytest.importorskip("warp")

    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    from deploy.go2_constants import (
        DEFAULT_POSE_POLICY, DEFAULT_POSE_SDK, POLICY_TO_SDK, SDK_TO_POLICY,
        ACTION_SCALE,
    )

    env = Go2WarpSplitbeltEnv()
    m = env.get_control_metadata()

    np.testing.assert_array_almost_equal(
        m["default_pose_policy"], DEFAULT_POSE_POLICY,
        err_msg="splitbelt_spawn keyframe ≠ deploy/go2_constants.DEFAULT_POSE_POLICY",
    )
    np.testing.assert_array_almost_equal(
        m["default_pose_sdk"], DEFAULT_POSE_SDK,
        err_msg="env-derived default_pose_sdk drifted from constants",
    )
    np.testing.assert_array_equal(m["policy_to_sdk"], POLICY_TO_SDK)
    np.testing.assert_array_equal(m["sdk_to_policy"], SDK_TO_POLICY)
    assert np.isclose(m["action_scale"], ACTION_SCALE), (
        f"action_scale drift: env={m['action_scale']} vs constants={ACTION_SCALE}"
    )
```

> **Implementer note:** `get_control_metadata` is overridden on `Go2WarpSplitbeltEnv` in Task 3.3 Step 1.5 (because base class shape-checks `nu=14` against `_default_pose` len-12). This test exercises that override. If the override is missing, the test will fail with `AssertionError: shape (14,) != (12,)` — fix in 3.3, not here.

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

### Task 4.4: Base PPO + FastSAC presets (S§5.6, in-scope)

**Files:**
- Modify: `jax_rl/configs/env_presets.py`

Spec §5.6 requires base presets; only the per-protocol variants (A1-A4) are deferred (§11.3). Without these, no smoke train can be invoked by the standard CLI.

- [ ] **Step 1: Read existing pattern**

```bash
grep -n "Go2WarpJoystickFlat\|_FAST_SAC_BASE_CFG\|_PPO_BASE_CFG" jax_rl/configs/env_presets.py | head
```

Note: each env gets a `(TrainConfig, AlgoConfig)` pair per algo, registered into the dispatch dict.

- [ ] **Step 2: Add splitbelt presets**

Add entries cloning the joystick presets, switching `env_name` and adjusting `episode_length=1250`. Same DR / reward / num_envs as joystick. Concrete example (verify shape against joystick precedent):

```python
# In env_presets.py, near the joystick presets:
def _splitbelt_fast_sac():
    train_cfg, algo_cfg = _go2_warp_joystick_fast_sac()
    train_cfg = dataclasses.replace(train_cfg, env_name="Go2WarpSplitbelt", episode_length=1250)
    return train_cfg, algo_cfg

def _splitbelt_ppo_fast():
    train_cfg, algo_cfg = _go2_warp_joystick_ppo_fast()
    train_cfg = dataclasses.replace(train_cfg, env_name="Go2WarpSplitbelt", episode_length=1250)
    return train_cfg, algo_cfg

PRESETS["Go2WarpSplitbelt", "fast_sac"] = _splitbelt_fast_sac
PRESETS["Go2WarpSplitbelt", "ppo_fast"] = _splitbelt_ppo_fast
```

- [ ] **Step 3: Smoke-load presets**

```bash
JAX_PLATFORMS=cpu uv run python -c "
from jax_rl.configs.env_presets import PRESETS
print('fast_sac:', PRESETS[('Go2WarpSplitbelt', 'fast_sac')]())
print('ppo_fast:', PRESETS[('Go2WarpSplitbelt', 'ppo_fast')]())
"
```

Expected: prints both `(TrainConfig, AlgoConfig)` tuples without error.

- [ ] **Step 4: Add hermetic preset test**

Modify `tests/test_env_presets.py` (or create one if missing). Pattern: lookup the preset, assert the returned `(TrainConfig, AlgoConfig)` has the right env_name + episode_length. Hermetic — no env construction.

```python
# In tests/test_env_presets.py, alongside existing joystick preset tests:
def test_splitbelt_fast_sac_preset():
    from jax_rl.configs.env_presets import PRESETS
    train_cfg, algo_cfg = PRESETS[("Go2WarpSplitbelt", "fast_sac")]()
    assert train_cfg.env_name == "Go2WarpSplitbelt"
    assert train_cfg.episode_length == 1250

def test_splitbelt_ppo_fast_preset():
    from jax_rl.configs.env_presets import PRESETS
    train_cfg, algo_cfg = PRESETS[("Go2WarpSplitbelt", "ppo_fast")]()
    assert train_cfg.env_name == "Go2WarpSplitbelt"
    assert train_cfg.episode_length == 1250
```

Run:

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_env_presets.py -k splitbelt
```

Expected: 2 tests pass.

- [ ] **Step 5: Regenerate preset docs (per CLAUDE.md cross-reference table)**

```bash
uv run python docs/scripts/gen_env_presets.py
```

- [ ] **Step 6: Commit**

```bash
git add jax_rl/configs/env_presets.py docs/api/env_presets.md tests/test_env_presets.py
git commit -m "feat(splitbelt): base PPO + FastSAC presets (S§5.6)"
```

---

### Task 4.5: Wire `splitbelt_traj.npz` sidecar in `record_video.py` (S§9.2)

**Files:**
- Modify: `scripts/record_video.py`

Spec §9.2: `record_video.py` writes `splitbelt_traj.npz` whenever `bundle.env_state.info` contains a `splitbelt` key. Without this, the offline analyzer (Tasks 1.8/1.9) has no input.

- [ ] **Step 1: Read existing collector pattern**

`record_video.py` collects per-step lists explicitly (e.g. `qpos_hist`, `cmd_hist`, `reward_components_hist`) by appending in the rollout loop. It does NOT keep a generic `info` dict per step. We must add a splitbelt-specific collector alongside the existing ones.

```bash
grep -n "_hist\|np.savez\|_traj.npz" scripts/record_video.py | head -30
```

Identify (a) the rollout-loop body where `qpos_hist.append(...)` etc. live, (b) the `np.savez` call that writes `_traj.npz`.

- [ ] **Step 2: Add a splitbelt collector loop**

Inside the rollout loop, alongside `qpos_hist.append(state.qpos)` and friends, gate on the splitbelt info key:

```python
# After the existing per-step appends in the rollout loop:
if "splitbelt" in state.info:
    if "splitbelt_hist" not in locals():
        splitbelt_hist = {k: [] for k in state.info["splitbelt"].keys()}
        splitbelt_hist["belt_schedule_at_step"] = []
    for k, v in state.info["splitbelt"].items():
        splitbelt_hist[k].append(np.asarray(v))
    splitbelt_hist["belt_schedule_at_step"].append(
        np.asarray(state.info["belt_schedule"][state.info["step_idx"]])
    )
```

After the rollout, alongside the existing `np.savez(out_dir / "_traj.npz", ...)`:

```python
if "splitbelt_hist" in locals():
    splitbelt_arrays = {k: np.stack(v) for k, v in splitbelt_hist.items()}
    splitbelt_arrays["belt_schedule"] = np.asarray(initial_state.info["belt_schedule"])
    np.savez(out_dir / "splitbelt_traj.npz", **splitbelt_arrays)
```

The conditional means non-splitbelt envs auto-skip (no key, no file).

- [ ] **Step 3: Smoke check by recording a 50-step rollout (manual, post Task 5.1 ckpt)**

```bash
MUJOCO_GL=egl uv run python scripts/record_video.py --checkpoint checkpoints/<latest_splitbelt_ckpt>/best --n-steps 50
ls checkpoints/<latest_splitbelt_ckpt>/best/splitbelt_traj.npz
```

- [ ] **Step 4: Verify offline analyzer can load the sidecar**

```bash
JAX_PLATFORMS=cpu uv run python -c "
import numpy as np
from jax_rl.envs.locomotion import splitbelt_analysis as sba
data = np.load('checkpoints/<dir>/best/splitbelt_traj.npz')
events = sba.detect_step_events(data['foot_in_contact'])
print('events:', events)
"
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(splitbelt): emit splitbelt_traj.npz sidecar from record_video"
```

---

## Stage 5: Validation + docs

### Task 5.1: Calibration smoke train — FastSAC + PPO (manual, post-implementation)

This is not a pytest test — it's a sanity training run that proves the env (a) is learnable, (b) doesn't bake in algo-specific failure modes (project rule: env must work cleanly across algos).

Per the audit: training only on FastSAC misses PPO entropy-collapse risk (joystick env had documented PPO-collapse precedent). We launch BOTH algos at modest budget; both must pass go/no-go before claiming the env works.

**Files:** none (logs only)

- [ ] **Step 1a: Tied-belt FastSAC smoke (1M steps)**

```bash
uv run python scripts/train_fast_sac.py \
    --env Go2WarpSplitbelt \
    --num-envs 1024 \
    --total-timesteps 1000000 \
    --reset-mode per_step \
    > /tmp/splitbelt_fastsac_smoke.log 2>&1 &
```

Expect ~30-60 min on 16 GB GPU. Go/no-go: end-of-run eval > 80 (proves env is learnable, not the calibration target — that's reserved for the longer 5M run).

- [ ] **Step 1b: Tied-belt PPO smoke (1M steps, parallel — wait for FastSAC to finish if GPU is shared)**

```bash
uv run python scripts/train_ppo_fast.py \
    --env Go2WarpSplitbelt \
    --num-envs 1024 \
    --total-timesteps 1000000 \
    > /tmp/splitbelt_ppo_smoke.log 2>&1 &
```

Go/no-go: eval > 80 AND `entropy/mean` log values do NOT collapse below 0.05. If entropy collapses early, the env's reward shape is biased toward exploit gait — flag for follow-up reward tuning before proceeding.

- [ ] **Step 2: Periodic checks**

```bash
grep "EVAL" /tmp/splitbelt_fastsac_smoke.log | tail -10
grep "EVAL\|entropy" /tmp/splitbelt_ppo_smoke.log | tail -10
# Per spec §11.X caveat: confirm treadmill_drift isn't drowning the smoothness terms.
# `pose` and `feet_air_time` should average above 0.01/step — if they stay near
# zero throughout, w_lat is over-tuned.
grep "reward/pose\|reward/feet_air_time\|reward/treadmill_drift" /tmp/splitbelt_fastsac_smoke.log | tail -20
```

Both algos must pass go/no-go thresholds. Compare to joystick env at the same step count for sanity (per-step mean reward, NOT raw eval — splitbelt episode_length=1250 vs joystick 1000, so raw returns scale differently).

If FastSAC passes but PPO fails entropy-collapse, document in journal — env is acceptable but with an algo-specific caveat.

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
### Splitbelt env tied-belt smoke (Task 5.1)
- FastSAC: 1M, num_envs=1024, schedule=tied(0.5). Eval/step: <value>. Joystick same-step: <value>.
- PPO: 1M, num_envs=1024. Eval/step: <value>. entropy/mean (final): <value>. Joystick same-step: <value>.
- Visual (FastSAC ckpt): PASS / FAIL — notes
- Wandb: <ids if logged>
- Verdict: env passes algo-agnosticism gate / NEEDS WORK (reason)
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

Expected: hermetic tests pass; collection lists the 3 gpu-marked files (env_smoke, bundle, control_metadata) — bundle test marker is `[gpu, warp, go2]` (lesson §1).

- [ ] **Step 4: Docs in sync**

```bash
git diff --name-only HEAD~10 HEAD | grep -E '\.(py|md|xml)$'
```

Inspect: every code change has an accompanying doc update where the cross-reference table in CLAUDE.md says it should.

- [ ] **Step 5: Open question for the user — proceed to A1 protocol benchmark or pause here?**

Splitbelt env build is complete. Next milestone (out of scope for this plan): per-protocol algo presets (PPO + FastSAC × A1/A2/A3/A4) and the offline analyzer CLI script (`scripts/analyze_splitbelt.py`) wrapping the analysis library. Either is a fresh brainstorm/spec/plan cycle.

---

## Out-of-scope items (cut intentionally per spec §11)

These were explicitly deferred per spec §11; do **not** add to this plan:

- Per-protocol algo presets (`Go2WarpSplitbelt` × {A1, A2, A3, A4} × {PPO, FastSAC}). Base `(PPO, FastSAC)` × `Go2WarpSplitbelt` presets ARE in scope (Task 4.4).
- Humanoid asset port (R3 step 2).
- Walking-cmd variant (`cmd_x > 0`).
- Composite limp index online (offline only).
- Belt-friction DR spec (off by default; the env reuses joystick DR specs verbatim — no new DR-composition test needed per lesson §7.4).
- `scripts/analyze_splitbelt.py` CLI shim (library exists; CLI when first protocol report is written).
- **Lesson §7.6 (off-policy loop smoke):** N/A — the base PPO + FastSAC presets land in Task 4.4, but the test pattern at `tests/_loop_helpers.py` is for stub-env smoke against the shared `run_offpolicy_loop`. Defer until per-protocol presets land (then the loop smoke makes sense as a contract test). Out of scope for this PR.

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
- `tests/test_splitbelt_bundle.py` — `[gpu, warp, go2]`
- `tests/test_splitbelt_control_metadata.py` — `[gpu, warp, go2, deploy]`

**Modified:**
- `jax_rl/training/env_backends/mjx_backend.py` — register Go2WarpSplitbelt (Task 4.1)
- `jax_rl/configs/env_presets.py` — base PPO + FastSAC presets (Task 4.4)
- `scripts/record_video.py` — splitbelt_traj.npz sidecar emission (Task 4.5)
- `docs/api/env_presets.md` — auto-regenerated from `gen_env_presets.py` (Task 4.4)
- `.context/AGENT_HANDOFF.md` — env list (Task 5.2)
- `.context/TODO.md` — completion + new entries (Task 5.2)
- `.context/journals/2026-05-02.md` — calibration smoke result (Task 5.1)

**Bundle test marker:** `[gpu, warp, go2]` (corrected from earlier `[gpu]` per lesson §1 — `make_env_bundle("Go2WarpSplitbelt")` constructs the Warp env so the file needs the full set).

**Total commits:** ~30 (one per task step), each 2-5 minute units.
