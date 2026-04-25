# Go2 Terrain Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a terrain curriculum to the Go2 Warp env — 4 terrain types on a 10×4 tile grid, per-env spawn+goal, goal-directed commands computed from world-frame goals, binary reach-based curriculum advancement, wandb per-terrain metrics.

**Architecture:** Procedural terrain generation produces a single composite MJCF with 40 tiles (10 difficulty rows × 4 terrain types). A new env `Go2WarpJoystickCurriculum` subclasses `WarpJoystick`, owns per-env `terrain_level` + `terrain_type` + `goal_xy` state. Each env has a world-frame goal point per episode; body-frame `(vx, vy, yaw_rate)` command is computed each step from goal + robot pose (P-controller on yaw, forward `vx`). Episodes don't terminate on reach — `episode_reached_goal` flag tracks whether robot got within `goal_radius` at any step. Curriculum advancement is binary: reached+not-fallen→promote, fallen→demote, timeout-with-no-progress→demote. Torque-speed actuator model is a separate registered variant (`...CurriculumTorqueSpeed`).

**Tech Stack:** JAX, MuJoCo Warp, MuJoCo Playground, Flax, Optax. MJCF assembled via XML string composition against a Go2 scene template. Body-frame command recomputation each step via rotation from world-frame goal vector.

---

## File Structure

New module: `jax_rl/envs/terrains/`

| File | Responsibility |
|------|---------------|
| `jax_rl/envs/terrains/__init__.py` | Exports public API |
| `jax_rl/envs/terrains/base.py` | `SubTerrainCfg` ABC + `TerrainOutput` dataclass |
| `jax_rl/envs/terrains/primitives.py` | 8 concrete terrain type generators |
| `jax_rl/envs/terrains/generator.py` | `TerrainGenerator` composes grid → MJCF string |
| `jax_rl/envs/terrains/config.py` | `TerrainGridCfg` dataclass + default configs |

Modified/new in env:

| File | Responsibility |
|------|---------------|
| `jax_rl/envs/locomotion/go2_warp_curriculum.py` (NEW) | `Go2WarpJoystickCurriculum` env — subclasses `WarpJoystick`, injects terrain MJCF, manages terrain state in `state.info` |
| `jax_rl/envs/locomotion/xmls/go2_warp_curriculum_scene_template.xml` (NEW) | Base scene template (robot + compiler options) — terrain injected at env init |
| `jax_rl/envs/wrappers/terrain_dr.py` (NEW) | `TerrainDomainRandWrapper` — DomainRandWrapper + terrain advancement logic integrated |
| `jax_rl/envs/wrappers/__init__.py` | Re-export |
| `jax_rl/training/env_setup.py` | Register `Go2WarpJoystickCurriculum` + `Go2WarpJoystickCurriculumTorqueSpeed` |
| `jax_rl/training/metrics_logger.py` | Add `log_terrain_metrics()` helper |
| `jax_rl/configs/env_presets.py` | PPO + FastSAC + FlashSAC presets |
| `tools/generate_terrain_preview.py` (NEW) | Standalone script: MJCF preview for Phase 1 deliverable |

Tests:

| File | Covers |
|------|--------|
| `tests/test_terrain_primitives.py` (NEW) | Each of 8 terrain types generates valid geoms |
| `tests/test_terrain_generator.py` (NEW) | Grid builder produces valid MJCF, correct origins |
| `tests/test_go2_warp_curriculum_env.py` (NEW) | Env loads, spawns correctly, advancement logic |
| `tests/test_terrain_dr_wrapper.py` (NEW) | Wrapper merges DR + terrain state correctly |

---

## Critical Design Decisions (locked)

- **Menu:** 4 terrain types: `RoughTerrainCfg`, `PyramidStairsTerrainCfg` (up), `InvertedPyramidStairsTerrainCfg` (bowl), `TiltedGridTerrainCfg`. Flat dropped (redundant — row 0 of every type is already flat at difficulty=0). Slope dropped (curriculum-varying tilt can't match row boundaries cleanly). DiscreteObstacles and SteppingStones omitted (navigation/stress-test, not deployment-relevant).
- **Grid:** 10 rows × 4 cols = 40 tiles; tile 9.6×9.6m (20% bigger than legged_gym 8m default, reduces boundary crossing); border 20m (4 strips AROUND grid, not a covering floor)
- **Aggressive grading per type:**
    - `RoughTerrainCfg(max_height=0.22)`
    - `PyramidStairsTerrainCfg(max_step_height=0.4)`
    - `InvertedPyramidStairsTerrainCfg(max_step_height=0.4)` — bowl shape: rim at z=0, descends to pit at center
    - `TiltedGridTerrainCfg(max_tilt_deg=25.0)` — with solid `base_depth=0.3` underneath to prevent fall-through gaps
- **Column assignment:** fixed per env (`env_id % num_cols`), specialization pattern from legged_gym
- **Command scheme:** goal-directed, body-frame computed from world-frame goal
    - At reset: sample `spawn_xy`, `goal_xy` per-terrain-type + random `yaw`; compute `initial_distance`
    - Rough / TiltedGrid: spawn on one edge of tile, goal on opposite edge
    - PyramidStairs / InvertedPyramidStairs: spawn on outer rim, goal at center
    - At each step: `cmd_vx = target_speed`, `cmd_vy = 0`, `cmd_yaw_rate = clamp(k_yaw * wrap_angle(atan2(dy, dx) - robot_yaw), ±max_yaw_rate)`
    - Target speed scales with level: `target_speed = 0.5 + (level / (num_rows - 1)) * 1.0` → 0.5 m/s (level 0) to 1.5 m/s (level 9)
    - Goal radius: 0.5m (reach criterion)
    - `k_yaw = 2.0`, `max_yaw_rate = 1.5 rad/s`
    - Episode does NOT terminate on reach — just flips `episode_reached_goal = True` in `state.info`
- **Curriculum advancement (at episode reset):**
    - `reached_goal AND NOT fallen` → promote (`level += 1`)
    - `fallen` → demote (`level -= 1`)
    - timeout AND NOT reached AND `episode_min_distance > 0.5 * initial_distance` (didn't get close) → demote
    - otherwise → stay
    - Level clamped to `[0, num_rows - 1]`
- **Initial level:** all envs start at row 0
- **Torque-speed:** separate registered env variant, not forced on
- **DR:** on (per_step, 8 specs — same as Go2WarpJoystickFlat)
- **Push forces:** deferred to separate future plan
- **Logging:** snapshot per training iter (not per episode-end). wandb metrics per terrain type: `mean_level`, `max_level`, `reach_rate`, `fall_rate`, `promote_rate`, `demote_rate`. Global: `global_mean_level`, `global_reach_rate`, `global_fall_rate`.
- **Delivery:** phased — terrain gen → env → advancement → polish, review between

---

## Phase 1 — Terrain Generation Module  **[COMPLETE as of 2026-04-16]**

**Goal:** Standalone module generates a grid MJCF. Zero env/training integration.

**Deliverable:** `tools/generate_terrain_preview.py` writes MJCF that loads in MuJoCo (1504 geoms, 40 tiles). `tools/render_terrain_screenshots.py` renders headless previews to `.temp/` (topdown + iso, with Go2-scale red reference boxes on every tile).

**Status:** 42 tests pass. User iterated with implementer on terrain visual quality — landed on 4-type menu, 9.6m tiles, bowl-shaped inverted pyramid, solid bases under tilted_grid, 4-strip border (not covering floor), aggressive grading values documented above.

**Note for future agents:** If you're reading this plan to implement Phases 2-4, Phase 1 is done. Skip to Phase 2.

### Task 1.1: Create `base.py` — SubTerrainCfg + TerrainOutput

**Files:**
- Create: `jax_rl/envs/terrains/__init__.py`
- Create: `jax_rl/envs/terrains/base.py`
- Test: `tests/test_terrain_primitives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_primitives.py
import numpy as np
from jax_rl.envs.terrains.base import TerrainOutput

def test_terrain_output_structure():
    """TerrainOutput holds a list of geom dicts + a spawn origin."""
    out = TerrainOutput(
        geoms=[{"type": "box", "size": (1, 1, 0.1), "pos": (0, 0, 0.05), "rgba": (0.5, 0.5, 0.5, 1)}],
        spawn_origin=np.array([0.0, 0.0, 0.3]),
    )
    assert len(out.geoms) == 1
    assert out.geoms[0]["type"] == "box"
    assert out.spawn_origin.shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_primitives.py::test_terrain_output_structure -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'jax_rl.envs.terrains'`

- [ ] **Step 3: Create package init**

```python
# jax_rl/envs/terrains/__init__.py
from jax_rl.envs.terrains.base import SubTerrainCfg, TerrainOutput
```

- [ ] **Step 4: Create base module**

```python
# jax_rl/envs/terrains/base.py
"""Base types for procedural terrain generation."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TerrainOutput:
    """Output of a SubTerrainCfg.generate() call.

    geoms: list of dicts, each describing one MJCF <geom>. Keys include
      "type" (box/plane/sphere), "size" (tuple), "pos" (tuple, tile-local),
      "quat" (optional tuple), "rgba" (tuple), "name" (optional str),
      "friction" (optional tuple).
    spawn_origin: (3,) tile-local spawn position for the robot on this tile.
    """
    geoms: list[dict]
    spawn_origin: np.ndarray


class SubTerrainCfg(ABC):
    """Abstract config for one terrain type. Subclasses implement generate()."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        difficulty: float,
        size: tuple[float, float],
        rng: np.random.Generator,
    ) -> TerrainOutput:
        """Generate tile geometry at given difficulty [0, 1] of given size."""
        raise NotImplementedError
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run python -m pytest tests/test_terrain_primitives.py::test_terrain_output_structure -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add jax_rl/envs/terrains/__init__.py jax_rl/envs/terrains/base.py tests/test_terrain_primitives.py
git commit -m "feat(terrains): add SubTerrainCfg ABC + TerrainOutput dataclass"
```

---

### Task 1.2: Flat terrain

**Files:**
- Create: `jax_rl/envs/terrains/primitives.py`
- Test: `tests/test_terrain_primitives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_primitives.py (append)
import numpy as np
from jax_rl.envs.terrains.primitives import FlatTerrainCfg

def test_flat_terrain_generates_one_plane():
    cfg = FlatTerrainCfg()
    rng = np.random.default_rng(0)
    out = cfg.generate(difficulty=0.0, size=(8.0, 8.0), rng=rng)
    assert len(out.geoms) == 1
    assert out.geoms[0]["type"] == "box"
    # Flat plane: large xy, thin z
    assert out.geoms[0]["size"][0] == pytest.approx(4.0)  # half-size
    assert out.geoms[0]["size"][1] == pytest.approx(4.0)
    assert out.spawn_origin[2] > 0  # above the plane

def test_flat_terrain_difficulty_ignored():
    cfg = FlatTerrainCfg()
    rng = np.random.default_rng(0)
    out_easy = cfg.generate(difficulty=0.0, size=(8.0, 8.0), rng=rng)
    out_hard = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=rng)
    # Difficulty has no effect on flat
    assert out_easy.geoms[0]["size"] == out_hard.geoms[0]["size"]
```

Add `import pytest` at top of test file if not present.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v
```
Expected: FAIL with `ModuleNotFoundError: ... primitives`

- [ ] **Step 3: Implement FlatTerrainCfg**

```python
# jax_rl/envs/terrains/primitives.py
"""Concrete terrain type generators."""
from dataclasses import dataclass
import numpy as np
from jax_rl.envs.terrains.base import SubTerrainCfg, TerrainOutput


@dataclass
class FlatTerrainCfg(SubTerrainCfg):
    """Level ground. Difficulty ignored."""
    name: str = "flat"
    thickness: float = 0.05  # box z half-size

    def generate(self, difficulty, size, rng) -> TerrainOutput:
        sx, sy = size
        geoms = [{
            "type": "box",
            "name": "flat_ground",
            "size": (sx / 2, sy / 2, self.thickness),
            "pos": (0.0, 0.0, -self.thickness),  # top surface at z=0
            "rgba": (0.5, 0.5, 0.5, 1.0),
            "friction": (0.6, 0.005, 0.0001),
        }]
        spawn = np.array([0.0, 0.0, 0.3])  # robot stands 0.3m above ground
        return TerrainOutput(geoms=geoms, spawn_origin=spawn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/terrains/primitives.py tests/test_terrain_primitives.py
git commit -m "feat(terrains): add FlatTerrainCfg"
```

---

### Task 1.3: Rough terrain

**Files:**
- Modify: `jax_rl/envs/terrains/primitives.py`
- Test: `tests/test_terrain_primitives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_primitives.py (append)
from jax_rl.envs.terrains.primitives import RoughTerrainCfg

def test_rough_terrain_generates_grid_of_boxes():
    cfg = RoughTerrainCfg(grid_size=(8, 8))
    rng = np.random.default_rng(0)
    out = cfg.generate(difficulty=0.5, size=(8.0, 8.0), rng=rng)
    # 8x8 grid = 64 boxes
    assert len(out.geoms) == 64

def test_rough_terrain_height_scales_with_difficulty():
    cfg = RoughTerrainCfg(grid_size=(8, 8), max_height=0.1)
    # Same rng state → deterministic heights
    out_easy = cfg.generate(difficulty=0.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    out_hard = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    # Easy: all boxes at base height (no variation)
    easy_z = np.array([g["pos"][2] for g in out_easy.geoms])
    hard_z = np.array([g["pos"][2] for g in out_hard.geoms])
    assert easy_z.std() < 1e-6  # no height variation at difficulty=0
    assert hard_z.std() > 0.01  # real variation at difficulty=1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_primitives.py::test_rough_terrain_generates_grid_of_boxes -v
```
Expected: FAIL

- [ ] **Step 3: Implement RoughTerrainCfg**

```python
# jax_rl/envs/terrains/primitives.py (append)
@dataclass
class RoughTerrainCfg(SubTerrainCfg):
    """Grid of boxes with per-cell height noise. Difficulty scales height range."""
    name: str = "rough"
    grid_size: tuple[int, int] = (8, 8)  # (rows, cols) of boxes per tile
    max_height: float = 0.1  # peak height at difficulty=1
    base_thickness: float = 0.05

    def generate(self, difficulty, size, rng) -> TerrainOutput:
        sx, sy = size
        nr, nc = self.grid_size
        cell_x, cell_y = sx / nc, sy / nr
        height_range = self.max_height * difficulty
        geoms = []
        for r in range(nr):
            for c in range(nc):
                cx = -sx / 2 + cell_x * (c + 0.5)
                cy = -sy / 2 + cell_y * (r + 0.5)
                dz = rng.uniform(-height_range, height_range) if difficulty > 0 else 0.0
                geoms.append({
                    "type": "box",
                    "size": (cell_x / 2, cell_y / 2, self.base_thickness),
                    "pos": (cx, cy, -self.base_thickness + dz),
                    "rgba": (0.55, 0.5, 0.4, 1.0),
                    "friction": (0.6, 0.005, 0.0001),
                })
        spawn = np.array([0.0, 0.0, 0.3 + self.max_height])
        return TerrainOutput(geoms=geoms, spawn_origin=spawn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/terrains/primitives.py tests/test_terrain_primitives.py
git commit -m "feat(terrains): add RoughTerrainCfg"
```

---

### Task 1.4: Slope terrain

**Files:**
- Modify: `jax_rl/envs/terrains/primitives.py`
- Test: `tests/test_terrain_primitives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_primitives.py (append)
from jax_rl.envs.terrains.primitives import SlopeTerrainCfg

def test_slope_generates_single_tilted_box():
    cfg = SlopeTerrainCfg(max_angle_deg=20.0)
    out = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    assert len(out.geoms) == 1
    # Quat should NOT be identity at difficulty > 0
    quat = out.geoms[0].get("quat", (1, 0, 0, 0))
    assert abs(quat[0] - 1.0) > 1e-3 or any(abs(q) > 1e-3 for q in quat[1:])

def test_slope_zero_difficulty_is_flat():
    cfg = SlopeTerrainCfg(max_angle_deg=20.0)
    out = cfg.generate(difficulty=0.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    quat = out.geoms[0].get("quat", (1, 0, 0, 0))
    # Identity quat
    assert abs(quat[0] - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v -k slope
```
Expected: FAIL

- [ ] **Step 3: Implement SlopeTerrainCfg**

```python
# jax_rl/envs/terrains/primitives.py (append)
@dataclass
class SlopeTerrainCfg(SubTerrainCfg):
    """Single tilted box. Difficulty scales tilt angle."""
    name: str = "slope"
    max_angle_deg: float = 20.0
    thickness: float = 0.05

    def generate(self, difficulty, size, rng) -> TerrainOutput:
        sx, sy = size
        angle = np.deg2rad(self.max_angle_deg * difficulty)
        # Tilt about x-axis (pitch up)
        quat = (np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0)
        # Spawn raised to compensate for tilt
        spawn_z = 0.3 + np.sin(angle) * sx / 4
        geoms = [{
            "type": "box",
            "size": (sx / 2, sy / 2, self.thickness),
            "pos": (0.0, 0.0, -self.thickness),
            "quat": quat,
            "rgba": (0.5, 0.55, 0.6, 1.0),
            "friction": (0.6, 0.005, 0.0001),
        }]
        spawn = np.array([0.0, 0.0, spawn_z])
        return TerrainOutput(geoms=geoms, spawn_origin=spawn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/terrains/primitives.py tests/test_terrain_primitives.py
git commit -m "feat(terrains): add SlopeTerrainCfg"
```

---

### Task 1.5: Stairs (pyramid, up)

**Files:**
- Modify: `jax_rl/envs/terrains/primitives.py`
- Test: `tests/test_terrain_primitives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_primitives.py (append)
from jax_rl.envs.terrains.primitives import PyramidStairsTerrainCfg

def test_pyramid_stairs_generates_multiple_steps():
    cfg = PyramidStairsTerrainCfg(num_steps=5)
    out = cfg.generate(difficulty=0.5, size=(8.0, 8.0), rng=np.random.default_rng(0))
    # 5 concentric rings of step boxes + 1 center platform
    assert len(out.geoms) >= 5

def test_pyramid_stairs_step_height_scales_with_difficulty():
    cfg = PyramidStairsTerrainCfg(num_steps=5, max_step_height=0.2)
    out_easy = cfg.generate(difficulty=0.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    out_hard = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    # At difficulty 0, all steps at same z (flat). At 1, staircased.
    easy_z = [g["pos"][2] for g in out_easy.geoms]
    hard_z = [g["pos"][2] for g in out_hard.geoms]
    assert max(easy_z) - min(easy_z) < 1e-6
    assert max(hard_z) - min(hard_z) > 0.1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v -k stairs
```
Expected: FAIL

- [ ] **Step 3: Implement PyramidStairsTerrainCfg**

```python
# jax_rl/envs/terrains/primitives.py (append)
@dataclass
class PyramidStairsTerrainCfg(SubTerrainCfg):
    """Concentric square steps rising to a central platform. Robot spawns on outer ring."""
    name: str = "stairs_up"
    num_steps: int = 5
    max_step_height: float = 0.2  # height per step at difficulty=1
    platform_width: float = 1.0
    base_thickness: float = 0.05

    def generate(self, difficulty, size, rng) -> TerrainOutput:
        sx, sy = size
        step_h = self.max_step_height * difficulty
        step_w_x = (sx - self.platform_width) / (2 * self.num_steps)
        step_w_y = (sy - self.platform_width) / (2 * self.num_steps)
        geoms = []
        for i in range(self.num_steps):
            # Each "step" is a hollow square ring made of 4 boxes (N/S/E/W)
            inner_x = sx / 2 - (i + 1) * step_w_x
            inner_y = sy / 2 - (i + 1) * step_w_y
            outer_x = sx / 2 - i * step_w_x
            outer_y = sy / 2 - i * step_w_y
            z = step_h * i
            # North strip
            geoms.append({"type": "box", "size": (outer_x, step_w_y / 2, self.base_thickness + z / 2),
                          "pos": (0, (inner_y + outer_y) / 2, z / 2 - self.base_thickness),
                          "rgba": (0.4, 0.5, 0.6, 1.0), "friction": (0.6, 0.005, 0.0001)})
            # South strip
            geoms.append({"type": "box", "size": (outer_x, step_w_y / 2, self.base_thickness + z / 2),
                          "pos": (0, -(inner_y + outer_y) / 2, z / 2 - self.base_thickness),
                          "rgba": (0.4, 0.5, 0.6, 1.0), "friction": (0.6, 0.005, 0.0001)})
            # East strip
            geoms.append({"type": "box", "size": (step_w_x / 2, inner_y, self.base_thickness + z / 2),
                          "pos": ((inner_x + outer_x) / 2, 0, z / 2 - self.base_thickness),
                          "rgba": (0.4, 0.5, 0.6, 1.0), "friction": (0.6, 0.005, 0.0001)})
            # West strip
            geoms.append({"type": "box", "size": (step_w_x / 2, inner_y, self.base_thickness + z / 2),
                          "pos": (-(inner_x + outer_x) / 2, 0, z / 2 - self.base_thickness),
                          "rgba": (0.4, 0.5, 0.6, 1.0), "friction": (0.6, 0.005, 0.0001)})
        # Central platform
        top_z = step_h * self.num_steps
        geoms.append({"type": "box",
                      "size": (self.platform_width / 2, self.platform_width / 2, self.base_thickness + top_z / 2),
                      "pos": (0, 0, top_z / 2 - self.base_thickness),
                      "rgba": (0.3, 0.4, 0.5, 1.0), "friction": (0.6, 0.005, 0.0001)})
        # Robot spawns at the outer edge on the ground level
        spawn = np.array([sx / 2 - step_w_x / 2, 0.0, 0.3])
        return TerrainOutput(geoms=geoms, spawn_origin=spawn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/terrains/primitives.py tests/test_terrain_primitives.py
git commit -m "feat(terrains): add PyramidStairsTerrainCfg (stairs up)"
```

---

### Task 1.6: Inverted Stairs (pyramid, down)

**Files:**
- Modify: `jax_rl/envs/terrains/primitives.py`
- Test: `tests/test_terrain_primitives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_primitives.py (append)
from jax_rl.envs.terrains.primitives import InvertedPyramidStairsTerrainCfg

def test_inverted_stairs_spawns_at_center_top():
    cfg = InvertedPyramidStairsTerrainCfg(num_steps=5, max_step_height=0.2)
    out = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    # Robot spawns on center platform (above everything)
    assert out.spawn_origin[0] == pytest.approx(0.0)
    assert out.spawn_origin[1] == pytest.approx(0.0)
    # Descending steps: outer ring should be lower than center
    z_positions = [g["pos"][2] for g in out.geoms]
    assert min(z_positions) < 0  # steps go below origin
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v -k inverted
```
Expected: FAIL

- [ ] **Step 3: Implement InvertedPyramidStairsTerrainCfg**

```python
# jax_rl/envs/terrains/primitives.py (append)
@dataclass
class InvertedPyramidStairsTerrainCfg(SubTerrainCfg):
    """Center platform with steps descending outward into a bowl.

    Robot spawns on central top platform, must walk down stairs outward.
    """
    name: str = "stairs_down"
    num_steps: int = 5
    max_step_height: float = 0.2
    platform_width: float = 1.0
    base_thickness: float = 0.5  # thicker base — steps descend below it

    def generate(self, difficulty, size, rng) -> TerrainOutput:
        sx, sy = size
        step_h = self.max_step_height * difficulty
        step_w_x = (sx - self.platform_width) / (2 * self.num_steps)
        step_w_y = (sy - self.platform_width) / (2 * self.num_steps)
        geoms = []
        # Floor at bottom (to prevent falling through)
        total_descent = step_h * self.num_steps
        geoms.append({"type": "box",
                      "size": (sx / 2, sy / 2, 0.05),
                      "pos": (0, 0, -total_descent - 0.1),
                      "rgba": (0.2, 0.2, 0.2, 1.0)})
        # Central platform (top)
        geoms.append({"type": "box",
                      "size": (self.platform_width / 2, self.platform_width / 2, 0.05),
                      "pos": (0, 0, -0.05),
                      "rgba": (0.3, 0.5, 0.4, 1.0), "friction": (0.6, 0.005, 0.0001)})
        # Descending rings (same 4-strip pattern as up-stairs but z goes down)
        for i in range(self.num_steps):
            inner_x = sx / 2 - (i + 1) * step_w_x
            inner_y = sy / 2 - (i + 1) * step_w_y
            outer_x = sx / 2 - i * step_w_x
            outer_y = sy / 2 - i * step_w_y
            z = -step_h * (i + 1)
            for (cx, cy, hx, hy) in [
                (0, (inner_y + outer_y) / 2, outer_x, step_w_y / 2),
                (0, -(inner_y + outer_y) / 2, outer_x, step_w_y / 2),
                ((inner_x + outer_x) / 2, 0, step_w_x / 2, inner_y),
                (-(inner_x + outer_x) / 2, 0, step_w_x / 2, inner_y),
            ]:
                geoms.append({"type": "box",
                              "size": (hx, hy, 0.05),
                              "pos": (cx, cy, z),
                              "rgba": (0.5, 0.3, 0.3, 1.0), "friction": (0.6, 0.005, 0.0001)})
        # Spawn on central platform
        spawn = np.array([0.0, 0.0, 0.3])
        return TerrainOutput(geoms=geoms, spawn_origin=spawn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/terrains/primitives.py tests/test_terrain_primitives.py
git commit -m "feat(terrains): add InvertedPyramidStairsTerrainCfg (stairs down)"
```

---

### Task 1.7: Discrete Obstacles

**Files:**
- Modify: `jax_rl/envs/terrains/primitives.py`
- Test: `tests/test_terrain_primitives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_primitives.py (append)
from jax_rl.envs.terrains.primitives import DiscreteObstaclesTerrainCfg

def test_obstacles_count_scales_with_difficulty():
    cfg = DiscreteObstaclesTerrainCfg(max_count=20)
    out_easy = cfg.generate(difficulty=0.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    out_hard = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    # Floor + obstacles. difficulty=0 means 0 obstacles, difficulty=1 means max_count
    n_easy = len(out_easy.geoms) - 1  # subtract floor
    n_hard = len(out_hard.geoms) - 1
    assert n_easy == 0
    assert n_hard == 20
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v -k obstacles
```
Expected: FAIL

- [ ] **Step 3: Implement DiscreteObstaclesTerrainCfg**

```python
# jax_rl/envs/terrains/primitives.py (append)
@dataclass
class DiscreteObstaclesTerrainCfg(SubTerrainCfg):
    """Flat floor with randomly placed box obstacles. Count and size scale with difficulty."""
    name: str = "obstacles"
    max_count: int = 20
    min_size: float = 0.1
    max_size: float = 0.4
    obstacle_height: float = 0.2
    spawn_clearance: float = 1.5  # no obstacles within this radius of spawn

    def generate(self, difficulty, size, rng) -> TerrainOutput:
        sx, sy = size
        count = int(self.max_count * difficulty)
        geoms = [{
            "type": "box",
            "size": (sx / 2, sy / 2, 0.05),
            "pos": (0, 0, -0.05),
            "rgba": (0.5, 0.5, 0.5, 1.0),
            "friction": (0.6, 0.005, 0.0001),
        }]
        placed = 0
        attempts = 0
        while placed < count and attempts < count * 10:
            attempts += 1
            cx = rng.uniform(-sx / 2 + self.max_size, sx / 2 - self.max_size)
            cy = rng.uniform(-sy / 2 + self.max_size, sy / 2 - self.max_size)
            if abs(cx) < self.spawn_clearance and abs(cy) < self.spawn_clearance:
                continue
            box_size = rng.uniform(self.min_size, self.max_size)
            geoms.append({
                "type": "box",
                "size": (box_size, box_size, self.obstacle_height / 2),
                "pos": (cx, cy, self.obstacle_height / 2),
                "rgba": (0.7, 0.3, 0.2, 1.0),
                "friction": (0.6, 0.005, 0.0001),
            })
            placed += 1
        spawn = np.array([0.0, 0.0, 0.3])
        return TerrainOutput(geoms=geoms, spawn_origin=spawn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/terrains/primitives.py tests/test_terrain_primitives.py
git commit -m "feat(terrains): add DiscreteObstaclesTerrainCfg"
```

---

### Task 1.8: Stepping Stones

**Files:**
- Modify: `jax_rl/envs/terrains/primitives.py`
- Test: `tests/test_terrain_primitives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_primitives.py (append)
from jax_rl.envs.terrains.primitives import SteppingStonesTerrainCfg

def test_stepping_stones_generates_grid_of_raised_boxes():
    cfg = SteppingStonesTerrainCfg(stone_count=(8, 8))
    out = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    # 64 stones + 1 pit floor
    assert len(out.geoms) >= 65

def test_stepping_stones_gap_scales_with_difficulty():
    cfg = SteppingStonesTerrainCfg(stone_count=(8, 8), max_gap=0.3, max_height_variation=0.1)
    out_easy = cfg.generate(difficulty=0.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    out_hard = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    # Harder = smaller stones (to create gaps)
    stone_sizes_easy = [g["size"][0] for g in out_easy.geoms if g.get("rgba", (0,))[0] > 0.6]
    stone_sizes_hard = [g["size"][0] for g in out_hard.geoms if g.get("rgba", (0,))[0] > 0.6]
    assert np.mean(stone_sizes_hard) < np.mean(stone_sizes_easy)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v -k stepping
```
Expected: FAIL

- [ ] **Step 3: Implement SteppingStonesTerrainCfg**

```python
# jax_rl/envs/terrains/primitives.py (append)
@dataclass
class SteppingStonesTerrainCfg(SubTerrainCfg):
    """Raised stones over a pit. Gaps + height variation scale with difficulty."""
    name: str = "stepping_stones"
    stone_count: tuple[int, int] = (8, 8)  # rows, cols
    max_gap: float = 0.3  # fraction of cell size
    max_height_variation: float = 0.08
    stone_height: float = 0.1
    pit_depth: float = 0.5

    def generate(self, difficulty, size, rng) -> TerrainOutput:
        sx, sy = size
        nr, nc = self.stone_count
        cell_x, cell_y = sx / nc, sy / nr
        gap = self.max_gap * difficulty
        stone_half_x = cell_x * (1 - gap) / 2
        stone_half_y = cell_y * (1 - gap) / 2
        h_var = self.max_height_variation * difficulty
        # Pit floor
        geoms = [{
            "type": "box",
            "size": (sx / 2, sy / 2, 0.05),
            "pos": (0, 0, -self.pit_depth),
            "rgba": (0.2, 0.2, 0.2, 1.0),
        }]
        # Stones
        for r in range(nr):
            for c in range(nc):
                cx = -sx / 2 + cell_x * (c + 0.5)
                cy = -sy / 2 + cell_y * (r + 0.5)
                dz = rng.uniform(-h_var, h_var) if difficulty > 0 else 0.0
                geoms.append({
                    "type": "box",
                    "size": (stone_half_x, stone_half_y, self.stone_height / 2),
                    "pos": (cx, cy, self.stone_height / 2 + dz),
                    "rgba": (0.7, 0.65, 0.55, 1.0),
                    "friction": (0.6, 0.005, 0.0001),
                })
        spawn = np.array([0.0, 0.0, 0.3 + self.stone_height])
        return TerrainOutput(geoms=geoms, spawn_origin=spawn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/terrains/primitives.py tests/test_terrain_primitives.py
git commit -m "feat(terrains): add SteppingStonesTerrainCfg"
```

---

### Task 1.9: Tilted Grid

**Files:**
- Modify: `jax_rl/envs/terrains/primitives.py`
- Test: `tests/test_terrain_primitives.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_primitives.py (append)
from jax_rl.envs.terrains.primitives import TiltedGridTerrainCfg

def test_tilted_grid_generates_grid_of_rotated_boxes():
    cfg = TiltedGridTerrainCfg(grid_size=(6, 6), max_tilt_deg=15.0)
    out = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    # 36 grid cells + pit floor + central platform
    assert len(out.geoms) >= 36

def test_tilted_grid_tilt_scales_with_difficulty():
    cfg = TiltedGridTerrainCfg(grid_size=(4, 4), max_tilt_deg=15.0)
    out_easy = cfg.generate(difficulty=0.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    out_hard = cfg.generate(difficulty=1.0, size=(8.0, 8.0), rng=np.random.default_rng(0))
    # Count non-identity quats
    def non_ident(geoms):
        count = 0
        for g in geoms:
            q = g.get("quat", (1, 0, 0, 0))
            if abs(q[0] - 1.0) > 1e-3:
                count += 1
        return count
    # At difficulty=0, no tilt; at difficulty=1, many tilted
    assert non_ident(out_easy.geoms) == 0
    assert non_ident(out_hard.geoms) > 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v -k tilted
```
Expected: FAIL

- [ ] **Step 3: Implement TiltedGridTerrainCfg**

```python
# jax_rl/envs/terrains/primitives.py (append)
@dataclass
class TiltedGridTerrainCfg(SubTerrainCfg):
    """Grid of tiles, each tilted at a random angle. Central spawn platform."""
    name: str = "tilted_grid"
    grid_size: tuple[int, int] = (6, 6)
    max_tilt_deg: float = 15.0
    platform_width: float = 1.0
    cell_thickness: float = 0.05

    def generate(self, difficulty, size, rng) -> TerrainOutput:
        sx, sy = size
        nr, nc = self.grid_size
        cell_x, cell_y = sx / nc, sy / nr
        tilt_max = np.deg2rad(self.max_tilt_deg * difficulty)
        geoms = []
        # Pit floor below
        geoms.append({"type": "box", "size": (sx / 2, sy / 2, 0.05),
                      "pos": (0, 0, -0.5), "rgba": (0.2, 0.2, 0.2, 1.0)})
        # Central platform
        geoms.append({"type": "box",
                      "size": (self.platform_width / 2, self.platform_width / 2, self.cell_thickness),
                      "pos": (0, 0, -self.cell_thickness),
                      "rgba": (0.3, 0.5, 0.4, 1.0), "friction": (0.6, 0.005, 0.0001)})
        # Tilted tiles
        for r in range(nr):
            for c in range(nc):
                cx = -sx / 2 + cell_x * (c + 0.5)
                cy = -sy / 2 + cell_y * (r + 0.5)
                # Skip if near central platform
                if abs(cx) < self.platform_width / 2 + cell_x / 2 and abs(cy) < self.platform_width / 2 + cell_y / 2:
                    continue
                ax = rng.uniform(-tilt_max, tilt_max) if difficulty > 0 else 0.0
                ay = rng.uniform(-tilt_max, tilt_max) if difficulty > 0 else 0.0
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_euler("xy", [ax, ay])
                q = tuple(rot.as_quat(scalar_first=True))
                geoms.append({
                    "type": "box",
                    "size": (cell_x / 2 * 0.9, cell_y / 2 * 0.9, self.cell_thickness),
                    "pos": (cx, cy, -self.cell_thickness),
                    "quat": q,
                    "rgba": (0.6, 0.55, 0.45, 1.0),
                    "friction": (0.6, 0.005, 0.0001),
                })
        spawn = np.array([0.0, 0.0, 0.3])
        return TerrainOutput(geoms=geoms, spawn_origin=spawn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_terrain_primitives.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/terrains/primitives.py tests/test_terrain_primitives.py
git commit -m "feat(terrains): add TiltedGridTerrainCfg"
```

---

### Task 1.10: Config + Terrain Generator

**Files:**
- Create: `jax_rl/envs/terrains/config.py`
- Create: `jax_rl/envs/terrains/generator.py`
- Test: `tests/test_terrain_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_terrain_generator.py
import pytest
import numpy as np
from jax_rl.envs.terrains.config import TerrainGridCfg, GO2_DEFAULT_CFG
from jax_rl.envs.terrains.generator import TerrainGenerator


def test_generator_produces_mjcf_string():
    cfg = GO2_DEFAULT_CFG
    gen = TerrainGenerator(cfg)
    mjcf, origins = gen.generate(seed=0)
    assert isinstance(mjcf, str)
    assert "<worldbody>" in mjcf
    assert "</worldbody>" in mjcf
    # 10 rows x 8 cols
    assert origins.shape == (10, 8, 3)


def test_generator_origins_reflect_grid_layout():
    cfg = GO2_DEFAULT_CFG
    gen = TerrainGenerator(cfg)
    _, origins = gen.generate(seed=0)
    # Row 0 tiles should all have the same y (same row)
    assert np.allclose(origins[0, :, 1], origins[0, 0, 1])
    # Column 0 tiles should all have the same x
    assert np.allclose(origins[:, 0, 0], origins[0, 0, 0])


def test_generator_is_seed_reproducible():
    cfg = GO2_DEFAULT_CFG
    gen = TerrainGenerator(cfg)
    mjcf1, _ = gen.generate(seed=42)
    mjcf2, _ = gen.generate(seed=42)
    assert mjcf1 == mjcf2


def test_generated_mjcf_parses_via_mujoco():
    """Smoke test: the generated terrain should load in MuJoCo without error."""
    import mujoco
    cfg = GO2_DEFAULT_CFG
    gen = TerrainGenerator(cfg)
    mjcf, _ = gen.generate(seed=0)
    # Wrap in a minimal scene
    full = f"""<mujoco>
<worldbody>
{mjcf}
</worldbody>
</mujoco>"""
    # Should not raise
    model = mujoco.MjModel.from_xml_string(full)
    assert model.ngeom > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_terrain_generator.py -v
```
Expected: FAIL with module import errors

- [ ] **Step 3: Create config module**

```python
# jax_rl/envs/terrains/config.py
"""Terrain grid configurations."""
from dataclasses import dataclass, field
from jax_rl.envs.terrains.primitives import (
    FlatTerrainCfg, RoughTerrainCfg, SlopeTerrainCfg,
    PyramidStairsTerrainCfg, InvertedPyramidStairsTerrainCfg,
    DiscreteObstaclesTerrainCfg, SteppingStonesTerrainCfg,
    TiltedGridTerrainCfg,
)
from jax_rl.envs.terrains.base import SubTerrainCfg


@dataclass
class TerrainGridCfg:
    """Grid layout for terrain curriculum."""
    num_rows: int = 10
    tile_size: tuple[float, float] = (8.0, 8.0)
    border_width: float = 20.0
    terrain_types: list = field(default_factory=list)  # list[SubTerrainCfg]

    @property
    def num_cols(self) -> int:
        return len(self.terrain_types)


# Go2 default: 8 terrain types, 10 difficulty levels
GO2_DEFAULT_CFG = TerrainGridCfg(
    num_rows=10,
    tile_size=(8.0, 8.0),
    border_width=20.0,
    terrain_types=[
        FlatTerrainCfg(),
        RoughTerrainCfg(),
        SlopeTerrainCfg(),
        PyramidStairsTerrainCfg(),
        InvertedPyramidStairsTerrainCfg(),
        DiscreteObstaclesTerrainCfg(),
        SteppingStonesTerrainCfg(),
        TiltedGridTerrainCfg(),
    ],
)
```

- [ ] **Step 4: Create generator module**

```python
# jax_rl/envs/terrains/generator.py
"""Procedural terrain grid MJCF generator."""
import numpy as np
from jax_rl.envs.terrains.config import TerrainGridCfg
from jax_rl.envs.terrains.base import TerrainOutput


def _geom_to_xml(g: dict) -> str:
    """Serialize one geom dict to MJCF <geom .../> string."""
    parts = [f'type="{g["type"]}"']
    if "name" in g:
        parts.append(f'name="{g["name"]}"')
    if "size" in g:
        parts.append(f'size="{" ".join(f"{s:.6f}" for s in g["size"])}"')
    if "pos" in g:
        parts.append(f'pos="{" ".join(f"{s:.6f}" for s in g["pos"])}"')
    if "quat" in g:
        parts.append(f'quat="{" ".join(f"{q:.6f}" for q in g["quat"])}"')
    if "rgba" in g:
        parts.append(f'rgba="{" ".join(f"{c:.3f}" for c in g["rgba"])}"')
    if "friction" in g:
        parts.append(f'friction="{" ".join(f"{f:.6f}" for f in g["friction"])}"')
    return f"<geom {' '.join(parts)} />"


class TerrainGenerator:
    """Generates a grid of terrain tiles as one composite MJCF string."""

    def __init__(self, cfg: TerrainGridCfg):
        self.cfg = cfg

    def generate(self, seed: int = 0) -> tuple[str, np.ndarray]:
        """Build the full grid. Returns (mjcf_xml_fragment, origins_array).

        mjcf_xml_fragment: XML string, wrapping a <body name="terrain"> containing
          all tile geoms, ready to inject inside <worldbody>.
        origins_array: shape (num_rows, num_cols, 3), world-frame spawn positions.
        """
        rng = np.random.default_rng(seed)
        sx, sy = self.cfg.tile_size
        nr, nc = self.cfg.num_rows, self.cfg.num_cols

        # Grid layout: tiles packed edge-to-edge in a rect
        grid_w = nc * sx
        grid_h = nr * sy
        x0 = -grid_w / 2 + sx / 2
        y0 = -grid_h / 2 + sy / 2

        origins = np.zeros((nr, nc, 3))
        all_geom_xmls = []
        geom_counter = 0

        for r in range(nr):
            difficulty = r / (nr - 1) if nr > 1 else 0.0
            for c in range(nc):
                terrain_cfg = self.cfg.terrain_types[c]
                tile_out: TerrainOutput = terrain_cfg.generate(
                    difficulty=difficulty,
                    size=self.cfg.tile_size,
                    rng=rng,
                )
                tile_x = x0 + c * sx
                tile_y = y0 + r * sy
                # Store spawn origin in world coords
                origins[r, c] = [
                    tile_x + tile_out.spawn_origin[0],
                    tile_y + tile_out.spawn_origin[1],
                    tile_out.spawn_origin[2],
                ]
                # Offset all geoms into world coords, write XML
                for g in tile_out.geoms:
                    offset_geom = dict(g)
                    offset_geom["pos"] = (
                        g["pos"][0] + tile_x,
                        g["pos"][1] + tile_y,
                        g["pos"][2],
                    )
                    offset_geom["name"] = f"t{geom_counter}"
                    geom_counter += 1
                    all_geom_xmls.append("    " + _geom_to_xml(offset_geom))

        # Outer border box (to prevent fall-off)
        bw = self.cfg.border_width
        border_xml = (
            f'    <geom type="box" name="border" '
            f'size="{grid_w / 2 + bw:.3f} {grid_h / 2 + bw:.3f} 0.05" '
            f'pos="0 0 -0.1" rgba="0.1 0.1 0.1 1" />'
        )
        all_geom_xmls.insert(0, border_xml)

        mjcf = '<body name="terrain" pos="0 0 0">\n' + "\n".join(all_geom_xmls) + "\n</body>"
        return mjcf, origins
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_terrain_generator.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add jax_rl/envs/terrains/config.py jax_rl/envs/terrains/generator.py tests/test_terrain_generator.py
git commit -m "feat(terrains): add TerrainGenerator + Go2 default grid config"
```

---

### Task 1.11: Preview script — Phase 1 deliverable

**Files:**
- Create: `tools/generate_terrain_preview.py`

- [ ] **Step 1: Write the script**

```python
# tools/generate_terrain_preview.py
"""Generate a terrain MJCF and write to file. Open in MuJoCo viewer to inspect.

Usage:
    uv run python tools/generate_terrain_preview.py [--seed N] [--out PATH]
"""
import argparse
from pathlib import Path

from jax_rl.envs.terrains.generator import TerrainGenerator
from jax_rl.envs.terrains.config import GO2_DEFAULT_CFG


MJCF_WRAPPER = """<mujoco model="terrain_preview">
  <compiler angle="radian" autolimits="true" />
  <option timestep="0.004" />
  <visual>
    <global offwidth="1920" offheight="1080" />
  </visual>
  <worldbody>
    <light pos="0 0 30" dir="0 0 -1" />
{terrain}
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="/tmp/terrain_preview.xml")
    args = parser.parse_args()

    gen = TerrainGenerator(GO2_DEFAULT_CFG)
    terrain_xml, origins = gen.generate(seed=args.seed)
    full_xml = MJCF_WRAPPER.format(terrain=terrain_xml)
    Path(args.out).write_text(full_xml)
    print(f"Wrote {args.out}")
    print(f"Grid: {origins.shape[0]} rows x {origins.shape[1]} cols")
    print(f"Open with: MUJOCO_GL=glfw uv run python -m mujoco.viewer --mjcf {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script + smoke test it loads**

```bash
uv run python tools/generate_terrain_preview.py --out /tmp/terrain_preview.xml
uv run python -c "import mujoco; m = mujoco.MjModel.from_xml_path('/tmp/terrain_preview.xml'); print(f'Loaded OK: {m.ngeom} geoms, {m.nbody} bodies')"
```
Expected: `Loaded OK: <some number> geoms, <some number> bodies`

- [ ] **Step 3: Commit**

```bash
git add tools/generate_terrain_preview.py
git commit -m "feat(terrains): add terrain preview script — Phase 1 deliverable"
```

---

### Phase 1 Review Checkpoint

After Task 1.11, pause for user review:

- Run `uv run python tools/generate_terrain_preview.py`
- User opens the MJCF in MuJoCo viewer
- Verifies: 80 tiles visible, each terrain type looks correct at appropriate difficulty, no overlaps / Z-fighting
- Any aesthetic or sizing issues should be fixed before Phase 2

---

## Phase 2 — Env Integration (goal-directed)

**Goal:** `Go2WarpJoystickCurriculum` env loads, robot spawns on a tile per `(terrain_type, terrain_level)`, and navigates toward a per-episode world-frame `goal_xy`. Body-frame command is recomputed each step from goal + robot pose. `episode_reached_goal` / `episode_min_distance` / `episode_fallen` tracked in `state.info`. Advancement rule is a no-op stub in Phase 2 — level stays at 0.

**Deliverable:** Train the env at fixed level=0 without curriculum. Verify robot turns toward goal and walks forward. No crashes, reward curves resemble `Go2WarpJoystickFlat`.

### Task 2.1: Base scene XML template

**Files:**
- Create: `jax_rl/envs/locomotion/xmls/go2_warp_curriculum_scene_template.xml`

Identical in spirit to `go2_warp_scene_flat.xml` but with a placeholder comment `<!-- TERRAIN_INJECT_POINT -->` inside `<worldbody>`. At env init, Python string-replaces this placeholder with the generated `<body name="terrain">...</body>` fragment from `TerrainGenerator.generate()`.

- [ ] **Step 1: Copy + adapt**

```xml
<mujoco model="go2_warp_curriculum_scene">
  <compiler angle="radian" autolimits="true" />
  <include file="unitree_go2/go2.xml" />
  <option timestep="0.004" iterations="6" ls_iterations="6">
    <flag eulerdamp="disable" />
  </option>
  <visual>
    <global offwidth="1920" offheight="1080" />
  </visual>
  <worldbody>
    <light pos="0 0 30" dir="0 0 -1" diffuse="0.7 0.7 0.7" />
    <!-- TERRAIN_INJECT_POINT -->
  </worldbody>
</mujoco>
```

- [ ] **Step 2: Commit**

```bash
git add jax_rl/envs/locomotion/xmls/go2_warp_curriculum_scene_template.xml
git commit -m "feat(curriculum): add scene template for curriculum env"
```

---

### Task 2.2: Env class skeleton — terrain injection + load

**Files:**
- Create: `jax_rl/envs/locomotion/go2_warp_curriculum.py`
- Test: `tests/test_go2_warp_curriculum_env.py`

Overview: `WarpJoystickCurriculum(WarpJoystick)` overrides `__init__` to:
1. Generate terrain MJCF via `TerrainGenerator(GO2_DEFAULT_CFG)`
2. Inject into the scene template → write to `xmls/_generated_curriculum_scene_{os.getpid()}.xml`
3. Call grandparent `Go2WarpEnv.__init__` with the generated path (skipping `WarpJoystick.__init__` which hardcodes flat scene)
4. Call `self._post_init()` (inherited)
5. Store `self._terrain_origins: jp.ndarray (10, 4, 3)` from generator
6. Store `self._num_rows = 10`, `self._num_cols = 4`

- [ ] **Step 1: Write failing test**

```python
# tests/test_go2_warp_curriculum_env.py
import jax
import jax.numpy as jnp


def test_env_loads():
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    assert env.action_size == 12
    assert env._terrain_origins.shape == (10, 4, 3)
    assert env._num_rows == 10
    assert env._num_cols == 4
```

- [ ] **Step 2: Run — FAIL** (module not found)

- [ ] **Step 3: Implement `go2_warp_curriculum.py`**

See full design in the file structure section — key points:
- Generate terrain MJCF at init, write with PID suffix to avoid multiprocess race
- Call `Go2WarpEnv.__init__` directly, not `super().__init__` (which uses flat scene)
- After `_post_init`, store `_terrain_origins` / `_num_rows` / `_num_cols`

- [ ] **Step 4: Add `.gitignore` entry for generated scene files**

```bash
echo "_generated_curriculum_scene_*.xml" >> jax_rl/envs/locomotion/xmls/.gitignore
```

- [ ] **Step 5: Test passes, commit**

```bash
git add jax_rl/envs/locomotion/go2_warp_curriculum.py \
        jax_rl/envs/locomotion/xmls/.gitignore \
        tests/test_go2_warp_curriculum_env.py
git commit -m "feat(curriculum): env loads with generated terrain MJCF"
```

---

### Task 2.3: `reset()` — spawn + goal per terrain type

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_warp_curriculum.py`
- Test: `tests/test_go2_warp_curriculum_env.py`

Overview: override `reset()` to sample spawn + goal per terrain type, place robot at spawn with random yaw, and store per-episode fields in `state.info`.

Per-terrain-type logic (implemented as a helper `_sample_spawn_goal(terrain_type, terrain_level, rng)`):

- **Rough (type 0) / Tilted (type 3):** "edge-to-opposite-edge traversal"
    - Sample axis (0=x, 1=y): random
    - Sample direction (+/-): random
    - Spawn edge = chosen (axis, direction), point within edge at random offset within ±0.5 × tile_size from axis midpoint
    - Goal edge = opposite (axis, direction)
    - Spawn xyz and goal xyz in tile-local frame, then offset by `terrain_origins[level, type]` to world frame
- **Pyramid (type 1) / Inverted (type 2):** "rim to center"
    - Spawn point: random angle around rim at radius = 0.9 × tile_half
    - Goal point: (0, 0, 0) in tile-local (tile center)
    - Spawn yaw: random (robot must orient toward goal)

`state.info` keys set at reset:
- `terrain_level: jp.int32` — read from previous state on reset, or 0 if first reset
- `terrain_type: jp.int32` — deterministic per env via `env_id % num_cols`, preserved across resets
- `goal_xy: jp.float32 (2,)` — world-frame goal
- `initial_distance: jp.float32` — `||spawn_xy - goal_xy||`
- `episode_reached_goal: jp.bool_` — False
- `episode_min_distance: jp.float32` — `initial_distance`
- `episode_fallen: jp.bool_` — False
- `target_speed: jp.float32` — `0.5 + level/9 * 1.0`

- [ ] **Step 1: Write tests**

```python
def test_reset_sets_goal_in_state_info():
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    for key in ["terrain_level", "terrain_type", "goal_xy", "initial_distance",
                "episode_reached_goal", "episode_min_distance", "episode_fallen",
                "target_speed"]:
        assert key in state.info


def test_reset_initial_level_is_zero():
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    assert int(state.info["terrain_level"]) == 0


def test_reset_spawn_near_tile_origin():
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    ttype = int(state.info["terrain_type"])
    tlevel = int(state.info["terrain_level"])
    expected_tile = env._terrain_origins[tlevel, ttype]
    actual_xy = state.data.qpos[:2]
    # Within tile footprint (9.6m tile, half-size 4.8m)
    assert float(jnp.abs(actual_xy[0] - expected_tile[0])) < 5.0
    assert float(jnp.abs(actual_xy[1] - expected_tile[1])) < 5.0


def test_reset_goal_matches_terrain_type_pattern():
    """Pyramid & Inverted goals should be at tile center; Rough & Tilted at opposite edge."""
    env = WarpJoystickCurriculum()
    # Run several resets; group by terrain_type
    for key in range(8):
        state = env.reset(jax.random.PRNGKey(key))
        ttype = int(state.info["terrain_type"])
        tile_origin = env._terrain_origins[int(state.info["terrain_level"]), ttype]
        goal_local = jnp.asarray(state.info["goal_xy"]) - tile_origin[:2]
        if ttype in (1, 2):  # pyramid / inverted
            # Goal at tile center
            assert float(jnp.linalg.norm(goal_local)) < 0.5
        else:  # rough / tilted
            # Goal at an edge
            assert float(jnp.max(jnp.abs(goal_local))) > 3.5
```

- [ ] **Step 2: Implement `reset()` override + helper**

Key structure:

```python
def reset(self, rng):
    rng, type_rng, spawn_rng, yaw_rng, cmd_rng = jax.random.split(rng, 5)

    # Terrain type assignment is fixed per env — the wrapper handles this.
    # env.reset() on its own samples a random type for standalone testing.
    terrain_type = jax.random.randint(type_rng, (), 0, self._num_cols)
    terrain_level = jp.int32(0)

    spawn_local, goal_local, spawn_yaw = self._sample_spawn_goal(
        terrain_type, terrain_level, spawn_rng
    )
    tile_origin = self._terrain_origins[terrain_level, terrain_type]
    spawn_world = spawn_local + tile_origin
    goal_world = goal_local[:2] + tile_origin[:2]

    # Override qpos[:3] for spawn, qpos[3:7] for yaw
    # Base reset to get a fresh state, then overwrite
    state = super().reset(rng)
    new_qpos = state.data.qpos.at[:2].set(spawn_world[:2])
    new_qpos = new_qpos.at[2].set(spawn_world[2])
    new_qpos = new_qpos.at[3:7].set(_yaw_to_quat(spawn_yaw))
    state = state.replace(data=state.data.replace(qpos=new_qpos))

    target_speed = 0.5 + terrain_level.astype(jp.float32) / (self._num_rows - 1) * 1.0
    initial_distance = jp.linalg.norm(spawn_world[:2] - goal_world)

    state.info["terrain_level"] = terrain_level
    state.info["terrain_type"] = terrain_type
    state.info["goal_xy"] = goal_world
    state.info["initial_distance"] = initial_distance
    state.info["episode_reached_goal"] = jp.bool_(False)
    state.info["episode_min_distance"] = initial_distance
    state.info["episode_fallen"] = jp.bool_(False)
    state.info["target_speed"] = target_speed

    return state
```

The `_sample_spawn_goal` helper dispatches on `terrain_type` using `jax.lax.switch` (traceable):

```python
def _sample_spawn_goal(self, terrain_type, terrain_level, rng):
    return jax.lax.switch(
        terrain_type,
        [
            lambda r: self._edge_to_edge(r, self._config.tile_size),       # Rough
            lambda r: self._rim_to_center(r, self._config.tile_size),      # Pyramid
            lambda r: self._rim_to_center(r, self._config.tile_size),      # Inverted
            lambda r: self._edge_to_edge(r, self._config.tile_size),       # Tilted
        ],
        rng,
    )
```

- [ ] **Step 3: Tests pass, commit**

```bash
git add jax_rl/envs/locomotion/go2_warp_curriculum.py tests/test_go2_warp_curriculum_env.py
git commit -m "feat(curriculum): per-terrain spawn + goal in reset()"
```

---

### Task 2.4: `step()` — goal-directed body-frame command + reach tracking

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_warp_curriculum.py`
- Test: `tests/test_go2_warp_curriculum_env.py`

Overview: at each step, compute body-frame command from `goal_xy` + current robot pose, write to `state.info["command"]`, delegate physics to `super().step()` (existing WarpJoystick step reads `info["command"]`). After step, update `episode_reached_goal` / `episode_min_distance` / `episode_fallen`.

Mid-step command computation (every env step):
```python
# Read current pose
robot_xy = state.data.qpos[:2]
robot_yaw = _quat_to_yaw(state.data.qpos[3:7])

# World-frame vector to goal
dx = state.info["goal_xy"][0] - robot_xy[0]
dy = state.info["goal_xy"][1] - robot_xy[1]
heading_world = jp.arctan2(dy, dx)
yaw_error = _wrap_angle(heading_world - robot_yaw)

cmd_vx = state.info["target_speed"]
cmd_vy = jp.float32(0.0)
cmd_yaw_rate = jp.clip(2.0 * yaw_error, -1.5, 1.5)
state.info["command"] = jp.array([cmd_vx, cmd_vy, cmd_yaw_rate])
```

Post-step tracking:
```python
# Delegate physics (WarpJoystick.step reads state.info["command"])
state = super().step(state, action)

# Update episode flags
dist = jp.linalg.norm(state.data.qpos[:2] - state.info["goal_xy"])
state.info["episode_min_distance"] = jp.minimum(state.info["episode_min_distance"], dist)
state.info["episode_reached_goal"] = state.info["episode_reached_goal"] | (dist < 0.5)
state.info["episode_fallen"] = state.done  # WarpJoystick sets state.done on fall
```

- [ ] **Step 1: Write tests**

```python
def test_step_updates_command_from_goal():
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    # Place robot far from goal, facing wrong direction
    state = state.replace(data=state.data.replace(
        qpos=state.data.qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))  # yaw=0
    ))
    action = jnp.zeros(12)
    next_state = env.step(state, action)
    # Command should have nonzero yaw_rate (robot needs to turn toward goal)
    cmd = next_state.info["command"]
    assert cmd.shape == (3,)
    # vx should be target_speed
    assert float(cmd[0]) == pytest.approx(float(state.info["target_speed"]), abs=0.01)


def test_step_sets_reached_goal_when_near():
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    # Teleport robot to goal
    goal = state.info["goal_xy"]
    state = state.replace(data=state.data.replace(
        qpos=state.data.qpos.at[:2].set(goal)
    ))
    state = env.step(state, jnp.zeros(12))
    assert bool(state.info["episode_reached_goal"])


def test_step_tracks_min_distance():
    env = WarpJoystickCurriculum()
    state = env.reset(jax.random.PRNGKey(0))
    initial_min = float(state.info["episode_min_distance"])
    for _ in range(10):
        state = env.step(state, jnp.zeros(12))
    # Min distance should only decrease or stay the same
    assert float(state.info["episode_min_distance"]) <= initial_min
```

- [ ] **Step 2: Implement `step()` override**

- [ ] **Step 3: Tests pass, commit**

```bash
git add jax_rl/envs/locomotion/go2_warp_curriculum.py tests/test_go2_warp_curriculum_env.py
git commit -m "feat(curriculum): goal-directed command in step() + reach tracking"
```

---

### Task 2.5: Register env in training pipeline

**Files:**
- Modify: `jax_rl/training/env_setup.py`

Register `Go2WarpJoystickCurriculum` via Playground's registry. Also register the TorqueSpeed variant (uses the baked-config pattern from the existing Flat env).

```python
from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
from jax_rl.envs.locomotion.go2_warp_curriculum import default_config as curriculum_default_config

if "Go2WarpJoystickCurriculum" not in pg_locomotion._envs:
    pg_locomotion.register_environment(
        "Go2WarpJoystickCurriculum",
        functools.partial(WarpJoystickCurriculum, task="flat_terrain"),
        curriculum_default_config,
    )

def _curriculum_ts_default_config():
    cfg = curriculum_default_config()
    cfg.torque_speed_model = True
    return cfg

if "Go2WarpJoystickCurriculumTorqueSpeed" not in pg_locomotion._envs:
    pg_locomotion.register_environment(
        "Go2WarpJoystickCurriculumTorqueSpeed",
        functools.partial(WarpJoystickCurriculum, task="flat_terrain"),
        _curriculum_ts_default_config,
    )
```

- [ ] **Step 1: Test env loads via registry**

```python
def test_env_registry_load():
    from jax_rl.training import env_setup
    from mujoco_playground import registry as pg_registry
    e1 = pg_registry.load("Go2WarpJoystickCurriculum")
    e2 = pg_registry.load("Go2WarpJoystickCurriculumTorqueSpeed")
    assert e1._torque_speed_model is False
    assert e2._torque_speed_model is True
```

- [ ] **Step 2: Commit**

```bash
git add jax_rl/training/env_setup.py tests/test_go2_warp_curriculum_env.py
git commit -m "feat(curriculum): register env + TorqueSpeed variant"
```

---

### Task 2.6: Smoke training (fixed level=0)

**Files:**
- No code change — smoke test only

Without the TerrainCurriculumWrapper (Phase 3), advancement is a no-op — all envs stay at level 0. But the env should still train correctly: robot turns toward goal, walks forward, tracking reward climbs over time.

- [ ] **Step 1: Short training run**

```bash
XLA_CLIENT_MEM_FRACTION=0.55 uv run python train_fast_sac.py \
  --env Go2WarpJoystickCurriculum --reset-mode per_step \
  --total-timesteps 500000 --seed 0 --num-envs 256 --wandb
```

Watch for:
- Training runs without NaN/Inf
- Robot learns to walk (reward climbs from ~0 toward ~200+ over 500k steps)
- `terrain/{type}/mean_level` stays at 0 (no curriculum yet)
- No crashes
- Compile time reasonable (<30s)

- [ ] **Step 2: Document result**

If training looks healthy, proceed to Phase 3. Otherwise debug.

---

### Phase 2 Review Checkpoint

- [ ] Env loads via CLI
- [ ] Short training run completes, rewards climb
- [ ] All Phase 2 tests pass
- [ ] `state.info` has all expected curriculum keys
- [ ] User reviews before Phase 3

---

## Phase 3 — Curriculum Advancement (goal-directed)

**Goal:** Add a wrapper (or extend `DomainRandWrapper`) that tracks per-env `terrain_level` across episodes and advances/demotes based on reach/fall outcomes. Fixed terrain_type assignment enforced at wrapper level (not random in env.reset).

**Deliverable:** Full training run shows `terrain_level` advancing over time per env. `mean_level` climbs in wandb.

### Task 3.1: `TerrainCurriculumDRWrapper` — skeleton

**Files:**
- Create: `jax_rl/envs/wrappers/terrain_curriculum_dr.py`
- Test: `tests/test_terrain_curriculum_dr_wrapper.py`

Overview: extends `DomainRandWrapper` to:
1. Enforce fixed `terrain_type` per env (assigned `env_id % num_cols` at init, preserved across resets)
2. On episode reset: compute new `terrain_level` using the advancement rule
3. Re-sample spawn + goal for new level
4. Reset episode flags (`episode_reached_goal`, `episode_min_distance`, `episode_fallen`)

Wrapper signature:
```python
class TerrainCurriculumDRWrapper(DomainRandWrapper):
    def __init__(
        self,
        env,
        episode_length=1000,
        mode="per_step",
        num_envs=1,
        goal_radius=0.5,
        min_distance_fraction=0.5,  # for demote-on-no-progress check
    ): ...
```

Core step logic:

```python
def step(self, state, action):
    prev_level = state.info["terrain_level"]
    prev_type = state.info["terrain_type"]
    prev_reached = state.info["episode_reached_goal"]
    prev_fallen = state.info["episode_fallen"]
    prev_min_dist = state.info["episode_min_distance"]
    prev_init_dist = state.info["initial_distance"]

    state = super().step(state, action)  # parent handles reset-on-done

    done = state.info[f"{self._KEY}_episode_done"].astype(jp.bool_)

    # Compute new level
    promote = prev_reached & ~prev_fallen
    demote_fall = prev_fallen
    demote_timeout = (~prev_reached) & ~prev_fallen & (prev_min_dist > 0.5 * prev_init_dist)
    stay = ~promote & ~demote_fall & ~demote_timeout

    delta = jp.where(promote, 1, 0) - jp.where(demote_fall | demote_timeout, 1, 0)
    new_level = jp.clip(prev_level + delta, 0, self._num_rows - 1)

    # On done, update level and preserve terrain_type (parent's where_done sets it to
    # reset-state value which randomly sampled in env.reset; we overwrite).
    state.info["terrain_level"] = jp.where(done, new_level, prev_level)
    state.info["terrain_type"] = prev_type  # always preserved (fixed per env)
    # Also re-generate spawn + goal for new level (done envs get new goal)
    # ... see below

    # Log advancement flags for wandb
    state.info["episode_promoted"] = promote & done
    state.info["episode_demoted"] = (demote_fall | demote_timeout) & done

    return state
```

For respawn on promoted/demoted env, the wrapper needs to re-sample spawn + goal at the NEW level. Options:
- Call `env._sample_spawn_goal(new_level, terrain_type, rng)` (needs to be exposed)
- Or: inner env.reset happens twice — once to get a "fresh state", then wrapper overwrites qpos with (new_level, terrain_type)-specific spawn

Second option simpler.

- [ ] **Step 1: Write tests** (preserve type, promote on reach, demote on fall)

- [ ] **Step 2: Implement wrapper**

- [ ] **Step 3: Tests pass, commit**

```bash
git add jax_rl/envs/wrappers/terrain_curriculum_dr.py tests/test_terrain_curriculum_dr_wrapper.py
git commit -m "feat(curriculum): TerrainCurriculumDRWrapper with advancement logic"
```

---

### Task 3.2: Wire wrapper into training pipeline

**Files:**
- Modify: `jax_rl/training/env_setup.py`

Currently `env_setup.py` wraps envs with `DomainRandWrapper`. Add conditional branch: if env is `WarpJoystickCurriculum` (or inherits from it), use `TerrainCurriculumDRWrapper` instead.

```python
from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
from jax_rl.envs.wrappers.terrain_curriculum_dr import TerrainCurriculumDRWrapper

if isinstance(env.unwrapped, WarpJoystickCurriculum):
    env = TerrainCurriculumDRWrapper(env, ..., num_envs=cfg.num_envs)
else:
    env = DomainRandWrapper(env, ...)
```

- [ ] **Step 1: Smoke train 500k steps**

```bash
XLA_CLIENT_MEM_FRACTION=0.55 uv run python train_fast_sac.py \
  --env Go2WarpJoystickCurriculum --reset-mode per_step \
  --total-timesteps 500000 --seed 0 --num-envs 256 --wandb
```

Watch `terrain/{type}/mean_level`: should climb above 0 for most types (rough first, harder types slower).

- [ ] **Step 2: Commit**

```bash
git add jax_rl/training/env_setup.py
git commit -m "feat(curriculum): training pipeline uses TerrainCurriculumDRWrapper for curriculum env"
```

---

### Phase 3 Review Checkpoint

- [ ] Curriculum tests pass
- [ ] 500k-step smoke run shows level advancement in wandb
- [ ] No crashes, reasonable sps
- [ ] User reviews before Phase 4

---


## Phase 4 — Logging + Polish

**Goal:** Wandb per-type metrics, TorqueSpeed variant, presets, docs.

### Task 4.1: log_terrain_metrics helper

**Files:**
- Modify: `jax_rl/training/metrics_logger.py`
- Test: `tests/test_metrics_logger.py` (new if doesn't exist, else append)

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics_logger.py (append or create)
import numpy as np


def test_log_terrain_metrics_returns_empty_dict_when_no_terrain():
    from jax_rl.training.metrics_logger import log_terrain_metrics
    info = {"some_other_key": np.zeros(10)}
    result = log_terrain_metrics(info, terrain_type_names=["rough"])
    assert result == {}


def test_log_terrain_metrics_computes_per_type_metrics():
    """4 envs per type; tests mean_level, reach_rate, fall_rate, promote_rate."""
    from jax_rl.training.metrics_logger import log_terrain_metrics
    info = {
        "terrain_level": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        "terrain_type":  np.array([0, 0, 1, 1, 0, 1, 0, 1]),
        "episode_reached_goal": np.array([True,  False, True,  False, True,  True,  False, False]),
        "episode_fallen":       np.array([False, True,  False, False, False, False, True,  True]),
        "episode_promoted":     np.array([True,  False, True,  False, False, True,  False, False]),
        "episode_demoted":      np.array([False, True,  False, False, False, False, True,  True]),
    }
    result = log_terrain_metrics(info, terrain_type_names=["rough", "pyramid_up"])
    # Type 0 (rough): envs indices [0,1,4,6], levels [0,1,4,6], mean=2.75
    assert abs(result["terrain/rough/mean_level"] - 2.75) < 1e-5
    # Type 0 reach_rate: [T,F,T,F] → 0.5
    assert abs(result["terrain/rough/reach_rate"] - 0.5) < 1e-5
    # Type 0 fall_rate: [F,T,F,T] → 0.5
    assert abs(result["terrain/rough/fall_rate"] - 0.5) < 1e-5
    # Type 0 promote_rate: [T,F,F,F] → 0.25
    assert abs(result["terrain/rough/promote_rate"] - 0.25) < 1e-5
    # Global mean_level: all 8 envs, mean=3.5
    assert abs(result["terrain/global/mean_level"] - 3.5) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_metrics_logger.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement helper**

```python
# jax_rl/training/metrics_logger.py (append)
import numpy as np


def log_terrain_metrics(info: dict, terrain_type_names: list[str]) -> dict[str, float]:
    """Extract per-terrain-type metrics from state.info at a snapshot.

    Snapshot approach: each env's values reflect its last completed episode.
    Aggregates over ~num_envs/num_types envs per type (noisy per-sample, smooth
    over training time).

    Returns empty dict if terrain_level not in info (non-curriculum envs).
    """
    if "terrain_level" not in info or "terrain_type" not in info:
        return {}

    levels = np.asarray(info["terrain_level"])
    types = np.asarray(info["terrain_type"])
    reached = np.asarray(info.get("episode_reached_goal", np.zeros_like(levels, dtype=bool)))
    fallen = np.asarray(info.get("episode_fallen", np.zeros_like(levels, dtype=bool)))
    promoted = np.asarray(info.get("episode_promoted", np.zeros_like(levels, dtype=bool)))
    demoted = np.asarray(info.get("episode_demoted", np.zeros_like(levels, dtype=bool)))

    result = {}
    for type_idx, name in enumerate(terrain_type_names):
        mask = types == type_idx
        if mask.any():
            result[f"terrain/{name}/mean_level"]   = float(levels[mask].mean())
            result[f"terrain/{name}/max_level"]    = int(levels[mask].max())
            result[f"terrain/{name}/num_envs"]     = int(mask.sum())
            result[f"terrain/{name}/reach_rate"]   = float(reached[mask].mean())
            result[f"terrain/{name}/fall_rate"]    = float(fallen[mask].mean())
            result[f"terrain/{name}/promote_rate"] = float(promoted[mask].mean())
            result[f"terrain/{name}/demote_rate"]  = float(demoted[mask].mean())

    # Global aggregates
    result["terrain/global/mean_level"] = float(levels.mean())
    result["terrain/global/reach_rate"] = float(reached.mean())
    result["terrain/global/fall_rate"]  = float(fallen.mean())
    return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python -m pytest tests/test_metrics_logger.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jax_rl/training/metrics_logger.py tests/test_metrics_logger.py
git commit -m "feat(curriculum): add log_terrain_metrics helper"
```

---

### Task 4.2: Wire log_terrain_metrics into training loop

**Files:**
- Modify: `jax_rl/training/offpolicy_loop.py` (or equivalent call site where wandb log happens)

- [ ] **Step 1: Identify log call sites**

```bash
grep -n "wandb_log\|wandb.log" jax_rl/training/*.py | head -20
```

- [ ] **Step 2: Add terrain metrics to log calls**

At each wandb log call that receives `state.info` or `episode_metrics`, add:

```python
from jax_rl.training.metrics_logger import log_terrain_metrics
from jax_rl.envs.terrains.config import GO2_DEFAULT_CFG

terrain_type_names = [t.name for t in GO2_DEFAULT_CFG.terrain_types]
metrics.update(log_terrain_metrics(state.info, terrain_type_names))
wandb_log(metrics, ...)
```

Gate on presence: the helper returns `{}` if terrain keys aren't in info, so flat-env training is unaffected.

- [ ] **Step 3: Smoke test**

Run a short training with curriculum env + wandb:

```bash
XLA_CLIENT_MEM_FRACTION=0.55 uv run python train_fast_sac.py \
  --env Go2WarpJoystickCurriculum --reset-mode per_step \
  --total-timesteps 200000 --seed 0 --num-envs 64 --wandb
```

Check wandb dashboard for `terrain/{type}/mean_level` metrics.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/training/offpolicy_loop.py jax_rl/training/train_flashsac.py
git commit -m "feat(curriculum): wandb per-terrain-type metrics"
```

---

### Task 4.3: ~~Register TorqueSpeed variant~~ (MOVED to Task 2.5)

The TorqueSpeed variant is now registered as part of Task 2.5 alongside the base curriculum env. Skipping this task.

---

<!--
Original content of Task 4.3 removed; TorqueSpeed registration happens in Task 2.5.
Placeholder below to maintain section numbering; content has been intentionally gutted:

- Step 1-3: n/a — see Task 2.5.

Expected:
  curriculum flag: False
  torque-speed variant flag: True
```

- [ ] **Step 3: Commit**

(Registration commit occurs in Task 2.5; no separate commit here.)
-->

---

### Task 4.4: Presets for PPO/FastSAC/FlashSAC

**Files:**
- Modify: `jax_rl/configs/env_presets.py`

- [ ] **Step 1: Add preset entries**

```python
# jax_rl/configs/env_presets.py (append after each existing Go2 preset dict)

# PPO
PRESETS["Go2WarpJoystickCurriculum"] = dataclasses.replace(
    PRESETS["Go2WarpJoystickFlat"], env_name="Go2WarpJoystickCurriculum"
)
PRESETS["Go2WarpJoystickCurriculumTorqueSpeed"] = dataclasses.replace(
    PRESETS["Go2WarpJoystickFlat"], env_name="Go2WarpJoystickCurriculumTorqueSpeed"
)

# FastSAC
FAST_SAC_PRESETS["Go2WarpJoystickCurriculum"] = (
    dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2WarpJoystickCurriculum"),
    _FAST_SAC_BASE_ALGO,
)
FAST_SAC_PRESETS["Go2WarpJoystickCurriculumTorqueSpeed"] = (
    dataclasses.replace(_FAST_SAC_BASE_CFG, env_name="Go2WarpJoystickCurriculumTorqueSpeed"),
    _FAST_SAC_BASE_ALGO,
)

# FlashSAC
FLASH_SAC_PRESETS["Go2WarpJoystickCurriculum"] = (
    dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name="Go2WarpJoystickCurriculum"),
    _FLASH_SAC_BASE_ALGO,
)
FLASH_SAC_PRESETS["Go2WarpJoystickCurriculumTorqueSpeed"] = (
    dataclasses.replace(_FLASH_SAC_BASE_CFG, env_name="Go2WarpJoystickCurriculumTorqueSpeed"),
    _FLASH_SAC_BASE_ALGO,
)
```

- [ ] **Step 2: Verify**

```bash
uv run python -c "
from jax_rl.training import env_setup
from jax_rl.configs.env_presets import get_fast_sac_preset
cfg, _ = get_fast_sac_preset('Go2WarpJoystickCurriculum')
print('FastSAC curriculum env_name:', cfg.env_name)
cfg, _ = get_fast_sac_preset('Go2WarpJoystickCurriculumTorqueSpeed')
print('FastSAC TS env_name:', cfg.env_name)
"
```

- [ ] **Step 3: Regenerate docs/reference/env-presets.md**

```bash
uv run python docs/scripts/gen_env_presets.py
```

Verify curriculum entries appear.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/configs/env_presets.py docs/reference/env-presets.md
git commit -m "feat(curriculum): add presets for curriculum env variants"
```

---

### Task 4.5: Docs — docs/api/envs.md

**Files:**
- Modify: `docs/api/envs.md`

- [ ] **Step 1: Add curriculum env section**

```markdown
<!-- docs/api/envs.md — append after WarpJoystick / Actuator models section -->

## WarpJoystickCurriculum

Procedurally generated terrain with per-env curriculum advancement. Robot spawns on one of 80 tiles (10 difficulty levels × 8 terrain types). Each env's `terrain_level` advances when its mean tracking reward exceeds a threshold.

**Variants:**
- `Go2WarpJoystickCurriculum` — default actuator (ideal PD).
- `Go2WarpJoystickCurriculumTorqueSpeed` — with linear torque-speed curve on actuator.

**Terrain types:** Flat, Rough, Slope, Stairs (up), Inverted Stairs (down), Discrete Obstacles, Stepping Stones, Tilted Grid.

**Column assignment:** each env is fixed to one terrain type for the entire training run (specialization pattern from legged_gym). With 1024 envs, ~128 envs per type, each advancing independently through 10 difficulty levels.

**Curriculum signal:** mean tracking_lin_vel reward per episode. If ≥ 0.8 → promote, ≤ 0.3 → demote.

**Usage:**
```bash
uv run python train_fast_sac.py --env Go2WarpJoystickCurriculum --reset-mode per_step --wandb
```

See [lessons/terrain_curriculum.md](https://github.com/stevenwman/jax-learning/blob/main/.context/lessons/terrain_curriculum.md) for design rationale.
```

- [ ] **Step 2: Commit**

```bash
git add docs/api/envs.md
git commit -m "docs(curriculum): add WarpJoystickCurriculum to envs page"
```

---

### Task 4.6: Lesson — .context/lessons/terrain_curriculum.md

**Files:**
- Create: `.context/lessons/terrain_curriculum.md`

- [ ] **Step 1: Write lesson**

Content: architecture, design decisions + rationale, integration gotchas (esp. terrain_type-preservation-across-resets trick), performance notes (compile time, sps impact), wandb metric interpretation, references to legged_gym / MJLab.

- [ ] **Step 2: Commit**

```bash
git add .context/lessons/terrain_curriculum.md
git commit -m "docs(curriculum): lesson — terrain curriculum design + gotchas"
```

---

### Task 4.7: Journal — .context/journals/YYYY-MM-DD.md

**Files:**
- Create: `.context/journals/<date-at-completion>.md`

- [ ] **Step 1: Write journal entry**

Content: what was built, phased delivery notes, A/B results (if a full run was done), next steps (push-force curriculum, additional terrain types).

- [ ] **Step 2: Commit**

```bash
git add .context/journals/<date>.md
git commit -m "docs(curriculum): journal entry for terrain curriculum landing"
```

---

### Task 4.8: TODO.md + AGENT_HANDOFF.md updates

**Files:**
- Modify: `.context/TODO.md`
- Modify: `.context/AGENT_HANDOFF.md`

- [ ] **Step 1: Mark curriculum work done in TODO.md**

- [ ] **Step 2: Add benchmark row to AGENT_HANDOFF.md (once a real run lands)**

- [ ] **Step 3: Commit**

```bash
git add .context/TODO.md .context/AGENT_HANDOFF.md
git commit -m "docs(curriculum): TODO/handoff updates"
```

---

### Phase 4 Review Checkpoint

After Task 4.8:

- All docs current
- Wandb dashboard shows per-type terrain metrics during a training run
- Lesson + journal written
- User confirms ready to merge

---

## Final Verification

Before wrapping:

- [ ] All tests pass: `uv run python -m pytest tests/test_terrain_primitives.py tests/test_terrain_generator.py tests/test_go2_warp_curriculum_env.py tests/test_terrain_dr_wrapper.py tests/test_metrics_logger.py -v`
- [ ] Existing tests still pass: `uv run python -m pytest tests/ -v` (terrain changes shouldn't break flat env)
- [ ] End-to-end smoke train: `uv run python train_fast_sac.py --env Go2WarpJoystickCurriculum --reset-mode per_step --total-timesteps 2000000 --seed 42 --wandb` runs cleanly for 2M steps; wandb shows per-type metrics
- [ ] User reviews final state; no lingering concerns

---

## Known Risks / Open Questions

1. **MJCF size / Warp compile time.** 80 tiles × ~20-50 geoms each = 1000-4000 geoms. Warp graph capture may add 10-30s to startup. Measured during Phase 2 smoke test; if too slow, trim grid to 10×6 or 5×8.

2. **Sim-to-sim gap from terrain.** Robot trained on curriculum terrain may overfit to specific tile geometries. Deployment on real terrain is TBD. Consider random tile roughness per env (via DR on tile geom params) as a follow-up.

3. **Curriculum signal miscalibration.** `tracking_reward_max=1.0` and thresholds `[0.3, 0.8]` are initial guesses. May need re-tuning based on observed reward distributions. Monitor wandb `terrain/global/mean_level` — if most envs stay at level 0 or jump to level 9 quickly, thresholds are wrong.

4. **`TerrainDomainRandWrapper` vmap complexity.** The wrapper modifies `state.info` per-env and re-spawns robots on tile advancement. Vmap semantics need careful handling (especially `state.data.qpos` updates). Phase 3 has integration tests but real training may surface subtle issues.

5. **Checkpoints broken across terrain_seed.** Different `terrain_seed` values produce different geometry. Checkpoints from one seed's terrain won't transfer cleanly to another. Document this; consider hashing terrain_seed into checkpoint dir name.

---

## Execution Guide

Each phase is independently reviewable. After Phase N completes:
1. Run all new + existing tests
2. Smoke-test end-to-end behavior
3. User reviews before Phase N+1

Commit frequency: one commit per task (5-10 commits per phase).

Do NOT bulk-merge phases without review. Each phase stands on its own.
