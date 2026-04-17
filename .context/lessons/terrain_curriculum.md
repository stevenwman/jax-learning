# Terrain Curriculum — Design + Gotchas

**Status:** Phases 1-4 complete (2026-04-17). Env registered, tests pass, presets wired.

---

## 1. Architecture Overview

One big MJCF scene. 10 difficulty rows × 4 terrain type columns = 40 tiles, each 9.6×9.6m. The entire grid is compiled into a single Warp scene at env construction time and written to a PID-suffixed XML file to avoid multiprocess race conditions.

```
_terrain_origins: (10, 4, 3)   — world-frame XYZ origin of each tile
terrain_level[i]: int32         — current difficulty row for env i
terrain_type[i]: int32          — fixed column for env i (env_id % 4)
```

Env `i` is **permanently assigned** to terrain type `i % 4`. This is the legged_gym pattern: specialization means each type's difficulty gradient is learned independently, not blurred together. At episode reset, the robot spawns on tile `(terrain_level[i], terrain_type[i])`.

**Class hierarchy:**
- `Go2WarpJoystickCurriculum` — subclasses `WarpJoystick` (inherits full reward spec, DR, flat env logic), extends `__init__` to build terrain MJCF, overrides `reset()` and `step()` for goal-directed commands and curriculum advancement.
- `TerrainCurriculumDRWrapper` — thin wrapper around `Go2WarpJoystickCurriculum` that applies domain randomization and stores per-env curriculum state in `state.info`.

---

## 2. Key Design Decisions

### 4 terrain types (not 8)

Original plan had 8: Flat, Rough, Slope, PyramidStairs, InvertedPyramidStairs, Obstacles, SteppingStones, TiltedGrid.

Dropped:
- **Flat** — redundant with `Go2WarpJoystickFlat`; if you want flat, train on that env.
- **Slope** — tried alternating hills. Hill magnitude varies across rows (harder = steeper) but tile width is fixed, so height transition creates flat spots at row boundaries. Alternating pattern adds complexity for marginal benefit. Dropped entirely.
- **Obstacles** — deployment relevance unclear. Curriculum signal (reach_goal through obstacles) hard to define cleanly.
- **SteppingStones** — same issue. Stepping stone spacing tuning across 10 levels is its own sub-project.

Kept 4 that cover the space efficiently: height variation (rough), stair negotiation (pyramid up/down), tilt stability (tilted grid).

### 10 rows × 4 cols, 9.6m tiles

9.6m (not 8m) to reduce boundary crossing mid-episode. Robot walks 1-1.5 m/s, episode 1000 steps = 20s. At 1.5 m/s that's 30m — robot will reach the far edge and potentially cross terrain boundaries. 9.6m gives 19.2m round-trip before boundary crossing; goal-directed commands (see below) further reduce boundary risk by pulling the robot toward a fixed goal rather than wandering.

### Goal-directed commands (not legged_gym Bernoulli)

legged_gym uses a Bernoulli random command that resamples mid-episode. This works for curriculum advancement based on tracking reward mean, but is weak for a binary reach/fall signal — if the robot gets lucky and wanders toward the goal, it advances without actually learning.

Our scheme: sample a fixed world-frame goal at episode start (opposite edge for rough/tilted, center for pyramid/bowl). Per-step command is computed from `(goal_pos - robot_pos)` via a P-controller on yaw error and a linear speed proportional to distance. The robot knows where to go and must consistently get there to advance.

**GOTCHA:** `WarpJoystick.step()` overwrites `state.info["command"]` via its Bernoulli sampler on every step. The goal-directed command must be injected BOTH before and after calling `super().step()`. Pattern:

```python
def step(self, state, action):
    state = self._inject_goal_command(state)   # before: so physics sees our command
    state = super().step(state, action)
    state = self._inject_goal_command(state)   # after: so Bernoulli sampler doesn't clobber it
    return state
```

Missing either injection → Bernoulli command leaks through in either physics or obs.

### Binary reach/fall curriculum (not tracking reward)

Tracking reward mean as curriculum signal (legged_gym approach) has two problems:
1. Requires per-episode buffering and smoothing — more state.
2. Reward scale varies across terrain types (bowl terrain is intrinsically harder to get high reward on than rough).

Binary outcome (reached_goal OR fallen) is terrain-agnostic, easy to compute, and directly measures the capability we care about: "can the robot navigate this tile?"

Advancement rule:
- `reached_goal AND NOT fallen` → promote
- `fallen` → demote
- `timeout AND min_distance > 0.5 * initial_distance` → demote (robot barely moved)
- otherwise → stay

Level clamped to `[0, num_rows-1]`.

### Inverted pyramid = actual bowl, not inverted pyramid

First implementation: inverted pyramid = pyramid with height negated. This looks identical to the standard pyramid because the visual difference is hidden under the ground plane. The correct construction: set the rim at z=0 and build descending rings inward. The robot enters at the rim and must navigate down into the pit — mechanically distinct from pyramid-up.

### Solid bases under tilted_grid

Tilted tiles have random yaw + tilt angle. Without a solid base geom, there are visible gaps between tiles at the seams. Fixed by adding a flat base box under each tilted tile that fills the gap.

---

## 3. Gotchas

### Command clobbering (most common bug)

As described above: always inject goal command before AND after `super().step()`. Any refactor that loses the post-step injection silently degrades to random Bernoulli commands.

### Generated scene file PID suffix

MJCF is written to `/tmp/go2_terrain_<pid>.xml`. Multiple processes (e.g., eval env + train env) each write their own file — no race. Clean up `/tmp/go2_terrain_*.xml` if training is killed mid-run.

### VRAM: ~1500 geoms, not ~100

Flat env: ~100 geoms. Curriculum env: 40 tiles × ~37 geoms each = ~1500 geoms. Warp compiles all geoms into the collision graph at init. Memory scales with geom count.

Consequence: reduce `num_envs` from 1024 to 64 on a 16GB GPU. 64 envs is still enough throughput for curriculum (episodes complete in ~20s each = 3 eps/min per env = ~192 eps/min total, sufficient signal).

### Eval OOM at end-of-training

`final_eval_and_checkpoint` creates a separate Warp env instance for eval. This roughly doubles VRAM usage (two separate Warp graphs, each with ~1500 geoms). On a 16GB GPU this often OOMs.

Workarounds:
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — allows dynamic allocation, avoids block reservation
- Skip final eval: `--eval-every 9999999` to suppress eval entirely, or patch `final_eval_and_checkpoint` to be a no-op on curriculum envs
- Use train env for eval (reuse the graph) — requires refactoring eval runner

---

## 4. wandb Metrics Interpretation

`log_terrain_metrics()` in `jax_rl/training/metrics_logger.py` extracts per-type and global metrics from a snapshot of `state.info` at log time. Snapshot = "whatever each env's last completed episode looked like".

Key metrics per terrain type:
- `terrain/{type}/mean_level` — average curriculum level. Should climb from 0 toward 9 over training.
- `terrain/{type}/reach_rate` — fraction of envs where last episode reached goal. Should track ~0.5-0.7 in steady state (curriculum self-regulates to the frontier).
- `terrain/{type}/fall_rate` — fraction of envs where last episode ended in fall. High early (hard terrain), should drop.
- `terrain/{type}/promote_rate` — fraction that leveled up. Tracks advancement velocity.
- `terrain/{type}/demote_rate` — fraction that leveled down. High early, should decay.

`terrain/global/mean_level` is the key diagnostic. If it plateaus near 0 after 1M steps, something is wrong (commands not working, reward not propagating). Should see monotonic rise to ~5-7 by 20M steps on a healthy run.

---

## 5. References

- **MJLab terrain module** (mujocolab/mjlab `mjlab/envs/terrain/`) — Isaac Lab API + MuJoCo Warp, terrain generator reference. See `.context/references/mjlab_audit.md`.
- **legged_gym curriculum** (leggedrobotics/legged_gym `legged_gym/envs/base/legged_robot.py`) — original source for terrain grid layout + column-specialization pattern.
- **Plan:** `.superpowers/plans/2026-04-15-terrain-curriculum.md` — full design doc with implementation phases.
- **Journal:** `.context/journals/2026-04-17.md` — implementation log, test results, smoke training results.
