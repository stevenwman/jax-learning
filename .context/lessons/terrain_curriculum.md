# Terrain Curriculum — Design + Gotchas

**Status:** Phases 1-4 + design iterations complete (2026-04-20). Env registered, tests pass, presets wired, v5 pilot running.

---

## 1. Architecture Overview

One big MJCF scene. 10 difficulty rows × 4 terrain type columns = 40 tiles, each 9.6×9.6m. The entire grid is compiled into a single Warp scene at env construction time and written to a PID-suffixed XML file to avoid multiprocess race conditions.

```
_terrain_origins: (10, 4, 3)   — world-frame XYZ origin of each tile
terrain_level[i]: int32         — current difficulty row for env i
terrain_type[i]: int32          — fixed column for env i (env_id % 4)
```

Env `i` is **permanently assigned** to terrain type `i % 4` (legged_gym pattern — specialization means each type's difficulty gradient is learned independently, not blurred together). At episode reset, the robot spawns on tile `(terrain_level[i], terrain_type[i])`.

**Class hierarchy:**
- `Go2WarpJoystickCurriculum` — subclasses `WarpJoystick` (inherits full reward spec, DR, flat env logic), extends `__init__` to build terrain MJCF, overrides `reset()` and `step()` for goal-directed commands.
- `TerrainCurriculumDRWrapper` — thin wrapper around `Go2WarpJoystickCurriculum` that applies domain randomization, computes promote/demote, and teleports robot to new tile on episode boundary.

---

## 2. Key Design Decisions

### 4 terrain types (not 8)

Dropped Flat (redundant with `Go2WarpJoystickFlat`), Slope (alternating hills created flat-spot artifacts at row boundaries), Obstacles (curriculum signal hard to define cleanly), SteppingStones (stepping-stone spacing tuning across 10 levels is its own project).

Kept 4 that cover the space efficiently: height variation (rough), stair negotiation (pyramid up/down), tilt stability (tilted grid).

### 10 rows × 4 cols, 9.6m tiles

9.6m (not 8m) to reduce boundary crossing. Robot walks 0.5-1.5 m/s, episode 1000 steps = 20s. At 1m/s that's 20m; 9.6m half-tile = 4.8m margin from center. Goal-directed commands (see below) further reduce boundary risk by pulling the robot toward tile center rather than wandering.

### Unified goal-directed design (2026-04-20 redesign)

**Earlier attempts:** dual-class scheme (Class A random-cmd locomotion robustness for rough/tilted, Class B goal-directed nav for pyramids) had two practical problems:

1. Class A random Bernoulli cmd causes robot to wander out of the tile mid-episode (5m+ travel in 20s on a 9.6m tile → past boundary onto neighbor's physics).
2. Class A promote signal went through several rewrites: `max_dist_from_spawn > 2m` was a net-displacement proxy that didn't work with direction-flipping cmds; mean-tracking-error required a threshold that felt ad-hoc.

**Resolution:** all 4 types use the same logic. Spawn on tile rim, holonomic body-frame linvel cmd toward tile center (the "goal"), yaw_rate from parent's Bernoulli sampler (random rotation = body-frame linvel DR). A single promote rule: `reach_goal & ~fall_at_done`. A single demote rule: `fall | no_progress_toward_goal`.

**Why this preserves omnidirectional DR:** as yaw_rate rotates the robot continuously, the fixed world-frame goal direction projects into cmd_vx/cmd_vy rotating through every body-frame direction. This is richer DR than Bernoulli step-jumps — continuously varying rather than discretely switching.

**Holonomic body-frame cmd (from world-frame goal vector + robot yaw):**

```python
world_vel = unit(goal - robot) * target_speed       # world frame
cmd_vx =  cos(yaw)*world_vx + sin(yaw)*world_vy     # body frame
cmd_vy = -sin(yaw)*world_vx + cos(yaw)*world_vy
```

### Zero linvel after reach (2026-04-20)

Once the robot enters the goal zone (within 0.5m), `episode_reached_goal` latches True via OR-accumulation. For the rest of the episode, linvel cmd flips to zero. Robot stays at the target and tracks stand-still + parent's random yaw_rate — exercises the yaw controller at zero linvel, teaches explicit "stop at target" behavior, and avoids the dist≈0 numerical degeneracy in the goal-direction normalization.

Episode doesn't terminate on reach — it runs to truncation (1000 steps) or fall. Reach is a **promote signal only**.

### DR flags: force_zero_linvel + force_zero_yaw

Per-episode Bernoulli flags sampled at env reset:
- `force_zero_linvel` (P = 0.15): zero cmd_vx, cmd_vy for the whole episode. Tests stand-still on uneven terrain.
- `force_zero_yaw` (P = 0.5 if `force_zero_linvel` else 0.15): zero cmd_yaw_rate. Higher prob when linvel is already zero → true stand-still episodes.

These override goal-directed logic: when `force_zero_linvel=True`, robot doesn't walk to goal, it just stands.

### Tile-origin semantics (fixed 2026-04-18)

Initial `TerrainGenerator` stored `origins[r, c] = [tile_x + primitive_spawn_x, tile_y + primitive_spawn_y, spawn_z]`. This meant `_terrain_origins[r,c]` was primitive-dependent (pyramid's rim, rough's edge). Code that used it as "tile center reference" for goal offset produced wrong world positions (goals at tile boundaries).

Fixed: `origins[r, c] = [tile_x, tile_y, 0]` — strict tile centers. Primitive `spawn_origin` is strictly tile-local; env code adds `tile_origin` to tile-local offsets to derive world positions.

### Inverted pyramid = actual bowl, not inverted pyramid

First implementation: inverted pyramid = pyramid with height negated. Looks identical to standard pyramid (visual difference hidden under ground plane). Correct construction: set the rim at z=0 and build descending rings inward. Robot enters at the rim and must navigate down into the pit — mechanically distinct from pyramid-up.

### Solid bases under tilted_grid

Tilted tiles have random yaw + tilt angle. Without a solid base geom, there are visible gaps between tiles at the seams. Fixed by adding a flat base box under each tilted tile.

### Contact-based termination (2026-04-20)

Parent `WarpJoystick._get_termination` uses `flipped | base_z < 0.18m` (world frame). The world-frame base_z check **fails on terrain**: standing in a pyramid pit has negative world-z (base_z ≈ -0.7m at L5), triggering auto-termination on every episode regardless of policy; standing on elevated rings has high world-z, missing actual falls where the torso collapses but stays above 0.18m in absolute frame.

Override in `WarpJoystickCurriculum._get_termination`: `flipped | base_contact > 0`. The `base_contact` sensor is `<contact body1="base_link" ...>` added to the scene template — fires whenever the torso body touches any geom. Terrain-agnostic: works identically on flat, pyramid, bowl, or tilted ground. Confirmed working on both CPU MuJoCo and Warp backends.

Deliberate design: base_contact misses "lying on side with legs propped" configurations (torso not touching, calves are). These are not treated as falls — leg contact is fine, the policy may still recover. The flipped check (`upvector_z < 0`) catches full inversions.

### Fall detection via truncation flag

In the wrapper, after `super().step()` runs the inner env's step and `where_done` merges, `state.info["episode_fallen"]` has been wiped to False for done envs. Inferring fall from it directly fails silently.

Fix: read `truncation` (which survives where_done). `fall_at_done = (done > 0) & (truncation < 0.5)` — done-via-env-termination = fall; done-via-episode-length = timeout.

---

## 3. Gotchas

### Preset `reset_mode` must be `per_step`

`TerrainCurriculumDRWrapper` is only applied in `env_setup.make_envs` when `cfg.reset_mode == "per_step"`. `TrainConfig` defaults to `"legacy"`, which silently runs the env with standard AutoReset + no curriculum logic. Fixed 2026-04-20: all curriculum presets (PRESETS, FAST_SAC, FLASH_SAC) now set `reset_mode="per_step"` explicitly. CLI launches without `--reset-mode` still work.

**Symptom if this regresses:** terrain_type distribution is skewed random (not `[num_envs/4]*4`), `mean_level` stays at 0, nothing promotes. The wrapper isn't running.

### Command clobbering

`WarpJoystick.step()` overwrites `state.info["command"]` via its Bernoulli sampler on every step. The command must be overridden BOTH before and after calling `super().step()`. Unified design does this via a single `_override` closure called twice:

```python
def _override(cmd):
    reached = state.info["episode_reached_goal"]  # re-read from closure
    zero_linvel = force_zero | reached
    body_vx, body_vy = self._goal_linvel_body(state)
    new_vx = jp.where(zero_linvel, 0.0, body_vx)
    new_vy = jp.where(zero_linvel, 0.0, body_vy)
    new_yaw = jp.where(force_zero_yaw, 0.0, cmd[2])
    return cmd.at[0].set(new_vx).at[1].set(new_vy).at[2].set(new_yaw)

state.info["command"] = _override(state.info["command"])
state = super().step(state, action)
state.info["command"] = _override(state.info["command"])  # re-read captures new state
```

Closure re-reads `state.info["episode_reached_goal"]` at each call, so the post-step override correctly sees reach flipping True mid-step.

### Generated scene file PID suffix

MJCF is written to `jax_rl/envs/locomotion/xmls/_generated_curriculum_scene_<pid>.xml`. Multiple processes (eval + train) each write their own file — no race. Clean up stale files if training is killed mid-run (`rm xmls/_generated_curriculum_scene_*.xml`).

### VRAM: ~1500 geoms, not ~100

Flat env: ~100 geoms. Curriculum env: 40 tiles × ~37 geoms each = ~1500 geoms. Warp compiles all geoms into the collision graph at init. Memory scales with geom count.

Consequence: reduce `num_envs` from 1024 (flat) to 32 (curriculum) on a 16GB GPU. Training throughput drops ~30× because 32 envs poorly amortizes Warp kernel dispatch overhead. Per-env throughput is competitive with MJLab/RSL-RL (~10-15 steps/s/env); total throughput is memory-bound, not FLOP-bound.

### Eval OOM at end-of-training

`final_eval_and_checkpoint` creates a separate Warp env instance for eval. Roughly doubles VRAM (two separate Warp graphs, each with ~1500 geoms). On a 16GB GPU this often OOMs.

Workarounds:
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — allows dynamic allocation
- Skip eval: `--eval-every 9999999`
- Use train env for eval (reuse the graph) — requires refactoring eval runner

### Don't bump `njmax` / `naconmax` to silence the "nefc overflow" warning

On terrain env init, MuJoCo Warp prints `nefc overflow - please increase njmax to 123`. The natural fix (`cfg.njmax = 256, cfg.naconmax = 8*8192`) **causes OOM** at forward-pass time — solver kernels allocate memory proportional to these limits.

Keep the defaults (`njmax=100`, `naconmax=32768`). Warning is cosmetic — sim functions correctly.

### Warp kernel cache accumulates across in-process env creations

Creating multiple curriculum env instances in one `python` process (sequential unit tests, stress probes) accumulates Warp graph captures that aren't garbage-collected. Typically fails around the 3rd-4th instantiation with `Warp CUDA error 2: out of memory (wp_cuda_graph_create_exec)`.

Workaround: run each test in a fresh subprocess. Not an issue for training — training uses one env for the whole run.

---

## 4. wandb Metrics Interpretation

`log_terrain_metrics()` in `jax_rl/training/metrics_logger.py` extracts per-type and global metrics from a snapshot of `state.info` at log time. Snapshot = "whatever each env's episode state looked like at the moment".

Key metrics per terrain type:
- `terrain/{type}/mean_level` — average curriculum level. Should climb from 0 toward 9 over training.
- `terrain/{type}/reach_rate` — fraction of envs where `episode_reached_goal` is True at snapshot. Noisy: catches mid-episode OR just-reset state; compute as "fraction of envs currently in post-reach phase of their episode" not "episodes-ending-in-reach".
- `terrain/{type}/fall_rate` — fraction of envs where last episode ended in fall. Should be >0 at the curriculum frontier; always 0.00 means either termination is under-firing (world-frame z check on elevated terrain) or policy never fails (terrain too easy).
- `terrain/{type}/promote_rate` — fraction that leveled up.
- `terrain/{type}/demote_rate` — fraction that leveled down.

`terrain/global/mean_level` is the key diagnostic. Should see monotonic rise to ~5-7 by 20M steps on a healthy run.

**Reach=0 with mean_level>0 isn't a bug.** Means envs previously reached (to get promoted) but are currently mid-episode at the harder level where they haven't reached yet. If rough/tilted show reach>0 and pyramids show reach=0, the pyramid envs are stuck at the new level, not that reach is broken.

**Reach=1 for all envs (pre-2026-04-20 artifact).** When Class A had `goal_xy = spawn_xy` placeholder, `dist_to_goal = 0 < 0.5` at every step → reach always True. Unified design removed this artifact: reach is only True after genuinely entering the goal zone.

---

## 5. References

- **MJLab terrain module** (mujocolab/mjlab `mjlab/envs/terrain/`) — Isaac Lab API + MuJoCo Warp, terrain generator reference. See `.context/references/mjlab_audit.md`.
- **legged_gym curriculum** (leggedrobotics/legged_gym `legged_gym/envs/base/legged_robot.py`) — original source for terrain grid layout + column-specialization pattern.
- **Plan:** `.superpowers/plans/2026-04-15-terrain-curriculum.md` — full design doc with implementation phases.
- **Journals:** `.context/journals/2026-04-17.md` (implementation), `2026-04-20.md` (fix marathon + unified redesign + pilot comparison).
