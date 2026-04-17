# Push Environments

Planar pushing benchmark designed for **algorithm adaptability evaluation**: train on one block shape (push-T), zero-shot evaluate on others (push-circle, push-L, push-plus). Obs is shape-agnostic so a generalizing policy learns pushing physics, not T-geometry.

| Environment | Block shape | Obs | Actions |
|-------------|-------------|-----|---------|
| [PushEnv(shape="T")](#pushenv) | T-block | 16d | 2 (pusher XY) |
| [PushEnv(shape="L")](#pushenv) | L-block | 16d | 2 |
| [PushEnv(shape="circle")](#pushenv) | Cylinder | 16d | 2 |
| [PushEnv(shape="plus")](#pushenv) | Plus-block | 16d | 2 |

Runs at **~360,000 sps on 1024 parallel envs (MuJoCo Warp)** — 28× faster than Go2 locomotion, cheap to ablate.

!!! info "Backend requirement"
    Pusher is a cylinder, blocks are boxes or cylinders. MJX's JAX backend doesn't implement cylinder-box collisions — this env is **Warp-only**. See [lessons/manipulation.md](https://github.com/stevenwman/jax-learning/blob/main/.context/lessons/manipulation.md) §"Cylinder-Box Collisions".

---

## PushEnv

```python
from jax_rl.envs.manipulation.push_env import PushEnv, default_config

cfg = default_config()
cfg.shape = "T"               # "T" | "L" | "circle" | "plus"
cfg.action_mode = "velocity"  # "position" | "velocity" | "teleport"
env = PushEnv(config=cfg)
```

Push a 2D block to a target pose using a circular pusher on a flat table. Pusher is constrained to XY (two slide joints); block has XY + yaw (two slides + one hinge). Walls contain both within the table.

**Key properties:**

- Shape swapped via one config flag — same obs, same reward, different block geometry
- Three action modes for action-parameterization ablations
- Per-episode randomized block pose + target pose
- Dense or sparse reward

### Observation (16d, shape-agnostic)

| Idx       | Field              | Dim | Source                                               |
|-----------|--------------------|-----|------------------------------------------------------|
| `[0:2]`   | `pusher_xy`        | 2   | Pusher position from `qpos`                          |
| `[2:4]`   | `block_xy`         | 2   | Block center from `qpos`                             |
| `[4:6]`   | `block_angle`      | 2   | `[sin(yaw), cos(yaw)]` — continuous angle encoding   |
| `[6:8]`   | `target_xy`        | 2   | Target pose (mocap body), randomized per episode     |
| `[8:10]`  | `target_angle`     | 2   | `[sin(goal_yaw), cos(goal_yaw)]`                     |
| `[10:12]` | `pusher_vel`       | 2   | Pusher XY velocity from `qvel`                       |
| `[12:14]` | `block_vel`        | 2   | Block XY velocity                                    |
| `[14:16]` | `last_action`      | 2   | Previous policy output (for PD-lag awareness)        |

**Intentionally shape-agnostic:** same 4 dims for block pose regardless of T vs circle vs star. Forces the policy to infer contact dynamics from block response, not memorize geometry.

### Action (2d)

Interpreted differently based on `action_mode`:

```python
action: jax.Array  # shape (2,), values in [-1, 1]
```

| Mode | `ctrl` mapping | Feel |
|------|----------------|------|
| `position` (default) | `ctrl = action * 0.28` (absolute XY target) | PD chases target. kp=80 gives ~35ms time constant → lag + slight overshoot. Jittery. |
| `velocity` | `ctrl = current_pos + action * 0.01` (delta) | Smooth 1 cm/step delta. Standard for manipulation RL. |
| `teleport` | `ctrl = current_pos + clipped_delta` (speed cap 0.25 m/s) | Matches gym-pusht / Diffusion Policy baselines. |

All three modes share the exact same obs, reward, and physics — only the `action → ctrl` mapping changes.

### Reward

**Dense (default):**

$$r = -\|block\_xy - target\_xy\| - 0.3 \cdot d_\theta(block\_yaw, target\_yaw) + 5 \cdot \mathbb{1}[success]$$

where $d_\theta$ is the shortest angular distance.

**Sparse:** `r = 1.0` if success else `0.0`.

**Success threshold:** pos_error < 2 cm AND angle_error < 0.15 rad (about 8.6°).

### Physics

| Parameter | Value | Why |
|-----------|-------|-----|
| `sim_dt` | 2 ms | 500 Hz physics |
| `ctrl_dt` | 20 ms | 50 Hz policy |
| substeps | 10 | sim_dt × substeps = ctrl_dt |
| Pusher mass | 0.1 kg | Light enough to be responsive |
| Block mass | 0.05 kg | Half of pusher |
| Pusher-block friction | 0.3 | Low → smooth rolling |
| Block-table friction | 0.4 | Moderate — block slides but not frictionless |
| `solref` | `"0.004 1"` | 4 ms contact resolution — clean, no visible penetration |
| `solimp` | `"0.98 0.995 ..."` | Nearly rigid contact |
| `iterations` | 50 | Newton solver at the stiffer problem |
| Integrator | `implicitfast` | Standard for stiff contacts |

### Shape Registry

```python
SHAPES: dict[str, list[dict]] = {
    "T":      [stem_box, top_box],
    "L":      [vertical_box, horizontal_box],
    "circle": [disc_cylinder],
    "plus":   [horizontal_bar, vertical_bar],
}
```

Adding a new shape = add an entry. XML is built programmatically by `build_xml(shape)` — template contains the shared structure (pusher, walls, lights, target ghost), shape-specific geoms are injected. Target ghost (transparent, no collisions) mirrors the same geoms for visual goal reference.

### Config

```python
default_config() → ConfigDict(
    ctrl_dt=0.02,
    sim_dt=0.002,
    episode_length=200,           # ctrl steps per episode (4 seconds real-time)
    shape="T",                    # see SHAPES registry
    action_mode="position",       # see Action section
    reward_type="dense",          # "dense" | "sparse"
    pos_threshold=0.02,           # m
    angle_threshold=0.15,         # rad
    randomize_target=True,
    randomize_block=True,
    table_half=0.22,              # spawn region half-size
    max_pusher_speed=0.25,        # m/s, used by teleport mode
    impl="warp",
)
```

### Methods

`reset(rng) → State`
: Sample block pose + target pose (if randomized). Place pusher at fixed offset behind block.

`step(state, action) → State`
: Map action to `ctrl` per action_mode, run `n_substeps` sim steps, compute reward.

## Adaptability Benchmark

Core use case:

1. Train policy on `PushEnv(shape="T")` with randomized target.
2. Zero-shot evaluate on `PushEnv(shape="L" | "circle" | "plus")` with the same policy weights.
3. Compare return distributions — a policy that memorized T-geometry will fail; a policy that learned pushing physics will transfer.

Observation space is identical across shapes. Only the block's response to contact differs. This is the minimal perturbation that isolates dynamics-generalization from representation-generalization.

## Comparison to gym-pusht

This env is **not** a drop-in replacement for the Diffusion Policy gym-pusht benchmark:

| Aspect | Our PushEnv | gym-pusht |
|--------|-------------|-----------|
| Simulator | MuJoCo Warp | pymunk (2D) |
| Batch | GPU-vmap 1024+ envs | 1 env per process |
| Throughput | 360k sps @ 1024 envs | ~10k sps single |
| Physics | Rigid body + soft contact | 2D impulse-based |
| Default action | `position-PD` | Absolute target with speed cap |
| Obs | 16d (center + angle + vels + last_act) | Keypoints (pixel-space) |
| Shape-agnostic | ✅ | ❌ (keypoints encode shape) |

For fair algorithm comparisons against published gym-pusht results, use `action_mode="teleport"` (speed-capped) and ignore the shape-transfer axis. For our adaptability benchmark, the differences are features not bugs.
