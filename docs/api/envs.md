# Environments

MuJoCo-based environments for quadruped locomotion. All environments use MuJoCo Warp (`impl="warp"`) for GPU-accelerated physics. For the planar-pushing manipulation benchmark, see [PushT](pusht.md).

| Environment | Task | Obs (state) | Obs (privileged) | Actions |
|-------------|------|-------------|-------------------|---------|
| [WarpJoystick](#warpjoystick) | Track velocity command | 48d | 122d | 12 (joint targets) |
| [WarpJoystick (+torque-speed)](#actuator-models) | Same, with motor saturation curve | 48d | 122d | 12 (joint targets) |
| [WarpJoystickCurriculum](#warpjoystickcurriculum) | Goal-directed locomotion on 4-type × 10-level terrain grid | 48d | 122d | 12 (joint targets) |
| [WarpJoystickCurriculum (+torque-speed)](#warpjoystickcurriculum) | Same, with motor saturation curve | 48d | 122d | 12 (joint targets) |
| [BongoHandstand](#bongohandstand) | Handstand on bongo board | 42–46d | ~96d | 12 (joint targets) |

Reward and observation specs are data-driven — swap terms without touching environment internals.

!!! note "Class names vs registered names"
    The class `WarpJoystick` is registered as environment `Go2WarpJoystickFlat` (flat terrain variant). Use the registered name in CLI flags (`--env Go2WarpJoystickFlat`) and the class name for imports.

---

## WarpJoystick

```python
from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick
```

Track a joystick velocity command (vx, vy, yaw rate) with the Unitree Go2. The primary locomotion environment — used for all benchmarks and sim-to-real deployment.

**Key properties:**

- Velocity command sampled via Bernoulli/Uniform process (changes mid-episode)
- Domain randomization: friction, damping, mass, motor strength

???+ note "17 reward terms"
    | Term | What it does |
    |------|-------------|
    | `tracking_lin_vel` | Gaussian bonus for matching commanded forward/lateral velocity |
    | `tracking_ang_vel` | Gaussian bonus for matching commanded yaw rate |
    | `lin_vel_z` | Penalizes vertical base velocity |
    | `ang_vel_xy` | Penalizes roll/pitch rotation |
    | `orientation` | Penalizes body tilt from upright |
    | `base_height` | Penalizes deviation from target standing height (0.27m) |
    | `torques` | Penalizes motor effort (L2 + L1) |
    | `action_rate` | Penalizes consecutive action changes (smoothness) |
    | `energy` | Penalizes joint velocity × actuator force |
    | `dof_pos_limits` | Penalizes joint positions near soft limits |
    | `pose` | Rewards proximity to default stance pose |
    | `stand_still` | Penalizes pose deviation when command is zero |
    | `feet_air_time` | Rewards swing phase >0.1s during locomotion |
    | `feet_slip` | Penalizes foot horizontal velocity during ground contact |
    | `feet_clearance` | Penalizes deviation from target swing height |
    | `feet_height` | Penalizes peak foot height error during swing |
    | `termination` | Fixed penalty on early termination |

    Weights are configured separately in the env's reward config — see [Custom Rewards](../tutorials/custom-rewards.md).

**Methods**

`reset(rng) → State`
: Initialize episode with randomized base position/yaw and sample velocity command.

`step(state, action) → State`
: Execute one physics step, update command, compute rewards.

`sample_command(rng, x_k) → command`
: Sample next velocity command (3d: vx, vy, wz).

`get_domain_randomization_spec() → list[DRSpec]`
: Declare per-episode randomization ranges.

### Actuator models

By default, WarpJoystick uses **ideal PD control**: `τ = Kp(q* − q) − Kd·q̇`, clipped by MJCF `actuator_ctrlrange` at peak torque (23.7 Nm hip/thigh, 45.43 Nm calf). Simple, fast, and sufficient for most training.

An optional **linear torque-speed curve** is available — the approximation used by MJLab's `DcActuator` and similar frameworks:

$$\tau_\text{limit} = \tau_\text{stall} \cdot \max\!\left(1 - \frac{|\dot{q}|}{\dot{q}_\text{max}},\ 0\right)$$

Peak torque decreases linearly with joint velocity, reaching zero at the velocity limit (30.1 rad/s hip/thigh, 20.07 rad/s calf — from the Unitree URDF).

Enable via the registered variant env:

```bash
uv run python train_fast_sac.py --env Go2WarpJoystickFlatTorqueSpeed \
    --reset-mode per_step --total-timesteps 20000000 --wandb
```

The variant uses identical hyperparameters to `Go2WarpJoystickFlat` — clean A/B comparable.

!!! note "When it matters"
    Joint velocities during flat-terrain walking at 1 m/s peak around 15 rad/s — well below the 20–30 rad/s limit. The clip is essentially dormant during steady gait (0% saturation in 1000-step rollouts). It activates under sprint commands, jumps, recovery motions, or push-force disturbances. Adopt as default when those regimes are on the menu; skip otherwise.

    Full analysis and quantitative trajectory data: [lessons/actuator_models.md](https://github.com/stevenwman/jax-learning/blob/main/.context/lessons/actuator_models.md).

---

## WarpJoystickCurriculum

Procedurally generated terrain with per-env curriculum advancement. Robot navigates toward a per-episode world-frame goal; curriculum advances when the robot reaches the goal, demotes when the robot falls or fails to make progress.

**Variants:**
- `Go2WarpJoystickCurriculum` — default actuator (ideal PD).
- `Go2WarpJoystickCurriculumTorqueSpeed` — with linear torque-speed actuator curve.

**Terrain grid:** 10 difficulty levels × 4 terrain types = 40 tiles, each 9.6×9.6m.

**Terrain types:**

| Type | Description | Max height |
|------|-------------|-----------|
| Rough | Grid of boxes with per-cell height variation | 0.22m |
| PyramidStairs (up) | Concentric rings rising to center platform | 0.4m step |
| InvertedPyramidStairs (bowl) | Rim at ground, descends to pit | 0.4m step |
| TiltedGrid | Grid of tiles with random tilt | 25° |

**Column assignment:** each env is fixed to one terrain type for the entire training run (`env_id % 4`). Specialization pattern from legged_gym.

**Goal-directed commands:** each episode samples a world-frame goal (opposite edge for rough/tilted, center for pyramid/inverted). Body-frame command is computed per-step from goal + robot pose via P-controller on yaw; `target_speed` scales linearly with `terrain_level` (0.5 m/s at level 0 → 1.5 m/s at level 9).

**Curriculum advancement:** at episode end,

- `reached_goal AND NOT fallen` → level += 1
- `fallen` → level -= 1
- `timeout AND min_distance > 0.5 × initial_distance` → level -= 1
- otherwise → stay

Level clamped to `[0, num_rows-1]`.

**wandb metrics:** terrain metrics are logged automatically when `--wandb` is active. Keys: `terrain/{type}/mean_level`, `terrain/{type}/reach_rate`, `terrain/{type}/fall_rate`, `terrain/{type}/promote_rate`, `terrain/{type}/demote_rate`, and `terrain/global/*` aggregates.

**Usage:**
```bash
uv run python train_fast_sac.py --env Go2WarpJoystickCurriculum --reset-mode per_step --num-envs 64 --wandb
```

!!! note "VRAM budget"
    Curriculum env has ~1500 geoms (vs ~100 for flat) — needs fewer parallel envs than flat. `--num-envs 64` fits in ~13 GB VRAM.

See [lessons/terrain_curriculum.md](https://github.com/stevenwman/jax-learning/blob/main/.context/lessons/terrain_curriculum.md) for design rationale and gotchas.

---

## BongoHandstand

```python
from jax_rl.envs.locomotion.go2_bongo_handstand import BongoHandstand
```

Go2 handstand balance on a bongo board. Applies antagonistic pushes to the robot base and board during training for robustness.

???+ note "10 reward terms"
    | Term | What it does |
    |------|-------------|
    | `survival` | Constant +1.0 per step for staying alive |
    | `orientation_cost` | Penalizes gravity vector error from inverted stance |
    | `board_tilt_cost` | Penalizes bongo board tilt magnitude |
    | `com_offset_cost` | Penalizes COM horizontal distance from board center |
    | `height_cost` | Penalizes COM height error from target (0.55m) |
    | `roller_cost` | Penalizes roller slider displacement |
    | `torque_cost` | Penalizes motor effort relative to max torque |
    | `action_rate_cost` | Penalizes consecutive action changes |
    | `joint_vel_cost` | Penalizes joint velocity magnitude |
    | `termination` | Fixed penalty on early termination |

**Methods**

`reset(rng) → State`
: Initialize from "handstand" keyframe (no position randomization).

`step(state, action) → State`
: Physics step with perturbation pushes.

`get_domain_randomization_spec() → list[DRSpec]`
: Randomize friction, mass, torso COM, motor strength, damping, armature.

---

## Reward Spec

```python
from jax_rl.envs.reward_spec import RewardTerm, compute_rewards
```

Data-driven reward composition. Each environment defines a list of `RewardTerm`s; weights are applied downstream in `step()`.

**RewardTerm** (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique identifier (e.g., `"tracking_lin_vel"`) |
| `fn` | `Callable` | Receives kwargs (`data`, `action`, `info`, `done`, ...), returns scalar |

**compute_rewards**(terms, \*\*kwargs) → dict[str, Array]
: Evaluate all terms, return `{name: unweighted_scalar}` dict.

---

## Observation Spec

```python
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs
```

Data-driven observation composition with per-term noise injection.

**ObsTerm** (dataclass)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Unique identifier |
| `fn` | `Callable` | — | Receives kwargs (`data`, `info`, ...), returns array |
| `noise_scale` | `float` | `0.0` | Multiplied by global `noise_level` |

**IncludeGroup** (dataclass)
: Reference another group's computed output (e.g., privileged includes state). Field: `group_name: str`.

**compute_obs**(groups, noise_level, rng, \*\*kwargs) → (obs_dict, new_rng)
: Compute all observation groups. Groups processed in insertion order; noise is uniform in `[-noise_level * scale, +noise_level * scale]`.
