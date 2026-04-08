# Environments

MuJoCo-based environments for quadruped locomotion. All environments use MuJoCo Warp (`impl="warp"`) for GPU-accelerated physics.

| Environment | Task | Obs (state) | Obs (privileged) | Actions |
|-------------|------|-------------|-------------------|---------|
| [WarpJoystick](#warpjoystick) | Track velocity command | 48d | 122d | 12 (joint targets) |
| [BongoHandstand](#bongohandstand) | Handstand on bongo board | 42–46d | ~96d | 12 (joint targets) |

Reward and observation specs are data-driven — swap terms without touching environment internals.

---

## WarpJoystick

```python
from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick
```

Track a joystick velocity command (vx, vy, yaw rate) with the Unitree Go2. The primary locomotion environment — used for all benchmarks and sim-to-real deployment.

**Key properties:**

- 17 reward terms (tracking, stability, regularization, feet contact)
- Velocity command sampled via Bernoulli/Uniform process (changes mid-episode)
- Domain randomization: friction, damping, mass, motor strength

**Methods**

`reset(rng) → State`
: Initialize episode with randomized base position/yaw and sample velocity command.

`step(state, action) → State`
: Execute one physics step, update command, compute rewards.

`sample_command(rng, x_k) → command`
: Sample next velocity command (3d: vx, vy, wz).

`get_domain_randomization_spec() → list[DRSpec]`
: Declare per-episode randomization ranges.

---

## BongoHandstand

```python
from jax_rl.envs.locomotion.go2_bongo_handstand import BongoHandstand
```

Go2 handstand balance on a bongo board. Applies antagonistic pushes to the robot base and board during training for robustness.

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
