# Actuator models beyond ideal PD

## Three levels of fidelity

Frameworks in the wild use one of three actuator models, in order of complexity:

| Model | What it captures | Data needed |
|---|---|---|
| **Ideal PD** (ours, most papers) | `τ = Kp(q* - q) - Kd·q̇`, clipped by MJCF `actuator_ctrlrange` at peak torque. | None — just PD gains. |
| **Torque-speed curve** (MJLab `DcActuator`, ours optional) | Peak torque decreases linearly with joint velocity: `τ_limit = τ_stall · max(1 - |q̇|/q̇_max, 0)`. | Stall torque + velocity limit per joint. |
| **Learned MLP actuator** (MJLab `LearnedMlpActuator`, ETH Legged Gym) | TorchScript net trained on real robot (position error, velocity history) → torque. Captures delays, backlash, friction nonlinearly. | Real robot data. |

"DC motor" in most frameworks is **not** an electromechanical simulation (no back-EMF, no current, no voltage). It's the linear torque-speed curve — a kinematic approximation of motor saturation.

## Our implementation

Optional, off by default. Flag lives at `config.torque_speed_model`. Enabled via registered env variant `Go2WarpJoystickFlatTorqueSpeed`.

Source of truth:
- **Stall torque per joint:** read from MJCF `actuator_ctrlrange[:, 1]` at init, remapped to joint order. For Go2: hip/thigh = 23.7 Nm, calf = 45.43 Nm (MJCF). No hardcoded value.
- **Velocity limit per joint:** from Unitree URDF (`unitree_rl_gym/resources/robots/go2/urdf/go2.urdf`). Hip/thigh = 30.1 rad/s, calf = 20.07 rad/s. Stored in [`go2_constants.py`](../../jax_rl/envs/locomotion/go2_constants.py) as `MOTOR_VELOCITY_LIMIT_PER_JOINT_TYPE`.

Shared helper `Go2WarpEnv._apply_torque_speed_limit(tau_joint, dq)` called from each env's substep. Zero overhead when flag off (Python bool evaluated at trace time → branch baked out of JIT).

## Flag propagation — the one trap

The flag is a Python bool read at env `__init__`, then closed over inside JIT-traced `substep`. Consequence:

- **Safe:** set via config at construction → flows through trace → compiles in the right branch.
- **Silent foot-gun:** swapping the actuation after construction (`env._actuation = MotorModel()`, or mutating `config.torque_speed_model`) has **no effect** on an already-traced `step()`. The JIT cache holds the graph compiled with the old value. (The actuation is now an `Actuation` component — `env._actuation`, TorqueOnly/MotorModel, built from config at `__init__` by `actuation_from_config`; the old `env._torque_speed_model` bool is gone.)

Rule: toggle via env name (use the registered variant), never by attribute mutation. The registry path is deterministic — two env names produce two distinct compilations.

## Quantitative finding: the clip rarely fires during walking

Trajectory analysis from 1000-step rollout (FastSAC, Go2 joystick flat, `vx ≈ 1 m/s`):

| policy → env | peak |q̇| (rad/s) | % steps > 50% limit | % saturated | mean scale |
|---|---|---|---|---|
| baseline → baseline | 15.6 | 0.38% | **0%** | 0.918 |
| baseline → TS | 15.6 | 0.38% | **0%** | 0.918 |
| TS-trained → TS | 13.7 | 0.45% | **0%** | 0.920 |

- Peak joint velocity ~15 rad/s vs limits of 20–30 rad/s → always sub-saturation.
- Mean torque-speed scale 0.92 — the curve reduces allowance by ~8% average, which is below the MJCF peak-torque hard clip (which also almost never fires during steady gait).
- `baseline → baseline` and `baseline → TS` trajectories are identical to 4 digits — same policy, dormant clip, same dynamics.
- The TS-trained policy settles at **slightly lower peak velocities** (13.7 vs 15.6). Small, but suggests the model is shaping behavior at the margin even when rarely firing.

## When the torque-speed model matters

It's essentially dormant during flat 1 m/s joystick walking. It wakes up under:

- **Commanded sprint speeds** (>2 m/s) — joint velocities scale with gait frequency.
- **Jumps, recovery motions** — transient spikes well above steady-state.
- **Push-force disturbance rejection** — external forces drive brief high-velocity events.
- **Aggressive angular velocity tracking** — yaw rate commands push hip joints hard.

For flat terrain + moderate speeds, it is ~free insurance. For push curriculum + high-speed deployment, it starts to shape the policy measurably.

## Adoption recommendation

- **Not** as default for baseline joystick tasks — no eval gain, adds nothing it changes.
- **Yes** as default when introducing push forces, sprint commands, or any regime that drives joints near velocity limits.
- **Yes** for sim-to-real runs where transient disturbances are expected on hardware.

Switch via `--env Go2WarpJoystickFlatTorqueSpeed` in training scripts. Same hyperparams as the base variant; clean A/B comparable.

## When to upgrade to learned actuator

Only when you have real robot data. The linear torque-speed curve is a kinematic approximation; the learned MLP captures:
- Actual torque transfer function (including nonlinear gear effects)
- Backlash
- Motor controller delays
- Coulomb friction curve

Requires recording real Go2 position commands + joint sensors over a variety of gaits, then fitting an MLP. Deferred until we have a physical robot.
