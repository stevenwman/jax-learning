# Go2 Sim-to-Real Plan

**Created:** 2026-03-22
**Status:** Draft — Q&A complete, ready for implementation planning
**North Star Alignment:** This is Phase 5 (sim-to-real) of the framework plan. The Go2 is the first hardware target, but the pipeline should generalize to Go2W, robot arms, and other platforms. The same env/training infrastructure will be used for Phase 6 skill discovery (DIAYN/METRA) on real robots.

---

## 1. High-Level Pipeline

```
[MJCF Model] → [MJX Env] → [JAX Training] → [Policy Export] → [Unitree SDK2 Deploy]
     |              |             |                |                    |
 Menagerie      Playground    PPO/SAC/        JAX→numpy or         CycloneDDS
 go2_mjx.xml    MjxEnv        FastSAC         JAX→ONNX (TBD)      50Hz loop
```

### Sources & References
- Go2 MJCF: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2) — `go2_mjx.xml`
- Env template: [MuJoCo Playground Go1](https://github.com/google-deepmind/mujoco_playground) — `_src/locomotion/go1/`
- Reference implementation: [unitree-go2-mjx-rl](https://github.com/alexeiplatzer/unitree-go2-mjx-rl) — Go2 + MJX + JAX + teacher-student
- Standard recipe: [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) — Unitree's official Go2 RL pipeline
- Reward/domain-rand reference: [legged_gym](https://github.com/leggedrobotics/legged_gym) — ETH RSL foundation
- Gait diversity: [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) + [Go2 fork](https://github.com/Teddy-Liao/walk-these-ways-go2)
- Actuator models: [MJLab](https://github.com/mujocolab/mjlab) — DC motor, delay, learned MLP ([paper](https://arxiv.org/html/2601.22074v1))
- Deployment: [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) — CycloneDDS interface
- Sim validation: [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) — same DDS interface as real robot
- Playground Go2 discussion: [Issue #270](https://github.com/google-deepmind/mujoco_playground/issues/270)

---

## 2. Environment Design

### File: `jax_rl/envs/go2.py`

Subclass Playground's `MjxEnv` (same pattern as `Go1Env` in `mujoco_playground/_src/locomotion/go1/base.py`). Load `go2_mjx.xml` from Menagerie.

**Why MjxEnv, not custom PipelineEnv:** The go2-mjx-rl repo built their own `PipelineEnv` with 4+ inheritance layers. Playground's `MjxEnv` is simpler, already integrated with our training scripts, and handles MJX physics setup. We keep our framework consistent.

**Composable reward/obs pattern (inspired by [MJLab](https://arxiv.org/html/2601.22074v1)):** Instead of a monolithic `step()` method, decompose env behavior into pluggable functions registered by name:

```python
REWARD_TERMS = {
    "tracking_lin_vel": (tracking_lin_vel_fn, 1.0),
    "orientation": (orientation_fn, -5.0),
    "torques": (torques_fn, -0.0002),
    ...
}
```

Each reward term is a standalone function `(state, action, config) → scalar`. The env sums `weight * fn(...)` for all active terms. Swapping between legged_gym and Playground reward sets = changing the weights dict. Adding a term for a new robot = adding one function. Same pattern applies to termination conditions and obs components.

This avoids MJLab's full manager framework (too heavy for us) while getting the composability benefit. Reward terms like `torques`, `action_rate`, `feet_air_time` are reusable across Go2, Go2W, and future robots.

**OPEN QUESTION:** Playground's `MjxEnv` may be too opinionated for custom reward functions. Verify by subclassing `Go1Env` with a modified reward and confirming it works before committing to this approach.

### Observation Space (31 dims per step, no base_lin_vel)

Based on [unitree-go2-mjx-rl obs construction](https://github.com/alexeiplatzer/unitree-go2-mjx-rl) and [Genesis 45-dim approach](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/locomotion/go2_train.py):

| Component | Dims | Scale | Source |
|---|---|---|---|
| Commanded velocity (vx, vy, yaw_rate) | 3 | [2.0, 2.0, 0.25] | Joystick input |
| Yaw rate | 1 | 0.25 | IMU gyroscope |
| Projected gravity (body frame) | 3 | 1.0 | IMU quaternion → rotation |
| Joint position offsets (q - q_default) | 12 | 1.0 | Encoders |
| Previous action | 12 | 1.0 | Stored from last step |
| **Total per step** | **31** | | |

**Frame stacking:** 3-5 frames (configurable). Total obs = 31 × N. Start with N=3 (93 dims), ablate.
Implementation: `jnp.roll` FIFO buffer (same pattern as go2-mjx-rl repo).

**Why no base_lin_vel:** Not measurable on hardware without GPS. Dropping it eliminates a sim-to-real gap. The policy infers velocity from joint position changes across the frame stack.
Sources: [Genesis Go2 example](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/locomotion/go2_train.py), [unitree-go2-mjx-rl](https://github.com/alexeiplatzer/unitree-go2-mjx-rl)

**CRITICAL: No online obs normalization for off-policy algos (SAC/FastSAC Phase B).** Use raw obs + Q LayerNorm. Online normalization with replay buffers causes distribution shift that explodes training. See LESSONS.md "Off-Policy Obs Normalization Explodes Replay." Sample-time normalization (`--obs-norm`) is the safe alternative if needed.

**Observation noise** (applied during training, from [legged_gym](https://github.com/leggedrobotics/legged_gym) and [Playground Go1 config](https://github.com/google-deepmind/mujoco_playground)):
- Joint position: ±0.03 rad
- Joint velocity: ±1.5 rad/s
- Gyroscope: ±0.2 rad/s
- Gravity vector: ±0.05
- Linear velocity (if used): ±0.1 m/s

### Action Space (12 dims)

| Component | Details | Source |
|---|---|---|
| Type | Joint position offsets | All standard pipelines |
| Dims | 12 (4 legs × 3 joints: hip, thigh, calf) | Go2 12-DOF |
| Scaling | `q_target = q_default + action * action_scale` | legged_gym pattern |
| action_scale | **PINNED: 0.25 (Unitree official) vs 0.3 (go2-mjx-rl)** | Test both |
| Clipping | To joint limits per MJCF | Safety |

**Default joint angles** (from [unitree_rl_gym Go2 config](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)):
- FL/FR hip: ±0.1 rad
- FL/FR thigh: 0.8 rad, RL/RR thigh: 1.0 rad
- All calf: -1.5 rad

### PD Controller

**RESOLVED: Kp=20, Kd=0.5** — Unitree's official RL deployment value from [unitree_rl_gym go2_config.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py).

| Source | Kp | Kd | Notes |
|---|---|---|---|
| **[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)** | **20** | **0.5** | **Official Unitree — USE THIS** |
| [Playground Go1](https://github.com/google-deepmind/mujoco_playground) | 35 | 0.5 | Playground's choice, community reports vibration on Go2 |
| [go2-mjx-rl](https://github.com/alexeiplatzer/unitree-go2-mjx-rl) | 35 | 0.5 | MJX-based, not validated on hardware |
| [Playground Issue #270](https://github.com/google-deepmind/mujoco_playground/issues/270) | — | — | Vibration at Kp=35/Kd=0.6 on Go2 |

Kp=35 is Playground's Go1 value, not Go2. Unitree's own repo uses Kp=20 for Go2 specifically.

### Motor Model

Simple PD first, then upgrade. Inspired by [MJLab actuator tiers](https://github.com/mujocolab/mjlab):

**Tier 1 (start here):** Simple PD — `torque = Kp * (q_target - q) + Kd * (0 - dq)`

**Tier 2 (add when needed):** DC motor model — `torque = min(torque_pd, effort_limit * (1 - |dq| / vel_limit))`
- Go2 effort limits: hip/thigh 23.7 Nm, knee 45.43 Nm ([Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2))
- Velocity limit: 30 rad/s ([Isaac Lab Go2 config](https://github.com/isaac-sim/IsaacLab))
- Implementation: ~5 lines of JAX

**Tier 3 (future):** Learned MLP actuator — train on real motor response data. Requires hardware data collection.

**Tier 0.5 (add when needed):** Actuator delay — circular buffer of N commands, apply command from N steps ago. Simulates communication latency (~4ms for Go2 at 500Hz internal loop).

**Upgrade criteria (Tier 1 → Tier 2):** After deploying Tier 1 (simple PD) policy on real Go2, measure: (a) step frequency, (b) motor current draw, (c) joint temperature after 1 min walking. If step frequency matches sim within 10% and current draw within 20%, keep Tier 1. If differences exceed these thresholds or the robot shows jerky/oscillatory behavior at high velocities, implement Tier 2 (DC motor torque-speed curve).

### Reward Function

Configurable source via `reward_source` flag. Start with legged_gym terms (proven for Go2 sim-to-real).

**legged_gym / Playground reward terms:**

| Term | Weight (legged_gym) | Weight (Playground Go1) | Description |
|---|---|---|---|
| tracking_lin_vel | 1.0 | 1.0 | `exp(-error² / σ²)`, σ=0.25 |
| tracking_ang_vel | 0.5 | 0.5 | Same formulation |
| lin_vel_z | -2.0 | -0.5 | Vertical velocity penalty |
| ang_vel_xy | -0.05 | -0.05 | Roll/pitch rate penalty |
| orientation | 0.0 | -5.0 | Non-upright penalty |
| torques | -0.0002 | -0.0002 | Energy efficiency |
| dof_acc | -2.5e-7 | — | Smoothness |
| action_rate | -0.01 | -0.01 | Action jerk penalty |
| feet_air_time | 1.0 | 0.1 | Encourage swing phase |
| collision | -1.0 | — | Self-collision (thigh/calf) |
| dof_pos_limits | -10.0 | -1.0 | Joint limit avoidance |
| feet_slip | — | -0.1 | Foot slip penalty |
| feet_clearance | — | -2.0 | Foot ground clearance |
| energy | — | -0.001 | Energy expenditure |
| termination | — | -1.0 | Episode termination |
| stand_still | — | -1.0 | Joint deviation when commanded to stop |
| pose | — | 0.5 | Default pose tracking |

Sources: [legged_gym LeggedRobotCfg](https://github.com/leggedrobotics/legged_gym), [Playground Go1 joystick.py default_config()](https://github.com/google-deepmind/mujoco_playground)

**Variable posture rewards** (from [MJLab](https://arxiv.org/html/2601.22074v1)): speed-dependent joint penalties — standing, walking, running have different ideal poses. Add as a future enhancement.

---

## 3. Domain Randomization

### File: `jax_rl/envs/domain_rand.py` (robot-agnostic wrapper)

Uses `jax.vmap` over MJX model parameters — each parallel env gets different physics. Same approach as Playground's `randomize.py` and MJLab's `DomainRandomizationVmapWrapper`.

**Parameters to randomize:**

| Parameter | Range | When | Source |
|---|---|---|---|
| Ground friction | [0.5, 1.5] | Reset | [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) |
| Base mass | ±1-3 kg | Reset | [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) |
| Motor strength (Kp) | ±15% | Reset | [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) |
| COM offset | ±0.03m | Reset | [MJLab](https://github.com/mujocolab/mjlab), [Isaac Lab](https://github.com/isaac-sim/IsaacLab) |
| Velocity kicks | up to 1.0 m/s | Every 10-15s | [legged_gym](https://github.com/leggedrobotics/legged_gym) |
| Restitution | [0.0, 0.4] | Reset | [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) |

**Obs noise** is applied in the obs construction (not the wrapper) since it's obs-space specific.

---

## 4. Training

### Phase A: PPO validation
Train Go2 flat terrain walking with PPO using our existing `train_ppo.py`. This validates the env, reward, and domain rand independently from the algo choice. If the humanoid falls over, it's the env, not the algo.

**Config:** 1024-2048 envs, 50Hz control, 200Hz physics (decimation=4), gamma=0.97

### Phase B: SAC/FastSAC
Swap to off-policy once PPO validates the pipeline. The research question: **can SAC/FastSAC match or beat PPO for quadruped locomotion?** If yes, the path to DIAYN/METRA (which wraps SAC) is natural.

**Why this matters for the north star:** DIAYN = SAC + skill discriminator. If SAC can't train a walking policy, DIAYN can't discover walking as a skill. Validating SAC on Go2 is a prerequisite for skill discovery on real robots.

**Phase B prerequisites:**
- **Auto-reset handling:** Playground's `auto_reset=True` means next_obs after done is the new episode's first obs. Off-policy buffer must only bootstrap on truncation, not done. We handle this with `handle_truncation=True` — verify it works correctly with the Go2 env before Phase B training.
- **SAC sanity check:** After PPO reaches stable walking, run a quick SAC baseline on the same task. If SAC matches PPO reward within 2x the steps, proceed. If SAC diverges, debug before committing to Phase B.
- **Memory budget:** With frame stacking (3-5), obs_dim ~93-155. At 1024 envs with 400K buffer, replay buffer uses ~1.5-2GB. Monitor GPU memory — our 16GB RTX 5080 has ~13GB available after VRAM overhead. May need to reduce buffer or envs.
- **No online obs normalization** — see note in Section 2.

---

## 5. Policy Export

**RESOLVED: ONNX export (or pure numpy for simplicity).**

Go2 EDU has **Jetson Orin Nano 8GB** (SM 8.7). JAX does NOT run on it — aarch64 wheels don't target SM 8.7, building from source is painful ([JAX issue #22723](https://github.com/jax-ml/jax/issues/22723)).

| Option | Status | Notes |
|---|---|---|
| ~~JAX direct~~ | **Not viable** | SM 8.7 not supported on Jetson Orin Nano |
| **ONNX Runtime** | **Recommended** | First-class Jetson support, CUDA execution provider |
| **Pure numpy** | **Viable fallback** | For a 3-layer MLP at 50Hz, ~0.1ms inference on ARM CPU |
| TensorRT | Overkill | Highest performance but unnecessary for small MLP |

**Pipeline:** Train in JAX → extract weights as numpy → export to ONNX → deploy with `onnxruntime` on Jetson.

---

## 6. Deployment

### Hardware: Unitree Go2 EDU
**PREREQUISITE: Confirm Go2 EDU edition in lab.** Air/Pro editions don't have low-level joint control.

### Interface: Unitree SDK2 ([unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python))
- CycloneDDS over Ethernet (192.168.123.x)
- Topics: `rt/lowcmd` (send), `rt/lowstate` (receive)
- 50Hz policy loop, 500Hz PD controller internally
- Joint ordering: FL_hip, FL_thigh, FL_calf, RL_*, FR_*, RR_* (12 total)

### Sim validation: [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
Same DDS interface as real robot — deploy script works in sim and on hardware with zero code change (just different DDS domain ID).

**RESOLVED:** `unitree_mujoco` has **zero ML framework dependencies** — it's pure MuJoCo + DDS ([source](https://github.com/unitreerobotics/unitree_mujoco)). No PyTorch required. Our deployment script subscribes to `rt/lowstate`, runs numpy/ONNX inference, publishes `rt/lowcmd`. Same script works in `unitree_mujoco` (sim) and on the real Go2 (hardware) — just change DDS domain ID.

### Deployment script pattern (from [unitree_rl_gym deploy](https://github.com/unitreerobotics/unitree_rl_gym)):
```python
# 50Hz loop
while running:
    state = read_lowstate()  # CycloneDDS
    obs = construct_obs(state)  # joint pos/vel + IMU + command + prev_action
    action = policy(obs)  # inference
    cmd = default_angles + action * action_scale
    send_lowcmd(cmd, kp=20, kd=0.5)  # CycloneDDS
    sleep(0.02)  # 50Hz
```

---

## 7. Sequence

### Step 1: Flat terrain walking (joystick velocity tracking)
- Load Go2 MJCF from Menagerie
- Implement Go2Env with legged_gym reward terms
- Train PPO to walk on flat terrain
- Validate with SAC/FastSAC
- Deploy to real Go2 via Unitree SDK2

### Step 2: Rough terrain locomotion
- Add terrain generation (heightfield, stairs, slopes)
- Terrain curriculum (progressive difficulty)
- Increase frame stack or add RNN for terrain estimation
- More aggressive domain randomization

### Step 3: Recovery behaviors
- Different reward structure (upright orientation, minimal energy)
- Fall detection + recovery trigger
- May need separate policy or multi-task training

---

## 8. Open Questions (Pinned)

| # | Question | Status | When to Resolve |
|---|---|---|---|
| 1 | Can Playground's `MjxEnv` support custom reward functions cleanly? | Verify by subclassing | Before env implementation |
| ~~2~~ | ~~Kp=20 vs 35 for Go2 PD controller~~ | **RESOLVED: Kp=20, Kd=0.5** | Unitree official, confirmed from [go2_config.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py) |
| 3 | Go2 EDU edition in lab? | **Ask Steven to check** | Before deployment work |
| ~~4~~ | ~~Policy export format (JAX vs ONNX)~~ | **RESOLVED: ONNX** | Jetson Orin Nano can't run JAX. ONNX Runtime has first-class support. |
| 5 | Frame stack count (3 vs 5 vs 15) | Ablation | During training experiments |
| 6 | action_scale=0.25 vs 0.3 | Test both | During training experiments |
| ~~7~~ | ~~`unitree_mujoco` DDS compatibility with JAX policies~~ | **RESOLVED: Compatible** | No PyTorch dependency. Numpy/ONNX inference in DDS loop works. |
| 8 | DC motor model — when to upgrade from simple PD? | After hardware test, use concrete metrics | Step freq ±10%, current ±20% thresholds |
| 9 | Actuator delay modeling — needed? | After first hardware test | If sim policy oscillates on hardware |

---

## 9. Infrastructure Impact on Framework

This plan requires building:

| Component | Location | Reusable? |
|---|---|---|
| Go2 env | `jax_rl/envs/locomotion/go2.py` | Template for other robots |
| Domain rand wrapper | `jax_rl/envs/wrappers/domain_rand.py` | Yes — robot-agnostic |
| Obs frame stacking | `jax_rl/utils/frame_stack.py` | Yes — also used for vision RL |
| DC motor model | `jax_rl/envs/actuators.py` | Yes — robot-agnostic |
| Policy export | `jax_rl/utils/export.py` | Yes |
| Deploy script | `deploy/deploy_go2.py` | Template for other robots |

These components serve both the Go2 target AND the broader north star:
- **Frame stacking** → vision RL (Phase 4)
- **Domain rand wrapper** → any sim-to-real task
- **Policy export** → any hardware deployment
- **Env template** → Go2W, robot arms, future platforms

---

## 10. Cross-Plan Alignment Notes

**Builders unification:** Not needed for PPO Phase A (PPO already uses builders). Required before SAC Phase B — SAC/TD3/FastSAC build encoders inline. Order: Go2 env → PPO validation → builders unification → SAC Phase B.

**Frame stacking:** Create `jax_rl/utils/frame_stack.py` as a shared utility. Used by Go2 (state obs) AND vision RL (pixel obs). Playground's `vision=True` handles rendering, NOT frame stacking — our code does frame stacking separately.

**Vision integration path:** Go2Env is a standalone MjxEnv subclass for state-based RL. When vision RL lands, pixel-based Go2 training would use Playground's `vision=True` loader or a wrapper on Go2Env — separate code path, same frame stacking utility.

**Memory budget for future pixel variant:** State-based Go2 at 400K buffer = ~150MB. Pixel-based (84×84×9) at 400K = ~25GB (OOM). Future pixel Go2 needs 50K-100K buffer. See `vision_rl_design.md`.

---

## 11. What NOT to Build

- **Isaac Lab integration** — PyTorch-only, would abandon our JAX framework
- **MJLab adoption** — PyTorch-only, but steal their actuator model patterns
- **quad-sdk integration** — Spirit 40 only, no Go2 support, ROS1 dependency
- **Full teacher-student pipeline** — defer until flat terrain with simple obs stack works
- **Terrain generation** — defer to Step 2 (rough terrain)
- **RNN/Transformer encoder** — defer, frame stacking first
