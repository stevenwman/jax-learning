# Go2 MJCF Comparison: Training (Menagerie MJX) vs Deployment (unitree_mujoco)

**Created:** 2026-03-26
**Purpose:** Document exactly why policies trained in our MJX env don't transfer to unitree_mujoco, and what needs to change.

---

## Summary

Two completely different physics environments that model the same robot. Same keyframe, same joint names, different everything else.

---

## Actuator Model (THE KILLER)

| Property | Training (Menagerie MJX) | Deployment (unitree_mujoco) |
|----------|------------------------|-----------------------------|
| **Type** | `general` (biastype="affine") | `motor` (pure torque) |
| **PD location** | Inside MuJoCo actuator (gainprm/biasprm) | External — DDS bridge computes PD, writes torque to ctrl |
| **gainprm[0]** | 35 (overridden from Menagerie's 50) | 1.0 (unity — ctrl IS the torque) |
| **biasprm** | [0, -35, -0.5] (position + velocity feedback) | [0, 0, 0] (no built-in feedback) |
| **ctrlrange** | Joint angle ranges (radians) | Torque limits (Nm: ±23.7 hip, ±45.43 knee) |
| **What ctrl means** | Position target → MuJoCo applies PD | Raw torque → applied directly |

**Why this matters:** Our policy outputs position targets that MuJoCo's affine actuator converts to torque via built-in PD. In unitree_mujoco, our deploy code correctly computes PD externally (robot_interface.py), but the effective dynamics differ because:
- Training: `force = Kp*(target - q) - Kd*qvel` computed by MuJoCo with its specific integration
- Deploy: `ctrl = tau + kp*(q_target - q_sensor) + kd*(dq_target - dq_sensor)` computed in Python, then MuJoCo applies ctrl as raw torque

The PD math is the same, but the integration timing differs (training PD is inside the physics substep, deploy PD is outside at DDS rate).

---

## Joint Dynamics

| Property | Training | Deployment | Ratio |
|----------|----------|------------|-------|
| **dof_damping** | ~~0.5~~ → **0.1** (fixed 2026-03-26) | 0.1 (XML default) | **NOW MATCHED** |
| **armature** | 0.01 | 0.01 | 1x |
| **frictionloss** | 0.2 | 0.2 | 1x |

**Joint damping was 5x different — NOW FIXED.** Changed training Kd from 0.5 to 0.1 to match hardware/unitree_mujoco.

---

## Joint Ranges

| Joint | Training | Deployment | Diff |
|-------|----------|------------|------|
| All hip (abduction) | [-1.0472, 1.0472] | [-1.0472, 1.0472] | Same |
| Front thigh | [-1.5708, 3.4907] | [-1.5708, 3.4907] | Same |
| **Rear thigh** | **[-1.5708, 3.4907]** | **[-0.5236, 4.5379]** | **Different** |
| All knee | [-2.7227, -0.8378] | [-2.7227, -0.8378] | Same |

Rear thigh range differs — unitree_mujoco uses the actual Go2 rear hip range which is asymmetric from the front. Menagerie MJX uses the front range for all legs.

---

## Contact Physics

| Property | Training (after overrides) | Deployment | Match? |
|----------|---------------------------|------------|--------|
| **Foot solimp** | [0.9, 0.95, 0.023] | [0.9, 0.95, 0.001] | Close |
| **Foot condim** | 3 | 6 | **NO** — 3D vs 6D friction |
| **Foot friction** | [0.6, 0.005, 0.0001] | [0.4, 0.02, 0.01] | **NO** |
| **Friction cone** | pyramidal | elliptic | **NO** |
| **Calf geom friction** | 0.6 | 0.4 | Different |

condim=3 (our training) vs condim=6 (unitree_mujoco) is significant. condim=6 enables rolling and spinning friction which affects foot contact dynamics during walking.

---

## Actuator Ordering

| Index | Training | Deployment |
|-------|----------|------------|
| 0 | FL_hip | FR_hip |
| 1 | FL_thigh | FR_thigh |
| 2 | FL_calf | FR_calf |
| 3 | FR_hip | FL_hip |
| 4 | FR_thigh | FL_thigh |
| 5 | FR_calf | FL_calf |
| 6 | RL_hip | RR_hip |
| 7 | RL_thigh | RR_thigh |
| 8 | RL_calf | RR_calf |
| 9 | RR_hip | RL_hip |
| 10 | RR_thigh | RL_thigh |
| 11 | RR_calf | RL_calf |

Pattern: FL↔FR and RL↔RR swapped. **Already handled** by deploy/go2_constants.py remapping.

---

## Sensor Layout

| Index | Training (Menagerie) | Deployment (unitree_mujoco) |
|-------|---------------------|----------------------------|
| 0-11 | Joint pos (FL,FR,RL,RR order) | Joint pos (FR,FL,RR,RL order) |
| 12-23 | Joint vel (same order) | Joint vel (same order) |
| 24+ | gyro, accel, orientation, global_pos/vel/angvel, local_linvel, upvector, fwdvector, foot pos, foot vel, floor contacts | Joint torques (12), then imu_quat, gyro, accel, frame_pos, frame_vel |

Sensor ordering within joint groups follows actuator ordering — different between models. **Already handled** by deploy/obs_builder.py remapping.

---

## Physics Solver

| Property | Training | Deployment |
|----------|----------|------------|
| **timestep** | 0.004s | 0.005s |
| **ctrl_dt** | 0.02s (5 substeps) | ~0.02s (DDS at 50Hz) |
| **friction cone** | pyramidal | elliptic |
| **impratio** | 100 | 100 |
| **CCD iterations** | 20 (overridden) | default |
| **solver iterations** | 1 | default |
| **euler damp** | disabled | default (enabled) |

---

## What Matches

- Keyframe default pose: `[0, 0.9, -1.8] * 4` — identical
- Base standing height: 0.27m — identical
- Joint names: same (FL_hip_joint, etc.)
- Mesh geometry: same Go2 model
- Gravity: -9.81 m/s²
- Hip/knee joint ranges (except rear thigh)
- Armature, frictionloss

---

## Reconciliation Options

### Option A: Match training to unitree_mujoco (recommended for sim2real)
Change go2_base.py runtime overrides to match unitree_mujoco:
1. ~~`dof_damping[6:] = 0.1` (from 0.5)~~ — **DONE** (2026-03-26)
2. Foot condim = 6 (from 3) — DEFERRED (sim approximation, not hardware property)
3. Foot friction = [0.4, 0.02, 0.01] — DEFERRED (sim approximation)
4. Friction cone = elliptic (from pyramidal) — DEFERRED (sim approximation)
5. ~~Rear thigh range = [-0.5236, 4.5379]~~ — **DONE** (2026-03-26)
6. Retrain PPO/SAC with these physics — **IN PROGRESS** (seed 3100, 50M steps)

**Pro:** Policy trained on matching physics has best shot at transferring.
**Con:** May need reward retuning (the Go2 reward balance saga again).

### Option B: Domain randomization over both
Randomize damping [0.1, 0.5], friction [0.4, 0.6], condim {3, 6}, etc. during training.

**Pro:** Most robust — covers both models and real hardware.
**Con:** Most work, slower training, may need more steps to converge.

### Option C: Adapt unitree_mujoco to match training
Modify unitree_mujoco's go2.xml to use general actuators with our gainprm/biasprm, our damping, our contacts.

**Pro:** No retraining needed.
**Con:** Defeats the purpose — we want to deploy to a model that matches real hardware.
