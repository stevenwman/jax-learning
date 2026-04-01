# Go2 Bongo Board Handstand — Design Spec

**Date:** 2026-04-01
**Status:** Draft
**Branch:** `new_slate_linen`

## Goal

Train a Unitree Go2 to maintain a handstand (front legs down, rear legs up) on a bongo board. Pure balance/skill task, not locomotion.

Two planned difficulty tiers:
- **Phase 1A: `Go2BongoHandstand`** — robot spawns already inverted on the board. Balance-only.
- **Phase 1B: `Go2BongoHandstandFull`** (future) — robot spawns standing normally, must approach, mount, and invert. Multi-stage task requiring curriculum or staged rewards.

This spec covers Phase 1A only.

---

## Bongo Board MJCF

### Physical model

Board and roller are one kinematic tree — the roller is a child body of the board connected by joints, not a separate free body.

| Component | Dimensions | Notes |
|-----------|-----------|-------|
| Board | 30" x 8" x ~1.2" (0.762 x 0.2032 x 0.03m) | Free joint, 2kg |
| Roller (main hub) | 4.6" diam x 3" wide | Ground contact surface |
| Roller (extensions) | 3.9" diam x 8" wide | Full-width visual body |

### Roller DOFs

The roller has two joints relative to the board:
- **Slide** (along board long axis X): range ±9" (±0.2286m), damped
- **Hinge** (spin around roller axis Y): free rotation, lightly damped

These are coupled by a **no-slip equality constraint**:

```
slide_pos = -radius × spin_angle
```

Enforced via `<equality><joint>` with tight `solref="0.001 1"` / `solimp="0.99 0.999 0.001"`. This gives the roller 1 effective DOF — rolling it translates it along the board, and vice versa.

### Contact

- **Roller ↔ ground:** real contact with friction (0.8). The roller sits on the floor.
- **Board ↔ roller:** no contact pair. They're in the same kinematic tree, connected by joints. The equality constraint replaces friction.
- **Robot feet ↔ board:** real contact. Front feet (FL, FR) land on `board_top` geom. Scene XML includes contact sensors for `FL`/`FR` against `board_top` (not floor).

### Files

- `jax_rl/envs/locomotion/xmls/bongo_board.xml` — board + roller model (included by scene)
- `jax_rl/envs/locomotion/xmls/go2_bongo_scene.xml` — scene combining Go2 + bongo board + floor + sensors
- `jax_rl/envs/locomotion/xmls/bongo_test_scene.xml` — standalone board test scene (no robot)

---

## Scene XML

`go2_bongo_scene.xml` includes:
1. `unitree_go2/go2.xml` **first** (robot qpos occupies indices 0–18)
2. `bongo_board.xml` **second** (board freejoint at qpos[19:26], roller joints at qpos[26:28])
3. Floor geom with friction
4. Go2 IMU sensors (same set as `go2_warp_scene_flat.xml`)
5. Board-specific contact sensors (`FL_board_found`, `FR_board_found` referencing `geom2="board_top"`)
6. `handstand` keyframe with robot inverted on board

### qpos/qvel layout (include order: Go2 first, board second)

```
qpos[0:7]   = Go2 base freejoint (pos xyz + quat wxyz)
qpos[7:19]  = Go2 12 joint angles (FL,FR,RL,RR × hip,thigh,calf)
qpos[19:26] = Board freejoint (pos xyz + quat wxyz)
qpos[26]    = roller_slide
qpos[27]    = roller_spin
```

**Index resolution strategy:** The env resolves ALL body/joint/geom indices by name at `_post_init()` time using `mj_model.joint("name").qposadr` etc. No hardcoded offsets beyond the robot joint range `qpos[7:19]` / `qvel[6:18]` which are guaranteed by Go2 being the first included body. The board/roller indices are looked up by name:

```python
self._board_jnt_qposadr = self._mj_model.joint("board_joint").qposadr
self._roller_slide_qposadr = self._mj_model.joint("roller_slide").qposadr
self._roller_slide_dofadr = self._mj_model.joint("roller_slide").dofadr
self._board_body_id = self._mj_model.body("board").id
```

---

## Env Class: `Go2BongoHandstand`

### Inheritance

Subclasses `Go2WarpEnv` (same as `WarpJoystick`). Inherits:
- MJCF loading + Warp backend setup
- PD control with external substep loop
- Joint→actuator remapping (`_act_to_joint`)
- Forcerange fix (ctrlrange → forcerange)
- IMU sensor helpers (gyro, gravity, etc.)

### Config

```python
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=500,       # 10s — falls are fast
        Kp=20.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.25,        # smaller — fine balance corrections
        soft_joint_pos_limit_factor=0.95,
        observe_board_state=True,  # expose board tilt + roller pos in obs
        target_handstand_height=0.45,  # placeholder — refine after keyframe
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                inverted_orientation=10.0,
                board_level=8.0,
                com_above_support=5.0,
                height=-5.0,
                roller_centered=-2.0,
                torques=-0.0002,
                action_rate=-0.01,
                termination=-1.0,
            ),
        ),
        impl="warp",
        contact_mode="training",
        naconmax=4 * 8192,   # may need tuning — board adds roller-ground
        naccdmax=5000,        # bumped from 4000: cylinder-plane contacts
        njmax=150,            # bumped: equality constraint + extra contacts
    )
```

### Observation

**Config flag:** `observe_board_state` (default `True`)

`"state"` (policy obs):

| Component | Dim | Condition |
|-----------|-----|-----------|
| gyro (noised) | 3 | always |
| gravity (noised) | 3 | always |
| joint_pos - handstand_default_pose (noised) | 12 | always |
| joint_vel (noised) | 12 | always |
| last_action | 12 | always |
| board_tilt_xy | 2 | if `observe_board_state` |
| roller_slide_pos | 1 | if `observe_board_state` |
| roller_slide_vel | 1 | if `observe_board_state` |
| **Total** | **42 or 46** | |

`"privileged_state"` (critic obs):

| Component | Dim | Notes |
|-----------|-----|-------|
| state | 42/46 | noised policy obs |
| gyro (unnoised) | 3 | |
| gravity (unnoised) | 3 | |
| joint_pos (unnoised) | 12 | |
| joint_vel (unnoised) | 12 | |
| actuator_force | 12 | |
| FL/FR board contact | 2 | foot-on-board contact flags |
| board_tilt_xy (unnoised) | 2 | always in privileged, even if not in state |
| roller_slide_pos (unnoised) | 1 | |
| roller_slide_vel (unnoised) | 1 | |
| board_angular_vel_xy | 2 | board tilt rate |
| CoM position xy | 2 | horizontal CoM relative to board center |
| **Total** | **~94 or ~98** | |

### Board state extraction

**Board tilt:** extracted from the board body's rotation matrix as the z-column of xmat, projected to xy. This gives `sin(tilt_angle)` which is linear near zero and saturates at extreme angles:

```python
board_xmat = data.xmat[self._board_body_id].reshape(3, 3)
board_tilt_xy = board_xmat[:2, 2]  # x,y components of board's z-axis in world frame
# board_tilt_xy = [0, 0] when level, magnitude increases with tilt
```

**Roller state:** read directly from joint qpos/qvel:

```python
roller_slide_pos = data.qpos[self._roller_slide_qposadr]
roller_slide_vel = data.qvel[self._roller_slide_dofadr]
```

**Board center (world position):** `data.xpos[self._board_body_id][:2]` — the board body's world-frame xy position.

### Rewards

| Term | Type | Formula |
|------|------|---------|
| `inverted_orientation` | reward | `exp(-sum((gravity_body - [0, 0, 1])²))` — full distance to target, creates gradient toward inverted state |
| `board_level` | reward | `exp(-sum(board_tilt_xy²) / 0.1)` |
| `com_above_support` | reward | `exp(-sum((subtree_com[torso_id][:2] - xpos[board_id][:2])²) / 0.05)` |
| `height` | cost | `(subtree_com[torso_id][2] - target_handstand_height)²` |
| `roller_centered` | cost | `roller_slide_pos²` |
| `torques` | cost | `sqrt(sum(torques²)) + sum(abs(torques))` (same as joystick) |
| `action_rate` | cost | `sum((action - last_action)²)` |
| `termination` | cost | `done` flag |

All rewards weighted by `config.reward_config.scales`, summed, then `clip(total * dt, 0, 10000)`.

Note: `inverted_orientation` uses full 3D distance `(gravity_body - [0,0,1])²` rather than just `gravity_body[:2]²`. This distinguishes between inverted (gravity_body_z ≈ +1, distance ≈ 0) and upright (gravity_body_z ≈ -1, distance ≈ 4), creating a proper gradient toward the handstand.

### Termination

- **Board tilt too extreme:** `sum(board_tilt_xy²) > 0.25` (corresponds to ~30° tilt, since `sin(30°) = 0.5`, and `0.5² = 0.25` for a single axis)
- **Robot not inverted:** `gravity_body[2] < 0` (should be positive when inverted)
- **Robot base too low:** `subtree_com[torso_id][2] < 0.15`
- **Roller at limit:** `abs(roller_slide_pos) > 0.22` (near ±9" limit)

### Step

Same pattern as `WarpJoystick.step()`:
1. Compute motor targets: `handstand_default_pose + action * action_scale`
2. External PD substep loop with `lax.scan` (Kp/Kd, act_to_joint remap)
3. Compute obs, rewards, termination
4. Update info dict (last_act, step_count, etc.)

No velocity commands. No random kicks (the board instability IS the perturbation).

### Contact mode

The base class overrides foot contact parameters for `contact_mode="training"`. For the bongo env, the `board_top` geom should also get training-mode contact overrides (firm solimp, condim=3) so foot-on-board contact behaves consistently with foot-on-floor. This is done in `_post_init()`:

```python
if self._config.contact_mode == 'training':
    board_gid = self._mj_model.geom("board_top").id
    self._mj_model.geom_solimp[board_gid, :3] = [0.9, 0.95, 0.023]
    self._mj_model.geom_condim[board_gid] = 3
    self._mj_model.geom_friction[board_gid] = [0.8, 0.005, 0.001]
```

### Reset

1. Load `handstand` keyframe qpos (covers both robot and board initial state)
2. Small random perturbations:
   - Joint angles (±0.05 rad)
   - Robot base xy offset from board center (±2cm)
   - Robot yaw (±5°) — not perfectly aligned with board axes
   - Board tilt (±2°)
3. `mjx.forward()` to initialize derived quantities

### Initial Handstand Pose

The hardest design challenge. The robot must be:
- Rotated 180° around pitch (Y axis): quat ≈ `[0, 0, 1, 0]`
- Front feet on the board surface
- Rear legs pointing up
- Joint angles set so the pose is statically stable (even briefly)

Approach: compute geometrically, then iterate with single-frame PNG renders for visual validation. The keyframe will be refined empirically.

---

## Registration

- Add `BONGO_SCENE_XML` path to `go2_constants.py`
- Register `Go2BongoHandstand` in `env_setup.py` via `pg_locomotion.register_environment()`
- Pattern: `functools.partial(BongoHandstand, task="bongo_handstand")`

---

## Tests

`tests/test_go2_bongo_env.py`, following `test_go2_warp_env.py` pattern:

- Model loads, correct body/joint/geom counts (includes board + roller bodies)
- `action_size == 12`
- `reset()` returns dict obs with correct shapes (42 or 46 state, ~94-98 privileged)
- `step()` with zero action: no NaN, finite reward
- `step()` with random action: correct shapes
- Board and roller bodies exist in model
- Equality constraint present
- `observe_board_state=False` gives 42d state
- Board/roller joint indices resolved correctly by name
- `make_envs` integration test (batched, 4 envs)

---

## Domain Randomization (Future)

### Roller radius DR

Randomizing roller radius requires updating coupled quantities at reset time:

| Quantity | Runtime field | Formula |
|----------|--------------|---------|
| Board initial qpos z | `data.qpos` | `2 * r + board_half_t` |
| Roller body offset | `model.body_pos[roller_id, 2]` | `-(board_half_t + r)` |
| Main roller geom radius | `model.geom_size[main_id, 0]` | `r` |
| Extension geom radius | `model.geom_size[ext_id, 0]` | `r - step_delta` |
| No-slip constraint ratio | `model.eq_data[eq_id, ...]` | `-r` |

**Critical:** board spawn height MUST be computed from the randomized radius. Spawning in air (even 1mm) on a bongo board is catastrophic — the drop destabilizes the handstand immediately. The `reset()` function must compute `board_z = 2 * randomized_r + board_half_thickness` and set qpos accordingly. Similarly, the robot's initial z must adapt to the new board height.

### Other DR options (composable with existing Go2 DR)

- Board mass and friction
- Roller-ground friction (`model.geom_friction[roller_main_id]`)
- Robot mass/COM jitter (existing `go2_randomize.py`)
- Kp/Kd scaling (existing, per-env)
- Board initial tilt range (already in reset perturbation)

---

## Open Items

- Handstand keyframe qpos — requires geometric calculation + visual iteration
- Reward weight tuning — initial weights are estimates, will need A/B testing
- Whether `action_scale=0.25` is appropriate for fine balance (may need tuning)
- `target_handstand_height` — placeholder 0.45m, refine after keyframe
- `naccdmax` / `njmax` buffer sizing — may need tuning with robot + board + floor contacts at 1024 envs
