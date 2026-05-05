# Splitbelt Treadmill — Lessons

> Splitbelt env build + first calibration: 2026-05-03 → 2026-05-05.
> Spec: `.superpowers/specs/2026-05-02-splitbelt-treadmill-env-design.md`
> Plan: `.superpowers/plans/2026-05-02-splitbelt-treadmill-env.md`
> Build journal: `.context/journals/2026-05-03-splitbelt-env-build.md`

---

## MuJoCo plane geoms are infinite — contact-pair `data="found"` margin-fires (2026-05-04)

**What happened:** Splitbelt env defined `fallback_floor` as `<geom type="plane" size="50 1.0 0.1" pos="0 0 -0.005">` and used `<contact name="FL_floor_found" geom1="FL" geom2="fallback_floor" reduce="mindist" num="1" data="found"/>` per foot to detect "off-belt" termination. Every foot at spawn (z=0.014, far above the plane at z=-0.005) registered `floor_found = 1.0`. Robot terminated immediately at step 1 → 1M episode-ends in 1M training steps → no learning signal, eval stuck at 0.3.

**Root cause:** MuJoCo's `<geom type="plane">` is INFINITE in its plane dimensions (the `size` attribute affects rendering only, not collision extent). Contact-pair sensors with `data="found"` use the geom-pair's contact margin — a plane being everywhere at z=-0.005 means it's near every foot, and the sensor fires.

**Fix:** Don't use `floor_found` for off-belt detection. Derive geometrically from foot xy + height:

```python
# splitbelt env _step:
foot_xy = foot_pos_world[..., :2]
foot_belt_id = geom.foot_belt_id(foot_xy, self._belt_layout)  # -1 if outside both belts
foot_off_belt_geom = foot_belt_id == jp.int32(-1)
foot_in_floor = foot_off_belt_geom & (foot_pos_world[..., 2] < 0.02)  # outside belts AND grounded
is_off_belt = jp.any(foot_in_floor)
```

**Lesson:** When a `<contact data="found">` sensor fires when it shouldn't, check whether the geom is a plane. Either replace with a finite box, OR use position-based termination. The contract `data="found"` returns true if margin is satisfied; for planes that's "any near-z geom." Same applies to the bongo lesson — `lessons/bongo.md` says "use contact sensors not position thresholds," but planes are the exception: position is more reliable than the sensor.

---

## `get_gravity()` returns body-frame gravity = -z when upright (2026-05-04)

**What happened:** `is_tilt = (gravity_body[2] < 0.5) | (base_pos_world[2] < 0.18)` always fired. Robot terminated every step.

**Root cause:** Confused the convention. `get_gravity(data)` returns the world gravity vector rotated into body frame. Gravity points "down" in world: `(0, 0, -9.81)` or unit `(0, 0, -1)`. When robot is upright, body z-axis aligns with world z, so gravity_body = `(0, 0, -1)`. Body z = -1, NOT +1. Check `gravity[2] < 0.5` → True → fires constantly.

**Fix:** Use upvector instead. `get_upvector(data)` returns the body z-axis IN world frame. When upright, upvector = `(0, 0, 1)` (body z aligns with world z). Joystick precedent at `go2_warp_joystick.py:409`:

```python
flipped = self.get_upvector(data)[-1] < 0.0  # < 0 = upside down
```

**Lesson:** "Body-frame gravity" and "world-frame upvector" point in opposite directions when the robot is upright. Use upvector for flipped detection. Default to checking what existing envs do before inventing a check.

---

## `Go2WarpEnv.__init__` clobbers `actuator_forcerange` before `mjx.put_model` (2026-05-04)

**What happened:** Splitbelt's belt actuators are `<velocity>` actuators with `ctrlrange="-3.0 3.0"` (m/s) and explicit `forcerange="-200 200"` (N) so they have enough force to drag the slab. After env init, `data.actuator_forcerange[belt_idx]` was `[-3, 3]` regardless of XML — belt actuators capped at ±3 N, can't track schedule under load.

**Root cause:** `Go2WarpEnv.__init__:64-67` runs:
```python
for i in range(self._mj_model.nu):
    self._mj_model.actuator_forcerange[i] = self._mj_model.actuator_ctrlrange[i]
```
For ALL `nu` actuators, BEFORE `mjx.put_model(self._mj_model, ...)`. The XML's `forcerange` is overwritten with `ctrlrange`, then snapshotted into MJX. The XML default is dead.

Even worse: post-`__init__` mutation of `self._mj_model.actuator_forcerange[idx]` does NOT propagate to `self._mjx_model` — MJX models are frozen snapshots.

**Fix (in subclass `_post_init`):**
```python
self._mj_model.actuator_forcerange[self._left_belt_act_id] = np.array([-200.0, 200.0])
self._mj_model.actuator_forcerange[self._right_belt_act_id] = np.array([-200.0, 200.0])
self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
assert float(self._mjx_model.actuator_forcerange[self._left_belt_act_id, 1]) > 100.0, (
    "belt actuator forcerange clobbered; mjx.put_model re-call failed"
)
```

**Lesson:** When adding non-leg actuators to a Go2 env that already has the base class clobber loop, re-mutate after `super().__init__` AND re-snapshot via `mjx.put_model`. Add a smoke assertion right after — if MJX snapshot semantics ever change, the assertion fails fast.

---

## Asymmetric AC: actor blindness to body translation in cross-policy transfer (2026-05-04 → 05)

**What happened:** Loaded a Go2WarpJoystickFlatNoAccel ckpt and ran it on the splitbelt env. Recorded video: robot stands still while belt drags it backward at 0.5 m/s. Policy never compensates.

**Root cause:** Joystick `state` (actor obs) does NOT include body lin vel. The `linvel_clean` term is in `privileged_state` (critic-only). Asymmetric AC training shapes a policy that consumes only proprio + cmd, with critic seeing translation. At inference, the actor never sees translation.

State actor sees on splitbelt at cmd=0 (rigid-body drag, feet planted on belt):
- gyro = 0 (no rotation)
- gravity = `(0, 0, -1)` (upright)
- joint_pos_offset ≈ 0 (legs at default pose, body translates rigidly)
- joint_vel ≈ 0
- cmd = 0
- last_act = 0

Identical to "standing on stationary ground" → identical policy output → no compensation.

**Counterpoint:** Splitbelt's own `blind` mode reaches eval 72 with the SAME actor obs. Difference is reward — splitbelt has `treadmill_drift` (penalty on world-frame xy from origin); joystick has `tracking_lin_vel` only (which is satisfied at cmd=0, body_vel=0). Splitbelt's reward signal during training shaped a policy that infers translation indirectly from proprio (subtle joint deflections, gyro micro-tilts during foot transitions). Joystick training never had drift pressure → policy never learned the inference.

**Lesson:**
1. Cross-policy transfer with same obs dim is NOT cross-deployable behavior. Check actor obs SCHEMA, not just dim.
2. Reward shape can substitute for sensor obs — splitbelt blind learning forward-stepping at cmd=0 from drift penalty alone is structurally similar to spinal CPG models (open-loop gait shaped by reward, not closed-loop kinematic tracking).
3. For interesting splitbelt cross-policy transfer: train splitbelt with `error` mode (actor sees `cmd_track_error + drift_xy`) — explicit translation signal — then drop on joystick env, see if it generalizes back. That's a follow-up experiment.

---

## Belt sign convention = biomech drag speed, NOT joint velocity (2026-05-05)

**What happened:** v1 calibration smoke (FastSAC 1M @ tied(0.5)) reached eval 105.5. Watched video — belt slabs moved WITH the robot's forward direction (+x), pushing robot forward. Robot didn't have to do anything; got a free ride. Eval was high because reward was easy.

**Root cause:** Schedule semantics unspecified. Default convention I picked: `schedule_table[t]` = joint velocity (signed). Belt slide joint axis is `(1, 0, 0)` in XML; ctrl=+0.5 means joint moves in +x. Robot faces +x → belt slab moves with robot.

**Fix:** Schedule semantics = **biomech drag speed**. Positive value means "belt drags foot backward at this speed." Belt slab actually moves -x. Negate ctrl + qvel writes:

```python
# step substep:
full_ctrl = full_ctrl.at[self._left_belt_ctrl_idx].set(-belt_vel_target[0])
full_ctrl = full_ctrl.at[self._right_belt_ctrl_idx].set(-belt_vel_target[1])

# reset:
qvel = qvel.at[self._left_belt_dofadr].set(-schedule_table[0, 0])
qvel = qvel.at[self._right_belt_dofadr].set(-schedule_table[0, 1])
```

`info["splitbelt"]["belt_vel"]` keeps the schedule (positive) value for analysis — analyst reads "drag speed" not "joint velocity."

After fix: v2 eval 72.3 (treadmill walking is genuinely harder than free-ride).

**Lesson:** Pin sign conventions at spec time. "Belt speed = 0.5" is ambiguous — biomech papers mean "foot drag rate"; physics simulators mean "joint angular velocity." Spec § 6.3 now documents: `qvel[belt_dofadr] = -schedule_table[0]` at reset; `data.ctrl[belt_act] = -schedule_table[t]` in step. Always test by playing back the recording after first 1k steps.

---

## Obs schema must align with sister envs to enable cross-deploy (2026-05-05)

**What happened:** Splitbelt's initial blind obs order was `[joint_pos, joint_vel, last_act, gravity, gyro, command]` = 45d. Joystick NoAccel: `[gyro, gravity, joint_pos_offset, joint_vel, last_act, command]` = 45d. Same fields, same dim, different ORDER and one different name.

User wanted to drop joystick policy on splitbelt without retraining. Worked mechanically (45d match) but policy reads splitbelt's `joint_pos` thinking it's `gyro` → garbage. Initial fix was a `joystick_compat` obs_mode (separate env variant). User pointed out: just align splitbelt's blind to joystick. Then BOTH directions work — joystick → splitbelt, AND future splitbelt → joystick.

**Fix:** Renamed `_PROPRIO_NAMES` to joystick order; renamed `joint_pos` → `joint_pos_offset` to match exactly. v2 ckpt invalidated (first-layer weights expect old order); did not retrain because no current use case demanded a splitbelt baseline.

**Lesson:**
1. Obs ORDER is a load-bearing axis for cross-policy transfer. Match sister-env exactly even if your env has more fields (just append the new ones).
2. The `joystick_compat` workaround was correct to remove — duplicating env config to paper over an arbitrary divergence is sediment. Fix the divergence at the source.
3. Plan-time spec didn't pin the obs name list against joystick. Future envs in the Go2 family: start by copying joystick's obs term factory and adding env-specific fields at the end.

---

## Pointers
- Belt-mech / sign-convention: spec §6.3, env file `_step()` substep block
- Asymmetric AC blindness: this lesson + `.context/lessons/offpolicy.md`
- MuJoCo plane gotcha: this lesson + `.context/lessons/bongo.md` (which previously argued "use sensors over position" — splitbelt is the exception)
- Forcerange clobber: env file `_post_init`, smoke assertion right after re-`put_model`
- Cross-deploy demo video: `checkpoints/20260428_085344_fast_sac_go2warpjoystickflatnoaccel_seed7002/best/20260505_183541_rollout.mp4`
