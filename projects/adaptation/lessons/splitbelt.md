# Splitbelt Treadmill — Lessons

> Splitbelt env build + first calibration: 2026-05-03 → 2026-05-05.
> Tunneling fix + retrain + OOD: 2026-05-06.
> Eval methodology lessons (reward hides, tilt dominant, DR doesn't extrapolate): 2026-05-07.
> Spec: `projects/adaptation/specs/2026-05-02-splitbelt-treadmill-env-design.md`
> Plan: `projects/adaptation/plans/2026-05-02-splitbelt-treadmill-env.md`
> Build journal: `.context/journals/2026-05-03-splitbelt-env-build.md` (chronological — stays in `.context/`)

---

## Foot-tunneling through thin slabs in MJX/MuJoCo Warp (2026-05-06)

**What happened:** Splitbelt belts are 10mm-thick boxes. Trained PoseDR policy looked OK in training (eval 280.6) but recorded rollouts showed FL foot center at z=-0.05 (5cm below belt top, INSIDE the slab body) for 95% of frames. Visual: robot looked stuck with one foot through the floor. v1 ckpt eval was inflated by foot-as-anchor exploit.

**Root cause:** MJX has no continuous collision detection (CCD). At foot vertical velocity 1.7 m/s and substep dt=0.005s, foot moves 8.5mm/substep — comparable to slab half-thickness. Per-substep position check sometimes catches foot ABOVE slab at step t and BELOW slab at step t+1 with no overlap detected at either step → contact missed → foot remains sub-slab indefinitely (no force pushing it back up because foot center is now BELOW slab center, slab isn't above the foot in the geom-distance check).

**Fix:** `<pair margin="0.02" .../>` on every foot×belt and foot×fallback_floor pair. Margin extends contact detection range — contact engages when foot bottom is within 20mm of belt top, generating soft repulsive force BEFORE the foot reaches the slab. Foot decelerates earlier → no fast-tunneling. Geometry unchanged (top of slab still at z=+0.005, resting foot center +0.027). 12 pairs total in `xmls/go2_warp_splitbelt_scene.xml`.

```xml
<pair geom1="FL" geom2="left_belt_geom"  margin="0.02"/>
<pair geom1="FL" geom2="right_belt_geom" margin="0.02"/>
... (8 foot×belt pairs total, every foot × every belt)
<pair geom1="FL" geom2="fallback_floor"  margin="0.02"/>
... (4 foot×fallback_floor pairs)
```

Tried alternatives:
- 50mm thick slab (bottom z=-0.045): catches most tunneling but ~0.2% leak; uglier off-belt visuals (50mm drop).
- 50mm slab + margin: slightly more tunneling than margin alone (1 frame). GPU non-determinism noise.

**Pre-existing related bug (same fix block):** Original spec assumed FL/RL stay on left belt and FR/RR on right belt → only those 4 collision pairs declared. Robot lateral drift carries any foot over either belt. Added 4 missing cross-belt pairs (`FL×right`, `FR×left`, `RL×right`, `RR×left`). Without these, even with margin, a foot crossing the centerline has no declared collision pair → tunnels.

**Lesson:**
1. **Always declare every leg-foot × every walkable-surface contact pair.** Don't assume gait keeps feet on a particular belt — drift breaks the assumption.
2. **For thin slabs in MJX/Warp, use `<pair margin>` to compensate for missing CCD.** 2× the slab thickness is a reasonable starting margin.
3. **Foot-as-anchor exploit:** policies trained on contact-deficient envs will exploit any tunneling for "free anchor." Eval scores can be inflated. Always sanity-check rollout videos before trusting an eval score.
4. **Closing the inboard belt-belt gap (5cm vestige) didn't help tunneling** — gap was a sensor-design fossil from `floor_found` era (replaced by position-based detection 2026-05-04). Closing it improved cosmetics but exposed the contact-pair bug.

---

## Belts must butt at y=0 (vestigial center gap removed 2026-05-06)

**What happened:** Original spec had a 5cm inboard gap between belts (y∈[-0.025, 0.025]) where neither belt was present and feet would land on `fallback_floor` 1cm below. Visually: feet bumped into 1cm vertical wall of belt slab when swinging inboard.

**Root cause:** The gap existed so the `<contact data="found">` sensor `floor_found` could distinguish "foot on belt" vs "foot in gap." That sensor was abandoned 2026-05-04 (margin-fires on plane geoms — see lesson above). Position-based off-belt detection (`foot_belt_id` from foot xy) doesn't need the geometric gap. Gap was a dead fossil.

**Fix:** Belts butt together at y=0 (left body pos `(0, -0.150, 0)`, right body pos `(0, +0.150, 0)`, each half-width 0.150). `BeltLayout` config: `left_y_min=-0.300 left_y_max=0.000`, `right_y_min=0.000 right_y_max=0.300`. `fallback_floor` now only catches outside-the-belts feet (|y|>0.300). Test `test_foot_at_boundary_assigns_to_belt`: y=0 lies on both belt edges; left wins by `where`-order.

**Lesson:** When you remove a sensor / detection mechanism, audit the geometry that was sized for it. Gaps, margins, special-case dimensions can be vestigial.

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

## Eval reward hides physics failures (2026-05-07)

**What happened:** PoseDR v2 OOD sweep showed (0.5, 1.0) — at the trained max ratio of 2× — scoring 276 reward (~70% of peak 393). Read as "in-dist edge, mild degradation." Same condition under physics-metric sweep: **75% termination rate**, 12/16 episodes died by tilt within ~16s. Across 144 episodes (9 OOD conditions), the only condition with 0% termination was tied (0.5, 0.5). Every other condition — including in-dist edges — had ≥19% termination, mostly tilt-out.

**Root cause:** The reward function blends survival reward + per-step pose-track terms. A policy that survives ~600 of 1250 steps cleanly accumulates more reward than one that immediately collapses, but both are "broken" by any deployment standard. Reading mean reward without the reward decomposition AND survival rate hides catastrophic failure modes behind a 70%-of-peak number.

**Fix:** Built `scripts/eval_splitbelt_physics.py` that reports per-condition termination rate (broken down by cause), mean survival steps, mean final and max |drift_x| (lag), max |drift_y| (sway). Ignores reward.

**Lesson:**
1. Reward is a *training* signal, not an *evaluation* signal. For deployment-relevant claims, evaluate on physical metrics (lag, sway, termination cause, survival).
2. Add a "termination rate" panel to every locomotion eval lane. A policy with 0 termination at one operating point and 75% at another is not "60% as good" — it's "broken outside one point."
3. When a policy reads as "moderate degradation" on reward, check survival before believing it. Reward-only OOD plots produce false-confidence narratives about generalization.

---

## Tilt is the dominant station-keeping failure (2026-05-07)

**What happened:** Across 144 PoseDR v2 episodes spanning 9 (vL, vR) conditions, termination causes broke down as:
- `term_cause=1` fall_torso (torso contact): **0** episodes
- `term_cause=2` off_belt (foot off belt span):  **7** episodes
- `term_cause=3` tilt (upvector_z<0.5 OR base_z<0.18): **~91** episodes

So even when belts dragged the robot 5.85m off origin in 17s (tied 1.5 condition), the failure was upright-loss, not contact crash. Robot doesn't crumple — it slowly tilts past 60° and the env terminates.

**Root cause hypothesis:** PoseDR's reward weighs `pose_pos_track = 5.0` and `pose_orient_track = 2.0`. Position dominates orientation 2.5×. When belt drag is sustained, the policy pours optimization budget into x-position correction (pushing back) and orientation drifts unchecked. Smoothness terms (action_rate, joint_vel) further suppress the *rapid* corrective torques needed to recover attitude near the tilt threshold (parallel to the bongo `lessons/bongo.md` finding that "regularization penalties can suppress necessary corrective actions").

**Lesson:**
1. For station-keeping under sustained drag/perturbation, **orientation reward must be heavier than position reward**, not the other way around. Falling over wastes all the position progress; staying upright lets you keep fighting.
2. Diagnostic: log `term_cause` distribution per OOD condition. If one cause dominates, the policy has a single load-bearing weakness; targeting it is higher-leverage than uniform retraining.
3. Belt span (50m × 1m here) is large enough that off-belt almost never triggers before tilt. Off-belt termination is mostly a backstop, not an active constraint.

---

## DR doesn't extrapolate, only interpolates (2026-05-07)

**What happened:** PoseDR v2 trained with `random_per_episode` over `vL ∈ U[0.3, 1.5]`, `ratio ∈ U[0.5, 2.0]`. Eval at conditions just past the training boundaries:
- (0.5, 1.5) ratio 3× (both speeds in-dist absolute, ratio just OOD): **81% term, max drift 1.0m**
- (0.3, 0.9) ratio 3× at slow magnitudes (both speeds in-dist, ratio just OOD): **100% term**
- (1.5, 1.5) tied at v=1.5 (in-dist absolute max, no asymmetry, ratio in-dist): **100% term, max drift 5.85m**

vs in-dist:
- (0.5, 0.5) tied: **0% term, ±10cm wander**

Going from train-max ratio 2× to ratio 3× is a 50% increase in the differential axis. Eval reward dropped 276 → 83 (-70%); termination went 75% → 81%; lag went 0.36m → 1.00m. The policy *did not* generalize — it cliff-dropped.

Direction asymmetry compounds the failure: ratio 3× R-faster (0.5, 1.5) → 81% term. Ratio 0.3× L-faster (1.0, 0.3), which is the inverse of ratio 3.3×, → only 44% term. Same magnitude differential but the policy is biased toward L-faster scenarios — likely the keyframe spawn places FL/RL on left belt, so the gait pattern that emerges in training is asymmetric to begin with.

**Root cause:** Uniform random_per_episode samples a *grid* of (vL, vR) within the support. The policy fits a function that interpolates the grid but has no inductive bias to extrapolate. Past the support boundary the function is whatever the network's smoothness prior says it is — usually wrong.

**Lesson:**
1. **DR over [a, b] gets you a policy that works on [a, b]** — period. Don't expect it to handle [a-ε, b+ε] without specific training. If you need (e.g.) ratio up to 3×, train with ratio range [0.5, 3.0] from the start.
2. **Symmetrize the training distribution explicitly.** Spawn keyframe + uniform DR ≠ symmetric exposure. To get symmetric robustness across L-faster vs R-faster, mirror-augment the data (spawn keyframe randomization + sometimes-flip-belt-assignment) or explicitly resample to 50/50 ratio>1 vs ratio<1.
3. **The training boundary is the policy's behavior cliff, not the failure point.** PoseDR was at 75% termination *inside* the boundary at ratio 2×. The boundary is where the policy stops being useful, not where it stops working entirely.

---

## Pointers
- Belt-mech / sign-convention: spec §6.3, env file `_step()` substep block
- Asymmetric AC blindness: this lesson + `.context/lessons/offpolicy.md`
- MuJoCo plane gotcha: this lesson + `.context/lessons/bongo.md` (which previously argued "use sensors over position" — splitbelt is the exception)
- Forcerange clobber: env file `_post_init`, smoke assertion right after re-`put_model`
- Cross-deploy demo video: `checkpoints/20260428_085344_fast_sac_go2warpjoystickflatnoaccel_seed7002/best/20260505_183541_rollout.mp4`
