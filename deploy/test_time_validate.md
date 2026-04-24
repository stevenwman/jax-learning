# Test-Time Validation — Deploy Sanity Checks

Run through this **before commanding any motion** every time you put a fresh checkpoint on hardware. Stance is the highest-information state because every obs slot has a known expected value. If anything in here is off, fix it before sending velocity commands — running an OOD-from-t=0 policy on a real robot causes the cascade described in §6.

---

## 1. Pre-flight: log the raw sensors

Add this near `Go2Interface.get_state()` (or in your ROS node before obs_builder):

```python
print(f"RAW quat (SDK [w,x,y,z]?):       {state.imu_state.quaternion}")
print(f"RAW gyro [rad/s]:                {state.imu_state.gyroscope}")
print(f"RAW accel [m/s², specific force]:{state.imu_state.accelerometer}")
print(f"RAW joint_pos_sdk [rad]:         {[state.motor_state[i].q  for i in range(12)]}")
print(f"RAW joint_vel_sdk [rad/s]:       {[state.motor_state[i].dq for i in range(12)]}")
```

Then at the first obs (t=0, after `interpolate_to_stand` completed):

| Raw signal | Expected at upright stance | If wrong |
|---|---|---|
| `quaternion` | `[w≈1, x≈0, y≈0, z≈0]` (close to identity) | Quat convention or IMU mount frame mismatch — see §3 |
| `gyro` | `[~0, ~0, ~0]` | IMU bias OR robot wobbling |
| `accel` | `[~0, ~0, +9.8]` | Sign / units mismatch — see §3 |
| `joint_pos_sdk` | matches `DEFAULT_POSE_SDK` in `go2_constants.py` within ±0.05 rad | Default pose calibration drift, or stand-up didn't finish — see §4 |
| `joint_vel_sdk` | `[~0]×12` | Robot still moving, or vel scaling wrong |

---

## 2. The 6 obs-slot expectations at stance

From `obs_printout.txt`, after `interpolate_to_stand`, with **command all zeros**, no policy stepping yet (or after a single zero-action step):

```
gyro     [0:3]   ≈ [0, 0, 0]
accel    [3:6]   ≈ [0, 0, +9.8]
gravity  [6:9]   ≈ [0, 0, -1]
jpos_off [9:21]  ≈ [0]×12       (joint_pos minus DEFAULT_POSE_POLICY)
jvel    [21:33]  ≈ [0]×12
last_act[33:45]  ≈ [0]×12       (first step) or previous policy output
command [45:48]  = [vx, vy, yaw_rate]
```

---

## 3. Cross-checks (do all four)

These catch mismatches that any single value can't:

**A. `||gravity|| ≈ 1.0`** — if not, quat isn't normalized (sensor pipeline broken).

**B. `||accel|| ≈ 9.81 m/s²`** at rest — if `≈ 1.0`, units are in `g` not m/s² (rare but seen in some firmware).

**C. `sign(accel_z)` and `sign(gravity_z)` MUST be opposite** when robot is upright.
- Specific-force convention: at rest, accel_body = `[0, 0, +9.81]` (sensor sees reaction force pushing UP through chassis).
- Projected gravity: at rest, gravity_body = `R^T @ [0,0,-1] = [0, 0, -1]`.
- **Same sign on both = the IMU's accel and quat are not in the same frame.** This was the bug observed in `.temp/obs_prinout.txt` (line 1: accel_z=+9.5, gravity_z=+0.57).

**D. `joint_pos_offset all small`** (each component within ±0.05 rad) → proves both:
- DEFAULT_POSE_POLICY values are correct
- SDK_TO_POLICY index map is correct (otherwise the subtraction lines up wrong joint pairs → garbage offsets)

If a single leg has offsets `[~0, ~0, ~0]` and another has `[~0, ±0.9, ∓1.8]` repeating, the SDK_TO_POLICY map has a 2-leg swap.

---

## 4. Decoding `gravity_body` geometry

If gravity is not `[0, 0, -1]`, decode the orientation:

```
body_up_in_world = -gravity_body
```

| `gravity_body` | Body-up direction in world | Robot orientation |
|---|---|---|
| `[0, 0, -1]` | `+z` | Upright (correct) |
| `[+1, 0, 0]` | `-x` | Pitched 90° forward (lying on chest) |
| `[-1, 0, 0]` | `+x` | Pitched 90° back (sitting upright on rump) |
| `[0, +1, 0]` | `-y` | Rolled right (lying on right side) |
| `[0, -1, 0]` | `+y` | Rolled left (lying on left side) |
| `[0, 0, +1]` | `-z` | Upside down |

Mixed components → tilted. Compute tilt angle: `θ_tilt = arccos(-gravity_z)`. If `θ_tilt > 5°` at "stance", something is wrong.

---

## 5. Schema check (for ckpts ≥ 2026-04-24)

```bash
python -c "
import json
m = json.load(open('CKPT_DIR/best/meta.json'))
print('obs_schema:', m.get('obs_schema'))
print('obs_dim:', m.get('obs_dim'))
print('n_frame_stack:', m.get('train_config', {}).get('n_frame_stack'))
"
```

- **`obs_schema` present**: deploy will use the saved layout. If `from_checkpoint` doesn't print a fallback warning at startup, you're golden — dim and layout both verified by ckpt.
- **`obs_schema` missing**: pre-2026-04-24 ckpt. Deploy falls back to `DEFAULT_STATE_SCHEMA = [gyro, accelerometer, gravity, joint_pos_offset, joint_vel, last_act, command]`. **Verify this matches what the env actually used at training time** by reading the env's `_obs_groups["state"]` from the same git commit the ckpt was trained at. If mismatch → retrain or hand-construct a custom `state_schema=[...]` arg to `ObsBuilder()`.

---

## 6. Cascade pattern (what bad-from-t=0 looks like)

When obs is OOD at t=0, the failure mode is recognizable:

1. **t=0**: obs has the bug (e.g., gravity wrong). gyro/accel/joints look reasonable but the *gravity* slot puts the policy in OOD-distribution.
2. **t=1**: policy outputs saturated actions (each joint near `±1.0`).
3. **t=2-10**: real robot tries to track wild joint targets → high real gyro/accel readings (this time legitimately, because robot is flailing).
4. **t>10**: gyro magnitudes ramp from O(0.01) to O(1.0+) rad/s; accel xy components O(10) m/s²; policy fully OOD; cascade.

Recognize from the printout:
- Look at the **first 3 obs lines** for slot sanity (no big numbers anywhere except command).
- If line 1 is suspect but accel/gyro look OK, the sensor pipeline is fine — the *interpretation* (gravity from quat, jpos_offset from default pose) is what's broken.
- If actions saturate on line 2 already → policy reacting to bad obs from line 1.

In `.temp/obs_prinout.txt` from 2026-04-24, this exact cascade played out: line 1 had accel ≈ rest but gravity tilted-on-side; line 2 actions hit ±0.99 on multiple joints; by line 5 accel ≈ ±27 m/s², robot violently flailing.

---

## 7. Action / actuator integrity

After fixing obs, before sending velocity command, send `vx=vy=yaw=0` and step the policy. Expected:

- Raw actions (policy output): close to zero (±0.1 typical for a stand-still cmd on a trained walking policy). **Saturated actions on a zero-cmd standing robot = bad sign.**
- Joint targets after scaling: `q_target = DEFAULT_POSE_SDK + action * ACTION_SCALE` should be very close to current pose.
- Robot motion: minimal twitching. Hold steady ±2cm at the feet.

If actions saturate on zero command:
- PolicyRunner missed obs normalization → check `runner.use_obs_norm == True` and `runner.norm_count > 0`.
- Schema mismatch silently giving the policy permuted input.
- Or the policy was trained on a different command distribution (e.g., curriculum policy expects goal-directed cmd with magnitude — zero cmd may be OOD).

---

## 8. Joint remap test (FR/FL/RR/RL ↔ FL/FR/RL/RR)

While robot is in default stance, lift ONE leg manually (e.g., FL hip) and watch which slots in `joint_pos_offset` change:

| Lifted leg | Expected slot to deviate (zero-indexed in `obs[9:21]`) |
|---|---|
| FL | `[0:3]` |
| FR | `[3:6]` |
| RL | `[6:9]` |
| RR | `[9:12]` |

If lifting FL changes slots `[3:6]`: SDK_TO_POLICY has FL/FR swapped. If lifting FL changes slots `[6:9]`: front/rear swapped. Each non-trivial mismapping needs constants in `go2_constants.py` corrected.

---

## 9. PD gain / actuator type sanity

Training (Warp env): `motor` actuator type, **Kp=20, Kd=0.5** (matches unitree_rl_gym). External PD applied per substep at 50Hz.

Verify deploy uses these:
```bash
grep -E "kp|kd" deploy/go2_constants.py deploy/robot_interface.py
```

If deploy default differs, robot will track joint targets with different stiffness → effective dynamics differ from sim → policy is OOD even with correct obs. Symptom: robot moves but is sluggish or jittery; locomotion looks "wrong" but not catastrophic.

---

## 10. Observation-norm sanity

PolicyRunner startup banner should print:
```
algo=fast_sac, obs=48d, act=12d
hidden=..., activation=..., squash=...
obs_norm=yes (count=N)        ← N should be large (~1e6-1e8 for a 20M-step training run)
```

If `obs_norm=no` or `count=0` for a checkpoint that trained with obs normalization (default for SAC/TD3 family), the deploy is feeding raw obs to a policy expecting whitened input → garbage actions even on perfect sensors.

Check `actor_params.npy`:
```python
import numpy as np
d = np.load("CKPT_DIR/best/actor_params.npy", allow_pickle=True).item()
print("norm_count:", d.get("norm_count"))
print("norm_mean[:8]:", d["norm_mean"][:8])
print("norm_mos[:8]:", d["norm_mean_of_squares"][:8])
```

For a healthy Go2 FastSAC ckpt at 20M steps, `norm_count` should be ~tens of millions, `norm_mean[0:3]` (gyro) close to 0, `norm_mos[0:3]` ~0.04 (since gyro is small, σ²~0.04 → σ~0.2 matching training noise).

---

## 11. Quick deploy checklist (TL;DR)

Print these every run, in order. Stop at the first ❌:

1. ☐ Raw quat at upright stance ≈ identity (`[1,0,0,0]` or `[0,0,0,1]`)
2. ☐ Raw accel ≈ `[0, 0, +9.8]`
3. ☐ Raw joint_pos within ±0.05 of `DEFAULT_POSE_SDK`
4. ☐ obs_builder fallback warning ABSENT (or schema explicitly verified)
5. ☐ obs[6:9] (gravity) ≈ `[0, 0, -1]`
6. ☐ obs[3:6] (accel) sign matches §3.C cross-check
7. ☐ obs[9:21] all within ±0.05
8. ☐ PolicyRunner says `obs_norm=yes (count > 0)`
9. ☐ At cmd=`[0,0,0]`, raw actions are non-saturated (each `< 0.5` in magnitude)
10. ☐ Joint targets after scaling stay within ±0.1 rad of current pose at zero cmd

If any fails, root-cause **before** sending walk commands. The cost of one bad run on real hardware is much higher than 5 minutes of validation.

---

## 12. Drift-history reference

For ckpts trained between **2026-04-10 and 2026-04-24** (the dates between when actor obs lost `linvel` and when schema-from-ckpt landed): deploy ran with hardcoded `[linvel(zeroed), gyro, gravity, ...]` while training used `[gyro, accelerometer, gravity, ...]`. Same total dim 48 → no warning. **Symptom**: gyro slot appeared in `[0:3]` of obs from training; deploy fed real gyro into the slot the policy interpreted as `gyro` (correct by accident on slot 0 if linvel was zeroed) but `accelerometer` slot was filled with hardware gyro instead of accel — broken.

Schema-from-ckpt (commit 437f530, 2026-04-24) prevents this by structure. **Any ckpt trained ≥2026-04-24 carries its own layout in `meta.json`.** Older ckpts: hand-verify the schema matches the env's `_obs_groups` at the training commit, or retrain.
