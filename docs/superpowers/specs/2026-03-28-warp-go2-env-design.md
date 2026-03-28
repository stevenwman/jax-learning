# MuJoCo Warp Go2 Environment — Design Spec

**Date:** 2026-03-28
**Status:** Draft
**Branch:** `new_slate_linen`

---

## Problem

MJX can't load unitree's Go2 MJCF — cylinder-box collisions are not implemented. We use Menagerie's `go2_mjx.xml` instead, which strips collision geometry to spheres/capsules. This creates an irreducible sim2sim gap: policies trained on Menagerie's simplified model don't transfer to unitree_mujoco's full model.

MuJoCo Warp supports all collision types including cylinder-box. Playground 0.2.0 added Warp as a backend via `impl="warp"`. Training on unitree's exact MJCF eliminates the sim2sim gap entirely.

## Goal

Create a new Go2 Warp environment that:
1. Trains on unitree_mujoco's go2.xml (full collision geometry) via the Warp backend
2. Supports a `contact_mode` flag: `"training"` (firm contacts for stable gaits) vs `"deploy"` (unitree-native physics for transfer validation)
3. Serves as a template for future Warp-based environments
4. Emits the same dict obs (`{"state": 48d, "privileged_state": 122d}`) for PPO asymmetric actor-critic compatibility

## Non-Goals

- Benchmarking Warp vs MJX performance
- Asymmetric critic support for off-policy algos (future work — theoretically sound per ICML 2025, but training scripts don't support it yet)
- Dropping the existing MJX env (keep it as a working fallback — Warp is Beta in Playground)

---

## Design

### File Structure

```
jax_rl/envs/locomotion/
├── go2_base.py                  # existing — MJX base (Menagerie go2_mjx.xml)
├── go2_joystick.py              # existing — MJX Joystick env
├── go2_warp_base.py             # NEW — Warp base (unitree go2.xml)
├── go2_warp_joystick.py         # NEW — Warp Joystick env
├── go2_sensors.py               # NEW — shared sensor helpers (extracted from go2_base.py)
├── go2_constants.py             # existing — shared constants (add WARP_ROOT_BODY)
├── go2_randomize.py             # existing — DR (works for both backends)
└── xmls/
    ├── go2_scene_flat.xml           # existing — includes Menagerie go2_mjx.xml
    ├── go2_warp_scene_flat.xml      # NEW — includes unitree go2.xml
    └── unitree_go2/                 # NEW — vendored from unitree_mujoco repo
        ├── go2.xml                  # patched: 4 foot sites added
        └── assets/                  # all mesh .obj files
```

### Vendored XML

Source: `unitree_mujoco/unitree_robots/go2/go2.xml` (from github.com/unitreerobotics/unitree_mujoco).

This is a **different file** from Menagerie's `go2.xml`. Key native values:

| Property | unitree go2.xml | Menagerie go2_mjx.xml |
|---|---|---|
| Actuator type | `motor` (direct torque) | `general` (affine PD) |
| Joint damping | 0.1 | 2.0 |
| Frictionloss | 0.2 | 0.0 |
| Geom friction | 0.4 | 0.6 |
| Foot condim | 6 | 6 (but we override to 3) |
| Friction cone | elliptic | pyramidal |
| Collision geometry | cylinder + box + sphere | spheres only (MJX-safe) |
| Rear thigh range | [-0.5236, 4.5379] | [-1.5708, 3.4907] (wrong) |
| Actuator order | FR, FL, RR, RL | FL, FR, RL, RR |
| Root body name | `base_link` | `base` |

**Patches applied to vendored XML:**

1. **Add 4 foot sites** (needed for privileged obs sensors). One line per leg inside each `*_calf` body:
```xml
<site name="FL_foot" pos="-0.002 0 -0.213" group="1"/>
```

2. **Verify `imu` site exists.** unitree's go2.xml has `<site name="imu" pos="-0.02557 0 0.04232"/>` on the `base_link` body. Confirmed present — no patch needed.

No other changes to the vendored XML.

**Mesh assets:** Vendored into `xmls/unitree_go2/assets/`. No dependency on Menagerie paths. Complete isolation.

### Scene XML (`go2_warp_scene_flat.xml`)

Same pattern as existing `go2_scene_flat.xml`:
- `<include file="unitree_go2/go2.xml"/>`
- Floor geom with `name="floor"`, `contype="1"`, `conaffinity="0"`, `priority="1"` (matching existing scene — ensures foot geoms collide with floor)
- Visual settings (skybox, ground texture, headlight)

**Sensor block — CRITICAL:** Unitree's go2.xml defines its own sensors (joint pos/vel/torque, imu_quat, imu_gyro, imu_acc, frame_pos, frame_vel) but is **missing 4 sensors** that exist in Menagerie's go2_mjx.xml and that our env code depends on. The new scene XML must define ALL of these:

```xml
<!-- Sensors from Menagerie go2_mjx.xml (NOT in unitree's go2.xml) -->
<gyro site="imu" name="gyro"/>
<accelerometer site="imu" name="accelerometer"/>
<framelinvel objtype="site" objname="imu" name="global_linvel"/>
<frameangvel objtype="site" objname="imu" name="global_angvel"/>

<!-- Sensors from existing go2_scene_flat.xml (also not in unitree's go2.xml) -->
<velocimeter site="imu" name="local_linvel"/>
<framezaxis objtype="site" objname="imu" name="upvector"/>
<framexaxis objtype="site" objname="imu" name="forwardvector"/>

<!-- Foot position/velocity sensors (reference patched foot sites) -->
<framepos objtype="site" objname="FL_foot" name="FL_pos" reftype="site" refname="imu"/>
<!-- ... FR, RL, RR ... -->
<framelinvel objtype="site" objname="FL_foot" name="FL_global_linvel"/>
<!-- ... FR, RL, RR ... -->

<!-- Floor contact sensors -->
<contact name="FL_floor_found" geom1="FL" geom2="floor" reduce="mindist" num="1" data="found"/>
<!-- ... FR, RL, RR ... -->
```

Note: unitree's existing sensors (joint pos/vel/torque, imu_quat, etc.) remain — extra sensors are harmless. Our env code doesn't reference them by name.

**`<contact>` sensor Warp compatibility:** Verify during implementation that `<contact>` sensors work under the Warp backend. If not, fall back to `mjx_env.get_contact` approach used by some Playground envs.

### Shared Sensor Helpers (`go2_sensors.py`)

Extract from `go2_base.py` into standalone functions that take explicit arguments instead of `self`:

```python
# go2_sensors.py — standalone functions, no class dependency

def get_sensor_by_name(mj_model, data, sensor_name):
    """Read sensor data by name. Thin wrapper around mjx_env.get_sensor_data."""
    return mjx_env.get_sensor_data(mj_model, data, sensor_name)

def get_gravity(data, imu_site_id):
    """Projected gravity in body frame from IMU site rotation matrix."""
    return data.site_xmat[imu_site_id].T @ jp.array([0, 0, -1])
```

The sensor-name-based helpers (`get_gyro`, `get_upvector`, `get_global_linvel`, etc.) all call `get_sensor_by_name` with the appropriate constant. `get_gravity` uses the IMU site ID directly.

Both `Go2Env` and `Go2WarpEnv` keep their existing instance methods as thin wrappers:
```python
# In Go2Env (unchanged public API)
def get_gyro(self, data):
    return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.GYRO_SENSOR)
```

This preserves backward compatibility — existing code calling `env.get_gyro(data)` still works. The extraction just eliminates the duplicated implementation.

### Go2WarpEnv Base (`go2_warp_base.py`)

Extends `mjx_env.MjxEnv`. Key differences from `Go2Env`:

**Asset loading:** New `get_warp_assets()` function loads from vendored `xmls/unitree_go2/` directory. Same pattern as existing `get_assets()` but points at the vendor path:
```python
def get_warp_assets() -> Dict[str, bytes]:
    assets = {}
    vendor_path = Path(__file__).parent / "xmls" / "unitree_go2"
    mjx_env.update_assets(assets, vendor_path, "*.xml")
    mjx_env.update_assets(assets, vendor_path / "assets")  # mesh .obj files
    # Also load the scene XML directory for the include
    mjx_env.update_assets(assets, Path(__file__).parent / "xmls", "*.xml")
    return assets
```
No Menagerie dependency.

**Runtime overrides — always applied:**

| Override | Value | Reason |
|---|---|---|
| `opt.timestep` | `config.sim_dt` (0.004) | Match training dt |
| `opt.ccd_iterations` | 20 | Match Go1 Playground |
| `vis.global_.offwidth/offheight` | 3840x2160 | Rendering |

**Runtime overrides — `contact_mode="training"` only:**

| Override | Value | Reason |
|---|---|---|
| Foot solimp | [0.9, 0.95, 0.023] | Firm contacts for crisp push-off |
| Foot condim | 3 | Basic friction (training stability) |
| Foot friction | [0.6, 0.005, 0.0001] | Match current training env |

**`contact_mode="deploy"`:** No contact overrides. XML-native values used as-is. This matches what unitree_mujoco does at runtime (only overrides timestep).

**NOT overridden (unitree XML already correct):**
- Actuator type (already `motor`)
- Joint damping (already 0.1)
- Frictionloss (already 0.2)
- Ctrl/force ranges (already ±23.7/±45.43)
- Rear thigh range (already correct)

**PD gains:** Stored as `self._kp`, `self._kd` from config (same as Go2Env).

**Model compilation:** `mjx.put_model(self._mj_model, impl=self._config.impl)`. Config-driven (not hardcoded) for consistency with `Go2Env`'s pattern.

**No actuator remapping.** Policy trains from scratch on unitree's FR-first ordering. The policy learns whatever ordering the model has. Remapping is only needed at deploy time (already handled by deploy code).

### WarpJoystick Env (`go2_warp_joystick.py`)

Extends `Go2WarpEnv`. Reward, obs, and termination logic is identical to current `Joystick`.

**`default_config()`:**
- `impl="warp"`
- `contact_mode="training"` (default)
- Same reward weights, noise config, command config as current Joystick
- `nconmax` and `njmax`: unitree's full collision geometry (cylinders + boxes on every leg) generates more contacts than Menagerie's sphere-only model. Start with `nconmax=4*8192` (same as MJX env) and `njmax=80` (2x current). If Warp errors on buffer overflow, increase. Warp handles dynamically varying contact counts better than MJX's fixed-size buffers, so this may not need tuning at all

**`_post_init()`:** Same as current Joystick but:
- Uses `consts.WARP_ROOT_BODY` (`"base_link"`) instead of `consts.ROOT_BODY` (`"base"`)

**`step()`:** Inherited from current Joystick. External PD via `lax.scan` substeps. No remapping needed.

**`_get_obs()`:** Same 48d state + 122d privileged_state structure. `data.actuator_force` used as-is in FR-first order (critic learns it from scratch).

**`_get_reward()`:** Inherited unchanged. All reward terms use `qpos`/`qvel`/sensor data (joint-ordered, FL-first in both XMLs) or sums over all joints (permutation-invariant).

**`_get_termination()`:** Inherited unchanged. Uses `base_link` body ID via constant.

### Registration

In `env_setup.py`:
```python
pg_locomotion.register_environment(
    "Go2WarpJoystickFlat",
    functools.partial(WarpJoystick, task="flat_terrain"),
    default_config,
)
```

Usage: `uv run python train_ppo_fast.py --env Go2WarpJoystickFlat`

### Domain Randomization

Existing `go2_randomize.py` has a compatibility issue: it hardcodes `TORSO_BODY_ID = 1` (assumes `"base"` is body ID 1). If unitree's body tree assigns a different ID to `base_link`, COM jitter and payload mass randomization will silently modify the wrong body.

**Fix:** Make `domain_randomize()` accept `torso_body_id` as a parameter instead of using a hardcoded constant. The calling code (`env_setup.py`) passes the ID looked up from the env's `mj_model`. This also makes the DR code more reusable for future envs.

```python
# Before (fragile):
TORSO_BODY_ID = 1

# After (explicit):
def domain_randomize(model, rng, torso_body_id=1):
```

Verify during implementation that `domain_randomize()` doesn't touch actuator-type-specific fields (it shouldn't — it randomizes friction, mass, COM, armature, frictionloss).

### Dependency Changes

```toml
# pyproject.toml
dependencies = [
    "warp-lang>=1.12.0",         # NEW — Warp compute backend (CUDA)
    "playground>=0.2.0",          # BUMPED from >=0.1.0 (Warp backend support)
    "jax[cuda13]>=0.9.0",        # unchanged (test compatibility)
    # ... rest unchanged
]
```

Note: there is no `mujoco-warp` pip package. Warp support comes through `warp-lang` (the compute backend) combined with `mujoco` (which already includes `mjx`). The `impl="warp"` codepath lives inside `mjx`, not a separate package. Playground 0.2.0 may pull `warp-lang` transitively — verify and add explicit dep if not.

**CUDA compatibility:** `warp-lang>=1.12` builds against CUDA 12. `jax[cuda13]` uses CUDA 13. CUDA is backward-compatible so this likely works, but must be verified during installation. Fallback: downgrade to `jax[cuda12]`.

---

## Name Compatibility Audit

Full audit of every name our env code references against unitree's go2.xml:

| Category | Name | Menagerie | unitree | Compatible? |
|---|---|---|---|---|
| Root body | `base` / `base_link` | `base` | `base_link` | NO — use constant |
| Joint names | `FL_hip_joint` etc. | yes | yes | YES |
| Foot geoms | `FL`, `FR`, `RL`, `RR` | yes | yes | YES |
| Foot sites | `FL_foot` etc. | site | body (no site) | NO — add sites to vendored XML |
| IMU site | `imu` | yes | yes (verified in XML) | YES |
| Floor geom | `floor` | in scene XML | in scene XML | YES (we define it) |
| Keyframe | `home` | yes | yes | YES |
| qpos order | FL, FR, RL, RR | yes | yes | YES (body tree order) |
| Actuator order | — | FL-first | FR-first | N/A (train from scratch) |

All incompatibilities resolved by: (1) vendored XML foot site patch, (2) separate `ROOT_BODY` constant.

**Constant addition to `go2_constants.py`:**
```python
WARP_ROOT_BODY = "base_link"   # unitree go2.xml (vs "base" in Menagerie)
```

---

## Verification Steps

1. **Install deps:** `warp-lang>=1.12` + `playground>=0.2.0` coexist with `jax[cuda13]`
2. **XML loads:** `mujoco.MjModel.from_xml_string(vendored_go2)` succeeds, `mjx.put_model(m, impl="warp")` succeeds
3. **Sensors work:** All sensors in scene XML resolve correctly (no missing site/geom references)
4. **Smoke test:** `train_ppo_fast.py --env Go2WarpJoystickFlat --total-timesteps 500000` runs without crash
5. **DR compatibility:** `--domain-rand` flag works with Warp env (verify `torso_body_id` is correct for `base_link`)
6. **Contact sensors:** Verify `<contact>` sensor type works under Warp backend
6. **Deploy mode:** `contact_mode="deploy"` loads without overrides, sim runs stable
7. **nconmax/njmax:** Tune values if Warp errors on contact buffer sizes

---

## Future Work (Out of Scope)

- **Asymmetric off-policy:** Add privileged critic support to SAC/TD3 training scripts. Theoretically justified (Pinto 2017, Lambrechts ICML 2025). The Warp env already emits privileged_state — just needs training script plumbing.
- **Warp backend for existing envs:** Abstract `impl` as a CLI flag so any env can run on either backend.
- **MuJoCo Warp benchmarking:** Compare sps, wall-clock time, memory usage vs MJX.
