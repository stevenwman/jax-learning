# mud_eval — HANDOFF / SPEC (authoritative, read this first)

**Goal:** evaluate jax-learning Go2 policies (joint-PD / OSC / variable-impedance,
the 2026-06-09 physical-motor retrains) in **Newton's triple-mud MPM** two-way-
coupled sim (thin/medium/thick mud), to see how compliant vs joint-PD control
copes on graded soft terrain. **Eval/rollout only — NOT training** (Newton physics
isn't JAX-traceable, so we hand-write the step loop and run the policy as
inference; see "Why a copy" below).

**Location:** `.worktrees/go2-osc-impedance/projects/mud_eval/` (in the worktree,
on branch `go2-osc-impedance`; moved here from the main repo on 2026-06-10).
Self-sufficient: vendored Newton + dedicated venv; no dependency on the
`Newton_stuff` clone at runtime.

---

## STATUS (2026-06-10)

| stage | state |
|---|---|
| M0 physics smoke (robot + 3-mud MPM, headless) | ✅ DONE |
| Visual gate (headless GL → PNG I can read + mp4) | ✅ DONE |
| M1 joint-PD policy eval (real jax_rl policy + obs adapter) | ✅ WORKS — stand/hold stable on mud; forward gait unstable (zero-shot) |
| **Migrate robot URDF → trained go2.xml (add_mjcf seam)** | ✅ DONE — couples with MPM, stands z~0.15 (gate PASS) |
| M1 clean traversal (orient down mud long-axis) | pending |
| M2 OSC controller (mjData Jacobian → Jᵀ·Λ·F torque injection) | pending — now natively wired to go2.xml |
| M3 variable-impedance (stiffness-tail decode) | pending |

---

## THE RUNTIME (hard-won — do NOT change versions blindly)

Dedicated venv at `projects/mud_eval/.venv`. The version combo took a long hunt;
it's pinned in `requirements.txt`. **Newton 0.1.3 needs the PRE-per-world-batching
mujoco-warp 0.0.2** (the 3.x line uses 2D `(nworld,ngeom)` geom arrays its
SolverMuJoCo can't consume → `geom_dataid expects 1D got 2D`):

```
warp-lang==1.12.0   mujoco==3.7.0   mujoco-warp==0.0.2   torch 2.11+cu128 (Blackwell/5080)
jax==0.9.0 (+ jax-cuda13, but jax runs the actor on CPU — fine, no GPU contention)
flax 0.12.2  optax 0.2.6  orbax-checkpoint 0.11.32  ml_collections 1.1.0  distrax 0.1.7  chex 0.1.91
trimesh + pycollada (URDF/.dae mesh load)  pyglet>=2.0 + imageio + imageio-ffmpeg (rendering)
newton 0.1.3 VENDORED under vendor/newton/ (on sys.path, NOT pip-installed)
```

**Critical install note:** installing jax must NOT bump mujoco off 3.7.0. Always
`uv pip install ... "mujoco==3.7.0" "mujoco-warp==0.0.2" "warp-lang==1.12.0"` as
constraints. And we import `jax_rl.algos.fast_sac` DIRECTLY (never
`jax_rl.training.*`, whose `__init__` imports `mjx_backend` → `mujoco_playground`,
which we can't install — it'd bump mujoco).

**Run anything** from `projects/mud_eval/` with:
```
WT=/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/go2-osc-impedance
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$WT" .venv/bin/python <script> ...
# rendering also needs:  PYGLET_HEADLESS=1
```

---

## APPROACH — drop-in policy + obs adapter, reuse the example

Newton's `mpm_go2_multi` example (vendored, forked-by-monkeypatch) already owns
the step loop (`apply_control → simulate_robot → simulate_sand`) and the whole
robot + 3-mud + two-way-coupling scene. We **swap its torch `Go2Policy` for our
`MudJaxPolicy`** (same `compute_joint_targets(state, command)` interface) — so we
reuse the entire sim and only own the controller.

**`MudJaxPolicy`** (`mud_jax_policy.py`):
- **Policy = the REAL jax_rl FastSAC** (no rebuild). `np.load` the checkpoint's
  `actor_params.npy` artifact → build `FastSAC(FastSACConfig(**meta['fast_sac_config']),
  obs_dim, act_dim, dummy_opt, dummy_opt)` → jit `select_action(p, obs, key,
  deterministic=True)` (= `tanh(mean)`). obs-norm is IDENTITY for these ckpts
  (asserted). jax on CPU.
- **Obs adapter** (the only real interface work): build jax-learning's 48-d
  `WarpJoystick.state` obs from Newton `State`:
  `[gyro(3), accelerometer(3), gravity(3), joint_pos_offset(12), joint_vel(12),
  last_act(12), command(3)]`. Newton joint order (FL,FR,RL,RR hip→thigh→calf) ==
  the policy's `policy_FL_FR_RL_RR` → **no remap**. Frame math via
  `quat_rotate_inverse` (matches the example). **The Newton free-joint quat is NOT
  unit-norm — normalize it.** accel = R^T(a_world − g_world), a_world = finite-diff
  of root linvel. **VERIFIED at home pose:** gyro 0, accel [0,0,9.81], gravity
  [0,0,-1], jpos_off 0, jvel 0.
- Action → joint targets: `default_pose + action*action_scale`, padded with 6
  free-joint zeros, → `wp.from_numpy(..., device=wp.get_device())` (NOT the torch
  device). joint-PD via the model's `joint_target_ke/kd` set from control.Kp/Kd.

**`patched_config()`** (in `mud_jax_policy.py`): writes a config.yaml override so
the Newton robot (a) **spawns over the mud** (see bug below), (b) spawns UPRIGHT
(example tilts via a non-z yaw axis), (c) spawns at the policy's default pose,
(d) uses the policy's Kp/Kd. Without these the policy sees OOD obs / wrong gains.

---

## THE FREE-FALL BUG (found via the visual+numeric gate; FIXED)

The robot fell through to z=−2.7 even with `|act|=0` (hold pose) → **not the
policy**. Cause: the example spawns at **y=−1.5**, but the mud (the robot's only
effective collider — the ground plane does NOT catch the rigid body) spans
**y[0,3]**, x[−1,1], z[0.01,0.12]. So it free-fell into the void. **Fix:**
`patched_config` sets `initial_position=[0, 1.5, 0.40]` (over the mud). Now it
settles + stays up.

**Zero-shot result (joint-PD, `…go2warpjoystickflatphysical`):** stand/hold =
stable but **sunk low** (torso z~0.13; firm-ground policy can't hold full height
on soft mud — feet are 2.2cm spheres → small contact patch). Forward command =
gait thrashes (`|act|~3.4`) + walks off the 2m-wide mud patch → unstable.

---

## VISUAL GATE (works — use it at every stage)

`record_mud.py <ckpt> <tag> [frames] [fwd|stand]` → ViewerGL headless (EGL),
**bounded** step+render loop (NOT `newton.examples.run`, which hangs headless),
`viewer.get_frame()` → PNG (PNG avoids the imageio-ffmpeg-vs-JAX-fork deadlock
that corrupts mp4). Frames + mp4 land in `recordings/`. **I read the PNGs.** Do
NOT flip the image (`get_frame` is already top-left origin). Toggle
`viewer.show_collision`/`viewer.show_visual` + `viewer.set_camera(pos,pitch,yaw)`
to render colliders / frame the robot (see `inspect_collision.py`).

---

## ROBOT MODEL MIGRATION — URDF → trained go2.xml (DONE)

The example loaded `go2_description.urdf`, NOT the `go2.xml` the policy trained on.
Migrated to go2.xml because: (1) OSC is wired natively for the in-house go2 (foot
sites + Jacobian + mass matrix come from OUR mjModel) — go2.xml makes M2 match;
(2) an MJCF loader = a morphology seam (new robot = new XML, not a new port).

**Seam = `mud_model.py`:** monkeypatch `newton.ModelBuilder.add_urdf` to dispatch
`.xml` paths to `add_mjcf` with `collider_classes=("collision","foot")` (feet are
class "foot" — missed by the default!), then set the home pose directly on
`joint_q[7:19]` (the example's `joint_key.index(key)+6` posing idiom OVERFLOWS on
go2.xml's 0-dof `*_foot_joint`). `patched_config(..., mjcf_model=abs_path)` points
the config at the model + EMPTIES `initial_joint_q` (so the buggy loop no-ops).
`record_mud.py` enables the seam by default. Model vendored at
`models/unitree_go2/` (go2.xml + assets, 28M).

**GATE (`gate_mjcf_coupling.py`):** hold-pose smoke → go2.xml stands on the mud
(z 0.84→0.15, finite) = MJCF colliders couple with the MPM. PASS.

**Collision compared (go2xml vs URDF):** feet IDENTICAL (SPHERE r=0.022 — the only
mud-contact shape); hips/thighs identical; URDF has 1 extra calf cylinder; go2.xml
torso richer (box+cyl+sphere vs box). So contact ~equivalent; migration's value is
the OTHER matched params + native OSC. Renders: `recordings/go2xml_collision.png`,
`collision_collision.png` (URDF), `mjcf_stand_f30.png` (go2.xml on mud).

**Caveat:** spawn z reads 0.84 at frame 0 (config says 0.40) — go2.xml base origin
differs from the URDF; robot still falls + catches on the mud fine, but for a clean
eval the spawn height could be retuned to ~0.2 (drop is currently ~0.7 m).

---

## CHECKPOINTS to eval (worktree `checkpoints/`, 2026-06-09 physical-motor retrains)

```
20260609_134104_fast_sac_go2warpjoystickflatphysical_seed0      joint-PD   (M1 uses this)
20260609_140251_fast_sac_go2warposcflatsoftphysical_seed0       OSC fixed-soft (M2)
20260609_142511_fast_sac_go2warposcvarflatphysical_seed0        var per-foot
20260609_144837_fast_sac_go2warposcvaraxisflatphysical_seed0    var per-axis
20260609_165254_fast_sac_go2warposcvardampingflatphysical_seed0 +damping foot
20260609_190852_fast_sac_go2warposcvardampingaxisflatphysical_seed0  +damping axis
```
Each has `actor_params.npy` (params + norm) + `meta.json` (control: Kp/Kd/
action_scale/default_pose_policy/policy_joint_names; fast_sac_config: hidden_dim
[512,256,128]/activation swish; obs_dim 48; action_dim 12/16/20/24/36).

---

## FILE MAP (projects/mud_eval/)

```
mud_jax_policy.py    MudJaxPolicy (drop-in) + obs adapter + patched_config(mjcf_model=)
mud_model.py         add_mjcf loader SEAM (URDF->go2.xml dispatch + home-pose set)
run_mud_eval.py      eval/rollout w/ per-frame z+|act| logging; cmds: fwd | stand | hold
record_mud.py        visual gate: headless GL → PNG (+mp4); enables the go2.xml seam
gate_mjcf_coupling.py GATE: go2.xml-on-mud hold-pose coupling smoke
inspect_collision.py URDF collision-shape summary + collider/visual render
inspect_go2xml.py    go2.xml collision-shape summary + render (standalone add_mjcf)
m0_smoke.py          M0 hold-pose physics smoke
mud_diag.py          frame-0 obs/quat/gains/joint-order diagnostic
requirements.txt     the pinned WORKING combo
models/unitree_go2/  VENDORED trained go2.xml + assets (the migration target)
README.md            overview · STATUS.md · HANDOFF.md (this)
vendor/newton/       vendored Newton 0.1.3 (lib + mpm_go2_multi example + go2_description, NO recordings/policies)
recordings/          PNG/mp4 outputs (gitignored)
.venv/               dedicated venv (gitignored)
```

## SETUP FROM SCRATCH (vendor/ + models/ + .venv/ are gitignored — reproducible)

0. **Vendor the trained model:**
   `cp -r ../../jax_rl/envs/locomotion/xmls/unitree_go2 models/`  (go2.xml + assets, 28M)
1. **Vendor Newton** (33M, no big artifacts):
   `rsync -a --exclude='recordings' --exclude='policies' --exclude='*.mp4' --exclude='*.pt' --exclude='__pycache__' /home/stevenman/Desktop/Work/Research/Newton_stuff/newton/ vendor/newton/`
2. **venv:** `uv venv --python 3.13 .venv` then
   `VIRTUAL_ENV=$(pwd)/.venv uv pip install -r requirements.txt` then add jax stack:
   `... uv pip install "jax==0.9.0" "jaxlib==0.9.0" "jax-cuda13-plugin==0.9.0" "jax-cuda13-pjrt==0.9.0" flax==0.12.2 optax==0.2.6 orbax-checkpoint==0.11.32 ml_collections==1.1.0 distrax==0.1.7 chex==0.1.91 pyglet imageio imageio-ffmpeg "mujoco==3.7.0" "mujoco-warp==0.0.2" "warp-lang==1.12.0"`
   (torch cu128: `uv pip install torch --torch-backend=cu128`). The trailing
   mujoco/mujoco-warp/warp constraints are MANDATORY so the jax install can't bump them.

## M2 OSC plan (the research point)
The OSC policies emit foot-position targets (trunk frame), not joint targets, and
need the OSC controller (Jᵀ·Λ·F). Plan: in MudJaxPolicy.apply, decode action →
foot deltas, compute the torque from the **mjData Jacobian + mj_fullM** of the
underlying mujoco model (Newton's SolverMuJoCo wraps mujoco_warp), and **inject
joint torques** (set ke=0 + use joint_target as force, or a joint_f path) instead
of position targets. Decided: get the Jacobian from mjData (user's call), not a
shadow-mjx. Reuse jax-learning's `impedance_gains` for M3's stiffness decode.
```
