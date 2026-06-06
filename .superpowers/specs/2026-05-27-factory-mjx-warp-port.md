# IsaacLab Factory → MuJoCo Warp Port — Spec

**Date:** 2026-05-27
**Source guide:** `/home/stevenman/Desktop/Work/Research/isaaclab-factory-jax/docs/mujoco_port_guide.md`
**Reference repo:** IsaacLab v2.3.2 (commit `37ddf626`) — `source/isaaclab_tasks/isaaclab_tasks/direct/factory/`
**Reference paper:** Tang et al., "Factory: Fast Contact for Robotic Assembly." RSS 2022. ArXiv `2205.03532`.
**Status:** Draft → Plan
**Target completion:** 3–5 weeks for full Factory suite (PegInsert / NutThread / GearMesh), functional-first

---

## Problem

The repo's manipulation story is PushT (planar, single rigid body, vendored gym-pusht through `gym_backend`). The end-of-line research targets — DIAYN / METRA / USD skill discovery on contact-rich manipulation — need a 6-DoF arm + assembly-task env class. IsaacLab Factory is the canonical benchmark suite (NutThread, PegInsert, GearMesh) but lives in Omniverse Kit, which collides with our JAX/MJX/Warp stack: numpy ABI mismatches, syntheticdata renderer dependency, kvdb lock contention, Kit-close hang on cleanup (documented in `isaaclab-factory-jax/docs/jax_learning_divergences.md`).

Porting Factory natively to MuJoCo Warp inside this repo eliminates the Omniverse dependency for our research line. We retain the task definitions, observation/action spec, reward shaping, and reset distribution from Factory while swapping the physics backend to mujoco-warp (the same backend already validated for Go2 locomotion).

## Goal

Bring **PegInsert, NutThread, and GearMesh** into this repo as first-class Warp envs that:

1. Route through `make_env_bundle` and the existing `mjx_backend`. The Warp physics backend is selected env-internally by setting `impl="warp"` on the env's mjx model (mirrors `jax_rl/envs/locomotion/go2_warp_joystick.py:74`). The bundle/backend layer needs no change — there is no `mjx_backend(impl=...)` kwarg; the routing is "backend is mjx, env opts into Warp."
2. Train end-to-end with vanilla SAC + asymmetric AC using the existing shared `offpolicy_loop`.
3. Honor repo contracts: `_obs_groups` schema, `stamp_meta` artifact contract, `DomainRandWrapper` per_step, `build_parser`, hermetic-first tests.
4. Beat random baseline by ~100× on PegInsert as the first functional milestone, then extend to NutThread / GearMesh.

**Success criteria (per task):**

1. `make_env_bundle("FactoryPegInsert", seed=0)` returns populated bundle with dict obs `{"state": (25,), "privileged_state": (~72,)}`.
2. `train_sac.py --env Factory<Task>` runs end-to-end via existing entry script — no new train script.
3. Scripted P-controller policy hits ≥0.5 reward/step (port guide baseline).
4. SAC @ 5M timesteps reaches ≥0.5 reward/step on PegInsert (functional bar). NutThread bar is ≥0.3 (contact-rich is harder).
5. Eval recording via `record_video.py` shows the actual task behavior (not reward-hacking).
6. Hermetic test suite + `[gpu, warp]` smoke tests pass.
7. Doc sweep: `.context/AGENT_HANDOFF.md` benchmark table updated, CLI ref regenerated, journal entry per phase, lessons captured in `.context/lessons/manipulation.md` if novel.

## Non-goals

- **Not implementing variable-impedance control (VIC) or admittance control.** Factory uses fixed-gain operational-space control; we port that exactly. VIC is flagged as future research in the controller taxonomy appendix.
- **Not implementing the FactoryAutomate / disassembly variants** in `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/automate/`. Same controller chain, different task definitions; can be added as a follow-up once Factory parity lands.
- **Not implementing sim2real for Factory tasks.** No real Franka arm in our hardware loop. Deploy artifacts (control metadata, ONNX export) are NOT stamped on Factory checkpoints in this plan.
- **Not implementing friction-based grasp in Phase 1.** Phase 1 uses `<equality><weld/>` to attach the held asset to the gripper. Friction grasp is a Phase 7+ extension; design accommodates via `grasp_mode: Literal["weld", "friction"]` config flag from day 1.
- **Not parity-strict from day 1.** Functional-first: train fresh, hit ≥0.5 reward/step. Numerical parity to Isaac configurations (reward at fixed pose, reset histograms) is a Phase 8 refinement, not a Phase 1 gate.
- **Not implementing FastSAC / FlashSAC / TDMPC2 baselines on Factory in this plan.** FastSAC's C51 atoms `[-20, 20]` clip Factory's discounted return distribution (per-episode max ~1350-2700); vanilla SAC scalar Q is the right fit. FlashSAC / TDMPC2 are deferred until vanilla SAC validates the env.
- **Not refactoring `mjx_backend.py` or `make_env_bundle`.** Factory routes through the existing Warp path (`impl="warp"` inside the mjx backend) — same path Go2WarpJoystick* uses.
- **Not bringing the IsaacLab observation history / `IsaacLabHistoryWrapper`** semantics. We replicate Isaac's action+prev_action concat directly into the obs vector at env level; no separate wrapper.

## Design

### File layout

```
jax_rl/envs/manipulation/factory/
├── __init__.py
├── factory_base.py             (~250 lines) shared obs/reward/reset math, controller hook
├── factory_peg_insert.py       (~200 lines) Phase 1-3 concrete env
├── factory_nut_thread.py       (~150 lines) Phase 5 — extends factory_base, swaps assets + rotation check
├── factory_gear_mesh.py        (~150 lines) Phase 6
├── controller/
│   ├── osc.py                  (~150 lines) Khatib operational-space controller
│   └── action_chain.py         (~50 lines)  EMA, denorm, clip, unidirectional_rot
└── assets/
    ├── franka_panda/           (menagerie panda.xml + fingertip_centered site override)
    ├── peg_insert/
    │   ├── scene.xml           (MJCF — panda + peg + hole + weld equality)
    │   ├── hole_assembly.xml
    │   └── meshes/             (CoACD-decomposed hole annulus OBJs)
    ├── nut_thread/             (Phase 5 — bolt + nut meshes from USD extract)
    └── gear_mesh/              (Phase 6)

jax_rl/configs/env_presets.py   (MODIFY) — register Factory<Task> presets, route via mjx_backend

scripts/
└── factory/                    (NEW dir for one-shot scripts)
    ├── spike_drop.py           (Phase 0 — Warp peg-in-hole drop test, no env scaffolding)
    └── extract_usds.py         (Phase 4 — pull Factory USDs from Nucleus, run nvidia-srl-usd-to-urdf)

tests/
├── test_factory_obs_schema.py          (hermetic)
├── test_factory_action_chain.py        (hermetic)
├── test_factory_reward_squashing.py    (hermetic)
├── test_factory_reset_distribution.py  (hermetic)
├── test_factory_osc_jacobian.py        (hermetic)
├── test_factory_osc_pose_error_sign.py (hermetic)
├── test_factory_osc_tracking.py        ([gpu, warp])
└── test_factory_env_smoke.py           ([gpu, warp])
```

No new training script. No new algo. No new backend. `train_sac.py --env FactoryPegInsert` is the canonical invocation.

### Phase plan

**2026-05-27 update:** Verified all 9 Factory USDs (peg, hole, nut, bolt, 4 gear variants, franka_mimic) are accessible via NVIDIA's public S3 mirror at `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/IsaacLab/Factory/<name>.usd`. No Isaac Sim or auth needed — pure `curl`. Per user preference, manipulands get USD parity from day 1 (the Franka still comes from menagerie since the arm doesn't benefit from PhysX-specific physics).

Phase 4 (asset extraction) split: **Phase 0a** pulls peg+hole USDs and validates the extraction pipeline before any env code; **Phase 4** (nut+bolt+gear) deferred to right before NutThread/GearMesh plans.

| Phase | Scope | Wall-clock | Cumulative |
|---|---|---|---|
| 0a | USD extract pipeline + peg+hole URDF/OBJ extraction (validates `nvidia-srl-usd-to-urdf` works) | 0.5-1 day | 1d |
| 0b | Warp physics spike using USD-extracted peg+hole geometry | 1 day | 2d |
| 1 | PegInsert env scaffold (obs/reward/reset, placeholder controller) | 2 days | 4d |
| 2 | OSC controller chain (Khatib, fp64 escape hatch) | 3-4 days | 8d |
| 3 | PegInsert SAC train baseline (dev + train wall-clock) | 3 days | 11d |
| 4 | Asset extraction for nut+bolt+gear (deferred — runs right before Phase 5/6 plans) | 1 day | 11d |
| 5 | NutThread (asset swap + reward coeff change + unidirectional_rot) | 1-2 weeks | ~21d |
| 6 | GearMesh (asset swap + engagement check) | 1 week | ~26d |

Each phase has explicit pass criteria (below). Going to the next phase before pass = anti-pattern; document the failure and back up.

### Phase 0a — USD extract pipeline (peg+hole)

**Goal:** Pull `factory_peg_8mm.usd` + `factory_hole_8mm.usd` from S3, run `nvidia-srl-usd-to-urdf`, produce loadable URDF + OBJ meshes for both. Validate the pipeline on the simplest USDs before Phase 0b uses them.

**Prereq:** Set up `usd-core` + `nvidia-srl-usd-to-urdf` in the project venv (`uv pip install usd-core nvidia-srl-usd-to-urdf`).

**Deliverable:**
```
jax_rl/envs/manipulation/factory/assets/peg_insert/extracted/
├── factory_peg_8mm.urdf
├── factory_hole_8mm.urdf
└── meshes/
    ├── peg_*.obj             (single piece — peg is convex)
    └── hole_decomp/
        └── hole_*.obj        (CoACD pieces — hole is non-convex)
```

**Pipeline script:** `scripts/factory/extract_usds.py`:
1. `curl` peg+hole USDs from S3 → `.tmp/factory_usds/`
2. Run `nvidia-srl-usd-to-urdf` on each USD → URDF + mesh OBJs
3. CoACD decomp on hole mesh (non-convex)
4. Stamp extracted assets into `jax_rl/envs/manipulation/factory/assets/peg_insert/extracted/`

**Pass criteria:**
- Both USDs download (HTTP 200, non-zero file size)
- Tool produces valid URDF + OBJ outputs
- `mujoco.MjModel.from_xml_path(<peg>.urdf)` loads peg without error
- `mujoco.MjModel.from_xml_path(<hole>.urdf)` loads hole without error
- CoACD decomp on hole produces ≤32 pieces

**Fallback cascade if tool fails:**
1. Isaac Sim's `omni_exporter_urdf` extension (sibling repo has Isaac Sim installed)
2. FreeCAD fab from `factory_tasks_cfg.py` dimensions (peg=cylinder 7.986mm × 50mm, hole=annulus 8.1mm × 25mm)
3. Hand-author MJCF primitives (loses Isaac geom parity but unblocks)

### Phase 0b — Warp physics spike

**Goal:** Prove Warp resolves cylinder-into-cylindrical-hole contact at 8mm scale with 114µm clearance, using the USD-extracted geometry from Phase 0a.

**Deliverable:** `scripts/factory/spike_drop.py` — single-file Warp script. No env scaffolding, no controller, no Franka.

**MJCF (minimal):**
```xml
<mujoco model="peg_hole_spike">
  <option timestep="0.002" integrator="implicitfast"/>
  <option iterations="100" tolerance="1e-10" solver="Newton"/>
  <worldbody>
    <body name="hole" pos="0 0 0">
      <!-- CoACD-decomposed hole annulus pieces -->
      <geom name="hole_001" type="mesh" mesh="hole_001" condim="6"
            friction="1.0 0.01 0.0001" solref="0.004 1" solimp="0.98 0.995 0.001"/>
      <!-- ... -->
    </body>
    <body name="peg" pos="0 0 0.080">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.019" diaginertia="..."/>
      <geom type="cylinder" size="0.003993 0.025" condim="6"
            friction="1.0 0.01 0.0001"/>
    </body>
  </worldbody>
</mujoco>
```

**Tests:**
1. Drop test: peg released at (0, 0, 80mm), zero velocity. Step 500 × 2ms = 1s sim.
2. Lateral nudge: same with (2mm, 0, 80mm) initial offset.
3. Tilt: same with 10° initial tilt.

**Pass criteria:**
- No NaN in qpos/qvel
- Final peg z ∈ [−1mm, +1mm]
- Final lateral ‖xy‖ < 2mm
- Sustained contact count > 0 for >50 steps
- 10° tilt self-rights via chamfered top of annulus

**Fail-mode response:**
- NaN → bump `iterations=200`, tighten `solref=0.002 1`
- Tunneling → CoACD `--threshold 0.05` → `0.02` (more pieces, denser contact)
- Contact flickering → check `condim=6` everywhere, bump `naccdmax`
- Drift instead of settling → bump friction tuple

**Escalation:** if still failing after solver + geom tuning by EOD, surface to user. Possible outcomes: widen clearance to 500µm (breaks parity but unblocks), or conclude MuJoCo can't do this geometry and Factory-class threading is out of reach for the repo.

### Phase 1 — PegInsert env scaffold

**Goal:** Loadable Warp env that resets, steps with placeholder actions, computes obs/reward/done correctly. NO real controller yet.

**File `factory_peg_insert.py` shape:**
```python
class FactoryPegInsert:
    obs_dim = 25            # 13 base + 6 actions + 6 prev_actions
    privileged_obs_dim = 72 # full state for asymmetric AC — verify against Isaac on first read
    action_dim = 6          # [dx, dy, dz, drx, dry, drz] in [-1, 1]

    _obs_groups = {
        "state": [
            ("fingertip_pos_rel_fixed", 3), ("fingertip_quat", 4),
            ("ee_linvel", 3), ("ee_angvel", 3),
            ("actions", 6), ("prev_actions", 6),
        ],
        "privileged_state": [...],  # ~72d, per port guide §Observation
    }

    def reset(self, key) -> State: ...
    def step(self, state, action) -> State: ...
    def get_domain_randomization_spec(self) -> List[DRSpec]: ...
    def get_control_metadata(self) -> dict: ...  # stub for Phase 7+ deploy
```

**MJCF composition (`assets/peg_insert/scene.xml`):**
```xml
<mujoco model="peg_insert">
  <include file="../franka_panda/panda.xml"/>
  <include file="hole_assembly.xml"/>
  <worldbody>
    <site name="fingertip_centered" pos="0 0 0.1034"/>
    <body name="peg" pos="0.6 0 0.10">
      <freejoint/>
      <geom type="cylinder" size="0.003993 0.025"
            friction="1.0 0.01 0.0001" condim="6"/>
    </body>
  </worldbody>
  <equality>
    <weld name="grasp" body1="peg" body2="panda_hand"
          relpose="0 0 0.025 1 0 0 0" active="true"/>
  </equality>
  <contact>
    <exclude body1="panda_leftfinger" body2="peg"/>
    <exclude body1="panda_rightfinger" body2="peg"/>
  </contact>
</mujoco>
```

**Action chain (`controller/action_chain.py`):** EMA (`ema_factor=0.2`), denorm to pose delta (`pos_threshold=[0.02,0.02,0.02]`, `rot_threshold=[0.097,0.097,0.097]`), clip relative to bolt position (`pos_action_bounds=[0.05,0.05,0.05]`). `unidirectional_rot` flag default False (PegInsert), True for NutThread.

**Reward (`_compute_reward`):** port Factory's `squashing_fn` from `factory_utils.py:105`:
```python
def squashing_fn(x, a, b):
    return 1 / (jnp.exp(a * x) + b + jnp.exp(-a * x))
```
Reward = `kp_baseline + kp_coarse + kp_fine + curr_engaged + curr_success`, no action penalties for PegInsert. Per-step max ~3.

**Reset distribution** via `DomainRandWrapper` per_step, declared specs:
- Bolt/hole pos noise: ±5cm cube
- Bolt yaw: 120° ± 30°
- Peg held offset: [0, ±3mm, ±3mm]
- Hand init: [0, 0, 1.5cm] ± [2cm, 2cm, 1cm]
- Hand init yaw: 1.83 rad ± 0.26 rad

**State.info auto-reset trap:** EMA `actions` and `prev_actions` carry across timesteps via `state.info`. Brax auto-reset does NOT clear info. Use `jp.where(done, 0, ema)` at step start. Dedicated test `test_factory_action_chain.py` verifies reset behavior.

**Future-work note (Phase 7+):** the weld equality is expedience. Long-term: friction grasp via stiff gripper PD + finger-peg contact for sim2real, grasp-force studies, mid-episode regrasp. `grasp_mode: Literal["weld", "friction"]` config flag stubbed from day 1 so the swap is a config change, not a refactor.

**Pass criteria:**
- `make_env_bundle("FactoryPegInsert", seed=0)` returns populated bundle with dict obs
- `train_sac.py --env FactoryPegInsert --total-timesteps 50000` runs end-to-end with placeholder controller (return is terrible — expected)
- Hermetic schema test passes
- Reward parity test (Phase 1.5, optional): set robot to fixed config, compute reward, compare to Isaac value at same config — within 1e-3
- **Lock exact `privileged_obs_dim`** — read Isaac's `_get_factory_obs_state_dict` resolution, count fields, update `obs_dim` constant on the env class, lock in PR. Downstream actor/critic net sizing depends on this number; merging Phase 1 with an estimate risks net-arch drift through Phases 2-3.

### Phase 2 — OSC controller chain

**Goal:** Port Factory's `compute_dof_torque` (Khatib operational-space control with dynamically-consistent nullspace) to JAX on top of MJX/Warp. Tracking error <1cm RMS on scripted trajectory.

**Controller pipeline** (per port guide §Layer 3):
```
action (6d, [-1,1])
  → EMA: actions = 0.2 * action + 0.8 * actions
  → denorm: pos_delta = actions[:3] * pos_threshold
            rot_delta = actions[3:] * rot_threshold
  → clip: ctrl_target_pos = clip(fingertip + pos_delta, bolt ± pos_action_bounds)
  → inner 8 substeps @ 1/120s (decimation=8):
        wrench = [Kp*pos_err - Kd*vel, Kp_rot*rot_err - Kd_rot*omega]
        tau_task = J^T @ wrench                              (7d arm)
        Lambda = inv(J @ M^-1 @ J^T)                          (6x6 task mass)
        J_eef_inv = Lambda @ J @ M^-1                         (6x7 DC pseudo-inverse)
        tau_null = (I - J^T @ J_eef_inv) @ M @ (kp_null*(q_def - q) - kd_null*qdot)
        tau = clamp(tau_task + tau_null, -100, +100)
        mjx.step(model, data.replace(ctrl=tau))
```

**Decimation:** stock Factory uses `decimation=8` (sim=120Hz, policy=15Hz). Sibling repo uses `decimation=4` (policy=30Hz). We pick **decimation=4** to match the sibling repo's PPO numbers (1570 reward at 5M) for direct comparison.

**Gains** (per `factory_env_cfg.py`):
- `default_task_prop_gains = [100, 100, 100, 30, 30, 30]` during episode
- `reset_task_prop_gains = [300, 300, 300, 20, 20, 20]` during reset settling
- `kd_task = 2 * sqrt(kp_task)` (critical damping)
- `kp_null = 10.0`, `kd_null = 6.3246`

**fp64 escape hatch:** `Lambda = inv(J·M⁻¹·J^T)` can ill-condition near singular configs. If fp32 produces NaN/spikes, wrap OSC math in `jax.config` block with `jax_enable_x64=True`. Env physics stays fp32. ~2× cost on controller, negligible vs physics.

**Inner loop:** **must use `jax.lax.fori_loop` or `lax.scan`, NOT Python loop.** Per `.context/lessons/warp.md`, Warp contact buffer OOMs in Python loops.

**Hermetic tests (CPU):**
1. `test_factory_osc_jacobian.py` — for known qpos+qvel, `J @ q̇` matches measured fingertip velocity from MuJoCo CPU model
2. `test_factory_osc_pose_error_sign.py` — robot at offset, OSC wrench points toward target

**GPU tests (`[gpu, warp]`):**
3. `test_factory_osc_tracking.py` — scripted trajectory (10cm fwd over 2s, 10cm down over 2s), assert final pos error <5mm, RMS over last 50 steps <2mm
4. Implicit in `test_factory_env_smoke.py` — no NaN at 256 envs over 1000 steps

**Pass criteria:**
- All 4 tests pass
- Scripted trajectory tracks within 1cm RMS over 30s (port guide milestone)
- No NaN at full training scale

**Risks (per Phase):**
- Jacobian convention mismatch (gripper joints in nv but not arm_dof_ids — slice carefully)
- Mass matrix slicing (`jnp.ix_(arm_dof_ids, arm_dof_ids)` for full submatrix)
- Decimation timing mismatch (inner finite-diff window must be 1/120s, not 1/30s)

### Phase 3 — PegInsert SAC train baseline

**Goal:** Validate env + controller learn together. Beat random by 100×.

**Algo:** vanilla SAC + asymmetric AC. Dict obs `{state, privileged_state}` auto-engages asymmetric critic via `make_offpolicy_state(dict_obs=True)`.

**Why not FastSAC:** C51 atoms `[-20, 20]` clip Factory's per-episode return (max ~1350-2700 with `decimation=4`). Manipulation lesson covers this explicitly. Scalar SAC has no such bound.

**Why not PPO:** Factory is short-horizon, dense-reward — off-policy data-efficient wins on our compute budget (RTX 5080, 256-512 envs after contact cap).

**Hyperparams (starting point):**
```python
SACConfig(
    num_envs=256,                # contact budget cap, validate in spike
    total_timesteps=5_000_000,
    batch_size=1024,
    buffer_size=1_000_000,
    gamma=0.99,                  # not 0.97 (Go2 special)
    tau=0.005,                   # standard SAC
    lr=3e-4,
    grad_updates_per_step=2,     # UTD=2, matches PushT recipe
    target_entropy=-6.0,         # -action_dim
    handle_truncation=True,      # 450-step TimeLimit must mask
    obs_norm=True,               # normalize at sample time, NOT pre-buffer
)
```

**Sanity gates (reward/step, port guide):**
| Baseline | reward/step | Total @ 450 steps |
|---|---|---|
| Random uniform | ~0.005-0.01 | ~4 |
| Scripted P-controller | 0.5-1.0 | ~225-450 |
| Trained SAC @ 5M (Phase 3 bar) | >0.5 | >225 |
| Isaac PPO @ 5M (parity bar, Phase 8 deferred) | ~1.74 | ~785 |

Isaac PPO parity number (1.74 reward/step, 1570 total at 5M, decimation=4, num_envs=1024) is sourced from `/home/stevenman/Desktop/Work/Research/isaaclab-factory-jax/docs/mujoco_port_guide.md` §Reference numbers. If/when Phase 8 (numerical parity) is revived, pin the sibling-repo commit hash and run config alongside the bar so the target stays reproducible.

**Telemetry:** `--wandb` always (project rule). Tag run `factory_peg_insert_v1`. Log per-component rewards (`r_baseline`, `r_coarse`, `r_fine`, `r_engaged`, `r_success`), `keypoint_dist`, `fingertip_height_above_hole`.

**Eval recording:** `record_video.py` works out of box (mjx backend dispatches). NPZ trajectory dump catches reward-hacking (`.context/lessons/bongo.md`).

**Pass criteria:**
- SAC @ 5M reaches ≥0.5 reward/step
- Recorded video shows actual peg-in-hole behavior (not z-axis tunneling)
- No mid-training NaN (Q-bias spike, controller blow-up)

### Research note — collision geometry for threaded / fine-feature meshes (Phase 5/6)

Investigation (2026-05-27) found that **MuJoCo Warp supports SDF collision but does not auto-bake mesh→SDF** — user must supply `@wp.func` for distance + gradient and register via `mjw._src.collision_sdf.user_sdf`. There is no built-in voxelizer.

**Isaac's Factory peg/hole uses convex decomposition + 192 solver iterations, NOT SDF.** Convex decomp is sufficient for an 8mm annular bore. Therefore: Phase 0a stays on CoACD path.

**For Phase 5 (NutThread) and Phase 6 (GearMesh)** — convex decomp is a known bad case for thread geometry (hundreds of pieces, contact-budget blowout). Two viable paths for our port:

1. **Convex decomposition (cheap, may not work):** Run CoACD on extracted nut/bolt/gear meshes. If piece count is reasonable (<64) and contact resolves, ship it. Same path as peg/hole. Bet on 192-iteration solver brute force.

2. **Analytical Warp SDF (research bet, higher payoff):** Hand-author screw-thread SDFs as Warp `@wp.func` kernels parameterized by thread pitch, diameter, lead-in. References:
   - Inigo Quilez's distance-function articles (`https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm`) — screw thread SDF formulas exist in literature
   - MuJoCo's own SDF plugin example (`bowl_sdf`) demonstrates the registration pattern
   - Once authored, a screw SDF generalizes to ANY thread (M16, M8, etc.) by changing parameters — write once, reuse for nut + bolt + future thread sizes

Decision deferred to Phase 5 kickoff. CoACD-on-threads is the fallback if SDF authoring proves too costly. SDF authoring is the long-term right answer for assembly-task research where threading is the central interesting physics.

### Phase 4 — Asset extraction for nut+bolt+gear (deferred; runs right before Phase 5/6 plans)

**Goal:** Extend the Phase 0a pipeline to nut+bolt+gear USDs. Gates NutThread (Phase 5) and GearMesh (Phase 6) plans.

**Prereq:** Phase 0a pipeline already proven on peg+hole.

**Test cascade (already cheap → potentially expensive):**
1. `factory_nut_m16.usd` (threaded internal)
2. `factory_bolt_m16.usd` (threaded external)
3. `factory_gear_medium.usd` (gear teeth)

**Pass criteria per asset:**
- Extracts to URDF + OBJ without error
- `mujoco.MjModel.from_xml_path` loads the URDF (or wrapped MJCF) without error
- Mesh visually matches USD in `meshlab` (eyeball)
- CoACD decomp produces ≤32 pieces (contact-budget guardrail)

**Fallback cascade if extraction fails on threaded geom:**
1. Isaac Sim's `omni_exporter_urdf` extension (heavier but proven)
2. FreeCAD fab from M16/gear dimensions in `factory_tasks_cfg.py` (uses `Part.makeHelix`)
3. Hand-author MJCF threaded geom via `<replicate>` of pitched helical segments

**Deliverable:** `.tmp/factory_assets/{peg,hole,nut,bolt,gear}/*.urdf|*.obj` + journal note documenting which tool worked, decomp piece counts, fallback if used.

**Does NOT block Phase 3.** Phase 5 starts after Phase 4 lands; if Phase 4 fails, Phase 5 uses FreeCAD fallback.

### Phase 5 — NutThread

**Delta from PegInsert:**

| Component | Change |
|---|---|
| Held asset | Cylinder peg → M16 nut (extracted URDF + CoACD decomp) |
| Fixed asset | Annulus hole → M16 bolt (extracted URDF + CoACD decomp) |
| Reward coeffs | `keypoint_coef_baseline=[100,2]`, `coarse=[500,2]`, `fine=[1500,0]`. `success_threshold=0.375`, `engage_threshold=0.5`, `keypoint_scale=0.05` |
| Action | `unidirectional_rot=True` — wrist yaw negative-only |
| Success check | `_get_curr_successes(check_rot=True)` — also requires yaw aligned to `ee_success_yaw=0` |
| Reset distro | Same shape, different magnitudes |
| Solver | May need `iterations=200` for threading — validate via mini-spike before training |
| Contact budget | ~24×24 decomp pairs × N envs — likely caps num_envs at 64-128 on 16GB |

**Pass criteria:**
- Mini drop+rotate spike on nut+bolt passes (threaded contact resolves, no tunneling)
- SAC @ 5M reaches ≥0.3 reward/step (lower bar than PegInsert — contact-rich is harder)
- Recorded video shows actual threading (not z-tunneling)

### Phase 6 — GearMesh

**Delta from NutThread:**
- Held = `factory_medium_gear`, Fixed = `factory_gear_base`
- Engagement = teeth interlock geometry (different success check)
- May need lower decimation (gear teeth contact is fast)

Same env+controller stack, mostly asset + reward config swap.

### Testing strategy

Per `.context/lessons/testing_new_envs.md` — default lane is hermetic CPU. GPU-marked tests excluded from default lane via `pyproject.toml addopts`.

| Test | Phase | Marker | Pattern |
|---|---|---|---|
| `test_factory_obs_schema.py` | 1 | hermetic | `_obs_groups` resolves, deploy `ObsBuilder` consumable |
| `test_factory_action_chain.py` | 1 | hermetic | EMA + denorm + clip math, unidirectional_rot flag, state.info auto-reset |
| `test_factory_reward_squashing.py` | 1 | hermetic | `squashing_fn`, `keypoint_dist`, fixed-pose reward computation |
| `test_factory_reset_distribution.py` | 1 | hermetic | DR spec composition over 1000 samples, histogram bounds |
| `test_factory_osc_jacobian.py` | 2 | hermetic | `J @ q̇` matches FK velocity (small CPU MJX model) |
| `test_factory_osc_pose_error_sign.py` | 2 | hermetic | Wrench points toward target |
| `test_factory_osc_tracking.py` | 2 | `[gpu, warp]` | Scripted trajectory <1cm RMS over 30s |
| `test_factory_env_smoke.py` | 1+2 | `[gpu, warp]` | reset + 100 steps random + 1 done, no NaN |
| `test_factory_reward_parity.py` | 1 (opt) | hermetic | Same robot config as Isaac → reward within 1e-3 (requires Isaac dump) |

**No module-level CUDA bombs, no hardcoded paths, no checkpoint/wandb scanning** per testing contract.

### Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Warp can't resolve 114µm peg-hole clearance | medium | high (blocks all phases) | Phase 0 spike answers in 1 day. Fallback: widen clearance to 500µm (breaks parity, unblocks) |
| R2 | OSC fp32 numerically unstable near singularities | medium | medium | fp64 escape hatch ready; `kp_null` keeps robot in dexterous region |
| R3 | Contact budget caps num_envs to <64 | high | medium (slower training) | Plan Phase 3 at 256 envs; eat wall-clock hit if forced lower |
| R4 | `nvidia-srl-usd-to-urdf` fails on threaded mesh | medium | medium (delays Phase 5) | Fallback: Isaac Sim exporter → FreeCAD fab |
| R5 | Reward function silently wrong (sign flips, EMA stale) | medium | high | Phase 1.5 reward parity test against fixed Isaac configs |
| R6 | State.info auto-reset trap on EMA actions | high | medium | Explicit `jp.where(done, 0, ema)` reset, dedicated test |
| R7 | Weld equality + Warp incompatibility | low | high | Validate in Phase 0 spike sub-test |
| R8 | Decimation 4 vs 8 mismatch with Isaac | low | low (functional-first) | Pick 4 (matches sibling repo PPO numbers) |

(R8 obs_dim 19 vs 25 was resolved during brainstorm — verified against `factory_env.py:_get_observations`, locked at 25 in Phase 1 class sketch. Removed from active register; see Appendix B for the remaining `privileged_obs_dim` lock-in note.)

### Doc sweep on completion of each phase

Per `CLAUDE.md` doc-sync rule:
- `.context/journals/YYYY-MM-DD.md` — what happened, results, blocker hits
- `.context/TODO.md` — mark phase complete, queue next
- `.context/AGENT_HANDOFF.md` — add Factory benchmark row, update Quick Reference commands
- `.context/lessons/manipulation.md` — capture any reusable lessons (contact-budget tuning, OSC fp32 quirks, weld/grasp tradeoff observations)
- `docs/scripts/gen_cli_reference.py` regen if any new flags surface (unlikely — we reuse `train_sac.py`)

### Appendix A — Cartesian controller taxonomy (reference for future picks)

| # | Name | τ formula sketch | Use case in this repo |
|---|---|---|---|
| 1 | Joint PD | `τ = Kp(q_t − q) − Kd·q̇` | Go2 (current). Direct joint tracking. |
| 2 | IK + Joint PD | IK → q_t, then #1 | Not in repo. Pick-and-place w/o contact. |
| 3 | Jacobian-transpose (naive) | `τ = J^T(Kp·err − Kd·ẋ)` | Cheap Cartesian, mass-coupled. **Phase 2 fallback if OSC fails.** |
| 4 | **Operational Space (OSC, Khatib)** | `τ = J^T·Λ·(Kp·err − Kd·ẋ) + N·τ_null`, `Λ = (JM⁻¹J^T)⁻¹` | **Phase 2 primary. Factory ports this exactly.** |
| 5 | Impedance (Cartesian) | `F = M_d·ẍ_d + B_d·ẋ_d + K_d·x_d` → J^T | Future research. Explicit dynamic behavior shaping. |
| 6 | Admittance | `ẍ_target = M⁻¹(F_ext − Bẋ − Kx)` | Needs F/T sensor. Skip until force feedback in obs. |
| 7 | Hybrid force-position (Raibert) | Selector per axis | Wiping, drilling. Out of Factory scope. |
| 8 | Variable Impedance (VIC) | #5 + policy outputs K_d, B_d | Future research. Natural extension once OSC works. |
| 9 | MPC / OCP | Solve QP/NLP per step | Trajopt-heavy, out of scope. |

Factory ≈ #4. OSC inherently produces impedance behavior (compliant in nullspace, stiff in task space), so OSC is sometimes called "task-space impedance," but strict #5 lets you tune M_d separately — Factory doesn't.

### Appendix B — Open implementation questions

- **Decimation 4 vs 8:** locked at 4 (matches sibling repo PPO benchmark). Document in Phase 1 PR.
- **`task_prop_gains` in obs:** include in privileged obs (Isaac does), even though OSC uses fixed gains — preserves obs schema parity for future VIC research.
- **Sim2sim recorder (analog to `deploy/sim2sim_direct.py`):** not until `grasp_mode="friction"` lands (Phase 7+).
- **Exact `privileged_obs_dim`:** doc estimate ~72; verify against Isaac `_get_factory_obs_state_dict` resolution in Phase 1 first read. Lock then.

### Appendix C — Cross-references

- Port guide: `/home/stevenman/Desktop/Work/Research/isaaclab-factory-jax/docs/mujoco_port_guide.md`
- Isaac source: `/home/stevenman/Desktop/Work/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/`
- Repo contracts:
  - `.context/lessons/algo_port_protocol.md` — bundle, artifact, CLI, test rules
  - `.context/lessons/env_backends.md` — backend contract
  - `.context/lessons/testing_new_envs.md` — marker discipline + hermetic patterns
  - `.context/lessons/warp.md` — Warp gotchas (lax.scan required, joint vs actuator ordering, PD gains)
  - `.context/lessons/manipulation.md` — PushT lessons (solref/solimp, condim, TimeLimit, action-repeat)
- Reference envs in repo:
  - `jax_rl/envs/locomotion/go2_warp_base.py` — Warp env pattern
  - `jax_rl/envs/wrappers/domain_rand.py` — per_step DR pattern
  - Vendored PushT env (in `jax_rl/training/env_backends/gym_backend.py`) — manipulation reward shaping reference
