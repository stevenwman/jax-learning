# Splitbelt Treadmill Env — Design

**Date:** 2026-05-02
**Branch:** `new_slate_linen`
**Status:** Brainstorming → spec (pending review)
**Author:** Steven + Claude (brainstorming session)

---

## 1. Problem

Build a benchmark locomotion environment in which a quadruped (and later a humanoid) walks on a split-belt treadmill — two parallel belts whose speeds can be commanded independently. The env is the substrate for an *adaptation benchmark* that aims to **differentiate four flavors of adaptation** (A1–A4 below) on the same physical task.

The env is algorithm-agnostic — must work cleanly with PPO, SAC, FastSAC, FlashSAC, TD-MPC2 — and embodiment-agnostic (Go2 first, humanoid later, sharing the treadmill apparatus).

This is a research env, not a deployment target.

---

## 2. Research framing — the adaptation continuum

Splitbelt locomotion is a canonical paradigm in motor-control neuroscience (Reisman/Bastian for humans; Frigon lab for quadrupeds). The interesting scientific question is not "can RL walk on a split belt" but **what kind of adaptation does a given policy/algorithm exhibit on a perturbation it has never seen, or has seen only in training?**

Four protocols, all runnable on the same env, span a continuum from purely reactive to fully meta-learned:

| Protocol | What's adapted | Where adaptation lives | Closest analog |
|---|---|---|---|
| **A1. Online within-episode** | Fixed policy weights; belts switch tied → split → tied mid-episode. Measures recovery time + after-effects. | Reactive control + memory | Bastian/Reisman human studies |
| **A2. Context-conditioned** | Policy reads (vL, vR) explicitly. Belts fixed per episode. Measures whether one network represents the (vL, vR) manifold. | Conditional generalization | Goal-conditioned RL |
| **A3. Meta-RL / fast adaptation** | Belts hidden from policy. Policy must infer latent context from proprio history (RNN, RMA-style adapter, RL²). | Online inference + latent representation | RMA, MAML, RL² |
| **A4. Pretrain-then-perturb** | Train on tied; switch to split; fine-tune. Measures plasticity vs stability. | Continual learning | Animal training protocols |

The benchmark contribution is the env + protocol set + metrics that distinguish these. Different metrics light up under different protocols (Section 9). We do **not** want a single scalar leaderboard — the value is in showing where each algorithm sits along the continuum.

---

## 3. Locked design decisions (brainstorm output)

The following were chosen during the brainstorming session; rationale captured here so reviewers don't re-litigate:

1. **Robot strategy: R3 — robot-agnostic apparatus, Go2 first.** The treadmill XML and schedule samplers are robot-blind. A humanoid variant is a follow-up env that reuses both. No humanoid asset work today.
2. **Belt mechanics: M1 — moving-geom belt slabs.** Each belt is a long box geom on a slide joint with a velocity actuator. Bongo-board precedent (`go2_bongo_handstand.py`) proves the pattern works on MuJoCo Warp. Cheaper alternatives (velocity-field force on contact, M2; reference-frame trick, M3) were rejected: M2 weakens the sim2real defensibility ("splitbelt becomes a friction model we made up"); M3 cannot represent per-foot belt frames. Slabs are gross-long (~50m) so a single episode does not require teleport.
3. **Task framing: T2 — joystick-style cmd.** Existing Go2 joystick env's reward shape ports near-zero. Stand/step-in-place corresponds to `cmd = 0`; walking-forward variants (`cmd_x > 0`) are a separate config later. The cmd vector stays in obs even when always zero (preserves obs/reward plumbing).
4. **Cmd frame: cmd is body-frame; treadmill_drift anchors world frame.** The IMU velocity sensor reads body-frame torso velocity, and the cmd vector is interpreted in body frame (matches `Go2WarpJoystickFlat` exactly). `cmd = 0` means body-frame torso velocity → 0; combined with `treadmill_drift` (penalty on world-frame `||base_xy − treadmill_center_xy||`), the two rewards jointly enforce "torso stationary in lab frame." Stride velocity is implicit — set by belt speed, not cmd. The treadmill is a stationary lab apparatus; the belt slabs translate under the robot but the apparatus frame is fixed at world origin (see `treadmill_center_xy` in §7.1).
   - **`treadmill_center_xy = (0.0, 0.0)`** (world origin). Defined here once; referenced by §7.1, §9.1.
   - The `cmd > 0` walking variant (§11.2) would conflict with `treadmill_drift` (forward translation vs anchor at origin) and is therefore out of scope for this spec. If walking is ever added, drift must be replaced with a band/region penalty.
5. **Obs modes: four supported via config switch (`cfg.obs_mode`).**
   - `blind` — proprio + cmd. (A1 fixed policy, A3 meta-RL, default.)
   - `informed` — proprio + cmd + (vL, vR). (A2.)
   - `error` — proprio + cmd + (cmd_track_error, drift_xy). Belt speeds hidden, error explicit. Tests value of explicit error signal alone.
   - `history` — k-step frame stack of proprio + cmd via the existing `FrameStackWrapper` (`jax_rl/envs/wrappers/frame_stack.py`). Wired via wrapper at env-bundle time when `obs_mode == "history"`. The env itself emits the same name layout as `blind`; the wrapper does the stacking. `k = cfg.history_len` (default 4).
   `privileged_state` content is defined once in §8 (single source of truth). Summary: full proprio set + (vL, vR) + cmd_track_error + drift_xy + (true) base_lin_vel + base_ang_vel. cmd is already in proprio. Nothing is privilege-hidden.
6. **Schedule API: S3 — per-episode schedule table.** A `(T, 2)` jax array filled at reset by a per-protocol sampler module (`splitbelt_schedules.py`). Env step is a pure index op, JIT/vmap clean. New protocol = new sampler, env doesn't change.
7. **Online metrics = primitives only.** Composite metrics (limp index, recovery time, after-effects) computed offline from logged time-series. Reduces bug surface in the env itself; matches the bongo lesson "verify visually, watch for silent reward hacking."
8. **No symmetry reward.** Step-length symmetry is a *measurement*, not a reward. Rewarding it would make the benchmark question circular. Reward = task (cmd tracking + stay-on-treadmill); asymmetry = what we measure.

---

## 4. Architecture

```
SplitbeltTreadmill env (mjx backend, Warp impl)
├── Asset: jax_rl/envs/locomotion/xmls/treadmill_splitbelt.xml   # robot-agnostic
│   ├── left_belt slab — slide joint along x, velocity actuator
│   ├── right_belt slab — slide joint along x, velocity actuator
│   └── fallback_floor — static, for off-belt + fall termination
├── Asset: jax_rl/envs/locomotion/xmls/go2_warp_splitbelt_scene.xml
│   ├── <include> unitree_go2/go2.xml (existing)
│   └── <include> treadmill_splitbelt.xml
├── Env: jax_rl/envs/locomotion/go2_warp_splitbelt.py
│   └── Go2WarpSplitbeltEnv(Go2WarpBase)
│       ├── _obs_groups gated by cfg.obs_mode ∈ {blind, informed, error, history}
│       ├── _post_init: schedule_table buffer init, sensor IDs, default_pose
│       ├── step: belt actuator vel = schedule_table[step_idx], physics, log primitives  # framework calls .step (no underscore)
│       └── reset: schedule_sampler(rng, cfg.schedule_kind) → (T, 2) buffer
├── Schedule samplers: jax_rl/envs/locomotion/splitbelt_schedules.py
│   ├── tied(v)
│   ├── split_constant(vL, vR)
│   ├── tied_split_tied(v_warm, vL_split, vR_split, t1, t2, t3)   # A1
│   ├── random_per_episode(v_range, ratio_range)                  # A2/A3 training
│   └── continual_phase(phase_id, ...)                            # A4
├── Backend registration: jax_rl/training/env_backends/mjx_backend.py
│   └── register Go2WarpSplitbelt with default_config factory
├── Presets: jax_rl/configs/env_presets.py
│   └── PPO + FastSAC presets, derived from joystick presets
└── Offline analysis: scripts/analyze_splitbelt.py
    ├── loads splitbelt_traj.npz + schedule_table
    └── outputs gait asymmetry, recovery time, after-effects, plots
```

Belt slab length: ~50m per side (configurable). Belts reset to origin each episode. Episode length cap: 25s default = 1250 steps × 0.02s. At max belt speed 1.5 m/s × 25s = 37.5m of slab translation; 50m gives margin.

---

## 5. Components & responsibilities

### 5.1 `xmls/treadmill_splitbelt.xml` (new asset)
- Two long box geoms (`left_belt`, `right_belt`) on slide joints (`left_belt_joint`, `right_belt_joint`) along world x.
- Velocity actuators (`left_belt_vel`, `right_belt_vel`) — kv tuned so commanded velocity is achieved within one ctrl_dt.
- Friction on belt geoms = standard Go2 floor friction (mu=1.0). No special tribology.
- Small gap between belts (~5 cm, configurable). Gap-bottom is the static `fallback_floor` geom. Off-belt feet fall onto it; that contact triggers termination.
- No walls. Off-belt = termination, not bouncing back.

### 5.2 `xmls/go2_warp_splitbelt_scene.xml` (new robot-specific scene)
- `<include>` go2.xml + treadmill_splitbelt.xml.
- Spawn keyframe: Go2 in standard pose, base centered between belts, all four feet on the appropriate belts (FL, RL on left; FR, RR on right).
- Contact filter:
  - feet × {left_belt, right_belt, fallback_floor}
  - base/torso × {left_belt, right_belt, fallback_floor} — all three count as fall contacts
  Per `lessons/bongo.md` (2026-04-02): use exact contact sensors, not position heuristics, for termination. The base/torso × belt filters catch the "robot collapses onto a belt" failure mode that would otherwise rely on tilt/height threshold backstop.

### 5.3 `go2_warp_splitbelt.py` (new env module)
- `Go2WarpSplitbeltEnv` subclasses `Go2WarpBase` (NOT joystick — joystick has flat-floor specifics; splitbelt scene replaces them).
- Inherits Go2 physics, action space, PD control, base obs/reward primitives.
- **Reward terms duplicated from joystick env, not factored.** Splitbelt env defines its own reward computation reusing the same term shapes (tracking_lin_vel_xy, tracking_ang_vel_z, lin_vel_z, etc.). If a third locomotion env later wants the same shape, factor into a shared helper at that point. Premature abstraction now is a YAGNI violation.
- Adds:
  - `_belt_step` — look up `state.info["belt_schedule"][state.info["step_idx"], :]`, write to belt actuators.
  - `_compute_gait_metrics` — foot-on-belt detection from contact sensors, foot world positions, base kinematics → `state.info["splitbelt"]` dict.
- `_obs_groups` is a config-gated dispatch:
  - `blind`: proprio + cmd
  - `informed`: proprio + cmd + (vL, vR)
  - `error`: proprio + cmd + (cmd_track_error, drift_xy)
  - `history`: k-step frame stack of (proprio + cmd)
  - All modes feed the same `privileged_state` group with full info.
- `_default_pose`: from spawn keyframe.
- DR specs: PD gains, friction, mass, ctrl_dt jitter — same patterns as `Go2WarpJoystickFlat`. Belt friction is a candidate DR spec; default off until baseline lands.

### 5.4 `splitbelt_schedules.py` (new module)
- One factory per protocol. Signature: `(rng, episode_length, cfg) -> (T, 2)` jax array.
- All factories pure — no env knowledge.
- Tested in isolation (`tests/test_splitbelt_schedules.py`).

### 5.5 Backend registration
- `mjx_backend.py`: register `Go2WarpSplitbelt` with `default_config` returning a `tied(v=0.5) + cmd=0` config (smoke baseline).
- Future variants (different default schedules, different obs modes) register as separate names if useful.

### 5.6 Presets (`env_presets.py`)
- `Go2WarpSplitbelt` × {PPO, FastSAC} starting from joystick presets, episode length adjusted, DR pared down to splitbelt-relevant.
- Per-protocol presets (A1-eval, A2-train, etc.) deferred until env smoke-tests.

### 5.7 Metrics integration
- Env writes per-step gait primitives into `state.info["splitbelt"]`.
- `metrics_logger` captures aggregate scalars; full per-step traj dumps to `splitbelt_traj.npz` at eval-rollout time (sidecar to existing `_traj.npz`).

---

## 6. Step lifecycle & data flow

### 6.1 Per env step
1. **PD inner loop** (existing Go2WarpBase pattern) for each substep in `action_repeat × sim_dt / ctrl_dt`:
   - Target joint pos = `action_scale · action + default_pose`.
   - PD torque = `Kp·(qpos_target − qpos) − Kd·qvel`.
   - Belt actuator velocity = `info["belt_schedule"][info["step_idx"], :]`.
   - `mjx.step(model, data)` advances physics; belt slabs translate, friction drags feet.
2. **Outer step:**
   - Increment `info["step_idx"]`.
   - Compute obs per `_obs_groups` (gated by `obs_mode`).
   - Compute reward terms (Section 7).
   - Check termination (Section 7).
   - Compute gait primitives → `info["splitbelt"]`.
   - Auto-reset on done via `DomainRandWrapper` (existing).

### 6.2 Per env reset
1. Sample initial qpos/qvel: spawn keyframe + small noise (existing pattern).
2. Sample belt schedule: `schedule_table = schedule_sampler(rng_sched, T_max, cfg.schedule_kind, cfg.schedule_params)` → `(T_max, 2)`.
3. Reset belt slab joint qpos to 0 (slab back to origin).
4. Reset belt slab joint qvel to `schedule_table[0]` (so velocity actuator does not have to spin up from rest).
5. Sample DR fields per `DomainRandWrapper`.
6. Sample cmd via `cfg.cmd_sampler(rng_cmd)`. Default = always-zero.
7. Zero `info["splitbelt"]["term_cause"] = 0` (vmap-safe — prevents stale term_cause from prior episode bleeding through). All other `info["splitbelt"]` fields recomputed on first step from fresh state.
8. Initial obs computed; `info["step_idx"] = 0`.

`_step` writes `term_cause` only when `done` fires; otherwise field passes through unchanged (which after reset is 0). Standard auto-reset pattern.

### 6.3 Invariants
- `schedule_table` set once at reset, frozen for the episode.
- Env step is a pure indexing op into `schedule_table` — JIT/vmap clean, no closures, no python state.
- `schedule_table` lives in `state.info` and flows through vmap/scan transparently. At T=1250 and N=1024 envs, ≈10 MB float32 — fine.
- **Static-shape lock-in:** `schedule_table.shape == (cfg.episode_length, 2)` always. `cfg.episode_length` defaults to `1250` (= 25 s / 0.02 s ctrl_dt). We do NOT introduce a separate `T_max` cap; the episode_length is the JIT-static shape. Same convention as existing locomotion envs.
- **Belt initial qvel:** `_reset` sets belt slide-joint qvel to `schedule_table[0]`. For tied schedules this is the warmup speed; for `random_per_episode` this means the robot's spawn keyframe lands feet on already-moving belts. Belt actuator gain (set in `treadmill_splitbelt.xml`, default `kv=200`) settles within one ctrl_dt either way; we accept the first-step transient rather than adding a settle window. (Documented because it's a research consequence: A1 `random` rollouts begin under perturbation immediately.)
- **`step_idx` indexing invariant:** `info["step_idx"]` is incremented AFTER each step that just executed. Reads (`schedule_table[info["step_idx"]]` inside the next step) are always pre-increment, valid range `[0, episode_length-1]`. The episode terminates (`info["truncation"]=1`, wrapper auto-resets) before `step_idx` ever reaches `episode_length`. As defense-in-depth, the env clamps `step_idx_for_index = jnp.minimum(step_idx, episode_length - 1)` before indexing — guards against off-by-one in wrapper compositions.

---

## 7. Reward & termination

### 7.1 Reward terms

**All weights ported verbatim from `Go2WarpJoystickFlat` (`jax_rl/envs/locomotion/go2_warp_joystick.py:47-69`).** The implementation must populate `default_config().reward_config.scales` with the joystick numerical defaults — empty/`get(name, default)` patterns are forbidden because they silently produce ~10× weaker tracking reward and lose the calibration target (see §10.5).

Tracking:
- `tracking_lin_vel_xy` — body-frame v vs cmd[:2], exp kernel. Default scale matches joystick.
- `tracking_ang_vel_z` — body yaw rate vs cmd[2], exp kernel. Default scale matches joystick.

Stay-on-treadmill (NEW; only this term is bespoke):
- `treadmill_drift` — quadratic penalty `−(w_lat · drift_y² + w_fwd · drift_x²)` where `drift = base_xy − treadmill_center_xy = base_xy − (0, 0)` (treadmill_center per §3.4). Default `w_lat = 4·w_fwd` (lateral drift is the dominant failure mode — robot off the side of the belt → off-belt termination; forward drift is bounded by the long slab and self-corrects). Concrete defaults: `w_lat = 2.0`, `w_fwd = 0.5`. Total reward contribution scaled by an outer `treadmill_drift` weight (default 1.0).

Stability/smoothness (port verbatim from joystick — the implementation MUST include the full set, not a subset). The exact term names and scales come from `go2_warp_joystick.default_config().reward_config.scales` (`go2_warp_joystick.py:47-69`):
- `lin_vel_z`, `ang_vel_xy`, `orientation`, `torques`, `action_rate`, `energy`, `dof_pos_limits`, `feet_air_time`, `feet_slip`, `feet_clearance`, `feet_height`, `pose`, `base_height`.

(Note: this spec previously listed `joint_vel` and `survival` here in error — neither exists in the joystick reward set. Implementer should follow the joystick env file as source of truth, not these bullet names.)

`stand_still` from the joystick set is **dropped** for splitbelt because cmd is always zero — joystick's `_cost_stand_still` evaluates the joint-pose deviation gated by `cmd ≈ 0`, which collapses to a permanent joint-pose anchor. We rely on the existing `pose` term (also a joint-pose anchor) for that role; the redundancy with a permanent `stand_still` would just double-count. If smoke training shows the leg posture drifts, restore `stand_still` (it harmlessly evaluates on cmd=0).

Termination:
- `termination` — large negative on early done. Default scale matches joystick (-1.0).

### 7.2 What is NOT in the reward
- **No step-length symmetry reward.** Symmetry is the measurement, not the target. Including it would bias the policy toward producing it and make the adaptation question circular.
- **No belt-speed-aware tracking term.** Belts are perturbations in the dynamics; they do not enter the reward. Reward stays in world/lab frame.

### 7.3 Termination conditions

Hard (contact-based — primary detectors, exact, threshold-free):
- Base/torso contact with `fallback_floor` (fall onto gap floor) → `term_cause = fall`.
- Base/torso contact with `left_belt` or `right_belt` (collapse onto a belt) → `term_cause = fall`.
- Any foot contact with `fallback_floor` (off-belt) → `term_cause = off-belt`.

Hard (threshold-based — backstop only, in case a contact pair is misconfigured):
- Excessive base roll/pitch (gravity-vector body-z below ≈ 0.5) → `term_cause = tilt`.
- Base height below ≈ 0.18 m → `term_cause = tilt` (shares cause code; height is a tilt proxy).

Soft:
- Episode timeout at `episode_length` steps. Sets `info["truncation"] = 1`.

### 7.4 Cmd sampling defaults
- Default `cfg.cmd_sampler` returns zero. Stand/step-in-place is the canonical splitbelt regime.
- Walking variant (`cmd_x > 0`) is incompatible with `treadmill_drift` (direct conflict between forward translation and stay-at-origin). It is a future env config, NOT part of this benchmark.

---

## 8. Obs modes

| Mode | obs["state"] = | Used by | Notes |
|---|---|---|---|
| `blind` | proprio + cmd | A1, A3 | Default. Adaptation must come from history (RNN) or implicit dynamics. |
| `informed` | proprio + cmd + (vL, vR) | A2 | Conditional controller; tests (vL, vR)-manifold representation. |
| `error` | proprio + cmd + (cmd_track_error, drift_xy) | (probe) | Tests whether explicit error signal alone suffices for adaptation, with the *cause* (belt speeds) hidden. |
| `history` | k-step frame stack of (proprio + cmd) | (probe) | Vanilla memory-via-stacking; subset of blind, distinct from RNN-meta. |

`obs["privileged_state"]` is always full info regardless of mode. Concrete contents (single source of truth — overrides any earlier mention in §3.5 or §5.3):

- All proprio terms that appear in `blind` mode's `state` (joint_pos, joint_vel, last_action, gravity, gyro, cmd).
- (vL, vR) belt speeds.
- cmd_track_error (3-vector).
- drift_xy (2-vector).
- True body-frame linear velocity (`base_lin_vel`, no obs-noise).
- True body-frame angular velocity (`base_ang_vel`, no obs-noise).

Asymmetric critics see all of these uniformly; symmetric critics auto-alias to `state` per the env-bundle layer.

---

## 9. Metrics & logging

### 9.1 Per-step primitives (in `state.info["splitbelt"]`)

```python
state.info["splitbelt"] = {
    # Foot kinematics (FL/FR/RL/RR order)
    "foot_in_contact": jp.bool_[4],
    "foot_pos_world": jp.float32[4, 3],
    "foot_belt_id": jp.int32[4],            # 0=left, 1=right, -1=off-belt

    # Body kinematics
    "base_pos_world": jp.float32[3],
    "base_vel_world": jp.float32[3],
    "base_yaw": jp.float32,

    # Belt state (echo of schedule_table[step_idx])
    "belt_vel": jp.float32[2],              # (vL, vR)

    # Tracking
    "cmd_track_error": jp.float32[3],       # body-frame velocity error
    "drift_xy": jp.float32[2],              # base_pos − treadmill_center

    # Termination cause. Set on the step that fires `done`; otherwise 0.
    # Truncation (timeout) lives separately in info["truncation"] (Brax convention).
    # Episode-end log readers should consult BOTH fields:
    #   info["truncation"] == 1 → timeout
    #   term_cause != 0          → hard termination (fall/off-belt/tilt)
    #   neither                  → mid-episode alive snapshot
    "term_cause": jp.int32,                 # 0=alive 1=fall 2=off-belt 3=tilt

    # Schedule pointer
    "step_idx": jp.int32,
}
```

`foot_belt_id` derives from contact sensors `(foot × belt_geom)` — a `(4, 2)` bool array; id is `argmax` over the belt axis or `-1` if neither.

### 9.2 Logging channels

| Channel | Frequency | Destination | Use |
|---|---|---|---|
| Reward terms | per step | `metrics_logger` | Training curves |
| `splitbelt.term_cause` histogram | per episode-end | `metrics_logger` | Health (off-belt/fall rate) |
| Full per-step `splitbelt` dict | per eval rollout | `splitbelt_traj.npz` sidecar | Offline analysis |
| `schedule_table` | per eval episode | sidecar npz | Identifies which protocol generated the trajectory |

`record_video.py` writes `splitbelt_traj.npz` whenever `bundle.env_state.info` contains a `splitbelt` key — auto-skipped for non-splitbelt envs.

### 9.3 Offline analysis (`scripts/analyze_splitbelt.py`)
- Loads `splitbelt_traj.npz` + `schedule_table`.
- Detects step events (touchdown, lift-off) from `foot_in_contact` time-series.
- Per-stride: step length, double-support fraction, stance time per side.
- Aligns with schedule phase boundaries (tied → split → tied for A1).
- Outputs scalar metrics + matplotlib plots of asymmetry-vs-time.
- Pure python/numpy. No env or training infra required.

### 9.4 Adaptation metric matrix

| Metric | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| `step_length_asymmetry` | time-series | scalar | time-series | both |
| `recovery_time` | ✓ | — | ✓ | — |
| `after_effect_magnitude` | ✓ | — | partial | ✓ |
| `generalization_gap` | — | ✓ | partial | — |
| `forgetting` | — | — | — | ✓ |
| `context_inference_accuracy` | — | — | ✓ | — |

The env logs primitives; the analyzer produces all rows.

---

## 10. Testing strategy

Per `.context/lessons/testing_new_envs.md` (full set [gpu, warp, go2] markers, hermetic-by-default, mock-first, no module-level CUDA bombs).

### 10.1 Hermetic CPU (default lane, no markers)

`tests/test_splitbelt_schedules.py` — pure jax/numpy
- Each schedule sampler returns shape `(T, 2)` with values in expected range.
- `tied_split_tied`: phase boundaries at exact step indices, values constant within phase.
- `random_per_episode`: distribution matches config (rng-seeded reproducibility).
- Determinism: same rng → same table.
- All PRNGKey/jnp construction inside test bodies or fixtures.

`tests/test_splitbelt_obs_schema.py` (lesson §7.1)
- Mock env via `types.SimpleNamespace` + `_obs_groups` dict + `compute_obs`.
- For each `obs_mode ∈ {blind, informed, error, history}`, assert `state` group keys + dims match schema.
- Assert `schema_from_obs_groups` round-trips through `deploy/obs_builder.ObsBuilder` for the deploy-critical mode.
- Pattern: `deploy/test_policy_runner.py::test_schema_extractor_resolves_include_group`.

`tests/test_splitbelt_dr_compose.py` (lesson §7.4, only if new DR specs introduced)
- Mock-wrapper pattern from `tests/test_domain_rand_compose.py:_mock_wrapper`.
- Skip if env reuses Go2 DR specs verbatim.

`tests/test_splitbelt_metrics.py` — pure numpy
- Synthesize fake `splitbelt_traj.npz` with hand-crafted contact patterns (perfect symmetry; known asymmetry).
- Run offline analyzer.
- Assert `step_length_asymmetry ≈ 0` for symmetric input, nonzero in expected direction for asymmetric, recovery time on synthetic A1 trace matches expectation.

`tests/test_splitbelt_belt_assignment.py` — hermetic
- Mock minimal model + data with known foot xy positions.
- Assert `foot_belt_id` returns correct (0/1/-1) for positions in left-belt range, right-belt range, gap, off-belt.
- The §5 risk-callout test; kept hermetic so it runs every default-lane execution.

### 10.2 GPU/Warp (excluded from default lane)

`tests/test_splitbelt_env_smoke.py`
- `pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]`.
- Real env: `registry.load("Go2WarpSplitbelt")`, reset + 1 step, assert obs shape + no NaN.
- Verify schedule plumbing: set known `schedule_table`, step, assert belt joint `qvel` matches.
- Verify off-belt termination: shift robot in y, assert `term_cause = off-belt`.
- All env constructions inside fixtures.

`tests/test_splitbelt_control_metadata.py`
- `pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2, pytest.mark.deploy]`.
- Assert `env.get_control_metadata()` returns Kp/Kd/action_scale/dts/contact_mode/joint_order matching `deploy/go2_constants.py`.

`tests/test_splitbelt_bundle.py`
- `pytestmark = [pytest.mark.gpu]`.
- `make_env_bundle(cfg, seed=0)` returns populated bundle, `backend_kind == "mjx"`, expected dims.

### 10.3 Skipped from §7
- Off-policy loop smoke (§7.6) — defer until SAC/PPO preset exists. Use `tests/_loop_helpers.py` pattern when added.

### 10.4 Verification before any commit (lesson §8)
```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_docs_drift.py tests/test_docs_code_blocks.py
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_*.py
uv run python -m pytest --collect-only -q -m gpu | grep splitbelt
```

### 10.5 Calibration baseline + visual verification (post-implementation, not pytest)
- **Tied-belt smoke train:** vL=vR=0.5 m/s, FastSAC 5M steps. Reward should look like joystick env at similar speed (eval ~280-290 if we are matching `Go2WarpJoystickFlat` behavior). If wildly off, env is wrong.
- **Visual verification (hard rule, from bongo lesson):** `record_video.py` over a tied-then-split episode. Confirm robot does not slide off belt, stepping looks like stepping, belt slabs are actually moving, asymmetric gait visibly emerges or visibly fails. No metric-based claims until visual passes.

### 10.6 Reward-hack watchlist

| Possible exploit | Mitigation |
|---|---|
| Robot rides one belt only (perpendicular stance) | `foot_belt_id` distribution check; require both belts covered |
| Belt slabs not actually moving (actuator misconfigured) | Per-step belt qvel assertion in test, log mean belt vel |
| Robot teleports off treadmill | Hard contact-based off-belt termination, no thresholds |
| Policy locks joints, gets dragged | Drift penalty + episode timeout truncation; survival weight tuned not to dominate |
| Schedule_table not actually being read | Unit test: vary schedule across N envs, assert belt qvel differs |

### 10.7 Hard rules baked in (lesson)
- No `jax.random.PRNGKey(...)` at module scope. Fixtures or test bodies.
- No env construction at collection time.
- No hardcoded `/home/stevenman/...` paths. `pathlib.Path(__file__).parent` or `tmp_path`.
- Markers listed in full each time; no marker inheritance.
- `uv run python` always.

---

## 11. Open questions / future work

1. **Humanoid asset selection.** R3 says humanoid second; we have not picked the asset (mujoco_playground Berkeley Humanoid, Unitree H1/G1, MuJoCo Humanoid-v5). Defer until Go2 splitbelt is calibrated.
2. **Walking variant (`cmd_x > 0`).** Conflicts with `treadmill_drift` as currently shaped. If we want it later, we either drop drift or replace with "stay within bounds" (band rather than point). Out of scope here.
3. **Per-protocol algo presets.** `Go2WarpSplitbelt × {A1, A2, A3, A4} × {PPO, FastSAC, ...}` is the full benchmark grid. Land env first, then build presets.
4. **Q-bias diagnostics on splitbelt.** `evaluate` (MJX path) supports it; analysis of per-protocol Q-bias is interesting but out of scope until baseline lands.
5. **Composite "limp index"** — left to offline analyzer; not online to keep bug surface small. Spec'd at the analysis-script layer when first protocol report is written.
6. **Belt-friction DR.** Listed as a candidate DR spec; off by default. Candidate for inclusion once baseline shows the policy is robust to nominal belt friction.

### 11.X Known caveats / surfaced from audit

These are not bugs — they are design consequences worth pre-warning anyone running the benchmark.

- **A1 recovery vs reward gradient.** Within-episode adaptation requires the policy to *want* to recover quickly after the belt switch. The reward set (tracking + drift + smoothness) penalizes deviation magnitude per step but does not directly reward fast recovery. A policy that drifts gradually back to good gait and a policy that snaps back may both achieve similar episode-summed reward. The `recovery_time` metric will measure both, but training pressure to *minimize* recovery_time only emerges if the post-switch reward gradient is steep relative to per-step survival/tracking.
- **A2 difficulty asymmetry.** Higher (vL, vR) magnitudes mean larger belt drag → harder to maintain torso position → systematically lower per-step reward at high speeds. A2's `generalization_gap` metric will conflate "policy doesn't generalize" with "task is harder at high speeds." Mitigations: report eval scores per (vL, vR) cell normalized by a tied-baseline at the same v_avg; or filter speed range narrowly enough that the difficulty asymmetry is small.
- **Replay-buffer bloat from `info["splitbelt"]`.** Per-step splitbelt primitives (~50 floats × 4 feet positions = ~50 floats) flow into off-policy replay through `info`. At buffer 1M × num_envs 1024 that's ~200 MB extra memory for fields the actor/critic never read. Mitigation (recommended): gate `info["splitbelt"]` writes behind `cfg.log_splitbelt: bool` (default True for eval/recording, False for training). If the bloat doesn't show up empirically, leave alone.
- **Episode-length 1250 vs joystick 1000 changes raw return shape.** Per-step training stats (entropy, reward components) are directly comparable; raw eval returns are not. Always normalize by step count when comparing splitbelt evals to joystick evals.
- **Treadmill_drift magnitude in early training.** With `w_lat=2.0`, drift_y of 0.2 m gives -0.08/step, larger than `pose=0.5` or `feet_air_time=0.1` would contribute at near-default poses. Drift dominates the smoothness terms early. Confirm at the calibration checkpoint: if `pose` or `feet_air_time` reward components stay near zero throughout training, the drift weight is over-tuned.

---

## 12. Pointers

- Bongo precedent: `jax_rl/envs/locomotion/go2_bongo_handstand.py`, `.context/lessons/bongo.md`
- Joystick env (reward/obs pattern): `jax_rl/envs/locomotion/go2_warp_joystick.py`
- Env backend contract: `.context/lessons/env_backends.md`
- Testing contract: `.context/lessons/testing_new_envs.md`, `.context/lessons/algo_port_protocol.md` §9, §11
- DR wrapper: `jax_rl/envs/wrappers/domain_rand.py`
- Go2 base: `jax_rl/envs/locomotion/go2_warp_base.py`
- Go2 constants for control-metadata test: `deploy/go2_constants.py`
