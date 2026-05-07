# Go2 Deploy Audit: Local FastSAC vs Unitree RL Lab

Date: 2026-04-27

Scope: explain why a Unitree RL Lab Go2 baseline can work on the same hardware
interface while a locally trained FastSAC policy on `Go2WarpJoystickFlat`
"freaks out", and turn that comparison into an execution-ready audit.

Primary local paths:

- `jax_rl/envs/locomotion/go2_warp_joystick.py`
- `jax_rl/envs/locomotion/go2_warp_base.py`
- `jax_rl/configs/env_presets.py`
- `jax_rl/training/checkpointing.py`
- `jax_rl/utils/export.py`
- `deploy/obs_builder.py`
- `deploy/robot_interface.py`
- `deploy/deploy_go2.py`
- `deploy/sim2sim_direct.py`
- `deploy/go2_constants.py`

External reference, checked on 2026-04-27:

- Unitree RL Lab Go2 velocity env: https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py
- Unitree RL Lab PPO config: https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py
- Unitree RL Lab robot asset config: https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
- Unitree RL Lab Go2 deploy runner: https://github.com/unitreerobotics/unitree_rl_lab/blob/main/deploy/robots/go2/src/State_RLBase.cpp
- Unitree RL Lab Go2 FSM config: https://github.com/unitreerobotics/unitree_rl_lab/blob/main/deploy/robots/go2/config/config.yaml
- Unitree RL Lab deploy joint action processor: https://github.com/unitreerobotics/unitree_rl_lab/blob/main/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h

## Bottom Line

The two training/deploy stacks are not effectively the same. The highest-risk
differences are not "FastSAC vs PPO" in isolation. The likely failure mode is a
deploy-contract mismatch: observation layout/scaling, action scaling, default
joint offset, joint order, and PD gains are partly implicit or hardcoded in the
local stack. Unitree RL Lab carries more of those semantics through Isaac Lab
observation/action managers and generated deploy YAML, but the public Go2
deploy YAML itself is not present in the repo and must still be checked for the
specific exported policy.

The local ONNX file is only the neural network. It does not contain the full
robot contract needed to execute the policy safely:

- observation terms, order, dimensions, and scale factors;
- action postprocessing, including default pose and action scale;
- joint order and SDK remapping;
- Kp/Kd and control rate;
- expected command convention and last-action convention.

Until that contract is made explicit and verified against the real robot input
stream, retraining on a different algorithm or terrain curriculum can mask the
root cause without proving deployment equivalence.

## Contract Diff

| Area | Local FastSAC joystick stack | Unitree RL Lab stack | Risk |
| --- | --- | --- | --- |
| Simulator | MuJoCo Warp over vendored `unitree_mujoco` MJCF; `sim_dt=0.004`, `ctrl_dt=0.02`; training contact overrides set foot `condim=3`, friction `[0.6, 0.005, 0.0001]` in `Go2WarpEnv`. | Isaac Lab/PhysX manager-based env; `sim.dt=0.005`, `decimation=4`, also 50 Hz policy. | Physics/contact and asset differences can matter, but are lower priority than obs/action contract mismatch. |
| Training env | User trained `Go2WarpJoystickFlat` with FastSAC, not local curriculum. Commands are sampled over `a=[1.5, 0.8, 1.2]` with Bernoulli gates `b=[0.9, 0.25, 0.5]`. | Go2 velocity task has manager-based command curriculum plus startup/reset/interval events. Terrain curriculum machinery is enabled, but the current public Go2 config has only the flat sub-terrain active. | The local flat policy may be less robust, but a stable flat policy should not instantly thrash if the deploy contract is correct. |
| Policy observations | Local actor obs is 48d: `gyro`, `accelerometer`, `gravity`, `joint_pos_offset`, `joint_vel`, `last_act`, `command`. Values are raw except training noise. | Unitree policy obs is 45d: base angular velocity scaled by `0.2`, projected gravity, velocity command, relative joint position, joint velocity scaled by `0.05`, last action. No accelerometer in policy obs. | P0. Same sensors do not imply same vector. A 48d policy run through a 45d/Unitree-style builder, or unscaled policy fed scaled values, will fail immediately. |
| Action transform | Local env uses `motor_targets = default_pose + action * action_scale`; default `action_scale=0.5`. Local ONNX outputs raw tanh action only. | Unitree training uses `JointPositionActionCfg(... scale=0.25, use_default_offset=True ...)`. C++ deploy loads generated `params/deploy.yaml`; `JointAction` applies YAML scale/offset/clip, and `State_RLBase` writes `processed_actions()` to motor `q` targets. | P0. A 2x action-scale mismatch or double/missing default offset is enough to make a decent policy violent. |
| Default pose | Local deploy constants and the current vendored XML home keyframe both use `[0.0, 0.9, -1.8]` for every leg in policy order, but that match is not serialized or enforced. | Unitree training asset default pose is asymmetric; public Go2 FSM `FixStand` pose is symmetric `[0.0, 0.8, -1.5] * 4`. Training offset, exported deploy offset, and stand pose must be checked separately. | P0. Default pose participates in both obs offsets and action offsets. A mismatch shifts every joint target and every joint-position observation. |
| Low-level gains | Local training default is `Kp=20.0`, `Kd=0.5`; real deploy constants are also `20.0/0.5`, but sim constants still include legacy `35.0/0.1`. | Unitree Go2 asset uses stiffness `25.0`, damping `0.5` in sim; public FSM config has separate Passive/FixStand gains. `State_RLBase::run()` writes only `q` targets; gains come from controller/FSM setup. | P1/P0 depending on deploy path. If hardware interface uses constants that differ from training, policy behavior changes before RL algorithm matters. |
| Algorithm | Local FastSAC preset: 100M steps, 1024 envs, gamma `0.97`, actor hidden `(512,256,128)`, critic `(768,384,192)`, swish, large replay, target entropy scale `0.0`. | Unitree RSL-RL PPO: 4096 envs, `num_steps_per_env=24`, gamma `0.99`, lambda `0.95`, ELU actor/critic `(512,256,128)`, adaptive LR, entropy `0.01`. | P2 after deploy contract. PPO/curriculum explains robustness differences, not an immediate nonsensical hardware response if inputs/actions are correct. |
| Export/deploy | `jax_rl/utils/export.py` exports FastSAC actor weights and validates JAX vs ONNX numerics only. `deploy/sim2sim_direct.py` reads partial checkpoint metadata for gains/action scale/timing and obs schema, but still uses constants for default pose/remaps. `deploy/deploy_go2.py` and `robot_interface.py` use constants for stand pose, timing, gains, and action transform. | Unitree deploy loads `policy.onnx` from an exported policy directory and sends action-manager `processed_actions()` to motor commands. | P0. Local ONNX has no self-contained deployment semantics. Unitree deploy expects sidecar action/observation semantics to match the export. |

## Findings

### P0 - Observation Contract Is Not Fully Specified

Evidence:

- Local training env declares a 48d policy obs with accelerometer included in
  `WarpJoystick._obs_groups["state"]`.
- Local deploy `ObsBuilder` reconstructs obs by term names from
  `meta["obs_schema"]["state"]`, but only supports raw local terms and does not
  store/apply per-term scale factors.
- `ObsBuilder.from_checkpoint()` falls back to a default schema when
  `meta["obs_schema"]` is missing. That is useful for legacy sim experiments
  but too permissive for real hardware.
- Unitree RL Lab uses a different policy obs contract: scaled base angular
  velocity, projected gravity, command, relative joint position, scaled joint
  velocity, and last action. Policy observation corruption/noise is enabled in
  Unitree's config.

Why this can cause the observed failure:

If the hardware interface was built around Unitree RL Lab deploy assumptions,
it may feed a Unitree-style 45d vector, scaled values, or different term order
to a local 48d FastSAC ONNX. Even if the tensor shape is somehow adapted, the
policy would interpret slices as the wrong physical quantities.

Required fix:

1. Extend checkpoint/export metadata to include `obs_contract`, not just
   `obs_schema`:
   - term name;
   - width;
   - order;
   - scale applied at inference;
   - noise-only-at-train flag;
   - source frame and units;
   - expected quaternion convention for gravity.
2. Make real deploy fail closed if the runtime obs builder cannot exactly
   satisfy the checkpoint contract. Legacy fallback should be allowed only for
   explicit sim/dry-run modes.
3. Add a real-state dry-run command that prints each obs term separately:
   min/max, first values, and final concatenated shape.
4. Add tests for:
   - current 48d joystick schema;
   - `WarpJoystickNoAccel` 45d schema;
   - wrong/missing schema rejection;
   - scaled vs unscaled gyro/joint velocity detection.

Acceptance check:

- `deploy.validate_contract --mode compare-fixtures` emits per-term diffs.
  CI asserts exact shape/order and finite values. Hardware fixture comparison
  is pass/fail only when configured per-term tolerances are supplied; otherwise
  it is a print-only diagnostic. The check must print raw action and final
  joint targets, not just ONNX output.

### P0 - Action Scale and Default Offset Are Outside the ONNX Artifact

Evidence:

- Local env applies `default_pose + action * action_scale`.
- Local FastSAC ONNX exports only `action in [-1, 1]`; the file itself does not
  encode `action_scale`, default pose, or joint order.
- Unitree RL Lab config uses `scale=0.25` with `use_default_offset=True`.
- Unitree deploy `JointAction` applies YAML `scale`, optional `offset`, and
  optional `clip`; public training action clipping is wide, so the meaningful
  bound is scale/offset.
- Local `deploy/robot_interface.py` applies `DEFAULT_POSE_SDK + action * ACTION_SCALE`
  from hardcoded constants.

Why this can cause the observed failure:

If local ONNX is dropped into Unitree RL Lab deploy without matching action YAML,
the raw action can be postprocessed with the wrong scale/default offset. If the
local deploy path is used, `robot_interface.py` can still disagree with the
checkpoint because it does not consume checkpoint `meta["control"]`.

Required fix:

1. Export a sidecar `deploy_contract.json` next to every ONNX:
   - `schema_version`;
   - `artifact_kind`;
   - `checkpoint_git_hash`;
   - `action_type: joint_position_delta`;
   - `action_scale`;
   - `default_pose_policy`;
   - `default_pose_sdk`;
   - `policy_joint_names`;
   - `sdk_joint_names`;
   - `policy_joint_order`;
   - `sdk_joint_order`;
   - `policy_to_sdk`;
   - `sdk_to_policy`;
   - `last_action_semantics: raw_post_tanh_pre_scale`.
2. Make `Go2Interface` accept a loaded contract and remove hard dependency on
   module constants for action scale and default pose.
3. Add a deterministic test that checks the same raw action maps to the same
   SDK joint targets in:
   - training env formula;
   - `deploy/sim2sim_direct.py`;
   - `deploy/robot_interface.py`;
   - generated ONNX sidecar.

Acceptance check:

- For raw action `zeros(12)`, every deploy path sends exactly the checkpoint's
  default pose in SDK order.
- For a one-hot action, every deploy path changes exactly one expected joint by
  exactly `action_scale`.

### P0 - Default Pose Mismatch Is Large Enough To Break Standing

Evidence:

- Local constants and the current vendored XML home keyframe both use the
  symmetric pose `[0.0, 0.9, -1.8]` for all four legs.
- That current match is not enforced, serialized, or consumed from checkpoint
  metadata.
- Unitree RL Lab's training Go2 asset has an asymmetric default pose, including
  hip offsets and rear thigh values different from front thigh values.
- Unitree's public Go2 FSM stand pose is a separate symmetric pose
  `[0.0, 0.8, -1.5] * 4`.

Why this can cause the observed failure:

The default pose is used twice:

- `joint_pos_offset = q - default_pose` in observations;
- `q_target = default_pose + action_scale * action` in control.

A mismatch means the policy is told it is offset from one pose while the robot
is commanded around another pose. This is a direct path to large corrective
actions and unstable initial motion.

Required fix:

1. Save `default_pose_policy` and `default_pose_sdk` in checkpoint metadata from
   the actual env instance, not from deploy constants.
2. Make `ObsBuilder` subtract the checkpoint default pose, not
   `deploy/go2_constants.py::DEFAULT_POSE_POLICY`.
3. Make `robot_interface.py` send targets around the checkpoint default pose.
4. Add a migration warning for old checkpoints with no default-pose metadata.

Acceptance check:

- A checkpoint trained with the current XML records the same default pose as
  `env._mj_model.keyframe("home").qpos[7:]`.
- Real deploy prints the loaded default pose before arming.
- Unitree-compatible deploy checks training default offset, exported action
  offset, and stand-state pose as three separate values.

### P0 - Real Robot Deploy Does Not Consume `meta["control"]`

Evidence:

- `Go2WarpEnv.get_control_metadata()` now returns Kp, Kd, action scale, policy
  dt, physics dt, action repeat, contact mode, torque-speed model, and string
  labels for joint/action order. It does not yet include default pose, joint
  names, or remap arrays.
- `save_checkpoint()` stamps this under `meta["control"]` when the env can
  provide it.
- `save_checkpoint()` currently swallows env metadata collection failures, so a
  deployable Go2 checkpoint can silently lose `obs_schema` or `control`.
- `deploy/sim2sim_direct.py` reads a partial `meta["control"]` block for gains,
  action scale, and timing. It still uses constants for default pose/remaps and
  ignores contact-mode and torque-speed semantics.
- `deploy/deploy_go2.py` uses constants for policy timing and interpolation to
  stand.
- `deploy/robot_interface.py` still chooses gains and action scale from
  `deploy/go2_constants.py`.

Why this can cause the observed failure:

Sim2sim may be closer to the training control contract than the real DDS path,
but it is still partial. That creates a false sense of deploy readiness if the
real entrypoint uses constants or if sim2sim ignores metadata fields that affect
the policy.

Required fix:

1. Load checkpoint metadata or generated `deploy_contract.json` in the real
   deploy entrypoint.
2. Thread Kp, Kd, action scale, default pose, and joint mapping into
   `Go2Interface`.
3. Extend `get_control_metadata()` or the contract builder to include
   `default_pose_policy`, `default_pose_sdk`, `policy_joint_names`,
   `sdk_joint_names`, and remap arrays.
4. Make metadata stamping failures explicit. For deployable Go2 envs, fail or
   emit a high-severity warning if `obs_schema` or `control` cannot be written.
5. Make sim2sim consume or assert `contact_mode` and `torque_speed_model`; if
   unsupported, fail closed or print a high-severity warning.
6. Fail closed on missing control metadata unless `--allow-legacy-contract` is
   explicitly passed in sim/dry-run mode. Real arming must not use legacy
   fallback by default.

Acceptance check:

- Running real deploy logs the same Kp/Kd/action_scale/default pose as the
  training checkpoint and the ONNX sidecar.
- The real deploy entrypoint does not send any nonzero-gain motor command until
  static contract validation passes. DDS startup may be used in a read-only
  dry-run mode to capture state before arming.

### P1 - ONNX Validation Does Not Validate Deployment Semantics

Evidence:

- `validate_export()` compares JAX and ONNX neural-network outputs on random,
  zero, and gravity-observation batches.
- That validation does not test obs construction from robot state or action
  postprocessing into joint targets.
- `PolicyRunner` validates broad shared-actor artifact kind but only implements
  PPO/SAC/FastSAC network reconstruction. TD3/FastTD3/FlashSAC shared
  checkpoints should fail with a clear compatibility message before real deploy.

Why this matters:

The ONNX can be numerically correct and still unsafe on hardware because the
wrong vector was fed into it or the output was postprocessed incorrectly.

Required fix:

1. Add an end-to-end deploy validation command:
   - load checkpoint;
   - validate `artifact_kind` and supported algorithm;
   - build obs from a recorded robot/sim state;
   - run PolicyRunner and ONNX;
   - map raw action to SDK q targets using the deploy contract;
   - print and optionally assert target deltas.
2. Store a small golden standing-state fixture for regression tests.

Acceptance check:

- JAX actor, ONNX actor, sim2sim action postprocess, and real deploy action
  postprocess agree on the same recorded input.

### P1 - Training Distribution Is Weaker Than Unitree RL Lab Baseline

Evidence:

- User trained FastSAC on the flat joystick env, not the local curriculum env.
- Unitree RL Lab Go2 config includes command curriculum, startup/randomization
  events, reset joint velocity noise, friction/mass randomization, and interval
  pushes. Terrain curriculum machinery is present, but the current public Go2
  terrain generator has only the flat sub-terrain active.
- Local flat env has pushes and domain-randomization specs, but the exact
  wrapper/application path must be verified per run from checkpoint metadata.

Why this matters:

After obs/action/default-pose parity is proven, the next likely reason Unitree
works better is broader training distribution. A flat joystick policy can be
valid in sim but brittle to real contact, latency, motor response, and initial
pose errors.

Required fix:

1. Verify the exact checkpoint's `meta.json`:
   - env name;
   - obs schema;
   - control metadata;
   - DR specs;
   - reset mode;
   - algorithm config.
2. Train a parity-targeted ablation before a broad curriculum run:
   - an explicit env/config target for Unitree-style 45d observation
     (`no accelerometer`, gyro scale `0.2`, joint velocity scale `0.05`);
   - action scale `0.25`;
   - default pose matching the hardware/deploy stack;
   - Kp/Kd matching the hardware interface;
   - same command convention used at deploy.
3. Then compare flat vs curriculum under the same deploy contract.

Acceptance check:

- A standing/slow-walk command remains stable in direct sim2sim and real-state
  dry-run before dynamic hardware tests.

### P1 - Local vs Unitree Joint Order Must Be Treated As A Contract, Not A Comment

Evidence:

- Local policy order is `FL, FR, RL, RR`; Unitree SDK order is `FR, FL, RR, RL`.
- `deploy/go2_constants.py` uses a symmetric remap array for both directions.
- Unitree asset declares SDK joint names in `FR, FL, RR, RL` order.
- Current local `meta["control"]` stores only string labels for joint/action
  order, not joint names or index arrays.

Why this matters:

Joint-order bugs often look exactly like "the policy freaked out": hips and
calves move on the wrong legs, while the neural net output range looks normal.

Required fix:

1. Put joint names in every contract, not just index arrays.
2. Add an action one-hot hardware-disabled test that prints target joint names
   and values before enabling motors.
3. Validate SDK state order from live `LowState_` against expected joint names
   where the SDK exposes names, or against controlled one-joint motion in a
   safe zero-torque/test stand mode.

Acceptance check:

- One-hot policy action on `FL_thigh` maps to exactly the `FL_thigh` SDK command
  index, and the obs offset for manually moved `FL_thigh` lands in the same
  policy index.

## Execution Plan For Other Agents

P0 phases fix deploy-contract correctness: obs construction, action
postprocess, default pose, gains, joint order, artifact compatibility, and
safety gates. P1/P2 phases study training distribution only after the same
contract validator passes.

### Phase 0 - Capture The Exact Failing Contract

Owner goal: no code changes until the current failing policy's contract is
known. This is a hard gate: do not change deploy behavior, retrain, or run
hardware experiments until the failing deploy path is identified.

Tasks:

1. Locate the failing checkpoint directory and exported ONNX.
2. Dump `meta.json` fields:
   - `algo`;
   - `obs_dim`;
   - `action_dim`;
   - `train_config.env_name`;
   - `train_config.n_frame_stack`;
   - `obs_schema`;
   - `control`;
   - `dr_specs`.
3. Record which deploy path was used:
   - local Python `deploy/robot_interface.py`;
   - local direct sim2sim;
   - Unitree RL Lab C++ deploy;
   - custom lab infrastructure.
4. Save one standing robot state sample and one direct-sim standing state sample
   as fixtures. Fixture format should be `.npz` with:
   - `joint_pos_sdk`, shape `(12,)`, radians, SDK order `FR, FL, RR, RL`;
   - `joint_vel_sdk`, shape `(12,)`, rad/s, SDK order;
   - `gyroscope`, shape `(3,)`, rad/s, body frame;
   - `accelerometer`, shape `(3,)`, m/s^2, body frame;
   - `quaternion`, shape `(4,)`, `[w, x, y, z]`;
   - `command`, shape `(3,)`, `[vx, vy, yaw_rate]`;
   - optional raw DDS/log payload for traceability.

Deliverable:

- `deploy_contract_observed.md` or JSON dump showing the actual runtime
  contract used by the failing policy.

### Phase 1 - Make The Contract Explicit

Owner goal: create a single source of truth consumed by ONNX deploy and Python
deploy.

Tasks:

1. Add `deploy/contract.py` with:
   - `load_checkpoint_contract`;
   - `write_deploy_contract`;
   - `validate_deploy_contract`.
2. Add a contract builder from checkpoint metadata:
   - input: checkpoint dir plus ONNX path;
   - output: `deploy_contract.json`.
3. Required schema:
   - `schema_version`;
   - `artifact_kind`;
   - `algo`;
   - `onnx_input_name` and `onnx_output_name`;
   - `obs_terms[]` with name, dim, scale, units, frame, and source;
   - `action` with type, dim, scale, default offset, clip, and last-action
     semantics;
   - `control` with Kp, Kd, policy dt, motor command dt, sim dt when relevant,
     contact mode, and torque-speed mode;
   - `joint_order` with policy names, SDK names, `policy_to_sdk`, and
     `sdk_to_policy`;
   - `default_pose_policy` and `default_pose_sdk`;
   - `safety_limits` with max target delta, first-run action clamp, and command
     ramp settings.
4. Update `ObsBuilder`, `robot_interface.py`, `deploy_go2.py`, and sim2sim
   entrypoints to load the contract.
5. Keep `deploy/go2_constants.py` only as a legacy fallback with explicit
   warning.

Deliverable:

- Contract file generated for every export.
- Real deploy refuses to arm if contract and checkpoint are missing or
  inconsistent. Legacy fallback may be allowed for sim/dry-run only.

### Phase 2 - End-To-End Parity Tests

Owner goal: prove the same robot state produces the same obs, raw action, and
q targets in every path.

Tests to add:

1. Obs schema test:
   - local 48d joystick schema;
   - optional Unitree-style 45d schema;
   - old checkpoint fallback warning.
2. Action mapping test:
   - zero action -> default pose;
   - one-hot action -> correct named joint and delta.
3. ONNX end-to-end unit test with a synthetic fixture and mocked interface:
   - `PolicyRunner` action equals ONNX action;
   - max absolute JAX-vs-ONNX diff is <= `1e-5`;
   - postprocessed q targets match contract within `1e-6`.
4. Recorded standing-state test:
   - builds obs from fixture;
   - prints per-term stats;
   - asserts finite obs/action;
   - asserts target deltas are below contract `safety_limits.max_target_delta`
     when that limit is configured.
5. Manual hardware-disabled DDS dry-run:
   - connects and reads state;
   - builds obs;
   - computes raw action and q targets;
   - prints target names and deltas;
   - does not send nonzero-gain motor commands.

Suggested CI command:

`uv run python -m pytest tests/test_deploy_contract.py deploy/test_policy_runner.py -q`

Deliverable:

- A single command that a hardware operator can run before arming:
  `python -m deploy.validate_contract --checkpoint ... --onnx ... --state-fixture ...`

### Phase 3 - Hardware Guardrails

Owner goal: prevent another unsafe first run.

Tasks:

1. Add an arming checklist that prints:
   - checkpoint git hash;
   - obs schema and raw dim;
   - action scale;
   - default pose;
   - Kp/Kd;
   - command;
   - max target delta from current q.
2. Add first-run clamps:
   - initial raw action clamp `abs(action) <= 0.25` unless explicitly
     overridden;
   - initial target delta clamp `abs(q_target - q_current) <= 0.25 rad` unless
     explicitly overridden;
   - command ramp from zero to requested command over at least 5 seconds;
   - zero-torque/stand fallback key.
3. Log first 10 seconds of:
   - per-term obs;
   - raw actions;
   - q targets;
   - measured q/dq;
   - motor command Kp/Kd.

Deliverable:

- A deploy dry-run log that can be compared against direct sim before enabling
  dynamic commands.

### Phase 4 - Training Parity Ablations

Owner goal: isolate deploy-contract fixes from learning-distribution fixes.

Do not start ablation training until Phases 0-3 pass for the current checkpoint.
Training-distribution work is P1 and must not be used to explain hardware
instability until deploy-contract parity is proven.

Run these in order:

1. Current FastSAC joystick checkpoint with fixed deploy contract.
2. An explicitly named local env/config variant for Unitree-style 45d obs,
   action scale `0.25`, hardware/export default pose, and hardware Kp/Kd.
3. Same as 2, but local curriculum.
4. PPO baseline using the local env with Unitree-like PPO hyperparameters.

Do not compare hardware behavior across these until each passes the same
contract validation and Phase 3 guardrails.

Deliverable:

- Table with sim return, direct-sim stability, contract-validation status, and
  hardware dry-run status.

## Most Likely Root Causes Ranked

1. Observation mismatch: 48d local raw obs with accelerometer vs 45d Unitree
   scaled obs without accelerometer, plus possible term-order and scaling drift.
2. Action postprocess mismatch: local `0.5` scale and local default pose vs
   Unitree `0.25` scale/default offset, or missing/double-applied offset.
3. Default pose mismatch across training offset, exported deploy offset, stand
   pose, and local constants. Current local XML/constants match today, but that
   match is not enforced by metadata.
4. Metadata split: sim2sim reads only partial checkpoint `control`, while real
   DDS deploy still uses constants.
5. Joint-order mismatch in lab infrastructure: policy order vs SDK order.
6. Training distribution: FastSAC flat joystick lacks Unitree's command
   curriculum and event randomization.
7. Simulator/actuator mismatch: MuJoCo Warp external PD vs Isaac Lab/PhysX
   actuator model and Unitree stiffness/damping.

## What Not To Do First

- Do not retrain a large curriculum policy before proving obs/action/default-pose
  parity.
- Do not judge ONNX safety from JAX-vs-ONNX numerical validation alone.
- Do not use Unitree RL Lab deploy YAML/action-manager defaults for a local ONNX
  unless they exactly match the local checkpoint contract.
- Do not keep default pose and action scale duplicated between env code,
  constants, sim2sim, and real DDS deploy.

## Minimal Handoff Checklist

Before another hardware attempt, require all checks below:

- Checkpoint `meta.json` has `obs_schema` and `control`; new exports or
  `deploy_contract.json` also have default-pose metadata. Old checkpoints fail
  closed for real deploy unless explicitly migrated.
- ONNX has a sidecar `deploy_contract.json`.
- Real deploy loads that contract and prints it.
- Standing-state fixture passes obs/action/q-target parity.
- Zero action maps to checkpoint default pose in SDK order.
- One-hot action maps to the intended named joint.
- Max initial target delta from current measured q is bounded and printed
  (default limit `0.25 rad` unless explicitly overridden).
- First run uses command ramp and action clamp (default raw action clamp `0.25`
  and at least 5 second command ramp).
- Logs capture obs terms, raw actions, q targets, measured q/dq, Kp/Kd.

## Notes For Refiner Agents

Please verify this file against current repo code and Unitree RL Lab sources.
Focus especially on:

- whether every P0 finding is grounded in code;
- whether proposed fixes are precise enough to execute;
- whether any current repo change already closes a listed gap;
- whether this audit incorrectly assumes the failing deploy path;
- whether the Unitree comparison overstates details not present in public config.
