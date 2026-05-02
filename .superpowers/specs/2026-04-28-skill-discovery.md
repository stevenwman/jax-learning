# Skill Discovery V2 Audit and Design

Date: 2026-04-28

This supersedes the implementation details in `.context/plans/skill-discovery-plan.md`.
The old plan remains useful for intent and algorithm sketches, but the repo has
changed enough that a fresh implementation plan is safer than another drift note.

## Bottom line

The concept is still right: use a `SkillManager` to sample skills, train
DIAYN/METRA auxiliary networks, feed `z` to the actor, and replace or augment
task reward with intrinsic skill reward.

The old implementation shape is stale. V2 must be built around:

- `make_env_bundle(cfg, seed) -> EnvBundle`, not the old `make_envs` 7-tuple.
- The current shared off-policy stack: `scripts/train_fast_sac.py` ->
  `jax_rl/training/offpolicy_loop.py::run_offpolicy_loop`.
- `ObsPipeline` sample-time normalization and frame-stack reconstruction.
- Go2 Warp envs registered in `jax_rl/training/env_backends/mjx_backend.py`.
- Schema-stamped checkpoints with `meta["obs_schema"]` and `meta["control"]`.
- A deploy contract for `skill_z`; current deploy/export code has no concept of
  skill-conditioned policies.

V2 should start with FastSAC + DIAYN on deployable Go2 observations, then add
D3-style factorization, METRA, style rewards, symmetry, and hardware skill
selection only after the contract is explicit.

## Existing docs status

| File | Status | Notes |
|---|---|---|
| `.context/plans/skill-discovery-plan.md` | Superseded for implementation | Still useful for DIAYN/METRA class sketches and `SkillManager` intent. Stale for paths, env setup, obs normalization, Go2 obs indices, and checkpoint/deploy expectations. |
| `.context/references/d3_skill_discovery.md` | Conceptual/current | D3 factorization, style, symmetry, and factor weighting remain the correct north star. Its claim that `EncoderConfig.context_dim` is available should be read as "reserved, not wired." |
| `.context/TODO.md` Phase 6 skill discovery | Partially current | The checklist still points in the right direction, but "skill vector in obs is one line" is too optimistic for off-policy, frame stack, checkpointing, export, and deploy. |
| `.context/references/mjlab_audit.md` | Conceptual/current | RewardSpec/ObsSpec recommendations have been implemented. Its "manager principle, not full framework" guidance still applies. |
| `.superpowers/plans/archive/2026-04-02-obs-spec.md` | Historical | The refactor landed. Do not follow the old task list as an implementation plan. |
| `.superpowers/plans/archive/2026-04-02-reward-spec.md` | Historical | The refactor landed. DIAYN reward swap still needs training-loop aux state and checkpointing. |
| `.context/go2/sim_to_real_plan.md` | Stale in details | MJX-era assumptions remain. Current path is Unitree XML via Go2 Warp envs, checkpoint-stamped control metadata, and stricter deploy obs schema. |
| `.context/plans/go2-sim2sim-sim2real-plan.md` | Historical/stale for implementation | Useful for deploy intent and joint-remap history only. Do not follow old 48d obs, constants-first deploy, or pre-`obs_schema`/`control` assumptions for skill deployment. |

## Current repo truth

### Training stack

- SAC and FastSAC scripts are thin wrappers around `run_offpolicy_loop`.
- FlashSAC remains standalone because BatchNorm, Zeta noise, and adaptive reward
  normalization do not fit the shared loop yet.
- `EnvBundle` is the canonical env boundary. It carries backend kind, actual
  `num_envs`, obs/action dims, dict-obs state, privileged critic dims, and eval env.
- The off-policy loop has no skill hook today. It:
  - extracts actor obs with `ObsPipeline`,
  - optionally normalizes sample-time obs,
  - stores `env_state.reward * cfg.reward_scaling`,
  - samples a batch,
  - normalizes the batch,
  - calls `algo.update(training_state, batch)`.
- `JaxReplayBuffer` can already store named extra obs fields, but
  `ObsPipeline.make_buffer(...)` only exposes this for privileged critic obs
  today. V2 should extend `ObsPipeline.make_buffer(...)` with generic
  `extra_obs_dims` and merge those with the existing critic extras. Store raw
  env reward in the existing `reward` slot until sample-time replacement.
- PPO/on-policy has `reward_augment_fn` and `extra_rollout_fn` patterns, but no
  mutable per-env `z` lifecycle. It is useful later, not the first V2 path.

### Env stack

- "Go2 Warp" is still a MuJoCo Playground `MjxEnv` using `impl="warp"`, not an
  independent Warp environment.
- Current default Go2 joystick actor obs is 48d:
  `gyro`, `accelerometer`, `gravity`, `joint_pos_offset`, `joint_vel`,
  `last_act`, `command`.
- `Go2WarpJoystickFlatNoAccel` and `Go2WarpJoystickUnitree` drop
  `accelerometer`, so their actor obs is 45d before any skill concat.
- Current Go2 joystick actor obs does not include base xy or base height.
  The old 51d plan with `[48:50] base_xy` and `[50] base_z` is stale.
- Go2 privileged critic obs is 122d for Flat and 119d for NoAccel/Unitree.
  It includes actor state plus clean gyro, accelerometer, gravity, local
  linear velocity, global angular velocity, clean joints, actuator force,
  contacts, feet data, and xfrc.
- Go2 action mapping is explicit:
  `motor_targets = default_pose + action * action_scale`.
  Default joystick action scale is `0.5`; the Unitree-style variant sets `0.25`.
- `RewardSpec` and `ObsSpec` are implemented and used by Go2.
- RewardSpec is env-owned and Go2 clips total reward after applying weights.
  Negative or unbounded intrinsic objectives should be computed in the training
  loop, not hidden inside the current Go2 reward clipping path.
- Wrapper order matters:
  action delay first, frame stack second, then training/reset wrappers.
  If `skill_z` is added inside env obs before frame stacking, it gets duplicated
  per frame.

### Network stack

- `EncoderConfig.context_dim` and `context_fusion` are reserved fields only.
- `MlpEncoder.__call__(obs, context=None)` can concatenate context, but current
  builders call `encoder(obs)` and no algo or deploy path passes context.
- V2 should use explicit obs concatenation for the first implementation.
  Structured context fusion can be a later refactor after training, checkpoint,
  export, and deploy all know about it.

### Checkpoint and deploy stack

- Go2 checkpoint save now stamps `obs_schema` and `control` metadata.
- `meta["control"]` includes Kp, Kd, action scale, policy dt, physics dt,
  joint names, default poses, remap arrays, contact mode, and torque-speed flag.
- Real `deploy_go2.py` refuses missing control/schema in non-sim mode.
- `ObsBuilder` is strict about known obs terms. It supports hardware terms only;
  there is no `skill_z` fetcher or skill selector today.
- `PolicyRunner.get_action()` takes only an obs vector. It has no skill input.
- ONNX export is FastSAC-only and emits an actor graph, not a deploy sidecar
  with obs/control/skill contract semantics.
- `sim2sim_direct.py` still duplicates some deploy action postprocessing. It
  reads partial control metadata but still uses deploy constants for default
  pose and remaps. Fix this before trusting sim2sim as a skill deploy gate.

## Design corrections from the old plan

### 1. Do not use fixed Go2 obs indices as the source of truth

Old plan:

```text
[48:50] base_xy
[50] base_z
```

Current repo:

```text
Go2WarpJoystickFlat actor obs is 48d; command is [45:48].
Go2WarpJoystickFlatNoAccel and Go2WarpJoystickUnitree actor obs are 45d;
command is [42:45].
```

Do not use either slice as the source of truth. V2 should define factors by
named extractors, not magic indices. The actor can stay deployable while the
auxiliary reward sees sim-only factors.

Example:

```python
FactorConfig(
    name="base_xy_delta",
    method="metra",
    skill_dim=2,
    source="sim_data",
    extractor="base_xy",
)
```

The policy does not need absolute base xy in its deploy obs for the auxiliary
reward to use base xy during training.

### 2. Do not rely on `EncoderConfig.context_dim` yet

The field exists but is not wired through `Actor`, `DeterministicActor`, algos,
checkpoint metadata, numpy deploy, or ONNX export.

V2 default:

```text
policy_input = concat(policy_obs_after_normalization, skill_z)
```

Later:

```text
policy_input = actor(obs, context=skill_z)
```

Only switch to the later form after builders, algos, export, deploy, and
checkpoint metadata support it.

### 3. Do not store `[normalized_obs, z]` as the replay buffer primitive

The old plan stores normalized augmented obs directly. That fights the current
off-policy stack, where `ObsPipeline` normalizes at sample time and frame stack
is reconstructed from raw frames.

V2 should store:

```text
obs          raw actor obs, as today
next_obs     raw actor next obs, as today
critic_obs   raw privileged critic obs, when available
skill_z      skill used for this transition
next_skill_z skill bound to the bootstrap state (see lifecycle rule below)
factor_obs   optional factor inputs needed for intrinsic reward recompute
next_factor_obs
reward       raw unscaled env reward until sample-time replacement
```

Skill lifecycle rule (SD-B default, `resample="episode"`):

```text
At collection time, for every transition (s_t, a_t, r_t, s_{t+1}, done_t):
    skill_z      = z_active for env i at step t
    next_skill_z = skill_z   (same value, always)

Resampling happens *between* transitions, not inside one. When done_t=1,
z_active for env i is resampled before the next collection step. The new z
becomes the skill_z of the *next* stored transition, not the next_skill_z of
the terminal one.

Why this works:
  - On done=1, SAC bootstrap multiplies Q(next_obs, next_skill_z) by (1-done)
    so next_skill_z is dead weight and never trains.
  - On done=0 within an episode, skill_z is constant by construction, so
    next_skill_z == skill_z is exactly correct.
  - Avoids cross-episode skill leakage into terminal Q targets.

For `resample="fixed_steps"` (deferred to SD-E), this rule no longer holds:
the resampling boundary can fall inside an episode, so next_skill_z must
reflect the post-boundary skill. Replay must also store the resample step
counter to recover skill history under frame stacking.
```

Then, after `ObsPipeline.normalize_batch(...)`, compose:

```python
task_reward = batch["reward"]
batch["obs"] = concat(batch["obs"], batch["skill_z"])
batch["next_obs"] = concat(batch["next_obs"], batch["next_skill_z"])
batch["critic_obs"] = concat(batch["critic_obs"], batch["skill_z"])
batch["critic_next_obs"] = concat(batch["critic_next_obs"], batch["next_skill_z"])
intrinsic_reward = skill_manager.compute_intrinsic_reward(aux_state, batch)
style_reward = skill_manager.compute_style_reward(batch)
safety_penalty = skill_manager.compute_safety_penalty(batch)
batch["reward"] = cfg.reward_scaling * (
    skill_cfg.intrinsic_weight * intrinsic_reward
    + skill_cfg.task_reward_weight * task_reward
    + skill_cfg.style_reward_weight * style_reward
    - skill_cfg.safety_penalty_weight * safety_penalty
)
```

The algo must be constructed with the augmented dimensions:

```text
actor_obs_dim = raw_actor_obs_dim + skill_dim
critic_obs_dim = raw_critic_obs_dim + skill_dim
```

This keeps sample-time normalization correct and avoids stale intrinsic rewards.
`ObsPipeline` and normalization state stay sized to the raw actor/critic obs;
only algo/network input dims and checkpoint policy contract use
`raw_dim + skill_dim`.

### 4. Compute off-policy intrinsic reward at sample time by default

Collection-time reward is simpler but stale: DIAYN/METRA auxiliary networks
change after the transition enters replay.

V2 default should recompute intrinsic reward on sampled batches. Replay must
store every factor input needed for that recompute; do not assume sim-only
factors can be recovered from deploy actor obs. Keep collection-time reward as
an ablation if compute cost becomes the bottleneck.

### 5. Treat deploy skill metadata as part of V2, not a hardware afterthought

A deployable skill policy needs the checkpoint/export contract to say:

- skill discovery is enabled,
- skill dimension,
- skill prior type,
- factor layout,
- resampling cadence,
- fixed/default deploy skill,
- how the operator selects a skill,
- whether skill is appended after frame stack or passed as context.

Without this, a trained skill policy can produce the right network shape but
the wrong hardware obs.

## V2 implementation shape

Use new phase names (`SD-A`, `SD-B`, ...) to avoid overloading existing "Phase 6"
references elsewhere in `TODO.md`.

### SD-A: Contract and scaffolding

Files to add:

```text
jax_rl/skill_discovery/__init__.py
jax_rl/skill_discovery/config.py
jax_rl/skill_discovery/factors.py
jax_rl/skill_discovery/prior.py
jax_rl/skill_discovery/diayn.py
jax_rl/skill_discovery/manager.py
tests/test_skill_discovery_config.py
tests/test_skill_discovery_factors.py
tests/test_skill_discovery_diayn.py
tests/test_skill_discovery_manager.py
tests/test_skill_discovery_prior.py
```

Core config should be explicit and serializable:

```python
@dataclass
class SkillDiscoveryConfig:
    enabled: bool = True
    mode: Literal["diayn", "metra", "factorized"] = "diayn"
    total_skill_dim: int = 0
    prior: Literal["one_hot", "dirichlet", "hypersphere"] = "one_hot"
    resample: Literal["episode", "fixed_steps"] = "episode"
    resample_steps: int | None = None
    reward_mode: Literal["sample_time", "collection_time"] = "sample_time"
    intrinsic_weight: float = 1.0
    task_reward_weight: float = 0.0
    style_reward_weight: float = 0.0
    safety_penalty_weight: float = 0.0
    factors: tuple[FactorConfig, ...] = ()
    deploy: SkillDeployConfig = field(default_factory=SkillDeployConfig)

@dataclass
class FactorConfig:
    name: str
    method: Literal["diayn", "metra"]
    skill_dim: int
    source: Literal["actor_obs", "critic_obs", "sim_data", "info"]
    extractor: str
    dim: int

@dataclass
class AuxNetConfig:
    """Architecture defaults for DIAYN discriminator and METRA phi.

    SD-A ships these as plain MLPs. SD-C/E may upgrade DIAYN discriminator
    to SimBa-style residual MLP (D3 reference uses SimBa[256,256] elu).
    """
    discriminator_hidden: tuple[int, ...] = (256, 256)
    discriminator_activation: str = "relu"        # D3 uses elu; pick one when wiring SimBa
    discriminator_lr: float = 3e-4                # DIAYN reference launcher (mujoco_all_diayn.py:26)
    phi_hidden: tuple[int, ...] = (256, 256)      # D3 reference (rsl_rl_usd_cfg.py:82-86)
    phi_activation: str = "elu"                   # D3 default
    phi_lr: float = 1e-4                          # METRA + D3 reference
    dual_lam_init: float = 30.0                   # METRA paper + D3
    dual_lam_lr: float = 5e-4                     # D3 lr_tau
    dual_slack: float = 1e-3                      # METRA reference (tests/main.py:132)
    dual_dist: Literal["one", "l2", "s2_from_s"] = "one"  # METRA default — constraint is ||Δφ||²≤1
```

Use `one_hot` for SD-A through SD-C. Move Dirichlet and hypersphere priors to
SD-E after the basic skill loop, checkpoint contract, and fixed-skill sim deploy
work.

Checkpoint metadata block:

```json
{
  "skill_discovery": {
    "schema_version": 1,
    "enabled": true,
    "mode": "diayn",
    "prior": "one_hot",
    "total_skill_dim": 8,
    "obs_injection": "concat_after_obs_pipeline",
    "resample": "episode",
    "hardware_ready": false,
    "factors": [],
    "deploy": {
      "skill_input_mode": "fixed",
      "default_skill": [1, 0, 0, 0, 0, 0, 0, 0]
    }
  }
}
```

Acceptance:

- Config round-trips through JSON.
- Skill priors sample expected shapes.
- `SkillManager.augment_actor_obs(obs, z)` is pure and shape-checked.
- `SkillManager.compute_intrinsic_reward(...)` is deterministic for a fixed aux
  state and batch.

### SD-B: FastSAC DIAYN training loop

Create a dedicated loop first instead of generalizing `run_offpolicy_loop`
prematurely:

```text
scripts/train_skill_discovery.py
jax_rl/training/skill_offpolicy_loop.py
```

It should reuse:

- `make_env_bundle`
- `ObsPipeline`
- `JaxReplayBuffer`
- `CheckpointManager`
- `metrics_logger`
- existing FastSAC construction from `scripts/train_fast_sac.py`

It should add:

- per-env `skill_z` lifecycle,
- replay storage for `skill_z` and `next_skill_z`,
- replay storage for `factor_obs` and `next_factor_obs` when intrinsic reward
  cannot be recomputed from actor/critic obs alone,
- sample-time reward replacement,
- aux network update after or before actor/critic update,
- skill metrics logging,
- checkpoint save/load for aux state and skill metadata.

Buffer work needed:

```text
Decision: extend `ObsPipeline.make_buffer(...)` with
`extra_obs_dims: dict[str, int] | None`, merge it with existing privileged
critic extras, and use `JaxReplayBuffer`'s existing named-extra support.
```

Do not hide `skill_z` inside the raw obs buffer; it must stay separable until
after normalization and frame-stack handling.

Reward contract:

```text
Replay stores raw unscaled env reward in the existing `reward` field.
SD-B simple-env smoke defaults to pure DIAYN: task_reward_weight=0.
SD-C Go2 defaults to a nonzero task/style/safety mix.
Apply cfg.reward_scaling exactly once to the final composed batch reward
immediately before algo.update(...).
```

Gradient-step order:

```text
1. Sample replay batch.
2. Normalize raw obs through ObsPipeline.
3. Append skill_z to actor and critic obs.
4. Compute intrinsic reward using the current aux state.
5. Compose final batch reward with task/style/safety terms.
6. Update actor/critic with that fixed reward.
7. Update aux params and aux optimizer state on the same batch.
```

Actor/critic gradients must not flow through aux networks.

Checkpoint decision:

```text
Add `jax_rl/skill_discovery/checkpointing.py` as a wrapper around the base
checkpoint writer. It should call the existing checkpoint path for actor/norm
state, then write `skill_aux/` through Orbax for discriminator/METRA params and
optimizer state, and patch `meta.json` with `skill_discovery`. Resume must load
both base training state and skill aux state.
```

Initial algorithm scope:

```text
algo: FastSAC only
env smoke: CheetahRun or WalkerWalk
Go2 target: Go2WarpJoystickUnitree
obs_norm: supported only if z is composed after ObsPipeline normalization
frame_stack: initially disallow or explicitly compose z after stack
```

For SD-B, allow only `resample="episode"` when `n_frame_stack > 1`. Fixed-step
resampling with frame stack needs skill history in replay and deploy, so defer it
until after the episode-skill path works.

Do not start with FlashSAC. Its standalone loop and BatchNorm path make it a
bad first target for skill discovery plumbing.

Acceptance (SD-B):

Plumbing gates:

- 10k-step CPU/GPU smoke on a simple env produces finite aux losses.
- Existing asymmetric-critic buffer tests still pass after generic extras land.
- Skill tests store and sample `skill_z`, `next_skill_z`, `factor_obs`, and
  `next_factor_obs` alongside critic extras.
- Skill checkpoint resume restores aux params and optimizer state.
- Existing FastSAC tests pass unchanged.
- No changes to existing `train_fast_sac.py` behavior.

Behavioral gates (paper-grounded — see `.context/references/skill_discovery_validation.md` Part 4 SD-B):

- Discriminator accuracy curve rises above chance (`1/num_skills`) within first
  100k steps on a simple continuous-control env (CheetahRun or WalkerWalk).
- At 1M steps: per-skill task-return histogram (M=10 episodes per fixed z).
  Spread (max − min across skills) > 50% of any single skill's return. Replicates
  DIAYN App. D.4 protocol.
- At 1M steps: render 1 video per skill. Skills should show qualitatively
  diverse gaits (subjective check, replicates DIAYN HalfCheetah figure intent).
- 3 seeds minimum, mean ± std reporting. Upgrade to 5 seeds if results
  contentious.

Open implementation questions — RESOLVED via source audits 2026-05-02 (see `.context/references/skill_discovery_source_extracts.md`):

- DIAYN discriminator MLP: `[300, 300]` ReLU plain in DIAYN reference;
  `SimBa[256, 256]` ELU in D3. Spec ships `[256, 256]` plain as `AuxNetConfig`
  default; SD-C/E may upgrade to SimBa.
- DIAYN reward feeds **current state s**, not next state `s'` (confirmed
  `diayn.py:175-180`). Confusable with DADS `q(s'|s,z)` — different paper.
- DIAYN Adam lr = `3e-4` (launcher overrides class default 3e-3).
- DIAYN num_skills reference default = 50; SD-A unit tests use 4.
- **Skip DIAYN's GMM K=4 policy** — DIAYN's TF1 reference uses a Gaussian
  mixture policy from the legacy SAC era. Modern SAC ships squashed Gaussian.
  Our SD-B uses FastSAC with standard squashed Gaussian — do NOT replicate
  DIAYN's GMM. (D3 itself is PPO with a `[512, 256, 256]` MLP actor + diagonal
  Gaussian, `init_noise_std=1.0`, log_std_range `(-5, 2)` — not relevant to our
  SAC choice but confirms GMM is not load-bearing.)

### SD-C: Go2 DIAYN with deployable obs

Start with deployable actor obs:

```text
env: Go2WarpJoystickUnitree
actor obs: current Unitree-style hardware terms (45d, no accelerometer) + skill_z
critic obs: privileged_state + skill_z
factor reward inputs: named sim-data/factor extractors, not actor obs indices
action scale: checkpoint-stamped, expected 0.25 for Unitree variant
```

Recommended first DIAYN factor:

```text
factor: command-conditioned behavior class or base velocity response
source: sim data / local velocity / command tracking diagnostics
method: DIAYN
reason: easier to validate than unbounded base position METRA
```

Do not begin with D3 full factorization. First prove that skill-conditioned
FastSAC can produce distinct stable Go2 behaviors without breaking deployment
contracts.

Acceptance (SD-C — paper-grounded, replicates D3 Table 2 + Fig 5 protocol on a single DIAYN factor; see validation doc Part 4 SD-C):

Plumbing gates:

- Fixed-skill eval uses skill-aware obs composition for action and Q diagnostics;
  do not route through `maybe_eval_and_checkpoint()` unless it grows a skill
  hook or forwards `action_fn_kwargs`.
- `meta.json` contains both normal Go2 deploy metadata and `skill_discovery`.
- A fixed-skill policy can be replayed in sim with the same obs/action contract
  as training.

Behavioral gates:

- **Per-skill state coverage** (D3 Table 2 protocol): 1000+ random skill samples,
  one rollout each; report std-of-mean-states for the factor's tracked dim.
  Should rise above no-skill (vanilla SAC) baseline by ≥ 2×.
- **Per-skill rollout aggregates** (M=10 episodes per fixed z, K=4 one-hot skills):
  - mean ± std episode return
  - mean ± std episode length (proxy for fall rate; full D3 illegal-contact %
    deferred to SD-E)
  - command-tracking error (since first DIAYN factor is command-conditioned)
  - per-leg torque RMS
  - **Fall rate ≤ 20% per skill** (looser than D3 hardware bar; sim-only here)
- **Skill-following fidelity** (D3-style cosine similarity): cosine(commanded z,
  realized base velocity direction) > 0.5 mean across skills.
- **Headline plot:** xy trajectory per skill, color-coded by skill index
  (METRA Fig 3 protocol).
- **Seeds:** 3-5 seeds, mean ± std.

### SD-D: Deploy/export contract for fixed skills

SD-D is a sim/deploy-contract gate only. It does not authorize real-robot skill
deployment. Real mode must reject skill-discovery checkpoints unless a later
hardware-readiness gate sets `meta["skill_discovery"]["hardware_ready"] = true`
after style/safety validation.

- Add a skill-aware deploy obs composer.
- The skill-aware deploy composer owns normalization:
  - build raw/frame-stacked hardware obs with `ObsBuilder`,
  - normalize only the policy obs using checkpoint norm stats,
  - append fixed `skill_z` after normalization,
  - call a `PolicyRunner` forward path that skips `normalize_obs`.
- Add an explicit obs dimension check:

```text
(obs_builder.raw_dim * runner.n_frame_stack) + skill_dim == runner.obs_dim
```

- Add deploy CLI flags:

```text
--skill-index N
--skill-vector path_or_csv
--skill-mode fixed
```

- Add ONNX sidecar export, or copy the checkpoint contract next to `actor.onnx`:

```text
actor.onnx
deploy_contract.json
```

- Fix `sim2sim_direct.py` to use checkpoint default poses and remaps, not
  deploy constants, before using it as the skill policy gate. Kp, Kd,
  action_scale, physics_dt, policy_dt, and ObsBuilder-side pose/remap are
  already metadata-driven; the remaining bug is action target construction.
- Before Unitree or skill ONNX export, remove Flat-only 48d/action_scale=0.5
  assumptions from `jax_rl/utils/export.py` comments and validation, and validate
  sidecar contract shape against `obs_schema`, `control`, and
  `skill_discovery` metadata.

Acceptance:

- `deploy_go2.py --sim` can run a fixed skill with strict metadata.
- `deploy_go2.py` refuses skill checkpoints without skill metadata in real mode.
- `deploy_go2.py` refuses all skill checkpoints in real mode unless
  `hardware_ready=true`.
- ONNX validation covers both network numerics and contract shape.

### SD-E: D3-style factorization

Only after SD-B through SD-D:

- Add METRA for unbounded factors.
- Add named factor extractors for base xy, heading, base height, roll/pitch.
- Add factor weighting lambda.
- Add style factor and safety penalties.
- Add symmetry augmentation for Go2 leg permutations.
- Add curriculum over skill priors and factor weights.

V2 should treat style/safety as required for hardware, not optional polish.
Hardware readiness belongs here or later, not in SD-D.

**Algo choice — we depart from D3.** D3 uses **PPO** with 6 fully separate
critic MLPs (one per factor + extrinsic) and symmetry-augmented PPO updates.
Our SD-B/E uses **FastSAC** (off-policy) for sample efficiency and to leverage
existing Go2 infra (Go2 Warp + DR + post-truncation-fix sweep validates
FastSAC at 286). The DIAYN/METRA aux modules are algo-agnostic (port D3's aux
faithfully). The per-factor *value-decomposition* trick is PPO-specific in
D3 reference — for our SAC variant, look to DUSDi's `StateMaskCritic` /
`FactoredValueHead` pattern (vmap'd ensemble of per-factor Q heads, twin Q,
TD loss summed per head). That's the SAC analog of D3's per-factor value
heads. Document this departure in the SD-E plan when written.

Acceptance (SD-E — replicates D3 Tables 1, 2, 3 + Fig 5; see validation doc Part 4 SD-E):

- **D3 Table 1 replication** (style on/off, sim): illegal-contact % per body part
  (base, shank, thigh) + per-skill task return. Style should reduce contacts ≥10×
  (D3 reports base 4.04% → 0.03%).
- **D3 Table 2 replication** (algo choice ablation, sim): DIAYN-only, METRA-only,
  Mixed (D3) on Go2 factors. Mixed should beat single-method on ≥ 2 factors by
  state-coverage std-of-means.
- **D3 Table 3 replication** (downstream nav, sim): hierarchical PPO over frozen
  skills on rough-terrain waypoint task. Report mean reward, heading error,
  position error, termination ratios.
- **D3 Fig 5 replication**: roll/pitch coverage map with/without symmetry.
- **DUSDi DCI score** (informational, not a gate): compute Disentanglement,
  Completeness, Informativeness on D3 factors. Diagnostic for "are our hand-picked
  factors actually independent on Go2."
- **Hardware readiness gate**: `meta["skill_discovery"]["hardware_ready"] = true`
  only after passing all sim ablations + a sim2real walkability test (no falls in
  60s sim2sim with operator-slider z).
- **Hardware deploy** (real Go2): per fixed skill, M=3 trials, report success/fall
  rate + qualitative video. Match D3's operator-slider deploy protocol.
- **Seeds:** 5 seeds for sim ablations (matches D3), 3 trials per skill on
  hardware (compute / wall-clock permitting).

Open implementation questions — RESOLVED via source audits 2026-05-02 (see `.context/references/skill_discovery_source_extracts.md`):

- **METRA target φ network: NONE** (only Q-targets). Confirmed `iod/metra.py`.
- **METRA `dual_lam_init = 30`**, parameterized as `log(lambda)` for positivity.
- **METRA `dual_slack = 1e-3`** (NOT 1e-5 from v1 plan — must update).
- **METRA `dual_dist = 'one'` default** — constraint is `||Δφ||² ≤ 1` (constant).
  L2 / s2_from_s are paper-text-style ablations. Ship 'one' as default; flag for L2.
- **METRA z continuous: `N(0, I) → unit-sphere project`**.
- **D3 lr = `1e-3` adaptive schedule** (NOT 1e-4 as paper text).
- **D3 λ sampling = `F.normalize(|N(0,1)|^skew)`** — half-normal, L2-normalized.
  NOT Dirichlet, NOT sum-to-1, NOT truncated Gaussian. Major departure from paper text.
- **D3 per-factor value: 6 fully separate critic MLPs** (not shared encoder + heads).
  UCB term wired but `beta_advantage_UCB=0.0` — pure weighted sum.
- **D3 already incorporates DUSDi-style negative-MI penalty** via
  `lambda_skill_disentanglement=0.1` and `skill_disentanglement=True`. DUSDi's
  contribution is partially baked into D3 reference.
- **D3 skill resampling = 375 steps (7.5s @ 50Hz)** for all factors.
- **D3 Dirichlet α curriculum is adaptive cosine-sim driven** (×1.01 if cos>0.7,
  ×0.99 if cos<0.6, clamped [0.05, 1.0]). NOT a fixed schedule.
- **D3 ships NO eval scripts** for Tables 1, 2, 3 — we build our own.
- **D3 has no `hardware_ready` flag** — our SD-D gate is novel and worth keeping.
- Per-skill state-coverage sample count: D3 paper says "10K+", D3 code uses
  pairwise distance over 512 envs (different protocol). Use 1K samples for
  first run, expand to 10K after sim2real gate (still our convention).

## Proposed first PR/task split

1. `feat(skill): add config, priors, and DIAYN aux module`
   - Pure unit tests only.

2. `feat(skill): add SkillManager and sample-time reward composition`
   - Synthetic batch tests.
   - No env or train script yet.

3. `feat(skill): add FastSAC skill off-policy loop`
   - New `train_skill_discovery.py`.
   - Smoke on non-Go2 env.

4. `feat(skill): add Go2 fixed-skill DIAYN preset`
   - Use `Go2WarpJoystickUnitree`.
   - Named factor extractors, no magic obs indices.

5. `feat(deploy): add skill contract and fixed-skill sim deploy`
   - Deploy dim checks.
   - Skill metadata in checkpoint.
   - Sim-only gate first.

6. `docs(skill): retire stale one-line DIAYN obs guidance`
   - Update `jax_rl/envs/obs_spec.py` and public docs that imply `skill_z` is
     just one `ObsTerm`. Keep the conceptual point, but add the frame-stack,
     replay, checkpoint, and deploy caveats from this v2 doc.

## Tests to add

```text
tests/test_skill_discovery_config.py
tests/test_skill_discovery_prior.py
tests/test_skill_discovery_diayn.py
tests/test_skill_discovery_manager.py
tests/test_skill_offpolicy_loop.py
tests/test_skill_checkpoint_contract.py
deploy/test_skill_contract.py
```

Important assertions:

- `skill_z` is not frame-stacked unless explicitly requested.
- Existing raw obs normalization remains sample-time.
- Actor and critic both receive compatible `z`.
- Factor inputs required for sample-time reward are stored or derivable by name.
- Sample-time intrinsic rewards change when aux params change.
- Checkpoint metadata contains enough to reconstruct policy input shape.
- Skill checkpoint resume restores aux state.
- Deploy refuses unknown skill obs terms or missing `skill_discovery` metadata.
- Real deploy refuses skill checkpoints unless `hardware_ready=true`.
- Legacy non-skill checkpoints still deploy exactly as before.

## Open design questions

1. Should initial Go2 DIAYN use discrete one-hot skills or a Dirichlet prior?
   Decision for SD-A through SD-C: use one-hot. The D3 Dirichlet prior moves to
   SD-E after the fixed-skill path works.

2. Should `z` be visible to the critic?
   For off-policy Q-learning, yes by default. If critic lacks `z`, Q estimates
   average over multiple reward functions.

3. Should task reward be mixed in from day one?
   For simple-env SD-B smoke, no: pure DIAYN. For Go2 SD-C, yes: use the reward
   contract above with nonzero task/style/safety weights. Pure DIAYN can
   discover unstable behaviors that are not worth deploying.

4. Should factor extractors read privileged sim state?
   Yes for intrinsic reward. The actor should still use deployable observations.

5. Should V2 modify `run_offpolicy_loop` or create a separate loop?
   Start separate. After one working skill loop, extract shared hook points if
   duplication becomes clear.

6. Should ONNX export support skill policies immediately?
   Only after checkpoint-side skill contract exists. Exporting a larger MLP
   without skill metadata is worse than not exporting.

## Audit conclusions

- The old docs had the right north star but are no longer executable.
- RewardSpec/ObsSpec are necessary infrastructure, but not sufficient for
  off-policy skill discovery.
- `context_dim` is not a usable feature yet.
- Go2 skill factors must be named extractors, not fixed obs indices.
- Deploy metadata is now strong enough to extend, but skill metadata is missing.
- The first useful V2 target is FastSAC + DIAYN + fixed skill deployment in sim,
  not full D3 or hardware.
