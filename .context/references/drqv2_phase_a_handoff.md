# DrQ-v2 Phase A — Handoff Notes

**Status:** Pre-spec. Brainstorm + audit done; no code written.
**Date:** 2026-04-27
**Pick up by:** invoking `superpowers:brainstorming` with the args at
the bottom and continuing from "Present design" — most discovery is
already captured here.

## Goal

Validate vision-RL pipeline + algo correctness against a published
benchmark by porting DrQ-v2 (Yarats et al 2021) to `jax_rl`.

**Why DrQ-v2 first:**
- Densest pixel-DMC reference numbers in the literature (Table 2,
  Figure 3: 12+ tasks at 100k/500k/1M/3M with multi-seed mean+std).
- Algorithmically simple — TD3 + image augmentation + CNN encoder.
  No world model, no MPPI, no latent dynamics. Bug surface tiny.
- DrM (current pixel-DMC SOTA) is a small delta on top: distributional
  Q + dormant-neuron resets. Becomes Phase B once DrQ-v2 matches paper.

**Out of scope:** locomotion vision (Go2 + MJWarp + asymmetric critic).
Existing `.context/references/vision_rl_design.md` was written for that
target; conflicts with paper-bench architecture (asymmetric critic vs
DrQ-v2's shared encoder). Add runtime warning if `n_frame_stack > 1`
is requested with pixel obs in locomotion envs (not implemented).

## Scope decision

| Phase | Algo | Bench target | Status |
|---|---|---|---|
| A | DrQ-v2 | DrQ-v2 paper Table 2 (Walker Walk, Cheetah Run, Quadruped Walk) | not started |
| B | DrM | DrM paper numbers (same DMC pixel suite) | depends on A |
| (skipped) | Asymmetric-critic locomotion vision | n/a | revisit if Go2 vision becomes priority |

## Architecture (per DrQ-v2 paper)

- Shared CNN encoder between actor + twin critics
- Critic-driven gradients into encoder; actor consumes
  `stop_gradient(encoder(obs))`
- Random shift augmentation: pad 84×84 → 92×92 replicate, random crop
  back to 84×84 per sample. Apply to `s` AND `s'` independently. K=2
  augmented copies, average for critic loss + TD target
- TD3 base: deterministic actor, twin critic, target nets, delayed
  actor update
- n-step returns (n=3) — same pattern as existing FastTD3
- Linear exploration noise schedule (σ: 1.0 → 0.1)
- Frame stack 3, 84×84 RGB → (84, 84, 9) channels-last
- uint8 obs storage in replay; float32 conversion + augmentation at
  sample time

## Audit findings (2026-04-27)

### What exists

- `.context/references/vision_rl_design.md` (2026-03-20, updated
  2026-03-30) — pre-existing design doc, locomotion-flavored. Contains
  useful component spec (CnnEncoderConfig, AugmentationConfig,
  PixelObsConfig) and the random_shift snippet at lines 191-195.
- `.context/lessons/vision.md` — frame-stack philosophy + asymmetric
  critic argument (mostly relevant to skipped Phase). The "uint8 is
  non-negotiable" lesson DOES apply.
- `.context/lessons/frame_stack.md` — companion.
- `.context/lessons/algo_port_protocol.md` — required reading. §1-§11
  apply to DrQ-v2.
- `jax_rl/utils/frame_stack.py` — sample-time stack reconstruction
  (state-obs use case).
- `jax_rl/buffers/jax_replay_buffer.py` — has `FrameStackConfig` (lines
  316-351) but obs storage **float32-only** (line 60: `dtype=jnp.float32`).
  uint8 path is aspirational, not implemented.
- `jax_rl/networks/encoders/mlp.py` — only encoder; no `make_encoder`
  factory. `builders.py` hardcodes `MlpEncoder` in 3 places.
- `jax_rl/algos/fast_td3.py` — full TD3 chassis. Closest reference for
  DrQ-v2 structure.
- MJWarp pixel render verified on RTX 5080 (pre-existing research).

### What's missing for DrQ-v2

- Pixel obs path through `make_env_bundle` for `dm_control` envs (gym
  backend). `cfg.env_kwargs={"from_pixels": True, "render_kwargs": ...}`
  or equivalent flag wiring.
- uint8 dtype path through `JaxReplayBuffer.add` and `sample`. Constructor
  takes `obs_dtype: jnp.dtype = jnp.float32` and threads through.
- `jax_rl/networks/encoders/cnn.py` — Nature CNN architecture
  (Conv 32 8×8 s4 → Conv 64 4×4 s2 → Conv 64 3×3 s1 → flatten →
  Dense feature_dim).
- `make_encoder(encoder_type, encoder_config)` factory in
  `jax_rl/networks/builders.py`. Replaces the 3 hardcoded `MlpEncoder()`
  call sites.
- `jax_rl/utils/augmentation.py` — `random_shift(images, key, pad=4)`
  vmappable. Reusable across vision algos.
- `jax_rl/algos/drqv2.py` (single file estimated ~500 LOC: encoder
  shared between actor+critics, augmentation in update_step,
  scheduled exploration, n-step returns).
- `scripts/train_drqv2.py` standalone loop (pixel obs shape doesn't
  fit `run_offpolicy_loop`, plus σ schedule needs step_idx). Mirror
  `train_fast_td3.py`.
- `tests/test_drqv2.py` hermetic.

## Algo-port-protocol implications

- §1 Bundle: `bundle = make_env_bundle(cfg, seed)`,
  `num_envs = bundle.num_envs`. DMC pixel routes through gym backend
  (cpu_count cap applies).
- §2 Backend gate: gym-only initially (until pixel route added to MJX
  backend). `if bundle.backend_kind != "gym": raise`. Eval via
  `evaluate_gym`.
- §3 Artifact: fold encoder into actor pytree; save as
  `KIND_SHARED_ACTOR`. Critics + targets are training-only, not in
  inference ckpt. `record_video.py` and `policy_runner.py` work
  unchanged.
- §4 CLI: `build_parser()` exposed; register in
  `docs/scripts/gen_cli_reference.py` SCRIPTS list.
- §5 Layout: single-file `jax_rl/algos/drqv2.py`. Encoder + augmentation
  factored to `networks/encoders/cnn.py` + `utils/augmentation.py` so
  algo file stays focused.
- §6 Shared loop: doesn't fit (pixel obs shape + σ schedule). Standalone
  loop. Comment why at top of `train_drqv2.py`.
- §7 `--resume-warmup {policy,random}` default `policy`. ~5 LOC.
- §9 Tests hermetic. No module-level `jax.random.PRNGKey`. Lazy-import
  `dm_control` inside fixtures.
- §10 README maturity = Experimental research until bench numbers land.
- §11 Self-audit via parallel Explore agents post-merge.

## Validation plan

DrQ-v2 paper Table 2 numbers (1M env steps, 3 seeds reported):

| Task | Paper score |
|---|---|
| Walker Walk | ~960 |
| Cheetah Run | ~660 |
| Quadruped Walk | ~775 |

Target: within 1σ of paper numbers on at least 2 of 3 tasks at 1M steps.

## When you pick this up

1. Re-read this doc + `.context/lessons/algo_port_protocol.md`.
2. Verify the audit is still accurate (`grep` checks at the bottom).
3. Invoke `superpowers:brainstorming` with the args block below. Skip
   ahead to "Present design" — discovery is captured here.
4. Brainstorm should produce a spec at
   `.superpowers/specs/YYYY-MM-DD-drqv2-design.md`.
5. Then `superpowers:writing-plans` for the implementation plan.
6. Then `superpowers:subagent-driven-development` to execute.

### Brainstorm args (paste into Skill tool)

```
Phase A: standard DrQ-v2 reimplementation in jax_rl, benchmarked
against the DrQ-v2 paper (Yarats et al 2021) on DMC pixel envs. Goal:
validate pixel pipeline + algo correctness against published numbers.
Architecture per paper: shared CNN encoder between actor + twin
critics, critic-driven gradients, actor stop-grad on encoder, random-
shift augmentation, n-step returns, linear exploration noise schedule.
Target envs: Walker Walk, Cheetah Run, Quadruped Walk via dm_control
built-in pixel obs (CPU render OK for bench validation). Build atop
existing FastTD3 chassis. Uses existing jax_rl/utils/frame_stack.py +
builders.py (needs make_encoder factory added). New files:
jax_rl/networks/encoders/cnn.py, jax_rl/utils/augmentation.py,
jax_rl/algos/drqv2.py, scripts/train_drqv2.py. Phase B (DrM extension)
reuses everything. Pre-existing handoff doc at
.context/references/drqv2_phase_a_handoff.md captures all audit
findings + algo-port-protocol implications — start there.
```

### Audit-still-valid checks

```bash
# §1 — pixel obs in bundle yet?
grep -nE "from_pixels|pixel_obs|render_kwargs" jax_rl/training/env_backends/*.py jax_rl/training/env_setup.py

# §2 — uint8 in replay buffer?
grep -nE "obs_dtype|jnp\.uint8|dtype.*uint" jax_rl/buffers/jax_replay_buffer.py

# §3 — make_encoder factory?
grep -nE "make_encoder|encoder_type" jax_rl/networks/builders.py

# Existing CNN encoder?
ls jax_rl/networks/encoders/ 2>&1

# Existing augmentation util?
ls jax_rl/utils/augmentation.py 2>&1
```

If any of those checks change (e.g., bundle gains a pixel path, or
replay buffer gains uint8), update the missing-list in this doc before
brainstorming.

## Why this doc exists

Stepped away from DrQ-v2 work after scope + audit but before spec.
Future-self / future-agent picking up the work needs the audit findings
+ scope decisions + architecture choices that would otherwise be
recreated from scratch (or worse, recreated wrong). This doc replaces
that re-discovery cost with one read.
