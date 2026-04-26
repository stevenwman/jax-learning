# TODO

## Medium priority — TD-MPC2 J3 Cheetah re-run with all fixes

J3 v1 (2026-04-25) hit mppi=837 with action_repeat=1, episode_length=1000, γ=0.995, AND pre-Bug-A/B. Now we have:
- Bug A (truncated≠terminated) fixed (c080d32)
- Bug B (Q dropout in policy/qscale paths) fixed (c080d32)
- action_repeat=2 + episode_length=500 (43b71ad)
- eval-key isolation (b5a8c70)

Re-run J3 with all fixes. Expected: ≥837 (might exceed if previously bottlenecked by Bug A on truncations or by under-coverage from action_repeat=1; or stay 837 if already at task ceiling). Also tests whether end-of-run collapse pattern recurs (J4 v3 didn't show it — eval-key fix likely helped).

```bash
PYTHONPATH=$PWD XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/train_tdmpc2.py --env CheetahRun --total-timesteps 1000000 \
  --seed 0 --num-envs 8 --eval-every 50000 --ckpt-dir .temp/tdmpc2_j3_v2
```

Not blocking — paper match already achieved with v1.

## Low priority — TD-MPC2 train-budget extension to paper scale (4-14M)

Paper trains DMControl for 4-14M env steps (Humanoid uses 14M per Fig.15). We've validated 1M trajectories that match paper Fig.15 band at the same env-step budget (Humanoid 559 at 1M). For full asymptotic comparison would need to extend training. Each 1M ≈ 5h on a single GPU; 14M ≈ 70h.

Best-ckpt-save protects deployable artifacts; the only reason to extend is settling per-task asymptotic numbers for publication.

## 🔥 High priority — Verify polyak refactor on Go2 FastSAC training run

Commit `45ad979` (2026-04-25) extracted `_soft_update` from sac/td3/fast_sac/fast_td3/flash_sac into `jax_rl/utils/polyak.py`. Within-run equivalence proven by `tests/test_polyak.py::test_arbitrary_tau_matches_inline_lambda` (bit-equal to old inline lambda for tau ∈ {0, 0.005, 0.125, 0.5, 0.9, 1.0}). All 322 unit tests pass across 7 files. **Behavioral seal-of-approval still pending** — needs a real training run.

**Plan:**
- `uv run python scripts/train_fast_sac.py --env Go2WarpJoystickFlat --reset-mode per_step --num-envs 1024 --total-timesteps 5000000 --seed 42 --wandb`
- Run pre-refactor (`git checkout 45ad979^`) and post-refactor (current HEAD) under same seed.
- Cross-run won't be bit-identical on GPU (memory: feedback_gpu_nondeterminism). Confirm eval curves converge to within seed variance. Pre-fix Go2 FastSAC seed=100 hit 283.8 (best in-loop, post-truncation-fix benchmark in AGENT_HANDOFF.md). Acceptable: post-refactor lands within ±5 of that.
- ~30-60 min GPU time. Background it.

If eval diverges by more than seed variance: bug in the polyak signature switch (tau capture, arg order, closure) — revert to `45ad979^` and re-investigate.

## 🔥 High priority — FlashSAC resume eval regression

After landing `reward_norm_state` persistence (commit 6a17f9b, 2026-04-24), verified the reward-norm fix works via direct ckpt inspection (G_count=499968, G_r_max=33.28, RewScale=8.4318 matches pre-resume log's 8.435 to 3 decimals). But: resume still shows a large eval drop — CartpoleBalance hit 996 pre-resume then 747 at first eval after resume.

`reward_norm_state` is persisted correctly, so the regression is from another non-persisted piece of state. Candidates:
1. **Replay buffer not persisted** — `JaxReplayBuffer` starts empty on resume. First 10k env steps = random-uniform warmup, then gradient updates start from a buffer that has no policy-on-distribution data. The pre-resume policy's behaviour (cartpole-balancing) drifts during this gap. Most likely root cause.
2. **EpisodeTracker reset** (`train_flashsac.py:215`) — fresh `n_episodes=0` means first eval fires when few episodes have completed post-resume; eval_mean could be noisier.
3. **BN running stats** — `actor_batch_stats`, `q{1,2}_batch_stats`, `target_q{1,2}_batch_stats` ARE persisted via TrainingState. But critic targets update their BN running stats via `train=True` forwards in-process; after resume these may drift before stabilizing.
4. **Zeta noise `count`/`repeat_n` fields** — persisted. Unlikely.

**Diagnostic plan:**
- Log `buffer_size` at each eval post-resume — confirm it starts at 0 and grows.
- Add a pre-resume "micro-eval" (5 eps) immediately on load, before any env steps, to isolate whether the policy itself regressed vs a warmup-gap artifact.
- Run the same resume experiment on a harder env (Go2WarpJoystickFlat) where cartpole's "easy to recover" dynamics don't mask regressions.

**Fix candidates (if root cause is replay buffer):**
1. Persist replay buffer alongside orbax ckpt. Reference (Holiday-Robot/FlashSAC) has `save_replay_buffer` / `load_replay_buffer`. ~50 LOC + test.
2. Skip the random-uniform warmup on resume (if buffer loaded) OR retain policy actions during resume warmup (safer than random).

Not blocking — resume is rare in practice — but should be understood before claiming "checkpoint+resume is correct" for FlashSAC.

## Completed (2026-04-24) — FlashSAC reward_norm_state persistence

Folded `RewardNormState` into `TrainingState` (matches `noise_state` pattern for per-env state). Zero changes to checkpointing.py / eval_runner.py — orbax serializes via existing `training_state` entry. Aligns with reference Holiday-Robot/FlashSAC which saves `reward_normalizer.pt` as first-class artifact. Verified via direct ckpt inspection post-save: G_count=499968 (matches 500k env steps), G_r_max=33.28, G_var=71.10, RewScale=8.4318 matches pre-resume log exactly. Commit: 6a17f9b.

## Completed (2026-04-24) — Privileged-obs normalization

Unified + persisted critic normalization for both on-policy and off-policy paths.

- `obs_pipeline.py`: added `init_critic_norm_state`, `update_critic_stats`, `normalize_critic`, `make_critic_norm_fn`. `normalize_batch()` accepts optional `critic_norm_state`.
- `offpolicy_loop.py`: threads `critic_norm_state` — init, update each step, pass to normalize_batch + save/load + q_fn eval lambdas. Fixes silent bug where off-policy critic obs passed raw (mixed-unit privileged obs destabilized value loss).
- `train_ppo_fast.py`, `train_ppo_contraction.py`, `onpolicy_collect.py`: persist `critic_norm_state` via ckpt. Removed BANDAID comments. Resume now restores critic stats.
- `checkpointing.py`: backwards-compat fallback — load_checkpoint retries without critic_norm_state key on orbax schema mismatch for pre-fix ckpts (3-tuple return + warning).
- Tests: 11 new in `test_obs_pipeline.py`, 2 in `test_checkpoint.py`. All green.
- Smoke (Go2WarpJoystickFlat, 256 envs, 2M):
  - FastSAC: eval 273.2 ± 3.7, Q stable ~8.8.
  - PPO: return 1.4 → 165, VLoss stable.
- Commit: 4562307.

**Follow-up (resolved 2026-04-24):** re-ran PPO baseline + ContractionPPO at ref HPs on new critic-norm code, 100M each. Result neutral (38.5 vs 37.8 5-seed mean on best ckpt). See `.context/journals/2026-04-24.md` and `.context/lessons/ppo.md`.

## Completed (2026-04-21) — ContractionPPO port
- [x] Read Zinage et al. ContractionPPO paper/repo, extract algorithm
- [x] Flax `ContractionMetric` with Lipschitz MLP → SPD output (`jax_rl/networks/contraction_metric.py`)
- [x] `ContractionConfig` dataclass, wired into `PPOConfig` (optional)
- [x] `PPOContraction` algorithm (`jax_rl/algos/ppo_contraction.py`) — baseline bit-identity test passes
- [x] Opt-in `observe_contraction` obs group on `go2_bongo_handstand`
- [x] `make_collect` factory (`jax_rl/training/onpolicy_collect.py`) — partial on-policy extraction with extras + reward_augment hooks
- [x] `train_ppo_contraction.py` + `Go2BongoHandstandContraction` env variant + PPO preset for both
- [x] Smoke run validated end-to-end: CPen 60% drop in 2 iters, no NaN, metric learning confirmed
- [x] 26 tests green

### Research work — CLOSED 2026-04-24

Final A/B on `Go2BongoHandstand` (100M, single seed, ref HPs `alpha=0.1, eps=1e-3, penalty_coef=0.005, constraint_coef=100, metric_hidden=[128,64]`): **baseline 38.5 vs contraction 37.8 mean (5-seed re-eval on best ckpt, deterministic)**. Neutral — not worse, not obviously better. Different failure-mode seeds suggest the contraction policy has a distinct strategy but not a superior one. Independent review confirms the port's contraction-specific math is faithful to ref; any residual gap is in the PPO base (KL-adaptive LR, clipped value loss — both intentional deviations in ours, see `lessons/ppo.md`).

Prior HP sweep at `penalty_coef ∈ {0.01, 0.1, 1.0}, constraint_coef=1` — mostly inert (those were 100× weaker than ref's regime). Kept for reference in journal.

**If revisiting:**
- Test ref's actual claim (wind-perturbation robustness, not nominal return). Paper reports 0.00 vs 0.99 failure ratio on bongo+wind.
- Try ref's `c = trunk_xy_world_position` coord (we used projected_gravity). Different stability signal.
- Sparse-reward stabilization (pendulum swingup) where shaping has more room to help — we tried CartpoleSwingupSparse briefly but with high training-noise; inconclusive.

## Completed (2026-03-22)
- [x] FastTD3 HumanoidRun — **665 eval** @ 100M steps
- [x] FastSAC HumanoidRun — **892 eval** @ 100M steps
- [x] FastDSAC rewrite — Huber loss from paper source code
- [x] FastDSAC Inf guard — `isinf()` added, survived 54M+ steps (past 53M crash)
- [x] FastDSAC buffer scaling — 51K→400K for 1024 envs
- [x] TrainConfig cleanup — PPO fields moved to PPOConfig (48 tests pass)
- [x] Checkpoints purge — 87→15 runs, 624→149MB
- [x] Go2 sim-to-real plan — comprehensive, cross-checked against all .context/ docs
- [x] Q diagnostics — `get_q_value()` on all off-policy algos, wired into eval with Q bias/RMSE/corr
- [x] Builders unification — Actor/DeterministicActor/VCritic in builders.py, all algos refactored. Encoder swappable.
- [x] Numpy replay buffer archived — JAX buffer is now the only buffer. `--jax-buffer` flag removed.
- [x] Frame stack utility (`jax_rl/utils/frame_stack.py`) — shared by Go2 and vision RL

## Completed (2026-03-23)
- [x] Go2 env (`jax_rl/envs/locomotion/go2_joystick.py` — deleted 2026-04-09) — MjxEnv subclass, dict obs (48d state + 116d privileged), 16 reward terms, 12 tests pass
- [x] env_setup.py → unified registry loading (Go2 + DM Control Suite both work)
- [x] Go2 PPO preset in env_presets.py (1024 envs, gamma=0.97)

## Completed (2026-03-24)
- [x] PPO fix: value loss 0.25x scaling + full-batch advantage normalization (matched Brax PPO)
- [x] train_ppo_fast.py — lax.scan collect, 110k sps on Go1 (3.4x speedup vs Python loop)
- [x] Brax PPO A/B baselines — Go1: 21.7 @ 50M, Go2: 17.9 @ 50M
- [x] Our fast PPO beats Brax: 27.3 eval @ 28.5M steps on Go1 (Brax: 18 at same point)
- [x] Go2 contact fix — solimp 0.015→0.9 (firm, matches Go1)

## Completed (2026-03-25)
- [x] Go2 PPO Phase A — DONE. Seed 2100: eval 233 @ 50M steps. Config: tracking_lin_vel=10.0, tracking_ang_vel=5.0, height_term=0.18m, calf_torque=45.43Nm. Robot stands at 0.31m and locomotes.
- [x] Reward breakdown in record_video.py — saved in _traj.npz as reward_* arrays
- [x] record_video.py dict obs support + command arrow overlay
- [x] Fix eval recompilation — norm_state passed as arg, JIT compiles once
- [x] Fix train_ppo_fast.py online return tracker — running_ep_return persists across collect calls
- [x] Print elapsed time + --eval-every CLI flag + line-buffered output

## Completed (2026-03-25, cont.)
- [x] Best-policy checkpointing — CheckpointManager saves to ckpt_dir/best/ on new eval high
- [x] Sync train_ppo.py with train_ppo_fast.py — frozen obs norm, CheckpointManager, eval fix, --eval-every, .3g format

## Completed (2026-03-30)
- [x] PandaPickCube SAC — **reward 1386, cube lifted 22cm** @ 10M steps. Preset added to env_presets.py.
- [x] Manipulation benchmark survey — MuJoCo Playground already has 10 tasks (PandaPickCube, LeapCubeReorient, AlohaSinglePegInsertion, etc.)

## Completed (2026-04-26) — TD-MPC2 J4 HumanoidRun paper-band match

- [x] **2 more correctness bugs** (commit c080d32):
  - Bug A: truncated treated as terminated in TD target — clip(dones - truncations, 0, 1)
  - Bug B: Q dropout disabled in policy_loss + qscale recompute paths — added rngs={"dropout": key} at both call sites
- [x] **Env source-parity** (commit 43b71ad): action_repeat=2 (TDMPC2Config new field) + episode_length=500 in DMC presets (was 1000). Discount auto-recomputes 0.995→0.99.
- [x] **scripts/record_video_tdmpc2.py** (commit fd3e814): MPPI + prior mode video capture via two-phase rollout+render. Camera tracks body_id=1.
- [x] **J4 HumanoidRun 1M v3 benchmark**: final mppi=559.68 (best at final, no end-of-run collapse). 5-round eval of best ckpt: **mppi 556.99 ± 4.06 over 40 episodes**. Within paper Fig.15 Humanoid Run trajectory band at the same env-step budget.

## Completed (2026-04-25) — TD-MPC2 J3 paper match + supporting infra

- [x] **3 correctness bugs fixed** (commit 1e56a7f, see `.context/journals/2026-04-25.md` afternoon):
  1. MPPI temperature inverted (was 4× sharper than source)
  2. Q dropout disabled in world-model value loss (regularizer was off)
  3. Replay buffer cross-env contamination (sample_sequence stride for multi-env)
- [x] **Runtime extraction + eval-only ckpt scoring** (commit 007e992): `jax_rl/algos/tdmpc2_runtime.py` (252 LOC shared init/eval), `scripts/eval_tdmpc2.py` (load ckpt + N-round eval).
- [x] **Eval-key isolation** (commit b5a8c70): training PRNG no longer perturbed by eval cadence; verified within-process via key_before/after match.
- [x] **Determinism diagnostics + lesson** (commit 391e718): `scripts/check_tdmpc2_determinism.py` (3 subchecks), `.context/lessons/determinism.md` documents JAX/XLA bit-ID limits + mujoco_warp non-det (officially ack'd, fix in flight Warp 1.14 ~Jun 2026).
- [x] **J3 CheetahRun 1M benchmark**: best mppi=**837.51 ± 1.35 over 40 episodes** at step 500k. Within 1.5% of paper ≈850.

## Completed (2026-04-24)
- [x] **TD-MPC2 port (Phases A-I)** — branch `tdmpc2-impl`. Full model-based RL pipeline: SimNorm/two-hot/qscale utilities, per-episode sequence buffer, networks (Encoder/Dynamics/Reward/QEnsemble/PolicyPrior), losses (world model + policy with iter-4 sign fix + iter-3 no-mask fix), MPPI planner (Gumbel single-elite + `_prev_mean` warm-start + horizon/horizon-1 loop asymmetry), TDMPC2State + multi_transform optimizer + update_step factory, standalone `train_tdmpc2.py` with warmup/collect/UTD/eval/checkpoint. ~80 unit tests passing; 5 spec + 3 plan review iterations caught 14+ silent-failure bugs before coding. See `.context/journals/2026-04-24.md`. Benchmarks (J2-J4) deferred to user-initiated multi-hour runs.

## Active
- [x] **Terrain curriculum (Phases 1-4)** — 2026-04-17. 4 types × 10 levels grid, goal-directed commands, binary reach/fall advancement. Tests pass (42 primitives + generator, 9 env, 7 wrapper, 3 metrics). Presets wired for PPO/FastSAC/FlashSAC.
- [x] **Curriculum fix marathon** (2026-04-20) — 8 fixes + unified redesign. Commits 07679e6, 7afadee, 8463b17, ceb72c2. See `.context/journals/2026-04-20.md`. Pilot v5 (5M) advancing at 4× v3 pace with unified rim-to-center design.
  1. Fall detect via `truncation` (episode_fallen wiped by where_done)
  2. 15% force_zero_linvel DR flag
  3. force_zero_yaw DR (0.5/0.15)
  4. Contact-based termination (`base_contact` sensor replaces world-frame base_z<0.18)
  5. Preset `reset_mode="per_step"` default (TC wrapper was silently not applied)
  6. Class-A tracking-error promote (obsoleted by #7)
  7. Unified goal-directed (all 4 types spawn rim, goal center; single promote/demote rule; rotating body frame = free omnidirectional linvel DR)
  8. Zero linvel after reach (stand/spin at goal for rest of episode)
- [x] **Validate terrain curriculum in full 20M run** (2026-04-23/24) — multiple 20M pilots:
  - v14 (4-col): eval **294.4**, mean_level 0.71
  - v16 (5-col incl flat, commit 0676e47): eval **290.3 ± 6.8** (best 294.7), mean_level 0.73 excl flat (rough 0.62, pyramid_up 0.27, pyramid_down 1.00, tilted 1.04), fall=0.00 everywhere
  - flat column stays at L0 by design (not goal-directed, no advancement) — trains on it via Bernoulli cmd for distribution coverage
  - pyramid_up remains hard case (stuck L0–L1). See journal 2026-04-24.
- [ ] **pyramid_up stall** — type consistently <0.4 mean_level at 20M. Root-caused 2026-04-24 via v16 L1 traj analysis: robot faces goal correctly, walks on flat ring 0, collides with 8cm ring-1 step face (quat stays upright, torso pitch spike at collision → `base_contact` termination). L0 is fully flat (step_h=0) so demotion doesn't teach climbing — oscillates L0↔L1. Fix candidates, cheap→expensive:
  1. Quadratic difficulty curve: `step_h = difficulty² × max_step_height` in `jax_rl/envs/terrains/primitives.py:233` (L1: 80mm→16mm, L5 unchanged). One-line change.
  2. Relative `feet_clearance` reward (foot-z above local ground plane median). Was zeroed in v13 to kill world-frame artifacts on tilted terrain; re-introduce a stance-relative version. ~20 LOC.
  3. Lower `max_step_height` 0.4→0.25 (changes semantics of L5 ceiling).
  4. Approach-spawn specialization: half-spawn robots already on ring 1 facing inward. Structural.
  Try #1 first; stack #2 if still stuck. See journal 2026-04-23 §pyramid_up or lessons/terrain_curriculum.md §"pyramid_up L1 failure mode".
- [ ] **Flat env robustness non-deterministic** — v16 on Go2WarpJoystickCurriculum flat col: seed 0 died 607, seed 1 died 871, seeds 2/3 full 1000. Warp seed behavior or residual policy fragility. Investigate.
- [x] **Investigate eval OOM on curriculum env** (2026-04-20) — **non-issue**. Probed directly: Warp compiles kernels per-env-class, not per-instance. Both `WarpJoystickCurriculum` instances share the cache. Total curriculum VRAM at num_envs=32: ~500 MiB (train+eval combined). Historic eval OOMs were from num_envs=64 + training buffer pressure, not eval-env-specific. See journal 2026-04-20.
- [ ] **Reduce num_rows 10→5** — halves geom count (~1500→750), potentially enables num_envs=64 on 16GB. Coarser curriculum steps but 2× throughput. Try after 20M baseline.
- [ ] **Push-force curriculum** — follow-up plan, combine with torque-speed variant.
- [x] **Re-benchmark Fast*/Flash* post-truncation-fix** (2026-04-13) — done. WandB project: `jax-rl-post-truncation-fix`. Results in AGENT_HANDOFF benchmark table.
  - FastTD3 CheetahRun 5M: 515.9
  - FastSAC Go2 + per_step DR 20M: 283.8 (best in-loop) / 283.5 final → new best reproducible (vs 279.2 pre-fix)
  - FlashSAC Go2 10M: 284.5 final → new best reproducible (vs 282.4 claimed on reverted obs)
  - FastTD3 Go2 + per_step DR 20M: still running at writing time (matrix-completing, first TD3-family Go2 result)
  - All `@pytest.mark.slow` tests pass on GPU (Go2Warp env bundle, SAC CheetahRun end-to-end).
- [x] **Render best-checkpoint videos** for the 3 Go2 runs (FastSAC, FlashSAC, FastTD3). Done 2026-04-13. Skipped CheetahRun. All 1000 steps no termination. Files in respective `checkpoints/.../best/*.mp4`.
- [x] **`record_video.py` memory fix** (2026-04-13): baked `XLA_PYTHON_CLIENT_PREALLOCATE=false` default + replaced `lax.scan` with Python loop over jitted `rollout_step`. Peak HBM drops ~500 MB–1 GB; also coexists with concurrent training on same GPU. NPZ schema unchanged. Lesson: `lessons/infrastructure.md` §"`record_video.py` Memory Fix: `PREALLOCATE=false` + Python Loop (Not `lax.scan`)".
- [x] **Code fix: `final_eval_and_checkpoint` now captures `is_best` + announces "New best!"** (2026-04-14). Checkpoint artifact was already correct — `save()` already received `eval_mean` and wrote to `best_dir` when final eval beat in-loop best. Only the stdout log was incomplete. Now matches the in-loop path. Grep "New best!" no longer undercounts peak.
- [x] **Removed hardcoded MEM_FRACTION from all training scripts** (2026-04-15). Rationale: territory knob, not fragmentation knob. PREALLOCATE=true handles fragmentation. Fraction value is config-dependent; tuning per-script limits future experiments. Users now set `XLA_CLIENT_MEM_FRACTION=0.7` (or 0.55 if Warp needs room) via shell. Lesson: `.context/lessons/memory_tuning.md` — complete memory tuning strategy, signals, and workflow.
- [x] **Torque-speed actuator model** (2026-04-15) — optional linear torque-speed curve (`tau_limit = stall_torque × max(1 - |dq|/vel_limit, 0)`) on Go2 Warp envs. Stall torques read from MJCF `actuator_ctrlrange`; velocity limits from Unitree URDF. Flag `config.torque_speed_model=False` by default (identical behavior to current). Enabled via registered env variant `Go2WarpJoystickFlatTorqueSpeed`. Shared helper on `Go2WarpEnv._apply_torque_speed_limit`. 12 unit tests pass. Inspired by MJLab's `DcMotorActuator`.
- [x] **A/B FastSAC torque-speed model vs baseline on Go2** (2026-04-15) — seed 42, 20M, per_step DR: **286.0 best / 280.9 final**. Matches full-config baseline (285.1 ± 3.2) despite stripped obs. Trajectory analysis: 0% saturations during 1 m/s walking, mean torque-speed scale 0.92, peak |q̇| 15.6 rad/s vs limits 20-30. Clip dormant at walking speeds. See `.context/lessons/actuator_models.md` and `.context/journals/2026-04-15.md`.
- [ ] **Second seed on torque-speed model** — single-seed result 286; need at least one more seed to firm up "recovers stripped-obs regression" claim.
- [ ] **Torque-speed validation under push curriculum** — clip is dormant at flat 1 m/s walking; real test comes with push-force disturbances driving transient joint velocity spikes.
- [x] **Push-T manipulation env** (2026-04-17) — `PushEnv` in `jax_rl/envs/manipulation/push_env.py`. 4 shapes (T/L/circle/plus), 3 action modes (position/velocity/teleport), 16d shape-agnostic obs. 361k sps @ 1024 envs Warp. Heuristic rollouts validate dynamics.
- [x] **Register PushEnv in `env_setup.py` / `pg_registry`** (2026-04-17) — `Push{T,L,Circle,Plus}_{Pos,Vel,Tele}[_Shaped]` all registered via `pg_manipulation`.
- [x] **Spawn-bug fix** (2026-04-18) — slide-joint body pos was being used as offset; world coords drifted outside walls. Re-anchored pusher + block bodies to origin. Documented in `lessons/manipulation.md`.
- [x] **Reward shaping iteration + diagnostics** (2026-04-18) — 3 shaping variants tested (v1/v2/v3). v1 pos-PD: eval +143, 47% success. Added per-component reward + velocity-magnitude metrics; found zero-action attractor bug in vel/tele modes. 5 new lessons.
- [x] **Vendor gym-pusht + expert demos** (2026-04-18) — copied pusht.py + pymunk_override.py + LICENSE to `jax_rl/envs/manipulation/pusht/`. Added `reward_mode` kwarg (coverage/sparse/shaped/approach). Packed 206 LeRobot demos into 0.29 MB `pusht_demos.npz`. Parity test (3/3 pass) vs pip gym-pusht.
- [x] **Train vendored PushTEnv — contact_gated reward + full stack + TimeLimit** (2026-04-19) — 84% mean / 89% peak sto coverage at 2M steps. Vanilla SAC with keypoint obs + frame_stack 3 + action_repeat 2 + obs norm. TimeLimit bug was the 6× blocker.
- [ ] **Close the final-mile position gap** — policy stops ~7 px short of perfect. Options: absolute position refinement bonus `exp(-pos_err/2)`, longer training (3-5M), or BC-pretrain.
- [x] **Ablate v9 knobs** (2026-04-20) — 9-run study. Action_repeat dominant (−41pp stripped), log_barrier reward +8pp (new ceiling 0.933 sto, `coverage_shape="log_barrier"` kwarg), keypoints help but log_bar compensates, frame_stack marginal. Minimal shape-agnostic config (state + FS=1) hits 0.939 sto — first success event observed. See `.context/journals/2026-04-20.md`.
- [ ] **Test sparse-coverage-only with TimeLimit** — maybe sparse reward alone works now that infra is right. One run to check.
- [ ] **BC pretrain → RL fine-tune on PushTEnv** — recipe from DP paper. Dataset bundled at `pusht/demos/pusht_demos.npz`. Probably gets past 95% threshold.
- [ ] **Action chunking for off-policy RL (revisit w/ flow-matching)** — K-step action chunks instead of single action. SAC mods needed: (1) actor outputs K·d flat or autoregressive, (2) exec K-step open-loop, (3) K-step returns for TD target `y = Σγⁱrᵢ + γᵏQ(s_{t+K}, a'_{1:K})`, (4) critic `Q(s, a_{1:K})`, (5) buffer stores chunked tuples, (6) target entropy scales with K. Nuance: independent Gaussian per step breaks action correlation → needs flow-matching or diffusion actor for proper chunking (Q-chunking NeurIPS 2025 uses FQL for this reason). Our current `action_repeat=2` is degenerate K=2 shared-action chunk — probably the floor. References: RL-with-Action-Chunking ([arxiv.org/abs/2507.07969](https://arxiv.org/abs/2507.07969)), Q-LAC (ICML 2025), Bidirectional Decoding (ICLR 2025). Plan to revisit once flow-matching off-policy infra lands.
- [x] **Cross-shape transfer eval** (OBSOLETE 2026-04-20) — was planned on MuJoCo-Warp `push_env.py`, but that env was deleted in favor of pymunk gym-pusht + shape registry (block_shape kwarg on PushTEnv). Replaced by letter-matrix (next).
- [x] **gym-pusht shape registry + zero-shot floor** (2026-04-20) — 4 new block shapes implemented via annular-sector decomposition (ellipse/iso-triangle/S/U, S uses 2× 270° rings overlapping, U uses 180° half-ring + arms). Recorder `tools/record_pusht_shapes.py` subclasses PushTEnv per shape. Zero-shot T-policy on non-T shapes → 0.025-0.13 cov (vs 0.87 in-distribution T). 5d pose-only obs doesn't generalize. Next: shapely `unary_union + buffer(0)` fix for overlapping rings; DR training over shape set.
- [x] **Letter-shape transfer matrix V1+V2** (2026-04-24) — 5×4 (T/L/K/S+DR train × T/L/K/S test) on letter shapes. V1 (N=11 zero-padded keypoints) hit 0% off-diagonal because zero-pad pattern leaked shape ID. V2 (N=10 dense arc-length KPs, no padding) recovered: DR row-mean 73.9% sto / 57.8% det, **beats 5d-state DR baseline by 11pp**. Specialists 87-94% own-cov. Zero-pad-leak diagnosis + fix in `.context/studies/2026-04-22_pusht_letter_matrix.md`. See journal 2026-04-24 §letter-matrix-v2.
- [ ] **Held-out letter zero-shot (M/W/U/V)** — true zero-shot generalization test. Train DR on {tee, l, k, s} as today; eval on never-seen letters. V2 DR should generalize if KP representation is genuinely shape-agnostic.
- [ ] **Set-transformer / masked attention over variable-N KPs** — natural generalization to shapes with variable KP counts (no fixed N constraint). May enable zero-shot to arbitrary letters.
- [ ] **Remove gym-pusht from pip deps?** — vendored version is primary. Keep pip for parity test only.
- [ ] **Add FastSAC preset for push-T** — in `env_presets.py`. Small networks (16d obs → 64/64 actor, 128/128 critic probably enough).
- [ ] **Train FastSAC on push-T** — first real baseline. 5M steps should be plenty for 16d obs + contact task.
- [ ] **Cross-shape transfer eval** — train on T, evaluate zero-shot on L/circle/plus. Core adaptability benchmark.
- [ ] **DR specs for push env** — block mass, block friction, pusher friction, table friction, block size scale (via `<geom size>` multiplier — careful with MJCF model surgery).
- [ ] **More shapes** — D (half-disc), star (radial boxes), irregular polygon from vertex list.

## Completed (2026-04-06)
- [x] Documentation site — MkDocs + Material theme, 20 pages, mkdocstrings autodoc, videos embedded
- [x] Auto-generators for CLI flags (`docs/scripts/gen_cli_reference.py`) and env presets (`docs/scripts/gen_env_presets.py`)
- [x] Moved superpowers specs/plans from `docs/superpowers/` to `.superpowers/` (CLAUDE.md override)
- [x] Context doc `.context/docs_site.md` for future agents

## Completed (2026-04-07)
- [x] FlashSAC JAX port — inverted residual blocks + BatchNorm + weight norm + adaptive reward scaling + Zeta noise repetition. 22 tests passing. New files: `reward_scaling.py`, `flash_sac_config.py`, `flash_blocks.py`, `flash_sac.py`, `train_flashsac.py`.

## Short-term — FlashSAC benchmarking
- [ ] A/B benchmark FlashSAC vs FastSAC on CheetahRun (5M steps) and Go2WarpJoystickFlat (100M steps)
  ```
  uv run python train_flashsac.py --env CheetahRun --total-timesteps 5000000 --seed 100
  uv run python train_fast_sac.py --env CheetahRun --total-timesteps 5000000 --seed 100
  ```

## Short-term — Documentation
- [ ] Finalize GitHub repo → set `repo_url` in mkdocs.yml, activate GH Actions workflow
- [ ] Content polish pass — second draft on tutorials and getting-started pages
- [ ] Add more training videos if available (e.g., CartpoleBalance, PandaPickCube)

## Short-term — Go2 robustness (ACTIVE)
- [x] Domain rand (Tier 1) — friction, mass, COM, armature, frictionloss. Now declared via `get_domain_randomization_spec()` on the env, applied by `DomainRandWrapper` when `--reset-mode per_step`. (Legacy `go2_randomize.py` + `--domain-rand` path removed 2026-04-09.)
- [x] CPU sister env — `go2_cpu.py` (deleted 2026-04-09), same MJCF + overrides, CPU mj_step. Policy walks 3s.
- [x] Velocity kicks — ±0.75 m/s every 350 steps, was in MJX `go2_joystick.py` (deleted 2026-04-09); superseded by `domain_rand.py` + Warp env.
- [x] Motor strength DR — ×U(0.9, 1.1) via actuator_gainprm scaling
- [x] Friction DR fix — randomize ALL geoms (MuJoCo max-combine), range [0.3, 1.5]
- [x] Action delay — `ActionDelayWrapper` (120ms FIFO), `--action-delay-ms` / `--action-delay-range-ms` CLI flags. Config-driven wrapper pipeline.
- [ ] **Wider DR ranges** — motor_strength, mass, friction. Use curriculum to expand ranges as policy stabilizes.
- [x] Frame stacking — universal `FrameStackWrapper` wraps any env, `--frame-stack 3` CLI flag, deploy ObsBuilder mirrors.
- [x] Go2 SAC Phase B — FastSAC eval 226. Off-policy validated on Go2.

## Short-term — Cleanup
- [x] Consolidate off-policy train scripts → unified dispatcher (commit 14a17df, later moved to `archive/train_offpolicy.py`).
- [x] Extract shared loop → `run_offpolicy_loop` helper (2026-04-12). Per-algo scripts (`train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py`) are thin wrappers (~110–130 lines each; grew from ~60 at split). Unified dispatcher moved to `archive/train_offpolicy.py`. FlashSAC stays standalone.
- [x] **Truncation bug fix (2026-04-12)** — FastSAC/FastTD3/FlashSAC were teaching `Q=r` at timeout steps (no loss mask). Now matches SAC/TD3 Brax convention: `target = r + γ(1-done)V_next`, `loss *= (1 - truncation)`. Systematic underestimation on long-horizon tasks should be gone. See `lessons/offpolicy.md` and `lessons/distributional.md` §3.
- [x] Drop dead `handle_truncation` constructor arg from 5 off-policy algos (was stored on `self`, never read). The real switch is `cfg.handle_truncation` in the training loop.
- [x] Integration debt — 7/7 resolved (select_action_eval, asymmetric PPO test, etc.)
- [x] `lax.scan` for gradient loops — benchmarked: 1.03x (no speedup)
- [x] MJX recompilation — root cause found, upstream issue, MEM_FRACTION=0.7 mitigates
- [x] Vendor training wrappers — Vmap, Episode, AutoReset, DR in `jax_rl/envs/wrappers/training.py`. Removed Brax training wrapper dependency. Parity-tested.

## Short-term — Experiment tracking
- [x] W&B integration — `--wandb` flag on all 3 train scripts, logs step + eval metrics. Tested: SAC (CheetahRun 200k), PPO (CartpoleBalance 500k), no-flag passthrough. All working.
- [ ] W&B HP tuning agent — Claude reads wandb curves via API, diagnoses stagnation/divergence, proposes HP changes (lr, entropy_coef, UTD ratio, reward weights). Could be a hook or a scheduled agent.

## Short-term (MuJoCo Warp migration — HIGH PRIORITY)
- [x] Brainstorm full spec — design doc at `docs/superpowers/specs/2026-03-28-warp-go2-env-design.md`
- [x] Install `warp-lang>=1.12` + `playground>=0.2.0`, verify unitree Go2 MJCF loads with `impl="warp"`
- [x] Vendor unitree_mujoco go2.xml + meshes, patch foot sites, create Warp scene XML
- [x] Implement `Go2WarpEnv` base + `WarpJoystick` env with `contact_mode` flag (training/deploy)
- [x] Extract shared sensor helpers (`go2_sensors.py`), parameterize DR body ID
- [x] Register `Go2WarpJoystickFlat`, smoke test PPO training (500k steps, 8.5k sps, no NaN/crash)
- [x] Joint→actuator ordering fix — root cause of Warp Go2 failure. `_act_to_joint` remap in go2_warp_base.py
- [x] FastSAC on Warp — **eval 276.5 @ 18M steps**. Surpasses MJX PPO 244. Sim2sim to CPU validated (walks 20s+).
- [ ] Train PPO on unitree MJCF via Warp — full 50M run (PPO hit 132, entropy collapsed)

## Short-term — Asymmetric off-policy critic
- [x] Asymmetric critic for all off-policy algos — actor 48d, critic 122d. A/B result: ~2x faster to 270+ (5M vs 9M), final 279 vs 276 (noise).

## Short-term — Bongo Board Handstand
- [x] Bongo board MJCF — board + roller, equality constraint, physics validated
- [x] Scene XML — Go2 + bongo board + floor + sensors
- [x] `Go2BongoHandstand` env — obs, reward, termination, step, reset
- [x] Registration + smoke tests (15 pass)
- [x] CMA-ES handstand keyframe optimization (gen69, PD-hold stable)
- [x] Step + integration tests
- [x] Contact-based termination — feet/board/head/arm on floor or board
- [x] Eval loop `lax.scan` — fixes Warp OOM from Python-loop buffer accumulation
- [x] Cost-based reward redesign — normalized quadratic costs, survival ceiling
- [x] Training runs A/B2 — best eval 119 (Run A), 101 (Run B2 w/ arm term)
- [x] Run C (cost-based) — eval 28, plateaued. Normalized quadratic costs.
- [x] PPO-C — eval 23.7, entropy collapsed, still climbing at 50M
- [x] PPO-C2 (torque/vel penalties) — eval 15.9, regressed to 11. Penalties hurt.
- [x] PPO-C3 (frame-stack 3) — **eval 46.9** (94% of max). Breakthrough.
- [ ] Domain randomization — feet-board friction DR via `<pair>` elements, board mass, robot mass/motor
- [ ] Push force curriculum — antagonistic pushes after stable balance converges
- [ ] Phase 1B: full approach + mount + handstand (future)

## Short-term — New Environments
- [ ] Split belt walking env
- [ ] Push-T env
- [ ] Multi-mass manipulation env
- [ ] Multi-leg-length ant env

## Mid-term (Vision RL)
- [ ] Verify MJWarp GPU renderer on RTX 5080 (`mjx.create_render_context` + `mjx.render`). Madrona MJX is gone — replaced by built-in Warp ray-tracer in mujoco>=3.6.0.
- [ ] Add render context to Go2WarpJoystick env (follow Playground CartpoleBalance vision pattern)
- [ ] CNN encoder (`jax_rl/networks/encoders/cnn.py`) + `CnnEncoderConfig`
- [ ] DrQ augmentation (`jax_rl/utils/augmentation.py`)
- [ ] `--vision` flag on train scripts
- [ ] ManiSkill / HumanoidBench integration — requires env factory abstraction in `env_setup.py` (currently only coupling point to Playground). Gymnasium adapter + DLPack bridge.
- [ ] Memory budget testing — pixel replay buffer on 16GB

## Mid-term (Go2 Deployment)
- [x] Deploy script (`deploy/deploy_go2.py`) — DDS loop, 50Hz, FSM, works for sim and real
- [x] Headless sim2sim (`deploy/sim_headless.py`) — runs over SSH, records video
- [x] Separate deploy venv (Python 3.12) — CycloneDDS + Unitree SDK, setup script
- [x] Match hardware properties — Kd 0.5→0.1, rear thigh range, motor actuators + external PD (was general/affine)
- [x] Retrain PPO with motor actuators — eval 244 @ 50M (seed 4000)
- [x] Sim2sim pipeline — sim2sim_direct.py (no DDS, PD per physics step) + DDS version
- [x] Sim2sim diagnosis complete — MJCF diff (collision geometry, solver) is the gap. MJX→CPU works (3s walking). MJX→unitree needs robustness. See `.context/go2/mjcf_comparison.md`.
- [x] **Sim2sim to unitree** — SOLVED by training on Warp (unitree MJCF directly). FastSAC 276.5 walks 20s+ on CPU. MJX→unitree gap was irreducible MJCF difference.
- [x] ONNX export utility (`jax_rl/utils/export.py`) — JAX weights → ONNX for Jetson. Hand-builds ONNX graph via onnx.helper (no jax2tf). Deterministic inference only.
- [ ] DC motor model (`jax_rl/envs/actuators.py`) — Tier 2, add if sim-to-real gap > threshold
- [ ] Confirm Go2 EDU edition in lab (ask Steven)

## Short-term — Ablations
- [x] **Per-frame obs normalization** — `normalize_stacked()` tracks stats on single-frame obs, normalizes each frame slice with shared stats. Wired into both train scripts. 3 new tests pass.
- [ ] **Frame-stack + obs-norm A/B** — CheetahRun FastSAC 5M steps: `--frame-stack 3` vs `--frame-stack 3 --obs-norm`. Tests whether per-frame normalization helps with stacked proprioceptive obs.
  ```
  uv run python train_fast_sac.py --env CheetahRun --frame-stack 3 --total-timesteps 5000000 --seed 100
  uv run python train_fast_sac.py --env CheetahRun --frame-stack 3 --obs-norm --total-timesteps 5000000 --seed 100
  ```
- [ ] **obs_normalization A/B** — FastSAC on Go2WarpJoystickFlat with `--obs-norm` vs without. Paper uses True (DM Control benchmarks), we default False. Quick 20M run each.

## Mid-term — Optimizer experiments
- [ ] Muon optimizer (`optax.contrib.muon`) — matrix-whitening via Newton-Schulz orthogonalization. Already in optax 0.2.6, drop-in `GradientTransformation`. Auto-routes 2D weights → Muon, biases/norms → AdamW internally. **RL caveat:** zero published RL benchmarks, untested on non-stationary targets + small MLPs. Start with actor-only Muon, keep critic on AdamW. First/last layer should stay Adam per author guidance.
- [ ] Shampoo / other second-order optimizers — evaluate if Muon shows promise on RL

## Mid-term — Contraction-theory off-policy extension

Port done 2026-04-21 for PPO (`jax_rl/algos/ppo_contraction.py`, plan at `.superpowers/plans/2026-04-21-contraction-ppo.md`). Paper (Zinage et al., Caltech, https://contractionppo.github.io/) is on-policy only. Off-policy extension is untested in literature — potentially novel contribution.

Why it should work: contraction signal enters via reward augmentation `R_c = (ε − ReLU(V̇ + αV + ε)) * penalty_coef`. Any RL algorithm maximizing E[Σ R_t] can absorb it. Metric loss is pure supervised on (c, ċ) pairs — off-policy agnostic.

- [ ] **SAC/FastSAC port** — `sac_contraction.py` (fork) or merge into `sac.py`. ~200-300 lines.
  - Replay buffer: add `contraction_c`, `contraction_c_dot` fields (20 lines in buffer code).
  - Decide: compute `R_c` at **storage time** (fixed in buffer, stale metric) vs **sample time** (always fresh metric, slight extra compute per minibatch). Recommend sample-time — matches "metric influence via rewards" principle with fresh params.
  - Metric training: runs on minibatches sampled from replay — gets ~batch_size×epochs more gradient steps per env sample than PPO version → faster metric convergence expected.
  - Tuning: Q magnitudes ≫ rollout rewards → rescale `penalty_coef`. Start ~0.01× of PPO value.
  - Ablation to run: SAC entropy bonus vs contraction's stability pull at high α_entropy — do they fight?
- [ ] **TD3 / FastTD3 port** — mostly identical to SAC modulo entropy term. Deterministic target policy aligns cleanly with contraction's determinism assumption.
- [ ] **Compare on-policy (PPO) vs off-policy (SAC) contraction** — same env (go2_bongo_handstand, same α/ε), same total steps. Does off-policy's replay-driven metric training actually converge faster?
- [ ] **Caveat to verify first:** PPO port must actually improve robustness (paper's claim) before scoping off-policy extension. If PPO port is neutral/negative, off-policy scope-cut.

## Short-term — MJX archival
- [x] MJX Go2 env deleted 2026-04-09 (go2_base.py, go2_joystick.py, go2_cpu.py removed outright; record_video_cpu.py moved to `archive/`). Doc cleanup done 2026-04-20 (audit `.context/tmp/audit_2026-04-19.md`). Spec: `docs/superpowers/specs/2026-04-06-archive-mjx-go2.md`.

## Mid-term — DR wrapper v2

`DomainRandWrapper` (in `jax_rl/envs/wrappers/domain_rand.py`), integrated into training pipeline. Per_step mode validated on Go2: 6% throughput cost, better sample efficiency.

### Completed
- [x] DomainRandWrapper with per_step reset mode (syncd mode dropped 2026-04-09)
- [x] Training pipeline integration (`--reset-mode` flag)
- [x] Benchmarked across 5 envs (Go2, Bongo, Cartpole, CheetahRun, Walker)
- [x] Per_step vs legacy training comparison on Go2 FastSAC (5M steps, wandb: drv2-comparison). Per_step: 6% slower throughput, ~19% better return at same wall clock.
- [x] **Wire DR specs into Go2** — `get_domain_randomization_spec()` on Go2WarpJoystick: 6 model specs (friction, damping, armature, frictionloss, mass, motor strength) + 2 runtime (kp_scale, kd_scale). Smoke tested on 4 envs.
- [x] **Go2 DR refresh (2026-04-10)** — removed `kp_scale`/`kd_scale` runtime DR (per_step PD gain DR was redundant with motor_strength DR), added `torso_com_jitter` (`body_ipos`) + `body_inertia` model DR. Now 8 model specs, 0 runtime.
- [x] **Clean up benchmark scripts** — deleted 10 bench/profile scripts from repo root.
- [x] **Remove legacy DR path + syncd mode** (2026-04-09) — deleted `go2_randomize.py`, `bongo_randomize.py`, `DomainRandomizationVmapWrapper`, `--domain-rand` flag, `_step_syncd`/`batch_reset`.

### Next
- [ ] **Validate DR in full training** — runs on wandb `drv2-comparison`:
  1. Go2 per_step + DR specs, seed 0, 5M — main test
  2. Go2 per_step + DR specs, seed 1, 5M — seed robustness
  3. CheetahRun per_step (no DR), seed 0, 5M — already showed 6,309 sps (not catastrophic!), but killed before first episode completed at step 1000. Need full run to confirm.
  Compare against: per_step-no-DR (eval 280) and legacy (eval 270).
- [ ] **Add DR specs to Bongo env** — mirror Go2's `get_domain_randomization_spec()`.
- [ ] **Investigate CheetahRun full_reset catastrophe** — 99% slowdown, not explained by forward() cost. Low priority.

## Mid-term — Env composability (from MJLab audit, prereq for DIAYN)
- [x] **RewardSpec** — `compute_rewards(spec, **kwargs)` returns unweighted dict. All 3 envs refactored (Warp 17 terms, MJX 16, Bongo 9). DIAYN swaps reward by replacing `env._reward_spec`.
- [ ] **Curriculum callback** — `curriculum_fn(mean_return) → {spec_name: multiplier}` hook in DomainRandWrapper. Expands DR ranges (motor_strength, mass, friction) as policy stabilizes. ~1 hr, ~50 lines.
- [x] **ObsSpec** — `compute_obs(groups, noise_level, rng, **kwargs)` with per-term noise. All 3 envs refactored. DIAYN appends `ObsTerm("skill_z", ...)` to "state" group — one line.

## Long-term (Phase 6 — Skill Discovery)
Informed by D3 paper (arXiv:2508.19953) and leggedrobotics/d3-skill-discovery. See `.context/references/d3_skill_discovery.md`.

### Phase 6A: DIAYN (foundation)
- [ ] Skill prior — `DirichletSkillPrior(n_skills, concentration_schedule)` with curriculum α ∈ [0.05, 1.0]
- [ ] Discriminator network — learned q_φ(z|s), 2-layer MLP, softmax output
- [ ] Intrinsic reward — r_DIAYN(s, z) = log q_φ(z|s) - log p(z), wired via RewardSpec
- [ ] Skill-conditioned policy — z as context input via `EncoderConfig.context_dim` (scaffolding exists)
- [ ] Skill vector in obs — append to "state" group via ObsSpec
- [ ] Benchmark on Go2 Warp — discover forward/backward/strafe skills. Success: 4-5 interpretable skills.

### Phase 6B: Symmetry augmentation
- [ ] Go2 morphology mirror functions — M_s^k (permute leg indices), M_z^k (permute skill components). 4-fold symmetry for quadruped.
- [ ] Augmentation in rollout collection — mirror transitions with prob 1/K before buffer storage
- [ ] A/B test symmetry on skill interpretability

### Phase 6C: Style factor + safety (required for hardware)
- [ ] Extrinsic reward terms — joint torques, contacts, height deviation, orientation penalties (D3 Table 9)
- [ ] Factor weighting λ — sample from truncated Gaussian, enforce Σλ=1, balance conflicting skills
- [ ] Regularization penalties — torque limits, contact bounds, joint velocity caps (D3 Table 10)
- [ ] These are NOT optional — D3 proves they're load-bearing for sim-to-real transfer

### Phase 6D: METRA + factorized skill discovery (D3 endpoint)
- [ ] `HypersphereSkillPrior(dim)` — z ~ U(S^d-1) for continuous directional skills (d ≤ 3)
- [ ] State transition predictor φ(s) + Wasserstein distance objective + learnable Lagrange multiplier
- [ ] Per-factor algorithm selection — METRA for position (unbounded), DIAYN for heading (bounded/discrete)
- [ ] State factorization — {base position (2D), heading (2D), base height (1D), roll/pitch (2D)} for Go2
- [ ] Skill resampling within episode (not just once per episode)

### Phase 6E: Sim-to-real with learned skills
- [ ] Deploy skill library on real Go2 with style factor active
- [ ] Zero-shot transfer test — walk to goal using learned skill primitives
- [ ] Compare vs direct PPO policy (no skill library)

### Infrastructure already in place
- [x] RewardSpec — DIAYN reward swap is one line
- [x] ObsSpec — skill vector z injection is one line
- [x] Asymmetric critic — critic sees privileged state
- [x] EncoderConfig.context_dim — skill z as context input (scaffolding exists, unused)
- [x] Action delay wrapper — sim2real latency simulation
- [ ] Goal-conditioned encoder fusion (concat/FiLM/cross_attn) — needed for skill z context
