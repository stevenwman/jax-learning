# TODO

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
- [x] Go2 env (`jax_rl/envs/locomotion/go2_joystick.py`) — MjxEnv subclass, dict obs (48d state + 116d privileged), 16 reward terms, 12 tests pass
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

## Active

## Completed (2026-04-06)
- [x] Documentation site — MkDocs + Material theme, 20 pages, mkdocstrings autodoc, videos embedded
- [x] Auto-generators for CLI flags (`docs/scripts/gen_cli_reference.py`) and env presets (`docs/scripts/gen_env_presets.py`)
- [x] Moved superpowers specs/plans from `docs/superpowers/` to `.superpowers/` (CLAUDE.md override)
- [x] Context doc `.context/docs_site.md` for future agents

## Short-term — Documentation
- [ ] Finalize GitHub repo → set `repo_url` in mkdocs.yml, activate GH Actions workflow
- [ ] Content polish pass — second draft on tutorials and getting-started pages
- [ ] Add more training videos if available (e.g., CartpoleBalance, PandaPickCube)

## Short-term — Go2 robustness (ACTIVE)
- [x] Domain rand (Tier 1) — friction, mass, COM, armature, frictionloss. `go2_randomize.py` + `--domain-rand` flag.
- [x] CPU sister env — `go2_cpu.py`, same MJCF + overrides, CPU mj_step. Policy walks 3s.
- [x] Velocity kicks — ±0.75 m/s every 350 steps, already in go2_joystick.py step()
- [x] Motor strength DR — ×U(0.9, 1.1) via actuator_gainprm scaling
- [x] Friction DR fix — randomize ALL geoms (MuJoCo max-combine), range [0.3, 1.5]
- [x] Action delay — `ActionDelayWrapper` (120ms FIFO), `--action-delay-ms` / `--action-delay-range-ms` CLI flags. Config-driven wrapper pipeline.
- [ ] **Wider DR ranges** — Kp/Kd scaling. May need curriculum.
- [x] Frame stacking — universal `FrameStackWrapper` wraps any env, `--frame-stack 3` CLI flag, deploy ObsBuilder mirrors.
- [x] Go2 SAC Phase B — FastSAC eval 226. Off-policy validated on Go2.

## Short-term — Cleanup
- [x] Consolidate off-policy train scripts → `train_offpolicy.py --algo sac|td3|fast_td3|fast_sac` (commit 14a17df)
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
- [ ] ONNX export utility (`jax_rl/utils/export.py`) — JAX weights → ONNX for Jetson (deferred — numpy inference at 50Hz is fine for now)
- [ ] DC motor model (`jax_rl/envs/actuators.py`) — Tier 2, add if sim-to-real gap > threshold
- [ ] Confirm Go2 EDU edition in lab (ask Steven)

## Short-term — Ablations
- [ ] **obs_normalization A/B** — FastSAC on Go2WarpJoystickFlat with `--obs-norm` vs without. Paper uses True (DM Control benchmarks), we default False. Quick 20M run each.

## Mid-term — Optimizer experiments
- [ ] Muon optimizer (`optax.contrib.muon`) — matrix-whitening via Newton-Schulz orthogonalization. Already in optax 0.2.6, drop-in `GradientTransformation`. Auto-routes 2D weights → Muon, biases/norms → AdamW internally. **RL caveat:** zero published RL benchmarks, untested on non-stationary targets + small MLPs. Start with actor-only Muon, keep critic on AdamW. First/last layer should stay Adam per author guidance.
- [ ] Shampoo / other second-order optimizers — evaluate if Muon shows promise on RL

## Mid-term — Env composability (from MJLab audit, prereq for DIAYN)
- [x] **RewardSpec** — `compute_rewards(spec, **kwargs)` returns unweighted dict. All 3 envs refactored (Warp 17 terms, MJX 16, Bongo 9). DIAYN swaps reward by replacing `env._reward_spec`.
- [ ] **Curriculum callback** — `curriculum_fn(env_ids, episode_returns) → dr_range_multipliers` in reset. Unblocks wider Kp/Kd DR ranges. ~1 hr, ~50 lines.
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
