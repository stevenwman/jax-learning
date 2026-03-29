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

## Active

## Short-term — Go2 robustness (ACTIVE)
- [x] Domain rand (Tier 1) — friction, mass, COM, armature, frictionloss. `go2_randomize.py` + `--domain-rand` flag.
- [x] CPU sister env — `go2_cpu.py`, same MJCF + overrides, CPU mj_step. Policy walks 3s.
- [ ] **Random external forces (velocity kicks)** — push robot every 10-15s during training. Standard in legged_gym/walk-these-ways. Goes in env step(), not DR wrapper. Key for bridging MJX→CPU drift.
- [ ] **Wider DR ranges** — research SOTA quadruped pipelines for ranges. Current friction U(0.3,1.2) may need widening. Add Kp/Kd scaling, motor strength variation.
- [ ] **Reward tuning for robustness** — research what reward terms help recovery (orientation penalty, base height tracking, energy penalty scaling). May need curriculum.
- [ ] Wire frame stacking into Go2 env (currently no frame stack — just raw obs)
- [x] Go2 SAC Phase B — FastSAC eval 226. Off-policy validated on Go2.

## Short-term — Cleanup
- [x] Consolidate off-policy train scripts → `train_offpolicy.py --algo sac|td3|fast_td3|fast_sac` (commit 14a17df)
- [x] Integration debt — 7/7 resolved (select_action_eval, asymmetric PPO test, etc.)
- [x] `lax.scan` for gradient loops — benchmarked: 1.03x (no speedup)
- [x] MJX recompilation — root cause found, upstream issue, MEM_FRACTION=0.7 mitigates

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
- [ ] Train PPO on unitree MJCF via Warp — full 50M run to get eval score

## Short-term — Asymmetric off-policy critic
- [ ] Add privileged critic support to SAC/TD3 training scripts (`train_offpolicy.py`). Actor sees `obs["state"]` (48d), critic sees `obs["privileged_state"]` (122d). Theoretically justified: Pinto 2017 (DDPG, the original asymmetric AC paper), Lambrechts ICML 2025 (unbiased policy gradients, algorithm-agnostic). No published system combines SAC + privileged critic + legged locomotion — this would be novel.

## Mid-term (Vision RL)
- [ ] Install `madrona_mjx`, verify Playground `vision=True` on RTX 5080
- [ ] CNN encoder (`jax_rl/networks/encoders/cnn.py`) + `CnnEncoderConfig`
- [ ] DrQ augmentation (`jax_rl/utils/augmentation.py`)
- [ ] `--vision` flag on train scripts
- [ ] ManiSkill integration (Gymnasium adapter + DLPack bridge)
- [ ] Memory budget testing — pixel replay buffer on 16GB

## Mid-term (Go2 Deployment)
- [x] Deploy script (`deploy/deploy_go2.py`) — DDS loop, 50Hz, FSM, works for sim and real
- [x] Headless sim2sim (`deploy/sim_headless.py`) — runs over SSH, records video
- [x] Separate deploy venv (Python 3.12) — CycloneDDS + Unitree SDK, setup script
- [x] Match hardware properties — Kd 0.5→0.1, rear thigh range, motor actuators + external PD (was general/affine)
- [x] Retrain PPO with motor actuators — eval 244 @ 50M (seed 4000)
- [x] Sim2sim pipeline — sim2sim_direct.py (no DDS, PD per physics step) + DDS version
- [x] Sim2sim diagnosis complete — MJCF diff (collision geometry, solver) is the gap. MJX→CPU works (3s walking). MJX→unitree needs robustness. See `.context/go2/mjcf_comparison.md`.
- [ ] **Sim2sim to unitree** — blocked on robustness (wider DR + velocity kicks). Once CPU env walks reliably, test unitree transfer again.
- [ ] ONNX export utility (`jax_rl/utils/export.py`) — JAX weights → ONNX for Jetson (deferred — numpy inference at 50Hz is fine for now)
- [ ] DC motor model (`jax_rl/envs/actuators.py`) — Tier 2, add if sim-to-real gap > threshold
- [ ] Confirm Go2 EDU edition in lab (ask Steven)

## Long-term (Phase 6 — North Star)
- [ ] DIAYN (skill discovery wrapping SAC)
- [ ] METRA (contrastive + metric-aware skills)
- [ ] Goal-conditioned RL (encoder `context_dim` + `context_fusion`)
- [ ] USD (Unified Skill Discovery)
