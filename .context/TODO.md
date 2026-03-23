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

## Active
- [ ] Builders unification — `make_encoder()` factory. PPO uses builders, SAC/TD3/Fast* build inline. **Prerequisite for SAC on Go2 (Phase B) and vision RL.**
- [ ] Create `jax_rl/utils/frame_stack.py` — shared utility for Go2 (state obs) and vision RL (pixel obs)

## Short-term
- [ ] Consolidate off-policy train scripts → `train_offpolicy.py --algo sac|td3|fast_td3|fast_sac`. Prerequisites: (1) builders unification, (2) `select_action` handles noise internally, (3) algo registry. Eliminates ~300 lines of duplication across 4 files. See builders_unification_plan.md.
- [ ] Go2 env (`jax_rl/envs/locomotion/go2.py`) — subclass MjxEnv, legged_gym rewards, 31-dim obs
- [ ] Domain rand wrapper (`jax_rl/envs/wrappers/domain_rand.py`) — robot-agnostic, vmap over MJX params
- [ ] Go2 PPO Phase A — flat terrain walking, validates env/reward/domain-rand
- [ ] Go2 SAC Phase B — off-policy validation (requires builders unification first)
- [ ] Revisit `lax.scan` for gradient loops — JAX buffer may fix carry overhead (see LESSONS.md, journal 03-18)
- [ ] MJX recompilation investigation — upstream `jit(while)`/`jit(scan)` recompile ~2x/min

## Mid-term (Vision RL)
- [ ] Install `madrona_mjx`, verify Playground `vision=True` on RTX 5080
- [ ] CNN encoder (`jax_rl/networks/encoders/cnn.py`) + `CnnEncoderConfig`
- [ ] DrQ augmentation (`jax_rl/utils/augmentation.py`)
- [ ] `--vision` flag on train scripts
- [ ] ManiSkill integration (Gymnasium adapter + DLPack bridge)
- [ ] Memory budget testing — pixel replay buffer on 16GB

## Mid-term (Go2 Deployment)
- [ ] ONNX export utility (`jax_rl/utils/export.py`) — JAX weights → ONNX for Jetson
- [ ] Deploy script (`deploy/deploy_go2.py`) — DDS loop, 50Hz, same interface for sim/real
- [ ] Validate in `unitree_mujoco` before real hardware
- [ ] DC motor model (`jax_rl/envs/actuators.py`) — Tier 2, add if sim-to-real gap > threshold
- [ ] Confirm Go2 EDU edition in lab (ask Steven)

## Long-term (Phase 6 — North Star)
- [ ] DIAYN (skill discovery wrapping SAC)
- [ ] METRA (contrastive + metric-aware skills)
- [ ] Goal-conditioned RL (encoder `context_dim` + `context_fusion`)
- [ ] USD (Unified Skill Discovery)
