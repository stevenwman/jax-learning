# TODO

## 2026-03-20

### Completed (2026-03-21)
- [x] FastTD3 HumanoidRun — **665 eval** @ 100M steps (tau=0.125, tapered nets, NaN guards)
- [x] FastSAC HumanoidRun — **892 eval** @ 100M steps (C51, obs-norm, paper recipe)
- [x] FastDSAC HumanoidRun — **NaN'd @ 6M** (Gaussian critic diverged, needs variance floor fix)

### Active (2026-03-22)
- [x] Fix and verify moved test files — 48 tests passing
- [x] Commit cleanup changes (__init__ exports, test consolidation)
- [x] FastDSAC rewrite — Huber loss (not Gaussian NLL), matches paper source code
- [x] FastDSAC Inf guard — MJX velocity overflow produces Inf, not just NaN
- [x] FastDSAC buffer scaling — 51K→400K for 1024 envs
- [x] Checkpoints purge — 87→15 runs, 624→149MB. benchmark_results.md created
- [ ] TrainConfig cleanup — code done (PPO fields moved to PPOConfig), needs GPU test before commit
- [ ] Q diagnostics — eval runner has optional `q_fn` for Q vs MC return. Needs GPU test + wiring into train scripts
- [ ] FastDSAC 1024-env NaN test — running with Inf guard + 400K buffer, ~2hr to 53M crash point

### Short-term
- [ ] Revisit `lax.scan` for gradient loops — JAX buffer (on-device) may fix the carry overhead that killed scanned_update.py (see LESSONS.md, journal 03-18). Carry would be `(TrainingState, key)` only, not 4M buffer arrays.
- [ ] MJX recompilation investigation — why does `jit(while)`/`jit(scan)` recompile with identical signatures ~2x/min? Is it MJX, Playground, or JAX?
- [ ] Builders unification — all algos use `make_encoder()` factory. Prerequisite for CNN encoder

### Mid-term (Vision RL)
- [ ] Install `madrona_mjx`, verify Playground `vision=True` loads on RTX 5080
- [ ] CNN encoder (`jax_rl/networks/encoders/cnn.py`) + `CnnEncoderConfig`
- [ ] `--vision` flag on train scripts (env loads with `vision=True`, encoder swaps to CNN)
- [ ] DrQ augmentation (`jax_rl/utils/augmentation.py`)
- [ ] Benchmark CartpoleBalance from pixels (Playground colab baseline: 57s on 4090)
- [ ] ManiSkill integration (Gymnasium adapter + DLPack bridge)
- [ ] Memory budget testing — pixel replay buffer sizing on 16GB

### Long-term (Phase 6 — North Star)
- [ ] DIAYN (skill discovery wrapping SAC)
- [ ] METRA (contrastive + metric-aware skills)
- [ ] Goal-conditioned RL (encoder `context_dim` + `context_fusion`)
- [ ] USD (Unified Skill Discovery)
