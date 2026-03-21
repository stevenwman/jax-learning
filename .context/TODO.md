# TODO

## 2026-03-20

### Completed (2026-03-21)
- [x] FastTD3 HumanoidRun — **665 eval** @ 100M steps (tau=0.125, tapered nets, NaN guards)
- [x] FastSAC HumanoidRun — **892 eval** @ 100M steps (C51, obs-norm, paper recipe)
- [x] FastDSAC HumanoidRun — **NaN'd @ 6M** (Gaussian critic diverged, needs variance floor fix)

### Active
- [ ] Fix and verify moved test files (test_ppo_setup, test_determinism, test_normalization) — waiting for GPU
- [ ] Commit cleanup changes (__init__ exports, test consolidation) — pending test verification

### Short-term
- [ ] MJX recompilation investigation — why does `jit(while)`/`jit(scan)` recompile with identical signatures ~2x/min? Is it MJX, Playground, or JAX?
- [ ] TrainConfig cleanup — move PPO-specific fields (`policy_hidden_dim`, `value_hidden_dim`, `squash`, `state_dependent_std`, `num_steps`, `num_updates_per_batch`, `anneal_lr`, `max_grad_norm`) to PPOConfig
- [ ] Checkpoints purge — 50 runs, 299MB. Keep best results + metrics, delete orbax weights from failed experiments
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
