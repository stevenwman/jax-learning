# TODO

## 2026-03-20

### Active
- [ ] FastTD3 HumanoidRun 100M steps — running with NaN guards v2 (action guard + C51 log_prob clamp)
- [ ] FastDSAC HumanoidRun — queue after FastTD3 finishes

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
