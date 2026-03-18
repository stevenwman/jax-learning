# Trainer Refactor — Tabled Discussion (2026-03-13)

## Problem
`train.py` is a monolith: env creation, optimizer construction, collection loop, PPO update, checkpointing, CSV logging. Breaks when adding:
- Second algo (SAC: per-step collection + replay buffer, not batch rollouts)
- Second env source (Robomimic/HumanoidBench: Gymnasium, not Brax)

## Proposed Architecture

```
Trainer (generic infrastructure)
├── creates env via adapter (dm_control, humanoid_bench, robomimic)
├── calls algo.collect() — algo knows if it's rollout or replay buffer
├── calls algo.update() — algo knows its own loss
├── handles: checkpointing, logging, eval, video
```

### Key decisions:
- On-policy vs off-policy distinction at **algorithm** level, not trainer level
  - PPO.collect() = N-step rollout → GAE → batch
  - SAC.collect() = 1-step + replay buffer → sample batch
- Env sources hide behind `JaxEnv` protocol + adapters
- Manager-based API (mjlab-style obs/reward/curriculum) is Phase 5 — separate concern

### What moves:
- train.py's collection loop → PPO.collect()
- Checkpointing/logging → Trainer infrastructure
- Env creation → adapter layer
- Config dispatch for `--algo ppo` / `--algo sac`

### What stays:
- PPO.update(), select_action(), init() — already clean
- Networks, buffers, normalization — untouched
- PPOConfig / TrainConfig — already refactored

## Status (updated 2026-03-18)

**We didn't do this.** Instead built SAC and TD3 with separate train scripts (`train.py`, `train_sac.py`, `train_td3.py`). Three scripts with ~70% shared code (env setup, episode tracking, logging, checkpointing, eval).

**Current pain:** Adding deterministic eval required identical changes to all three scripts. Same for any future logging/checkpoint changes.

**When to actually do this:**
- When adding FastTD3/FastSAC (5 train scripts = too much duplication)
- When adding Gymnasium adapter (collection loop diverges per env type)
- When adding Wandb (touching 3+ logging blocks)

Until then, the duplication is manageable and each script is self-contained and readable.
