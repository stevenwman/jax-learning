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

## When to do this
After CheetahRun hits ≥700. Before starting SAC. That's the natural breakpoint.
