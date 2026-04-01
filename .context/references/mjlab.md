# MJLab Reference

**Repo:** https://github.com/mujocolab/mjlab
**What:** GPU-accelerated robot learning framework merging Isaac Lab's manager-based API with MuJoCo Warp.
**Why we care:** Similar goals (legged loco, sim2real, skill discovery), different architecture. Good reference for composability, DR, curriculum — things we need for our north star.

## Architecture

Manager-based API where each env aspect is a separate composable module:

| Manager | Purpose | Our equivalent |
|---------|---------|----------------|
| EventManager | Domain randomization (4 trigger modes: startup/reset/step/interval) | `go2_randomize.py` (single function, applied once at init) |
| CommandManager | Goal/target generation (velocity commands, etc.) | Inline in `go2_warp_joystick.py` (`sample_command()`) |
| ActionManager | Action processing, scaling | Inline in env `step()` |
| ObservationManager | Sensor data aggregation, noise | Inline in env `_get_obs()` |
| RewardManager | Reward computation, per-term scaling | Inline in env `_get_reward()` |
| TerminationManager | Episode ending conditions | Inline in env `_get_termination()` |
| CurriculumManager | Difficulty adjustment during training | **We don't have one** |
| MetricsManager | Custom logging | Inline reward_components in state.info |

## Key Design Differences

### Vectorization
- **MJLab:** Native batched MuJoCo Warp arrays — scene compiled once, shared across envs. No vmap.
- **Us:** Per-env logic + `jax.vmap` via Playground/Brax wrapper. Simpler per-env code, vmap handles batching.

### Auto-reset
- **MJLab:** Selective `_reset_idx(env_ids)` — full reset per done env. No staleness problem.
- **Us:** Cached initial state replay (`jp.where(done, cached, new)`). Fast but `state.info` goes stale (we fixed for frame_stack with `jp.where(done, ...)`).

### Domain Randomization
- **MJLab:** `EventManager` with `@requires_model_fields()` decorator. Automatic per-env memory allocation. 4 trigger modes (startup, reset, step, interval). Tracks recomputation levels.
- **Us:** `go2_randomize.py` returns randomized model + in_axes. Applied via Playground's DR vmap wrapper at env init time. No mid-episode randomization, no interval triggers.

### Curriculum Learning
- **MJLab:** `CurriculumManager` adjusts task difficulty during reset. Can modify terrain complexity, command ranges, reward scales.
- **Us:** Nothing yet. TODO mentions "may need curriculum" for wider DR ranges.

### Truncation
- **MJLab:** `is_finite_horizon` config flag — distinguishes terminal done (no bootstrap) from truncated done (bootstrap continues).
- **Us:** `handle_truncation` flag in TrainConfig + EpisodeWrapper sets truncation flag.

## What We Can Learn

1. **Composable managers** — Our monolithic env classes work but don't scale. When we add DIAYN (skill vector as command), curriculum (terrain difficulty), and vision (CNN obs), the current `_get_obs()` / `_get_reward()` methods will get unwieldy. Manager pattern separates concerns.

2. **Event-driven DR** — Randomizing at intervals (not just reset) is more realistic. Real-world disturbances happen mid-episode. Their `interval` mode with randomized timing is exactly what velocity kicks try to approximate.

3. **Curriculum** — We need this for wider DR ranges, terrain difficulty, and eventually skill discovery. Their pattern (compute during reset, adjust per env) is clean.

4. **Owning the training infra** — They don't depend on Brax/Playground for wrapping. Everything is built-in. We should move the same direction (vendor the 80 lines of wrappers).

## When to Revisit

- Before implementing curriculum learning
- Before adding terrain environments
- Before DIAYN (skill discovery needs composable command/obs managers)
- When adding ManiSkill/HumanoidBench (need env factory abstraction anyway)
