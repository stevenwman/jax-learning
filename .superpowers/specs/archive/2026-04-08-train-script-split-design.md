# Design: Split train_offpolicy.py into Per-Algo Scripts

## Problem

`train_offpolicy.py` (498 lines) handles 4 algorithms via `if family ==` branching and an algo registry. The complexity grows with each new feature (DR syncd tracking, FlashSAC was already split out). Each algo's training loop should be readable top-to-bottom without branching.

## Decision

Split into 4 standalone scripts. Each duplicates ~60 lines of loop boilerplate but is self-contained. Shared infra stays in `jax_rl/training/` (already factored: `make_envs`, `EpisodeTracker`, `CheckpointManager`, logging, eval).

## New Files

| File | Algo | Family | ~Lines |
|------|------|--------|--------|
| `train_sac.py` | SAC | sac | ~180 |
| `train_td3.py` | TD3 | td3 | ~180 |
| `train_fast_sac.py` | FastSAC | sac | ~180 |
| `train_fast_td3.py` | FastTD3 | td3 | ~180 |

## Deleted

- `train_offpolicy.py` — replaced by the 4 scripts above

## Existing (unchanged)

- `train_flashsac.py` — already standalone
- `train_ppo.py` / `train_ppo_fast.py` — on-policy, separate concern
- `jax_rl/training/` — shared utilities (no changes needed)

## Per-Script Structure

Each script follows the same layout:

```
1. Docstring with usage example          (~5 lines)
2. Imports + env vars                    (~15 lines)
3. CLI argparse                          (~30 lines)
4. Algo + optimizer setup                (~20 lines)  ← ALGO-SPECIFIC
5. Env setup (make_envs)                 (~5 lines)
6. Dict obs detection + buffer init      (~20 lines)
7. Explore closure                       (~5 lines)   ← ALGO-SPECIFIC
8. Resume + tracker + checkpoint init    (~10 lines)
9. Training loop:
   a. Obs extraction + normalization     (~8 lines)
   b. Action selection (warmup/explore)  (~6 lines)
   c. Env step + DR reset               (~10 lines)
   d. Buffer add + tracker step          (~15 lines)
   e. Gradient updates                   (~15 lines)
   f. Metric handling                    (~5 lines)   ← ALGO-SPECIFIC
   g. Logging                            (~15 lines)
   h. Eval + checkpoint                  (~15 lines)
10. Final eval                           (~10 lines)
11. W&B finish                           (~1 line)
```

Total: ~180 lines per script. ~60 lines of loop body is duplicated across all 4.

## Algo-Specific Differences

### SAC / FastSAC (sac family)
```python
# Explore: stochastic policy
def explore(actor_params, obs, key):
    return algo.select_action(actor_params, obs, key)

# Logging: entropy + alpha
log_extra_fields = [("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")]
log_extra_keys = ["entropy", "alpha", "alpha_loss"]

# Metrics: direct assignment
last_metrics = step_metrics
```

### TD3 / FastTD3 (td3 family)
```python
# Explore: deterministic + noise
def explore(actor_params, obs, key):
    return algo.select_action(actor_params, obs, key,
        deterministic=False, exploration_noise=noise_std)

# Logging: no extra fields
log_extra_fields = []
log_extra_keys = []

# Metrics: policy_delay handling
if float(step_metrics.get("actor_loss", 0.0)) != 0.0:
    last_metrics = step_metrics
else:
    last_metrics = {**step_metrics, "actor_loss": last_metrics.get("actor_loss", 0.0)}
```

### SAC vs FastSAC / TD3 vs FastTD3
Only differ in optimizer setup:
- SAC/TD3: `optax.adam(lr)`
- FastSAC/FastTD3: `optax.adamw(cosine_decay_schedule, b2=0.95, weight_decay=0.001)`

## CLI Args

Each script has its own argparse. Common args across all 4:
```
--env, --seed, --resume, --num-envs, --total-timesteps, --lr,
--reward-scaling, --episode-length, --eval-every, --obs-norm,
--domain-rand, --wandb, --wandb-project, --frame-stack,
--action-delay-ms, --action-delay-range-ms, --reset-mode
```

Algo-specific args:
- SAC family: `--target-entropy-scale`
- TD3 family: `--exploration-noise`
- Fast variants: `--batch-size`, `--grad-updates-per-step`, `--buffer-size`

## DR Integration

Each script gets the full `--reset-mode` support (per_step, syncd, legacy). The DR syncd tracking code (~15 lines) is duplicated in each script's loop. This is intentional — it's simple enough that extraction isn't worth the indirection.

## Sync Strategy

No automated sync mechanism. The duplicated sections are:
1. Small (~60 lines of loop body)
2. Stable (rarely change — last change was DR integration)
3. Covered by existing tests (test_go2_warp_env.py tests make_envs integration)

If a shared concern changes (e.g., new buffer API), update all 4 scripts. The scripts are short enough that a diff review catches divergence.

## Migration

1. Create 4 new scripts from `train_offpolicy.py` template
2. Remove algo registry, `_make_algo()`, family branching
3. Inline the algo-specific code directly
4. Test each script with a smoke run
5. Delete `train_offpolicy.py`
6. Update docs (AGENT_HANDOFF, README, CLI reference)
