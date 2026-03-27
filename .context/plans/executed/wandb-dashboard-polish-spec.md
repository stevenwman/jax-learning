# W&B Dashboard Polish — Design Spec

**Date:** 2026-03-26
**Status:** Approved (pending implementation)
**Scope:** Improve W&B metric organization for at-a-glance readability by lab members

---

## Problem

All metrics are logged to W&B as flat keys (`q1_mean`, `eval_mean`, `sps`, `actor_loss`, ...). The dashboard shows 15+ unsorted charts. A new lab member has to know which chart matters. Comparing HP sweeps or seed runs means scanning every panel.

## Solution

Prefix metrics into 4 collapsible W&B sections + set `define_metric` defaults for summary behavior. Changes centralized in `metrics_logger.py` — train scripts only add one `wandb_setup_metrics()` call each. CSV format unchanged.

## Metric Sections

| Section | Keys (flat name -> W&B name) | Summary |
|---------|------------------------------|---------|
| `perf/` | `eval_mean`, `eval_std`, `eval_min`, `eval_max`, `avg_return`, `min_return`, `max_return` | `eval_mean`: max; `avg_return`: max |
| `critic/` | `q1_mean`, `q2_mean`, `q_bias`, `q_rmse`, `q_corr`, `q_mean`, `mc_mean` | `q_bias`: last; `q_corr`: last |
| `actor/` | `actor_loss`, `policy_loss`, `value_loss`, `entropy`, `alpha`, `alpha_loss`, `approx_kl`, `clip_fraction`, `log_std_mean`, `log_std_min`, `log_std_max` | `entropy`: last; `alpha`: last |
| `infra/` | `sps`, `elapsed`, `grad_steps`, `episodes`, `iteration`, `iter_time` | `sps`: last; `episodes`: max |

Keys not in the mapping pass through unchanged to W&B (future-proof).

## Implementation Details

### File: `jax_rl/training/metrics_logger.py`

**1. Module-level mapping dict:**

```python
_WANDB_PREFIX = {
    # perf
    "eval_mean": "perf/eval_mean",
    "eval_std": "perf/eval_std",
    "eval_min": "perf/eval_min",
    "eval_max": "perf/eval_max",
    "avg_return": "perf/avg_return",
    "min_return": "perf/min_return",
    "max_return": "perf/max_return",
    # critic
    "q1_mean": "critic/q1_mean",
    "q2_mean": "critic/q2_mean",
    "q_bias": "critic/q_bias",
    "q_rmse": "critic/q_rmse",
    "q_corr": "critic/q_corr",
    "q_mean": "critic/q_mean",
    "mc_mean": "critic/mc_mean",
    # actor
    "actor_loss": "actor/actor_loss",
    "policy_loss": "actor/policy_loss",
    "value_loss": "actor/value_loss",
    "entropy": "actor/entropy",
    "alpha": "actor/alpha",
    "alpha_loss": "actor/alpha_loss",
    "approx_kl": "actor/approx_kl",
    "clip_fraction": "actor/clip_fraction",
    "log_std_mean": "actor/log_std_mean",
    "log_std_min": "actor/log_std_min",
    "log_std_max": "actor/log_std_max",
    # infra
    "sps": "infra/sps",
    "elapsed": "infra/elapsed",
    "grad_steps": "infra/grad_steps",
    "episodes": "infra/episodes",
    "iteration": "infra/iteration",
    "iter_time": "infra/iter_time",
}
```

**2. New function: `wandb_setup_metrics()`**

Called once after `wandb_init()`. Same try/except ImportError guard as other wandb functions.

X-axis: We rely on the `step=` positional arg already passed to `wandb.log()` in all call sites. This sets W&B's global step to `total_steps`. No `define_metric("*", step_metric=...)` needed — it would conflict with eval dicts that don't contain a `total_steps` key.

Summary behavior (guarded with try/except for version compat):
- `wandb.define_metric("perf/eval_mean", summary="max")` — summary table shows best eval
- `wandb.define_metric("perf/avg_return", summary="max")`
- `wandb.define_metric("infra/sps", summary="last")`
- `wandb.define_metric("infra/episodes", summary="max")`

If `define_metric(summary=...)` raises TypeError (older wandb), skip silently — summaries are a nice-to-have.

**3. Modify `wandb_log()`**

Remap flat keys to prefixed keys before calling `wandb.log()`:

```python
def wandb_log(metrics: dict, step: int) -> None:
    try:
        import wandb
        if wandb.run is not None:
            remapped = {_WANDB_PREFIX.get(k, k): v for k, v in metrics.items()}
            wandb.log(remapped, step=step)
    except ImportError:
        pass
```

**4. Update `wandb_init()` call sites**

Add `wandb_setup_metrics()` call right after `wandb_init()` in all 3 train scripts. This is the only train script change — one line each.

## What Does NOT Change

- CSV column names (flat, no prefixes)
- stdout format
- `make_metrics_row()` or `log_training_step()`
- Any algo code
- `record_video.py`

## Testing

### Unit test (new: `tests/test_wandb_metrics.py`)
- Test `_WANDB_PREFIX` remapping: known keys get prefixed, unknown keys pass through
- Test that `total_steps` is NOT remapped (stays flat for x-axis)
- No wandb import needed — test the dict transform directly

### Integration tests (manual)
1. Run SAC + `--wandb` on CheetahRun (~200k steps), verify W&B dashboard shows 4 collapsible sections
2. Run PPO + `--wandb` on CartpoleBalance (~500k steps), verify PPO-specific metrics land in `actor/`
3. Run without `--wandb`, verify no breakage
4. `pytest tests/ -v` — all existing tests still pass

## Future Extensions (not in scope)

- Video logging (`wandb.Video`) — deferred, adds EGL dependency to training loop
- Custom panel layout templates — dashboard-side, not code-side
- W&B HP tuning agent — separate feature (see TODO.md)
