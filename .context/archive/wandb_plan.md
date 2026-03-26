# W&B (Weights & Biases) Integration Plan

## Context

All training metrics are already centralized through `metrics_logger.py` (stdout + CSV) and `eval_runner.py` (eval metrics + Q diagnostics). W&B adds cloud-hosted experiment tracking — real-time curves, config comparison, video artifacts — without changing the training logic. Prerequisite for DIAYN, where we'll compare many runs across seeds and hyperparameters.

## Approach: Thin wrapper, 3 integration points

No separate W&B module — just add `wandb.log()` calls alongside existing CSV logging.

### 1. Init (in each train script's `train()` function)
```python
if use_wandb:
    import wandb
    wandb.init(project="jax-rl", name=f"{algo}_{env}_seed{seed}", config={...meta dict...})
```

### 2. Step metrics (after `make_metrics_row()`)
```python
if wandb.run:
    wandb.log(metrics_row, step=total_steps)
```

### 3. Eval metrics (after eval completes)
```python
if wandb.run:
    wandb.log(eval_metrics, step=total_steps)
```

## Files to modify
- `train_offpolicy.py` — add `--wandb` flag, init, log step + eval metrics
- `train_ppo_fast.py` — same pattern
- `train_ppo.py` — same pattern
- `jax_rl/training/metrics_logger.py` — add `wandb_log()` helper (checks `wandb.run` before logging)
- `pyproject.toml` — add `wandb` as optional dependency

## What NOT to do
- Don't create a separate WandbLogger class — overkill for `wandb.log(dict)`
- Don't make wandb required — optional, gated by `--wandb` flag
- Don't log every gradient step — log at `log_interval` (same as CSV)
- Don't replace CSV — W&B is supplementary, CSV stays for offline access

## CLI
```bash
uv run python train_offpolicy.py --algo sac --env CheetahRun --wandb
uv run python train_offpolicy.py --algo sac --env CheetahRun --wandb --wandb-project my-project
```

## Config logged to W&B
Reuse the `meta` dict from `checkpointing.py` — already has env_name, algo, all hyperparams.

## Video logging (future, not in this PR)
`record_video.py` can log MP4 as `wandb.Video` artifact. Deferred.

## Verification
1. `uv run python train_offpolicy.py --algo sac --env CheetahRun --total-timesteps 200000 --wandb` — check wandb.ai dashboard shows curves
2. Without `--wandb` — verify no import error, no behavior change
3. Run tests — nothing should break (wandb is optional)
