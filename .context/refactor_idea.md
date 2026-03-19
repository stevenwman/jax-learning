# Codebase Refactor Plan — Shared Utilities (Brax-style)

**Status:** Planned (not started)
**Decision date:** 2026-03-19
**Trigger:** 6 train scripts with 35% verbatim duplication, record_video.py only supports 2/6 algos

## Problem

Six train scripts (`train.py`, `train_sac.py`, `train_td3.py`, `train_fast_td3.py`, `train_fast_sac.py`, `train_fast_dsac.py`) share ~710 lines of identical code out of ~2,400 total (35%). Every cross-cutting change (eval, checkpointing, logging, env setup) must be applied 6 times.

**Concrete pain points:**
- `_save_checkpoint()` copy-pasted 6 times (differs only in meta dict keys)
- Episode return tracking: identical 7-line block × 6
- Eval trigger + checkpoint logic: identical 15-line block × 6
- Env setup: identical 6-line block × 6
- Argparse boilerplate: ~45 lines × 6
- `record_video.py` only supports PPO and SAC — TD3, FastTD3, FastSAC, FastDSAC can't be visualized
- `TrainConfig` has PPO-specific fields (`policy_hidden_dim`, `num_steps`, etc.) polluting shared config
- `select_action` signatures differ across algos (PPO takes full state, others take just params)

**What's actually unique per off-policy script:** ~45 lines (action selection with exploration noise, gradient update call, which metrics to log, optimizer construction). The off-policy scripts share ~80% identical structure.

## Research: How Other Frameworks Do It

| Approach | Example | Core idea | Tradeoff |
|---|---|---|---|
| **No abstraction** | CleanRL | Single-file per algo variant | Easy to read, impossible to maintain at scale |
| **Shared utilities** | Brax | Algo owns its loop, shares stateless helpers | Sweet spot for JAX — self-contained but DRY |
| **Light hierarchy** | Tianshou | Base class + 2 abstract methods, Trainer owns loop | Easy to add algos, must fit framework's shape |
| **Deep hierarchy** | SB3 | BaseAlgorithm → OnPolicy/OffPolicy → Algo | Powerful but 5+ layers to trace a forward pass |
| **Delegation** | Isaac Lab | No algos — wraps external RL libraries | Clean separation but no algo control |

### What each framework shares vs owns

| Component | Brax | SB3 | Tianshou | CleanRL | Us (now) |
|---|---|---|---|---|---|
| Training loop | Per-algo | Base class | Trainer class | Per-file | Per-script (6x dup) |
| Network construction | Factory fns | Policy classes + extractors | User-provided | Inline | Mixed (builders vs inline) |
| Evaluation | Shared `Evaluator` | Shared `evaluate_policy()` | Shared `Collector` | Per-file | Shared `evaluate()` but recording 2/6 |
| Checkpointing | Per-algo | Base class `save()`/`load()` | Trainer handles | Per-file | Per-script (6x dup) |
| Replay buffer | Shared module | Base class creates it | Shared `Buffer` | Per-file | Shared module |
| Episode tracking | Shared in `acting.py` | Base class | `Collector` | Per-file | Per-script (6x dup) |
| Config | Function params | Constructor args | Dataclasses | argparse | Dataclasses (coupling issues) |
| Algo interface | `train() → (make_policy, params, metrics)` | `BaseAlgorithm.learn()` | `_preprocess_batch()` + `_update_with_batch()` | None | Informal `init/update/select_action` |

## Decision: Brax-Style Shared Utilities

**NOT a Trainer class** (SB3/Tianshou path). Reasons:
- Too many layers for a research project — new people can't trace the flow
- On-policy vs off-policy loops are fundamentally different; forcing both into one Trainer means either splitting into OnPolicyTrainer/OffPolicyTrainer (SB3's path) or stuffing conditionals into one class
- We want each train script readable top-to-bottom without tracing through inheritance

**Instead: shared stateless utility functions** that each train script calls. The algo owns its loop; shared code handles infra. Train scripts shrink from ~400 lines to ~100 lines of algo-specific logic.

### New shared modules: `jax_rl/training/`

**1. `env_setup.py`** — env creation + wrapping
```python
def make_env(env_name, episode_length, num_envs, seed):
    """Create training + eval envs, JIT env.step, reset with seed."""
    # Currently 6 lines copy-pasted × 6 scripts
    ...
    return env, env_step, env_state, eval_env
```

**2. `checkpointing.py`** — save + load checkpoints
```python
def save_checkpoint(ckpt_dir, training_state, norm_state, cfg, algo_cfg,
                    algo_name, obs_dim, action_dim, metrics_log, resume):
    """Save meta.json + metrics.csv + actor_params.npy + orbax."""
    # Currently 46 lines × 6 scripts, differs only in meta dict keys
    ...

def load_checkpoint(ckpt_dir, training_state, norm_state):
    """Load orbax checkpoint + return start_step from metrics CSV."""
    ...

def load_actor_for_inference(ckpt_dir):
    """Load actor_params + meta.json for recording/eval. Algo-agnostic."""
    ...
```

**3. `episode_tracker.py`** — reward tracking + return computation
```python
class EpisodeTracker:
    """Track per-env episode rewards, compute completed returns."""
    def __init__(self, num_envs): ...
    def step(self, rewards, dones) -> None: ...
    @property
    def completed_returns(self) -> list[float]: ...
    def recent_stats(self, n=100) -> dict: ...  # avg, min, max, n_eps
```

**4. `metrics_logger.py`** — CSV writing + stdout formatting
```python
def log_step(total_steps, tracker, last_metrics, sps, buffer_size=None, min_buffer=None):
    """Print training step to stdout. Handles warmup vs training format."""
    ...

def make_metrics_row(total_steps, tracker, last_metrics, grad_steps, sps, elapsed):
    """Build dict for CSV row from current metrics."""
    ...
```

**5. `eval_runner.py`** — eval trigger logic (wraps existing `evaluate()`)
```python
def maybe_eval_and_checkpoint(
    select_action_fn, actor_params, eval_env, tracker,
    cfg, ckpt_dir, training_state, norm_state, algo_cfg, algo_name,
    metrics_log, last_eval_eps, key, resume,
):
    """Run eval + checkpoint if enough episodes have passed."""
    ...
```

### Lightweight algo Protocol

```python
from typing import Protocol, Any
import jax

class RLAlgorithm(Protocol):
    def init(self, key: jax.Array) -> Any: ...
    def update(self, state: Any, batch: dict) -> tuple[Any, dict]: ...
    def select_action(self, params: Any, obs: jax.Array, key: jax.Array,
                      deterministic: bool = False) -> jax.Array: ...
```

**Not a base class** — just a Protocol for type checking and documentation. Algos don't inherit from it. The only enforcement: `record_video.py` and `evaluate()` expect the `select_action` signature.

**PPO `select_action` fix:** PPO currently takes full `TrainingState` (needs critic for value baseline during collection). All off-policy algos take just `actor_params`. Fix: split into two functions:
- `select_action(actor_params, obs, key, deterministic) → action` — matches Protocol, used for eval/recording
- `collect_action(training_state, obs, key) → (action, log_prob, value)` — PPO-only, used during rollout collection for GAE

The value computation only matters during training. Eval and recording just need the action — no critic. This is what Brax does: `make_policy(params)` returns a function that only needs actor params.

### What a refactored train script looks like

```python
# train_sac.py (after refactor) — ~100 lines
from jax_rl.training.env_setup import make_env
from jax_rl.training.checkpointing import save_checkpoint, load_checkpoint
from jax_rl.training.episode_tracker import EpisodeTracker
from jax_rl.training.metrics_logger import log_step, make_metrics_row
from jax_rl.training.eval_runner import maybe_eval_and_checkpoint

def train(cfg, sac_cfg, seed=0, resume=None, jax_buffer=True):
    env, env_step, env_state, eval_env = make_env(cfg, seed)

    # SAC-specific: optimizer + algo init (~15 lines)
    optimizer = optax.adam(cfg.lr)
    alpha_optimizer = optax.adam(sac_cfg.alpha_lr)
    sac = SAC(sac_cfg, obs_dim, action_dim, optimizer, alpha_optimizer, cfg.gamma)
    training_state = sac.init(key)

    buffer = JaxReplayBuffer(obs_dim, action_dim, max_size=sac_cfg.buffer_size)
    tracker = EpisodeTracker(cfg.num_envs)

    for outer_step in range(...):
        # SAC-specific: action selection (~5 lines)
        if len(buffer) < sac_cfg.min_buffer_size:
            action = random_action(key, cfg.num_envs, action_dim)
        else:
            action = sac.select_action(training_state.actor_params, obs, key)

        env_state = env_step(env_state, action)
        buffer.add_batch(...)
        tracker.step(env_state.reward, env_state.done)

        # SAC-specific: gradient updates (~5 lines)
        if len(buffer) >= sac_cfg.min_buffer_size:
            for _ in range(sac_cfg.grad_updates_per_step):
                batch = buffer.sample(sac_cfg.batch_size, key=sample_key)
                training_state, last_metrics = sac.update(training_state, batch)

        log_step(total_steps, tracker, last_metrics, sps)
        maybe_eval_and_checkpoint(sac.select_action, ...)
```

### record_video.py becomes algo-agnostic

```python
def record(checkpoint, env_name=None, ...):
    meta, actor_params, norm_state = load_actor_for_inference(checkpoint)
    algo_name = meta["algo"]

    # Reconstruct algo from meta (for select_action)
    algo = build_algo_from_meta(meta)  # registry: "sac" → SAC, "td3" → TD3, etc.

    # Rollout — same for ALL algos
    for _ in range(max_steps):
        action = algo.select_action(actor_params, obs, key, deterministic=True)
        env_state = env_step(env_state, action)
```

### Config cleanup

- Move PPO-specific fields out of `TrainConfig` into `PPOConfig` (or a separate `PPOTrainConfig`)
- `TrainConfig` keeps only truly shared fields: env_name, num_envs, total_timesteps, lr, gamma, episode_length, eval settings
- Each algo config owns its own network/training params

## What this enables for Phase 4-6

- **Gymnasium adapter (Phase 4):** Change `env_setup.py` once, not 6 scripts
- **Wandb (Phase 3):** Add to `metrics_logger.py` once
- **DIAYN wrapping SAC (Phase 6):** Compose at algo level, shared infra doesn't care
- **Goal conditioning (Phase 6):** Add context to `make_env` and `select_action` Protocol
- **New algo:** Write ~100-line train script calling shared utilities, not copy-paste 400 lines

## Readability goals

The refactor should make the codebase easier to onboard on, not just less duplicated.

- **One screen per algo:** A new contributor should understand the full training flow by reading one ~100-line train script. No scrolling through 400 lines of boilerplate to find the 45 lines that matter.
- **Infra is invisible:** Shared utility calls (`save_checkpoint`, `tracker.step`, `log_step`) should read as one-liners that don't demand attention. The algo-specific logic (action selection, gradient updates, optimizer construction) should visually dominate the script.
- **Protocol as documentation:** `RLAlgorithm` with `init/update/select_action` in one file tells a new reader "every algo has these three methods" without reading any implementation.
- **Self-documenting utilities:** Each function in `jax_rl/training/` has a one-line docstring explaining what it replaces (e.g. "env creation — replaces the 6-line block duplicated across all train scripts").
- **Three-layer mental model:** env → training script → algo. A brief comment at the top of each train script or a project-level doc makes this explicit. The algo never touches the env; the training script mediates.
- **No hidden magic:** Unlike SB3 where you trace through 5 inheritance layers, every function call in a train script is a direct call you can ctrl-click to. No base class dispatch, no implicit hooks.

## Execution order

1. Create `jax_rl/training/` with shared utilities (extract from existing scripts)
2. Refactor one off-policy script (e.g., `train_sac.py`) to use them — validate it works
3. Refactor remaining off-policy scripts (TD3, FastTD3, FastSAC, FastDSAC)
4. Refactor `train.py` (PPO) — slightly different since on-policy, but checkpointing/eval/logging still shared
5. Fix `record_video.py` to be algo-agnostic via Protocol
6. Clean up `TrainConfig` (move PPO-specific fields)

## What NOT to do

- No Trainer base class — shared functions, not shared control flow
- No algorithm registry with string dispatch — explicit imports are fine for 6 algos
- No config system migration (Hydra, etc.) — dataclasses work, just clean up coupling
- No BaseAlgorithm ABC — Protocol is sufficient
