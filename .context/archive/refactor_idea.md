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

## Detailed execution spec — line-by-line mapping from train_sac.py

Using `train_sac.py` (400 lines) as the reference. Each block maps to a utility or stays.

### Block 1: Env vars (lines 12-14) → stays in each script
```python
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.7")
```
Must run before JAX import. Can't move to a utility without import-order issues. Keep in each script (2 lines).

### Block 2: _save_checkpoint (lines 43-90) → `jax_rl/training/checkpointing.py`
```python
def save_checkpoint(
    ckpt_dir: str,
    training_state,
    norm_state: NormalizationState,
    cfg: TrainConfig,
    algo_cfg,              # SACConfig, TD3Config, etc. — any dataclass
    algo_name: str,        # "sac", "td3", "fast_td3", etc.
    obs_dim: int,
    action_dim: int,
    metrics_log: list[dict],
    resume: str | None,
) -> None:
    """Save meta.json + metrics.csv + actor_params.npy + orbax checkpoint."""

def load_checkpoint(
    ckpt_dir: str,
    training_state,
    norm_state: NormalizationState,
) -> tuple[Any, NormalizationState, int]:
    """Load orbax checkpoint. Returns (training_state, norm_state, start_step)."""

def load_actor_for_inference(ckpt_dir: str) -> tuple[dict, dict, NormalizationState]:
    """Load meta.json + actor_params.npy for recording/eval. Returns (meta, actor_params, norm_state)."""
```

The only diff across scripts is `algo_name` and `algo_cfg` type — solved by passing both as args. The `algo_cfg_key` in meta.json derives from `algo_name` (e.g., "sac" → "sac_config").

### Block 3: env setup (lines 96-105) → `jax_rl/training/env_setup.py`
```python
def make_envs(
    cfg: TrainConfig,
    seed: int,
) -> tuple[Any, Callable, Any, Any, int, int]:
    """Create training env + eval env, JIT env.step, reset training env.

    Returns: (env, env_step, env_state, eval_env, obs_dim, action_dim)
    """
```

### Block 4: Print header (lines 111-124) → stays (algo-specific)
Each algo prints different params. Keep in script. ~15 lines.

### Block 5: Algo init (lines 127-145) → stays (algo-specific)
Optimizer construction, algo class instantiation, param counting. ~20 lines. This is the core algo-specific setup.

### Block 6: Norm state (lines 149-153) → `jax_rl/training/env_setup.py`
```python
def make_identity_norm_state(obs_dim: int) -> NormalizationState:
    """Identity norm state for off-policy algos (no obs normalization)."""
```
3 lines → 1-line call. Trivial but removes import of NormalizationState from train scripts.

### Block 7: Buffer init (lines 156-157) → stays (1 line, algo chooses buffer type)

### Block 8: Resume (lines 160-175) → `jax_rl/training/checkpointing.py::load_checkpoint()`
16 lines → 1-line call.

### Block 9: Episode tracking init (lines 178-180) → `jax_rl/training/episode_tracker.py`
```python
class EpisodeTracker:
    def __init__(self, num_envs: int):
        self.episode_rewards = np.zeros(num_envs)
        self.completed_returns: list[float] = []

    def step(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        """Update episode rewards and track completed episodes."""
        self.episode_rewards += rewards
        done_mask = dones.astype(bool)
        if done_mask.any():
            self.completed_returns.extend(self.episode_rewards[done_mask].tolist())
            self.episode_rewards[done_mask] = 0.0

    def recent_stats(self, n: int = 100) -> dict:
        """Compute avg/min/max return over last n completed episodes."""
        if not self.completed_returns:
            return {"avg": float("nan"), "min": float("nan"), "max": float("nan"), "n_eps": 0}
        recent = self.completed_returns[-n:]
        return {
            "avg": float(np.mean(recent)),
            "min": float(np.min(recent)),
            "max": float(np.max(recent)),
            "n_eps": len(self.completed_returns),
        }
```
Replaces lines 178-180 (init) + 237-244 (step) + 264-272 (stats). Total: ~20 lines across script → 1 init + 1 step call + 1 stats call.

### Block 10: Eval env + ckpt dir (lines 183-190) → partially in env_setup, dir stays
Eval env creation moves to `make_envs()`. Ckpt dir pattern stays (3 lines, algo name differs).

### Block 11: Training loop variables (lines 196-201) → stays (5 lines)

### Block 12: The training loop body (lines 203-310)

**Lines 207-217 — action selection:** STAYS (algo-specific: random warmup vs policy)
**Lines 220-225 — env step + truncation:** STAYS (2 lines, same everywhere but could extract)
**Lines 228-235 — buffer add:** STAYS (differs: np.asarray for numpy buffer, direct for JAX buffer)
**Lines 237-244 — episode tracking:** → `tracker.step(rewards, dones)` (1 line)
**Lines 246-257 — gradient updates:** STAYS (algo-specific: batch_size, grad_updates_per_step)
**Lines 259-310 — logging + eval + checkpoint:** → shared utilities

### Block 13: Logging (lines 259-310) → `jax_rl/training/metrics_logger.py`
```python
def log_training_step(
    total_steps: int,
    tracker: EpisodeTracker,
    last_metrics: dict,
    sps: int,
    buffer_size: int | None = None,
    min_buffer: int | None = None,
    extra_metrics: list[tuple[str, str]] | None = None,
) -> None:
    """Print training step. extra_metrics = [(label, key), ...] for algo-specific fields."""

def make_metrics_row(
    total_steps: int,
    tracker: EpisodeTracker,
    last_metrics: dict,
    grad_steps: int,
    sps: int,
    elapsed: float,
    extra_keys: list[str] | None = None,
) -> dict:
    """Build dict for CSV logging. extra_keys = algo-specific metric keys to include."""
```

The tricky part: each algo logs different metrics (SAC: entropy/alpha, TD3: no entropy, FastDSAC: q_var). Solved via `extra_metrics` param — the script passes which fields to print/log.

### Block 14: Eval + checkpoint trigger (lines 312-332) → `jax_rl/training/eval_runner.py`
```python
def maybe_eval_and_checkpoint(
    select_action_fn: Callable,
    actor_params,
    eval_env,
    tracker: EpisodeTracker,
    cfg: TrainConfig,
    ckpt_dir: str,
    training_state,
    norm_state,
    algo_cfg,
    algo_name: str,
    obs_dim: int,
    action_dim: int,
    metrics_log: list[dict],
    last_eval_eps: int,
    key: jax.Array,
    resume: str | None,
) -> tuple[int, jax.Array]:
    """Run eval + save checkpoint if enough episodes completed.
    Returns updated (last_eval_eps, key)."""
```

### Block 15: Final eval (lines 334-353) → same function, called unconditionally after loop

### Block 16: argparse (lines 356-399) → stays (algo-specific flags)
Could extract common args into a helper but not worth it until the refactor proves itself.

## What train_sac.py looks like after (estimated ~120 lines)

```
[2]  env vars
[5]  imports
[20] algo-specific setup (optimizer, SAC init, print header)
[3]  buffer init
[3]  ckpt dir
[5]  loop variables
[~80] training loop:
     [5]  action selection (algo-specific)
     [2]  env step
     [3]  buffer add
     [1]  tracker.step()
     [5]  gradient updates (algo-specific)
     [1]  log_training_step()
     [1]  maybe_eval_and_checkpoint()
[2]  final eval
[40] argparse
```
~120 lines vs current 400. The 280 lines that move are: _save_checkpoint (48), episode tracking (20), logging (50), eval trigger (20), env setup (10), resume (16) = ~164 lines of logic + their associated imports/variables.

## What NOT to do

- No Trainer base class — shared functions, not shared control flow
- No algorithm registry with string dispatch — explicit imports are fine for 6 algos
- No config system migration (Hydra, etc.) — dataclasses work, just clean up coupling
- No BaseAlgorithm ABC — Protocol is sufficient
