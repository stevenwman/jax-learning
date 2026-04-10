# Training Script Cleanup — Design Spec

4 independent refactors to reduce conditionals and improve readability in training scripts.

## 1. ObsPipeline

**File:** `jax_rl/training/obs_pipeline.py`

Stateless class that encapsulates all obs extraction, normalization, and buffer setup logic. Built once from config, eliminates ~8 branches from every training loop.

```python
class ObsPipeline:
    def __init__(self, dict_obs, has_privileged, use_obs_norm, n_frame_stack, obs_norm_eps):
        ...

    def get_obs(self, env_obs) -> jax.Array:
        """dict['state'] or passthrough."""

    def get_critic_obs(self, env_obs) -> jax.Array:
        """dict['privileged_state'] or fallback to get_obs."""

    def update_stats(self, obs, norm_state) -> NormalizationState:
        """Update norm stats. No-op if obs_norm disabled. Handles frame stacking."""

    def normalize_for_action(self, obs, norm_state) -> jax.Array:
        """Normalize obs for actor. Handles frame stacking."""

    def normalize_batch(self, batch, norm_state) -> dict:
        """Normalize obs+next_obs in batch. Sets critic_obs keys if not privileged."""

    def make_buffer(self, obs_dim, action_dim, buffer_size) -> JaxReplayBuffer:
        """Create buffer with frame_stack_config + extra_obs_dims."""

    def make_obs_norm_fn(self, norm_state) -> callable:
        """Return obs normalize function for eval_runner."""
```

Immutable — norm_state passed in, returned. No internal state.

## 2. TrainContext

**File:** `jax_rl/training/train_context.py`

Dataclass bundling training state that gets threaded through eval/checkpoint functions.

```python
@dataclass
class TrainContext:
    cfg: TrainConfig
    algo_cfg: Any
    algo_name: str
    ckpt_dir: str
    obs_dim: int
    action_dim: int
    metrics_log: list[dict]
    ckpt_mgr: CheckpointManager
    resume: str | None
```

`training_state` and `norm_state` stay external (mutated every step).

Update `maybe_eval_and_checkpoint` and `final_eval_and_checkpoint` in `eval_runner.py` to accept `ctx: TrainContext` instead of 10+ individual args.

## 3. Algo-owned policy_delay

**Files:** `jax_rl/algos/td3.py`, `jax_rl/algos/fast_td3.py`

Modify `update()` to always return the most recent real `actor_loss` in metrics, even on critic-only steps. The algo tracks this internally (add `last_actor_loss` field to TrainingState). Training scripts unconditionally do `last_metrics = step_metrics`.

## 4. CLI override utility

**File:** `jax_rl/training/cli_utils.py`

```python
CFG_FIELDS = {"num_envs", "total_timesteps", "lr", "reward_scaling",
              "episode_length", "n_frame_stack", "action_delay_ms", "reset_mode"}
ALGO_FIELDS = {"batch_size", "grad_updates_per_step", "buffer_size",
               "exploration_noise_std", "target_entropy_scale"}
BOOL_CFG = {"domain_rand", "obs_normalization"}

def apply_cli_overrides(args, cfg, algo_cfg):
    """Apply CLI args to config dataclasses. Returns (cfg, algo_cfg)."""
```

Declarative mapping. Each script calls `cfg, algo_cfg = apply_cli_overrides(args, cfg, algo_cfg)` instead of 15+ `if` lines.

## Testing

### Unit tests (`tests/test_obs_pipeline.py`)
- `test_flat_obs_passthrough` — non-dict obs returns unchanged
- `test_dict_obs_extracts_state` — dict obs returns "state" key
- `test_critic_obs_privileged` — returns "privileged_state" when present
- `test_critic_obs_fallback` — falls back to actor obs when no privileged
- `test_update_stats_noop_when_disabled` — returns norm_state unchanged when use_obs_norm=False
- `test_update_stats_updates_when_enabled` — norm_state.count increases
- `test_normalize_for_action_noop` — passthrough when disabled
- `test_normalize_for_action_frame_stack` — normalizes per-frame with shared stats
- `test_normalize_batch_sets_critic_keys` — non-privileged batch gets critic_obs = obs
- `test_normalize_batch_preserves_critic_keys` — privileged batch keeps existing critic_obs
- `test_make_buffer_frame_stack` — buffer created with FrameStackConfig when n_frame_stack > 1
- `test_make_buffer_extra_obs_dims` — buffer created with critic_obs dim when privileged

### Unit tests (`tests/test_train_context.py`)
- `test_train_context_creation` — all fields set correctly
- `test_eval_runner_with_context` — `maybe_eval_and_checkpoint` works with TrainContext arg

### Unit tests (`tests/test_policy_delay.py`)
- `test_td3_update_returns_actor_loss_on_critic_step` — actor_loss in metrics even when policy_delay skips actor update
- `test_td3_update_returns_real_actor_loss_on_actor_step` — actor_loss matches actual loss on actor update steps
- `test_fast_td3_same_behavior` — same tests for FastTD3

### Unit tests (`tests/test_cli_utils.py`)
- `test_apply_overrides_none_ignored` — None args don't override config
- `test_apply_overrides_sets_values` — non-None args update config
- `test_apply_overrides_bool_flags` — boolean flags (domain_rand, obs_norm) handled
- `test_unknown_args_ignored` — args not in mapping don't crash

### Integration: smoke tests on all training scripts
After refactoring, run each script for 50k steps on CartpoleBalance to verify no regressions:
```bash
uv run python train_sac.py --env CartpoleBalance --total-timesteps 50000 --num-envs 4
uv run python train_td3.py --env CartpoleBalance --total-timesteps 50000 --num-envs 4
uv run python train_fast_sac.py --env CartpoleBalance --total-timesteps 50000 --num-envs 4
uv run python train_fast_td3.py --env CartpoleBalance --total-timesteps 50000 --num-envs 4
uv run python train_flashsac.py --env CartpoleBalance --total-timesteps 50000 --num-envs 4
```

### Integration: Go2 with DomainRandWrapper
Verify the obs pipeline + DR integration doesn't break:
```bash
uv run python train_fast_sac.py --env Go2WarpJoystickFlat --total-timesteps 50000 --num-envs 4 --reset-mode per_step
```

### Regression: existing test suite
```bash
uv run python -m pytest tests/ -v --timeout=120
```
All existing tests must pass. No regressions.

## Scope

- All 4 standalone scripts (train_sac/td3/fast_sac/fast_td3) + train_flashsac
- eval_runner.py (TrainContext integration)
- td3.py, fast_td3.py (policy_delay)
- No behavior changes. Pure refactor.
