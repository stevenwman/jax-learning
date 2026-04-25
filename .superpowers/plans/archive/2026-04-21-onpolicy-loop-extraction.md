# On-Policy Loop Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract the shared on-policy training loop into `jax_rl/training/onpolicy_loop.py` so PPO and future on-policy variants (PPOContraction next) share orchestration. Matches the existing off-policy extraction at [jax_rl/training/offpolicy_loop.py](jax_rl/training/offpolicy_loop.py) pattern.

**Architecture:** One new helper `run_onpolicy_loop()` takes the algo + EnvBundle + hook callbacks + log fields. Each per-algo script becomes a thin wrapper (~100 lines) showing only algo-specific choices. First consumers: `train_ppo_fast.py` (refactor) and `train_ppo_contraction.py` (new, thin wrapper).

**Tech Stack:** JAX, Flax, Optax, MJX/Warp, pytest.

**Scope decisions:**
- Extract **only the fast/scan path** (`train_ppo_fast.py`). `train_ppo.py` (Python-loop, 442 lines) is slow-path, rarely used. Decide case-by-case whether to migrate or delete after this extraction.
- Reward augmentation hook designed specifically for ContractionPPO, kept general enough for future shaping / intrinsic motivation work.

## Hook API

```python
def run_onpolicy_loop(
    cfg: TrainConfig,
    ppo_cfg: PPOConfig,
    algo,                          # PPO or PPOContraction instance, constructed
    algo_name: str,                # "ppo" | "ppo_contraction"
    env_bundle: EnvBundle,
    *,
    # Hooks — all optional, None = baseline behavior:
    extra_rollout_fn: Callable | None = None,
    #   Signature: (training_state, env_state, key) -> RolloutExtras (a NamedTuple)
    #   Called INSIDE collect_step BEFORE env_step. Returns a NamedTuple whose
    #   fields are packed into StepData.extras and RolloutBatch.
    extras_type: type | None = None,
    #   Required when extra_rollout_fn is provided. The NamedTuple type the hook
    #   returns. The loop uses this type to stack per-step results via
    #   jax.tree.map and to splat kwargs into RolloutBatch(**extras._asdict()).
    reward_augment_fn: Callable | None = None,
    #   Signature: (training_state, env_state, extras) -> jax.Array of shape (num_envs,)
    #   Called AFTER env_step AFTER reward_scaling is applied. Added to reward.
    log_extra_keys: list[str] = (),
    seed: int = 0,
    resume: str | None = None,
    use_wandb: bool = False,
    wandb_project: str = "jax-rl",
) -> None: ...
```

**Extras typing — REQUIRED NamedTuple:** `extra_rollout_fn` must return a concrete `NamedTuple` subclass (declared by the consumer), not a dict. Rationale:
- `lax.scan` needs static pytree structure; a dict with dynamic-looking keys is a footgun.
- NamedTuple gives typed access (`extras.contraction_c` not `extras["contraction_c"]`).
- Each consumer defines its own class and passes it via `extras_type=...`.
- Example consumer (ContractionPPO, declared in `train_ppo_contraction.py`):
  ```python
  class ContractionExtras(NamedTuple):
      c: jax.Array       # (num_envs, constraint_dim)
      c_dot: jax.Array   # (num_envs, constraint_dim)
  ```

Both hooks are **compile-time constants** — the loop's top-level `@jax.jit` closes over them. Changing the hook between runs retraces. None = fast-path baseline.

---

## File Structure

**Create:**
- `jax_rl/training/onpolicy_loop.py` — `run_onpolicy_loop()` + `StepData` NamedTuple (extended with `extras: dict`)
- `tests/test_onpolicy_loop.py` — CPU stub-env smoke test (matches off-policy pattern)
- `train_ppo_contraction.py` — thin wrapper building PPOContraction + contraction hooks, calling `run_onpolicy_loop`

**Modify:**
- `train_ppo_fast.py` — collapse to thin wrapper (target ~100 lines)
- `jax_rl/training/__init__.py` — export `run_onpolicy_loop`

**Do NOT touch:**
- `jax_rl/algos/ppo.py`, `jax_rl/algos/ppo_contraction.py` — algorithms already done
- `jax_rl/algos/sac.py`, `td3.py`, `fast_*.py` — unrelated
- `train_ppo.py` — Python-loop path, out of scope

---

## Task 1: Extract `run_onpolicy_loop` scaffold

**Files:**
- Create: `jax_rl/training/onpolicy_loop.py`

Copy the banner/W&B/timestamp/TrainContext setup from `train_ppo_fast.py:66-156` verbatim. Stop before the `_collect` definition. Commit as a scaffold with a `raise NotImplementedError` stub.

- [ ] **Step 1: Copy setup code** (env unpack, banner, W&B, optimizer, algo.init, norm_state, resume, tracker, ckpt_mgr, ctx)
- [ ] **Step 2: Write CPU stub test that imports `run_onpolicy_loop` and confirms the stub raises**
- [ ] **Step 3: Commit**

## Task 2: Port `_collect` with extras hook

**Files:**
- Modify: `jax_rl/training/onpolicy_loop.py`

Port `_collect` from `train_ppo_fast.py:191-275`. Modifications:

1. `StepData` gets a new `extras: dict` field (may be empty dict when `extra_rollout_fn=None`).
2. Inside `collect_step`: if `extra_rollout_fn` is not None, call it on `env_state` BEFORE `env_step` (off-by-one safe — matches ref `process_env_step` behavior). Pack result into `step_data.extras`.
3. If `reward_augment_fn` is not None, call it AFTER `env_step` and `reward_scaling`. Add to reward before emitting `step_data`.

- [ ] **Step 1: Write failing test** — CPU stub env, run 4 steps with a dummy `extra_rollout_fn` returning `{"foo": jnp.ones((num_envs, 3))}`. Assert `step_data.extras["foo"].shape == (4, num_envs, 3)`.
- [ ] **Step 2: Implement `_collect` with hook invocation**
- [ ] **Step 3: Implement reward_augment path** — separate test with `reward_augment_fn` returning constant 0.5; assert rewards increased by 0.5 after scaling
- [ ] **Step 4: Commit**

## Task 3: Port PPO update + training loop body

**Files:**
- Modify: `jax_rl/training/onpolicy_loop.py`

Port the outer `for iteration in range(...)` loop from `train_ppo_fast.py:297-500+`. Key change: build `RolloutBatch` with extras forwarded via `**rollout.extras` kwargs. Since `RolloutBatch` already accepts optional `contraction_c`/`contraction_c_dot` via NamedTuple defaults (Task 4 of contraction plan, already shipped), this is just a splat.

- [ ] **Step 1: Write failing test** — stub env, 2 iterations, assert `training_state.actor_params` changes
- [ ] **Step 2: Port loop body**
- [ ] **Step 3: Port eval, checkpointing, W&B logging**
- [ ] **Step 4: Commit**

## Task 4: Migrate `train_ppo_fast.py` to thin wrapper

**Files:**
- Modify: `train_ppo_fast.py`

Replace inline training code with `run_onpolicy_loop(cfg, ppo_cfg, algo=ppo, algo_name="ppo", env_bundle=env_bundle, ...)`. Target ~100 lines (matches `train_fast_sac.py`).

- [ ] **Step 1: Refactor train function**
- [ ] **Step 2: Run `uv run pytest tests/test_ppo_setup.py tests/test_onpolicy_loop.py -v`** — must all pass
- [ ] **Step 3: Run one real iteration of Go2 training** (`uv run python train_ppo_fast.py --env Go1JoystickFlatTerrain --total-timesteps 100_000 --seed 0`) — verify completes without crash
- [ ] **Step 4: Commit**

## Task 5: Create `train_ppo_contraction.py`

**Files:**
- Create: `train_ppo_contraction.py`

Thin wrapper (~100 lines) that:
1. Populates `constraint_dim` from env obs after `make_env_bundle`.
2. Validates `cfg.ppo.contraction` is set.
3. Builds `PPOContraction` instance.
4. Defines hooks:
   - `extra_rollout_fn`: extracts `contraction_state` from obs, splits into `c, c_dot`.
   - `reward_augment_fn`: calls `algo.compute_contraction_reward(metric_params, c, c_dot)`.
5. Calls `run_onpolicy_loop(..., extra_rollout_fn=..., reward_augment_fn=...)`.

- [ ] **Step 1: Write failing end-to-end test** — GPU-required, short bongo handstand run (5 iters, 4 envs), assert `metric_params` change + no NaN in metrics
- [ ] **Step 2: Implement the wrapper**
- [ ] **Step 3: Verify test passes** — also verify `train_ppo_fast.py` baseline on same env+seed is NOT regressed
- [ ] **Step 4: Commit**

## Task 6: Docs sync

- [ ] Update `.context/TODO.md` — mark on-policy extraction done
- [ ] Add `.context/journals/2026-04-21.md` — extraction + contraction wrapper shipped
- [ ] Update `docs/reference/training-loop.md` with on-policy pattern

---

## Risks

- **JIT retrace on hook change:** acceptable (one-time cost per training run)
- **Extra obs key plumbing:** env must emit `contraction_state`. Already gated behind `observe_contraction` config flag (Task 5 of contraction plan). No regression for baseline.
- **StepData.extras as typed NamedTuple:** required, not dict. Consumer declares the type, passes via `extras_type=...`. `lax.scan` sees a static pytree structure automatically.
- **train_ppo.py (Python-loop) left unmigrated:** intentional. Will be re-evaluated after fast-path migration stabilizes.
