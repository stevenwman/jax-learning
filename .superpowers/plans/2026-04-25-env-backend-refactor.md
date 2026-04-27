# Env-Backend Refactor — gym + IsaacLab + MJX unified pipeline

**Date:** 2026-04-25
**Author:** session
**Goal:** Make the training pipeline env-backend-agnostic so we can train SAC/PPO on (a) MJX/Warp envs (current), (b) gym envs (pusht, dm_control), and (c) IsaacLab envs (PyTorch GPU) through one entrypoint and one set of training harnesses. Labmate-portable.

**Non-goals:** No IsaacLab integration testing this pass — keep IsaacLab code Protocol-conformant + smoke-able but don't validate against a real install. Plan is theoretical for the IsaacLab branch.

**Building from:** rsl_rl conventions for vectorized env interface (`num_envs`, `num_obs`, `num_actions`, batched `step`/`reset`).

---

## Current state (verified via codebase audit)

What's already env-agnostic:
- `run_offpolicy_loop()` — **Python loop**, not lax.scan. Calls `env_step(state, action)` directly. No MJX coupling in the loop body itself.
- `JaxReplayBuffer.add_batch()` — auto-converts numpy → JAX via `jnp.asarray`. Supports batched `(N, dim)` ingest.
- `ObsPipeline` — dict obs structure, no MJX coupling.
- `TrainConfig` / `SACConfig` — env-agnostic fields.

What IS MJX-coupled (need to fix):
- `jax_rl/training/env_setup.py:make_envs()` — hardcodes `pg_registry.load()`.
- `jax_rl/training/env_setup.py:_make_nan_safe_step()` — JIT-wraps env.step assuming JAX in/out. Needed for MJX physics, but not for gym/IsaacLab.
- `scripts/record_video.py` — calls `pg_registry.load(env_name).render()`. Backend-specific render.

**Important:** `scripts/train_ppo.py` is **already** a Python-loop variant for non-JIT-able envs (docstring says "for envs that require Python-level operations per step (e.g., MJWarp)"). No PPO pyloop refactor needed — gym/IsaacLab backends route here, MJX routes to `train_ppo_fast.py` (which uses scan via `make_collect`).

What's a duplicate the refactor should eventually kill (out of scope this pass):
- `scripts/train_pusht.py` — 476 lines duplicating numpy buffer, action scale, normalize, action repeat, training loop, eval. Migration deferred per user.

---

## Design decisions (locked)

**Q1 (scope):** gym + IsaacLab + MJX. Brax not in scope.

**Q2 (vectorization):**
- gym: `gym.vector.AsyncVectorEnv` always, N capped at `os.cpu_count()` with one-line warning if cfg.num_envs higher. `SyncVectorEnv` auto-selected when N=1 (debug).
- IsaacLab: native batched (env's own num_envs at construction).
- MJX: unchanged (vmap'd N envs in JAX).

**Q3 (loop variants):**
- Off-policy: **single loop** (already Python-loop, env-agnostic). Just dispatch on env_step callable type — JIT'd for MJX, plain Python for gym/IsaacLab.
- On-policy: **two variants already exist** — `train_ppo_fast.py` (lax.scan via `make_collect`, MJX fast path) + `train_ppo.py` (Python loop, for non-JIT-able envs). Just need to route bundle.backend_kind to the right script.

**Q4 (algo coupling):**
- `EnvBundle` Protocol with `backend_kind`, `num_envs`, `num_obs`, `num_actions`, `num_critic_obs`, `has_privileged`. Same shape as rsl_rl's VecEnv.
- Replay buffer: existing one. Convert at ingest (numpy → jnp inline, torch → jnp via `jax.dlpack`).
- Critic obs: `has_privileged` flag on bundle; default `critic_obs = obs` aliasing if absent.
- Render: per-backend `bundle.render(state, env_idx) → np.uint8 | None`. Recording wrapper backend-agnostic.

**Q5 (pusht migration):** Defer until refactor + gym backend land. Keep `train_pusht.py` working in the meantime.

---

## File-by-file plan

### Phase 0 — Define the Protocol (no behavior change)

**New file:** `jax_rl/training/env_bundle.py`

```python
from typing import Protocol, Literal, Any
from dataclasses import dataclass

BackendKind = Literal["mjx", "gym", "isaaclab"]

@dataclass
class EnvBundle:
    backend_kind: BackendKind
    env: Any                # backend-specific handle
    env_step: Any           # callable (state, action) -> state
    env_state: Any          # initial batched state
    eval_env: Any           # single-env or single-batch eval handle
    obs_dim: int
    action_dim: int
    critic_obs_dim: int | None    # None when not privileged
    has_privileged: bool
    dict_obs: bool
    num_envs: int
    key: Any                # JAX RNG (mjx) or seed int (gym/isaaclab)

    def render(self, state, env_idx: int = 0) -> "np.ndarray | None": ...
```

This is what `make_env_bundle()` already returns minus `backend_kind`/`num_envs`/`render`. Add fields, current usage unaffected.

**Touch:** `jax_rl/training/env_setup.py` — move EnvBundle dataclass to new file, add fields, set `backend_kind="mjx"`, `num_envs=cfg.num_envs`, `render=lambda s, i: env.render(...)` for MJX path.

**Verify:** all existing call sites still work. Run smoke `train_sac --env Go2WarpJoystickFlat`.

---

### Phase 1 — Backend registry

**New file:** `jax_rl/training/env_backends/__init__.py` — registry.

```python
from typing import Callable
BACKEND_REGISTRY: dict[str, Callable[[TrainConfig, int], EnvBundle]] = {}

def register_backend(name: str, builder):
    BACKEND_REGISTRY[name] = builder

def detect_backend(env_name: str) -> str:
    # env_name format: "Go2WarpJoystickFlat" → mjx
    # "PushT" / "DmcCheetahRun" → gym
    # "IsaacLab/CartpoleDirect" → isaaclab (prefix)
    if env_name.startswith("IsaacLab/"): return "isaaclab"
    if env_name in MJX_REGISTRY: return "mjx"
    if env_name in GYM_REGISTRY: return "gym"
    raise ValueError(f"Unknown env {env_name!r} — not registered in any backend")
```

**New file:** `jax_rl/training/env_backends/mjx_backend.py`

Move current `make_envs()` body here as `make_mjx_env_bundle(cfg, seed) → EnvBundle`. Just relocates code, no behavior change.

**Refactor:** `jax_rl/training/env_setup.py:make_env_bundle()` — becomes a thin dispatcher:

```python
def make_env_bundle(cfg, seed):
    backend = detect_backend(cfg.env_name)
    return BACKEND_REGISTRY[backend](cfg, seed)
```

**Verify:** smoke `train_sac --env Go2WarpJoystickFlat`. Output identical to pre-refactor.

---

### Phase 2 — Gym backend

**New file:** `jax_rl/training/env_backends/gym_backend.py`

Build a `GymEnvBundle` adapter:

1. Construct `gym.vector.AsyncVectorEnv([make_env]*N)` where N = `min(cfg.num_envs, os.cpu_count())`. Warn if capped.
2. Wrap to expose `state`-like struct: a thin dataclass `GymState(obs, reward, done, truncation, info)`.
3. Define `env_step(state, action) -> state` as a Python closure that:
   - Calls `vec_env.step(action_np)` where `action_np = np.asarray(action)` (handles JAX → numpy)
   - Returns `GymState(...)` with numpy fields
4. `eval_env`: separate single-env `make_env()`. Single rollout for eval.
5. `render(state, env_idx) → np.uint8`: call `vec_env.envs[env_idx].render()` (works for SyncVecEnv; for AsyncVecEnv we need a separate eval env render).

**Gym env registry:** `GYM_REGISTRY = {"PushT": make_pusht_env, "DmcCheetahRun": ...}`. Each builder returns a thawed `gym.Env` factory.

**Special handling for pusht specifically:**
- Apply `NormalizeObsWrapper` and `ActionRepeatWrapper` (port from train_pusht.py)
- Apply `gym.wrappers.TimeLimit(..., max_episode_steps=300)` (critical, see existing lesson)
- Action scaling [-1,1] → gym space — handle in env_step, not in policy code

**Verify:** smoke `train_sac --env PushT` should reproduce within noise of train_pusht.py results (89% sto cov on T at 2M steps with same HPs).

---

### Phase 3 — IsaacLab backend (DEFERRED 2026-04-26 — no install/test target)

**Status:** plan only. Stubs rot without a smoke target; revisit when (a) a labmate has a concrete IsaacLab env to port, or (b) you install IsaacLab locally for a real test. Sketch below kept as the design-of-record.



**New file:** `jax_rl/training/env_backends/isaaclab_backend.py`

Implement Protocol but with deferred imports + clear NotImplementedError messages:

```python
def make_isaaclab_env_bundle(cfg, seed):
    try:
        from isaaclab.envs import IsaacEnv  # or whatever the actual import is
        import torch
    except ImportError:
        raise ImportError(
            "IsaacLab backend requires `pip install isaaclab` and a CUDA install. "
            "See https://isaac-lab.github.io for setup."
        )

    env = IsaacEnv(cfg.env_name.removeprefix("IsaacLab/"), num_envs=cfg.num_envs)

    def env_step(state, action):
        # action is JAX array (N, action_dim)
        action_torch = jax.dlpack.to_dlpack(action)
        action_torch = torch.utils.dlpack.from_dlpack(action_torch)
        obs_t, r_t, term_t, trunc_t, info = env.step(action_torch)
        # back to JAX
        obs = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(obs_t))
        ...
        return IsaacState(obs, r, term, trunc, info)

    return EnvBundle(backend_kind="isaaclab", ...)
```

Document at top:
- jax↔torch via dlpack zero-copy on same GPU
- Stay on GPU, do NOT call `.cpu()` in hot path
- Their config is Hydra; we accept env_name + num_envs only, ignore Hydra
- Their `step` returns dict-like obs typically — flatten for now, address dict obs in a follow-up

**Verify:** import-protected; doesn't run unless someone installs IsaacLab. Just verifies the interface compiles.

---

### Phase 4 — Wire PPO scripts to backend dispatch

**Already exists:** `scripts/train_ppo.py` is the Python-loop variant (`for step in range(num_steps): env_state = env_step(...)`). Built originally for MJWarp envs.

**Refactor:** both `train_ppo.py` and `train_ppo_fast.py` need to:
1. Switch `make_envs()` → `make_env_bundle()` (route through registry).
2. Validate bundle.backend_kind matches script:
   - `train_ppo_fast.py` requires `backend_kind == "mjx"` (uses scan).
   - `train_ppo.py` works for any backend.
3. Optional: add a thin `scripts/train_ppo_dispatch.py` that picks the right one based on env name. Or just document the rule and let users pick.

**Verify:** PPO smoke on Go2 (existing path unchanged) + PPO on PushT via `train_ppo.py` (new gym path) to validate the gym backend works for on-policy too.

---

### Phase 5 — Recording

**Touch:** `scripts/record_video.py`

Currently: `pg_registry.load(env_name).render()` — MJX-only.

Add backend dispatch:
- For `backend == "mjx"`: existing path.
- For `backend == "gym"`: rebuild gym env, replay actions from rollout, call `env.render(mode="rgb_array")` per step, ffmpeg.
- For `backend == "isaaclab"`: skip / NotImplementedError (their render is dev-time only).

**Verify:** `record_video.py --env PushT --ckpt ...` produces an mp4.

---

### Phase 6 — Pusht migration (deferred)

After Phase 2 lands and `train_sac --env PushT` reproduces train_pusht.py results: delete `scripts/train_pusht.py`. Update README + journal note.

Done last so we have a reference point during the refactor.

---

## Risks & mitigations

1. **Gym AsyncVecEnv pickle overhead** — pusht is fast (~100µs/step), IPC could bottleneck. Mitigate: SyncVecEnv option for small N, profile before declaring done.
2. **dlpack version mismatch jax↔torch** — JAX dlpack API changes between releases. Pin jax + torch versions in IsaacLab branch dependency note.
3. **Gym dict obs envs** — some gym envs (manipulation tasks) return dict obs. Handle via flattening in v1, dict-obs support in follow-up.
4. **Replay buffer with vectorized eval** — existing buffer assumes batched ingest; eval rollouts are single-env. Fine, just N=1.
5. **TimeLimit semantics differ** — gym `truncated` flag; SAC bootstrap needs to handle. Existing offpolicy_loop already splits done/truncation — verify it correctly propagates from gym.

---

## Test strategy per phase

- Phase 0: existing tests must pass unchanged.
- Phase 1: new test `test_backend_dispatch.py` — assert correct backend chosen for each env name.
- Phase 2: new test `test_gym_backend.py` — instantiate PushT bundle, call reset/step, verify shapes. Smoke training run separate (non-CI).
- Phase 3: new test `test_isaaclab_protocol.py` — verify Protocol compliance via `isinstance` checks; skip step/reset if isaaclab not installed.
- Phase 4: new test `test_collect_pyloop.py` — collect N steps, verify rollout dict shape matches scan variant.
- Phase 5: manual smoke (record_video.py).
- Phase 6: end-to-end pusht reproduction (sto cov at 2M steps within ±2pp of train_pusht.py).

---

## Estimated effort

- Phase 0 — 30 min (rename + dataclass tweaks)
- Phase 1 — 1.5 hr (registry + dispatch)
- Phase 2 — 4-6 hr (gym backend, port pusht wrappers, validate)
- Phase 3 — 2 hr (skeleton + Protocol verification, no smoke)
- Phase 4 — 1 hr (route train_ppo.py / train_ppo_fast.py through bundle dispatch — Python-loop PPO already exists)
- Phase 5 — 2 hr (recording dispatch)
- Phase 6 — 1 hr (delete + doc) — separate session

Total: 11-13 hr of focused work. Can be split across sessions, each phase merge-able independently.

---

## Open questions for user before starting

1. **rsl_rl namespace** — do you want `EnvBundle` matching rsl_rl's `VecEnv` exact field names (`num_obs`, `num_actions`) or our existing names (`obs_dim`, `action_dim`)? Convenience for labmates vs continuity for our codebase.
2. **Phase 3 IsaacLab depth** — do you want a stub-only Protocol, or a one-shot IsaacLab Cartpole demo end-to-end as proof-of-life (would need IsaacLab install on your machine, plus a basic env)?
3. **Phase order** — do Phase 4 (PPO pyloop) before Phase 2 (gym)? Currently planned 2→4 because gym backend has no JAX scan dependency, so easier validation. PPO refactor can wait.
