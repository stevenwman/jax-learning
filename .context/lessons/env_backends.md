# Env Backend Contract

> Codebase supports two execution modes for env↔algo interaction. This
> doc nails down what each backend must provide, what consumers can
> rely on, and which scripts route to which path. Audited 2026-04-26
> after the env-backend-refactor merge (commit `adf5a3a`).

---

## §1. Two execution modes

The env-backend layer separates "JAX-jittable env" from "Python env".
Both modes use the same outer Python training loop; what differs is
whether `env_step(state, action)` is a JAX op or plain Python.

**Mode A — Fully JAX-jittable env (`backend_kind = "mjx"`, future "warp"):**
- `env_step` is a JAX op (jit-compatible). All state arrays are
  `jax.Array`.
- Outer loop is currently Python in `offpolicy_loop`/`train_ppo`, but
  could be wrapped in `lax.scan` for full-scan training. PPO-fast
  already leverages full-scan collection inside its rollout.
- Buffer / norm / algo / eval all jit. Vmap-style parallelism via the
  underlying physics engine (MJX, future Warp).

**Mode B — Python env (`backend_kind = "gym"`, future "isaaclab"):**
- `env_step` is plain Python. State arrays are `numpy.ndarray`.
- Outer loop MUST be Python. Inner ops (action sampling, gradient
  update, eval rollout) jit'd individually with auto numpy↔jax
  conversion at the boundary.
- Vector-env parallelism via `gym.vector.{Sync,Async}VectorEnv` capped
  at `os.cpu_count()`.

The shared `offpolicy_loop.run_offpolicy_loop` is structured as
**Python outer + jit inner**, which works unchanged for both modes.
That's the canonical path. Mode-A-specific accelerations (full-scan
collection) are explicit, gated, and limited to PPO-fast for now.

---

## §2. EnvBundle contract

Defined in `jax_rl/training/env_bundle.py`. Each backend builder
returns a fully-populated `EnvBundle`; consumers read from these fields
and ONLY these fields.

### Required fields

| Field | Type | Mode A (mjx) | Mode B (gym) |
|---|---|---|---|
| `env` | Any | vmap'd MJX env | `gym.vector.VectorEnv` |
| `env_step` | `Callable[(state, action) -> state']` | jit op | Python |
| `env_state` | Any | flax.struct (jax arrays) | `GymState` (numpy arrays) |
| `eval_env` | Any | same MJX env (single rollout via vmap) | `gym.vector.SyncVectorEnv` (size 1) |
| `obs_dim` | `int` | from env spec | from `obs.shape[-1]` after reset |
| `action_dim` | `int` | from env spec | from `single_action_space.sample().shape` |
| `critic_obs_dim` | `int \| None` | privileged dim or None | privileged dim or None |
| `has_privileged` | `bool` | dict-obs has `"privileged_state"` | dict-obs has `"privileged_state"` |
| `dict_obs` | `bool` | obs is a dict | obs is a dict |
| `key` | `jax.Array` | env RNG (mjx) | policy RNG only (gym envs use their own seed at reset) |
| `backend_kind` | `Literal["mjx","gym","isaaclab"]` | `"mjx"` | `"gym"` |
| `num_envs` | `int` | as configured | clamped to `os.cpu_count()` |
| `render_fn` | `Optional[Callable]` | `None` (record_video has its own MJX path) | `lambda s, idx: eval_env.envs[idx].render()` |

### env_step invariants (both modes)

`env_step(state, action) → state'` must:

1. **Accept JAX-array action.** Mode B converts at the boundary
   (`gym_backend.env_step:242`).
2. **Return a state with attributes:** `.obs`, `.reward`, `.done`,
   `.info` (dict).
3. **Populate `info["truncation"]`.** Mode A reads from MJX env's own
   info dict; Mode B sets `truncation = truncated` (gym 5-tuple). If
   the env genuinely has no timeout, populate with zeros — `cfg.handle_truncation`
   gates whether the loss mask uses it.
4. **`done = terminated | truncated`** (Brax convention). Both backends
   conform; check before adding new ones.
5. **Auto-reset.** Mode A via `AutoResetWrapper`; Mode B is native to
   `gym.vector.VectorEnv`. Obs returned at terminal step is the *reset*
   obs (NOT the terminal obs). Algos that need the terminal obs must
   read it before `env_step` is called or use `next_obs` from buffer.

### dtype discipline

| Field | Mode A | Mode B |
|---|---|---|
| `state.obs` | `jax.Array` (or dict thereof) | `np.ndarray` (or dict) |
| `state.reward` | `jax.Array (N,)` | `np.ndarray (N,)` `float32` |
| `state.done` | `jax.Array (N,)` | `np.ndarray (N,)` `float32` (uint8 cast) |
| `state.info["truncation"]` | `jax.Array (N,)` | `np.ndarray (N,)` `float32` |

Consumers must NOT assume JAX-only or numpy-only — both arrive at the
same call sites.

---

## §3. Consumer contract

### `offpolicy_loop.run_offpolicy_loop`

- Reads `env_state.obs/reward/done/info["truncation"]` only.
- Calls `pipe.get_obs(env_state.obs)` for actor obs (handles dict
  case). Pipeline accepts both dtypes — internally uses `jnp.asarray`.
- Calls `tracker.step(np.asarray(env_state.reward), np.asarray(env_state.done))`
  — explicit numpy conversion since EpisodeTracker is numpy-only.
- Calls `buffer.add_batch(...)` with whatever dtype it has —
  `JaxReplayBuffer.add_batch` does `jnp.asarray(...)` per-arg.
- Calls `explore_fn(actor_params, obs_for_action, ak)` — returns JAX
  array action regardless of obs dtype (jit'd internally).

Net: Mode-A and Mode-B states pass through identically. The only
mode-aware piece is `env_step` itself, which is supplied by the bundle.

### `eval_runner._eval_fn_for(ctx)`

Dispatches `evaluate` (Mode A, full-scan rollout via JAX) vs
`evaluate_gym` (Mode B, Python loop with jit'd action selection) on
`ctx.backend_kind`. `TrainContext.backend_kind` is threaded from
`bundle.backend_kind` at loop init.

Asymmetry: `evaluate` returns Q-bias diagnostics
(`q_bias`/`q_rmse`/`q_corr`); `evaluate_gym` returns mean/std/min/max
only. Mode-B Q-bias is a TODO if a use case arises.

### `record_video.py`

Dispatches `_record_gym(...)` for `backend_kind == "gym"`; default
path is the MJX renderer. NPZ schema is identical.

### `train_context.TrainContext.backend_kind`

Source of truth for downstream consumers. Set from `bundle.backend_kind`
at loop init. Don't reach back into `bundle` from inside the loop —
bundle is for setup only.

---

## §4. Backend support per training script

| Script | Mode A (mjx) | Mode B (gym) | Routing |
|---|---|---|---|
| `train_sac` | ✅ | ✅ | `make_env_bundle` → bundle dispatch |
| `train_td3` | ✅ | ✅ | `make_env_bundle` → bundle dispatch |
| `train_fast_sac` | ✅ | ✅ | `make_env_bundle` → bundle dispatch |
| `train_fast_td3` | ✅ | ✅ | `make_env_bundle` → bundle dispatch |
| `train_ppo` | ✅ | ❓ untested | bundle, but no explicit gate; Python collect loop *should* work on gym |
| `train_ppo_fast` | ✅ | ❌ explicit guard | full lax.scan collection — Mode-A only |
| `train_ppo_contraction` | ✅ | ❌ legacy `make_envs` | migration to bundle pending (TODO) |
| `train_flashsac` | ✅ | ❌ legacy `make_envs` | migration to bundle pending (TODO) |
| `train_pusht` | ❌ | ✅ legacy gym path | Phase 6 deletion after parity reproduction via `train_sac --env PushT` |
| `train_tdmpc2` | ✅ | ❓ | TDMPC2 is on a separate dev cycle — out of scope for this audit |

**Mode-A-only escape hatches and why:**
- `train_ppo_fast`: full-scan rollout collection (`lax.scan` over env
  steps) requires JAX env_step.
- `train_ppo_contraction`, `train_flashsac`: use legacy `make_envs`
  re-export instead of `make_env_bundle`. These are pre-refactor
  scripts that were left alone because nobody had a gym-side use case
  yet. Migration is mechanical — see §5.

---

## §5. Adding a new env backend (checklist)

1. Create `jax_rl/training/env_backends/<name>_backend.py`.
2. Implement a builder `(TrainConfig, seed: int) -> EnvBundle` that
   conforms to §2 (all fields populated, `env_step` invariants).
3. At the bottom of the module, register:
   ```python
   from jax_rl.training.env_backends import register_backend
   register_backend("<name>", make_<name>_env_bundle)
   ```
4. If your backend is selected by env name (not a `<Name>/...` prefix),
   call `register_gym_env_name("EnvA")` for each detected name (or
   extend `detect_backend` in `__init__.py`).
5. Add the backend module to the side-effect import block at the
   bottom of `env_backends/__init__.py`.
6. Verify: `make_env_bundle(cfg, 0)` returns a populated bundle and
   `bundle.env_step(bundle.env_state, jnp.zeros((cfg.num_envs, action_dim)))`
   doesn't crash.
7. Smoke-train: `uv run python scripts/train_sac.py --env <YourEnv>
   --total-timesteps 5000` should complete and print "Collecting
   <min_buffer> samples...".
8. If your backend can't auto-reset, add an explicit gate in
   `offpolicy_loop` or document the limitation.

For migrating an existing legacy script (`train_flashsac`,
`train_ppo_contraction`):
1. Replace `make_envs(cfg, seed)` → `bundle = make_env_bundle(cfg, seed)`.
2. Unpack: `env, env_step, env_state, eval_env = bundle.env, bundle.env_step, bundle.env_state, bundle.eval_env`.
3. Use `bundle.obs_dim`, `bundle.action_dim`, `bundle.critic_obs_dim`,
   `bundle.has_privileged`, `bundle.dict_obs` instead of re-detecting.
4. Thread `bundle.backend_kind` into `TrainContext` for eval dispatch.
5. If the script does anything Mode-A-specific (e.g.,
   `env_state.replace(...)` — that's a flax.struct method gym
   doesn't have), gate behind `if bundle.backend_kind != "mjx": raise`
   and document why.

---

## §6. Open issues from this audit (2026-04-26)

1. **`train_ppo` lacks an explicit backend gate.** The Python collect
   loop *should* work on gym (uses `env_step` via bundle, jit'd inner
   ops, no `lax.scan` over the rollout), but it's never been tested
   on a gym env. Action items: either add a smoke run (`train_ppo
   --env HalfCheetah --total-timesteps 50000`), or add an explicit
   `if bundle.backend_kind != "mjx": raise` until verified.

2. **`train_ppo` doesn't use `_eval_fn_for(ctx)` dispatch.** It has
   its own eval call pattern that may assume JAX env_state.
   Inspection target: lines around eval invocation in `train_ppo.py`.
   If it's MJX-only, the explicit gate in #1 covers it; otherwise
   migrate to `_eval_fn_for(ctx)`.

3. **FlashSAC + PPOContraction migration to bundle.** Both scripts
   bypass the bundle (legacy `make_envs`). Cheap migration per §5
   migration recipe. See TODO entry "Migrate `train_ppo_contraction.py`
   + `train_flashsac.py` to bundle dispatch".

4. **`Bundle.key` field semantics ambiguous for gym.** For Mode A, it's
   the env RNG and is consumed by env_step internally. For Mode B,
   it's a JAX PRNGKey that's only ever used for *policy* action
   sampling (gym envs seed via `vec_env.reset(seed=...)`). The field
   name is misleading. Comment-only fix in `env_bundle.py:45` —
   clarify "for Mode A: env+policy RNG; for Mode B: policy RNG only".

5. **`evaluate_gym` lacks Q-bias diagnostics.** The MC-return Q-bias
   that `evaluate` computes via `lax.scan` would need a Python-loop
   port for gym. Not blocking — log it as a future enhancement when a
   use case shows up. (Already in TODO.)

6. **No `IsaacLab` backend yet.** Protocol exists; one-file
   implementation when an env target lands. PyTorch GPU obs would need
   dlpack bridging at the env_step boundary.

---

## Pointers

- `jax_rl/training/env_bundle.py` — `EnvBundle` dataclass + `BackendKind` literal.
- `jax_rl/training/env_backends/__init__.py` — registry + `detect_backend` + `build_env_bundle`.
- `jax_rl/training/env_backends/mjx_backend.py` — Mode A reference impl.
- `jax_rl/training/env_backends/gym_backend.py` — Mode B reference impl + per-env factories.
- `jax_rl/training/offpolicy_loop.py` — canonical Python-outer + jit-inner loop.
- `jax_rl/training/eval_runner.py` — backend-aware eval dispatch.
- `jax_rl/utils/eval.py` — `evaluate` (Mode A scan) + `evaluate_gym` (Mode B Python).
- `.context/branches/env-backend-refactor.md` — original refactor plan + merge coordination notes.
