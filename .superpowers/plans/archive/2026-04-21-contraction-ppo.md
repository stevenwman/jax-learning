# ContractionPPO Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port contraction-theory stability penalty from Zinage et al. ContractionPPO (CalTech, Chung group, [site](https://contractionppo.github.io/), [repo](https://github.com/contractionppo/ContractionPPO)) onto this repo's JAX PPO, first target = `go2_bongo_handstand`.

**Architecture:** The reference implementation sidesteps dynamics Jacobian `∂f/∂x` entirely. State `x` is a low-dim constraint residual `c = obs − desired`. Time derivative `ẋ` is **supplied by the env as an extra observation** (`contraction_c_dot`), using simulator-provided derivatives (`body_ang_acc`, `joint_acc`) and analytical kinematics (`ġ_b = −ω×g_b`, `q̇ = ½Ω(ω)q`). Lyapunov is `V(x) = xᵀM(x)x` with `M(x) = L(x)L(x)ᵀ` from a spectral-norm bounded MLP. `V̇ = ∇_x V · ẋ` uses `jax.grad` on the metric network (no simulator differentiation). Penalty `ReLU(V̇ + αV + ε)` trains the metric via a **separate optimizer**; PPO's policy loss is **unchanged**. Policy is influenced only via **reward augmentation** (per ref `contraction_ppo.py:905-908`) — safest integration.

**Tech Stack:** JAX 0.4+, Flax (linen), Optax, MJX/Warp via mujoco_playground, existing PPO in `jax_rl/algos/ppo.py`.

**Reference code (PyTorch) cached locally at `/tmp/cppo/`:**
- `contraction_metric.py:41-227` — Lipschitz MLP → tril → SPD
- `contraction_ppo.py:271-333` — `compute_lyapunov_derivative` (autograd on V)
- `contraction_ppo.py:335-355` — `compute_contraction_penalty` (ReLU of violation)
- `contraction_ppo.py:357-459` — `update_contraction_metric` (separate SGD step)
- `contraction_ppo.py:645,661` — PPO loss does **NOT** include contraction (line 661 commented)
- `contraction_ppo.py:905-908` — reward augmentation `(ε − penalty) * penalty_coef`
- `contraction_obs.py:104-212` — env-side ẋ observation helpers
- `contraction_cfg.py:19-103` — full hyperparam list

---

## File Structure

**Create:**
- `jax_rl/networks/contraction_metric.py` — Flax ContractionMetric module (~130 lines)
- `jax_rl/configs/contraction_config.py` — ContractionConfig dataclass (~40 lines)
- `jax_rl/envs/contraction_obs.py` — reusable `*_dot` ObsTerm helpers for MJX (~120 lines)
- `tests/test_contraction_metric.py` — SPD, Lipschitz, V̇ tests (~80 lines)
- `tests/test_contraction_penalty.py` — penalty math test (~40 lines)

**Modify:**
- `jax_rl/configs/ppo_config.py` — optional `contraction: ContractionConfig | None` field
- `jax_rl/configs/__init__.py` — export `ContractionConfig`
- `jax_rl/networks/__init__.py` — export `ContractionMetric`
- `jax_rl/algos/ppo.py` — extend `TrainingState`, add metric update in scan, plumb contraction batch
- `jax_rl/buffers/rollout.py` — extend `RolloutBatch` with optional contraction fields
- `jax_rl/envs/locomotion/go2_bongo_handstand.py` — add `contraction_state` obs group
- one `train_*.py` script targeting `go2_bongo_handstand` — wire config + rollout collection

**Do NOT touch (scope discipline):**
- `jax_rl/algos/sac.py`, `td3.py`, `fast_*.py`, `flash_sac.py` — off-policy, irrelevant
- `jax_rl/envs/manipulation/pusht/` — different task family
- `deploy/` — deploy interface unchanged

---

## Task 1: ContractionMetric network module

**Files:**
- Create: `jax_rl/networks/contraction_metric.py`
- Test: `tests/test_contraction_metric.py`

**Reference:** `/tmp/cppo/contraction_metric.py:41-227`

**Port decisions:**
- Use `flax.linen.Module` not `nn.Module`.
- Replace torch `spectral_norm` with explicit power-iteration approximation OR simpler `optax.adaptive_grad_clip`-style weight clipping. Paper uses `spectral_norm_bound=3.0` default — loose enough that we can start with raw orthogonal init + periodic weight rescaling. Document this deviation in a one-line comment.
- Output: pack flat vector of `n(n+1)/2` entries into `L` via `jnp.tril_indices`; apply `jax.nn.softplus` + `min_diagonal_value=0.1` on diagonal. `M = L @ L.T`.
- Shape contract: `call(x)` where `x: (batch, input_dim)` returns `M: (batch, constraint_dim, constraint_dim)`.

- [ ] **Step 1: Write failing SPD test**

```python
# tests/test_contraction_metric.py
import jax, jax.numpy as jnp
from jax_rl.networks.contraction_metric import ContractionMetric

def test_output_is_positive_definite():
    net = ContractionMetric(input_dim=6, constraint_dim=4, hidden_dims=(32, 32))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 6)))
    x = jax.random.normal(jax.random.PRNGKey(1), (16, 6))
    M = net.apply(params, x)
    assert M.shape == (16, 4, 4)
    eigs = jnp.linalg.eigvalsh(M)
    assert (eigs > 0).all(), f"min eig = {eigs.min()}"
    assert jnp.allclose(M, M.transpose(0, 2, 1), atol=1e-5)
```

- [ ] **Step 2: Run test, confirm it fails** (`uv run pytest tests/test_contraction_metric.py -v`)

Expected: `ModuleNotFoundError: jax_rl.networks.contraction_metric`.

- [ ] **Step 3: Implement minimal module**

```python
# jax_rl/networks/contraction_metric.py
from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

class ContractionMetric(nn.Module):
    input_dim: int
    constraint_dim: int
    hidden_dims: Sequence[int] = (128, 128)
    activation: str = "elu"
    min_diagonal_value: float = 0.1

    @nn.compact
    def __call__(self, x):
        act = {"elu": nn.elu, "relu": nn.relu, "tanh": jnp.tanh}[self.activation]
        h = x
        for d in self.hidden_dims:
            h = act(nn.Dense(d, kernel_init=nn.initializers.orthogonal())(h))
        n = self.constraint_dim
        out_dim = n * (n + 1) // 2
        flat = nn.Dense(out_dim, kernel_init=nn.initializers.orthogonal())(h)
        rows, cols = jnp.tril_indices(n)
        batch = flat.shape[0]
        L = jnp.zeros((batch, n, n)).at[:, rows, cols].set(flat)
        diag = jax.nn.softplus(jnp.diagonal(L, axis1=1, axis2=2)) + self.min_diagonal_value
        L = L.at[:, jnp.arange(n), jnp.arange(n)].set(diag)
        return L @ jnp.swapaxes(L, 1, 2)
```

- [ ] **Step 4: Run test, confirm PASS**

- [ ] **Step 5: Add V̇ chain-rule test**

```python
def test_v_dot_via_jax_grad():
    net = ContractionMetric(input_dim=4, constraint_dim=4)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 4)))

    def V(x_single):
        M = net.apply(params, x_single[None])[0]
        return x_single @ M @ x_single

    x = jnp.array([0.1, -0.2, 0.3, 0.05])
    x_dot = jnp.array([0.01, 0.02, -0.01, 0.0])
    grad_V = jax.grad(V)(x)
    V_dot = grad_V @ x_dot
    assert jnp.isfinite(V_dot)
```

- [ ] **Step 6: Run and commit**

```bash
uv run pytest tests/test_contraction_metric.py -v
git add jax_rl/networks/contraction_metric.py tests/test_contraction_metric.py
git commit -m "feat(contraction): add ContractionMetric network with SPD output"
```

---

## Task 2: ContractionConfig dataclass

**Files:**
- Create: `jax_rl/configs/contraction_config.py`
- Modify: `jax_rl/configs/__init__.py`, `jax_rl/configs/ppo_config.py`

**Reference:** `/tmp/cppo/contraction_cfg.py:19-103`

Fields mirror ref: `alpha`, `penalty_coef`, `metric_lr`, `constraint_dim`, `hidden_dims`, `activation`, `epsilon_contraction`, `min_diagonal_value`, `spectral_norm_bound`. Drop: `action_dim` (use `e_t` not needed — we stick with `x = c` only, matching `contraction_ppo.py:307` where `x = c.clone().requires_grad_(True)`). Drop: `lyapunov_regularization` for now (can add later if training unstable).

- [ ] **Step 1: Write config file**

```python
# jax_rl/configs/contraction_config.py
from dataclasses import dataclass, field

@dataclass
class ContractionConfig:
    alpha: float = 0.1
    epsilon_contraction: float = 1e-3
    penalty_coef: float = 1.0           # reward augmentation weight (ref: contraction_penalty_coef)
    constraint_coef: float = 1.0         # scales c and c_dot before metric/penalty eval (ref: contraction_constraint_coef, used at contraction_ppo.py:296,324)
    metric_lr: float = 1e-3
    constraint_dim: int = 0              # populated at runtime from env
    hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "elu"
    min_diagonal_value: float = 0.1
    spectral_norm_bound: float = 3.0     # not enforced in MVP; see metric module

    def validate(self):
        """Called from PPO.__init__ when contraction is enabled."""
        if self.constraint_dim <= 0:
            raise ValueError(
                f"ContractionConfig.constraint_dim must be populated at runtime "
                f"(got {self.constraint_dim}). See Task 8 Step 0."
            )
```

- [ ] **Step 2: Wire into PPOConfig**

Modify `jax_rl/configs/ppo_config.py` — add:
```python
contraction: ContractionConfig | None = None
```

- [ ] **Step 3: Export from `jax_rl/configs/__init__.py`**

- [ ] **Step 4: Commit**

```bash
git add jax_rl/configs/
git commit -m "feat(contraction): add ContractionConfig, wire into PPOConfig"
```

---

## Task 3: Env-side time-derivative ObsTerm helpers (generic only)

**Files:**
- Create: `jax_rl/envs/contraction_obs.py`

**Reference:** `/tmp/cppo/contraction_obs.py:104-212` (PyTorch/IsaacLab).

**Port decisions:**
- `ObsTerm.fn` contract (see [obs_spec.py:63](jax_rl/envs/obs_spec.py#L63)) is `(**kwargs) -> jax.Array` where kwargs passed by `_get_obs` are only `data=data, info=info`. **No `env` arg, no `prev_info`.** Helpers must match this.
- Anything needing env-specific state (e.g. `self.get_gyro`, `self.get_gravity`) lives **inline** in the env file (Task 5), not here.
- This module holds only helpers that can be computed from `data` alone: `joint_acc`, analytical quaternion derivative, etc.

- [ ] **Step 1: Write generic helpers**

```python
# jax_rl/envs/contraction_obs.py
"""Generic time-derivative helpers for contraction-theory observations.

All fns match ObsTerm contract: (**kwargs) -> jax.Array with data=mjx.Data."""
import jax.numpy as jp

def joint_acc_12dof(data, **_):
    """Leg joint accelerations for Go2-style 6-dof-base + 12-joint robots.
    Ref: contraction_obs.py:121-127."""
    return data.qacc[6:18]

def quat_derivative(q, omega):
    """q̇ = ½ Ω(ω) q. Ref: contraction_obs.py:58-92."""
    wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
    zeros = jp.zeros_like(wx)
    Omega = jp.stack([
        jp.stack([zeros, -wx, -wy, -wz], axis=-1),
        jp.stack([wx, zeros, wz, -wy], axis=-1),
        jp.stack([wy, -wz, zeros, wx], axis=-1),
        jp.stack([wz, wy, -wx, zeros], axis=-1),
    ], axis=-2)
    return 0.5 * (Omega @ q[..., None]).squeeze(-1)
```

- [ ] **Step 2: Write test that verifies `joint_acc_12dof` output shape + finiteness on a reset+step of `go2_bongo_handstand`.**

- [ ] **Step 3: Commit**

Note: for handstand, Task 5 uses only `ġ_b = −ω × g_b` which is trivially inline. `joint_acc_12dof` is a future-proofing helper, not used by the MVP handstand port. OK to skip if scope-cutting.

---

## Task 4: Extend rollout data structures

**Files:**
- Modify: `jax_rl/buffers/rollout.py` (RolloutBatch NamedTuple)
- Modify: `train_ppo_fast.py:43-52` (StepData NamedTuple + collect_step + RolloutBatch reassembly at line 315-323)
- Modify: `train_ppo.py:195-219` (Python rollout loop — if we also wire slow path)

**Reality check:** `train_ppo_fast.py` does NOT call `RolloutBuffer.add` — it constructs `StepData` inside a `lax.scan` and hand-rolls `RolloutBatch` after scan. Extending only `RolloutBatch` is insufficient; must also extend `StepData` and `collect_step`.

```python
# RolloutBatch (jax_rl/buffers/rollout.py) — append fields
contraction_c: jax.Array | None = None       # (T, E, constraint_dim)
contraction_c_dot: jax.Array | None = None   # (T, E, constraint_dim)

# StepData (train_ppo_fast.py:43-52) — append fields
contraction_c: jax.Array | None = None       # (E, constraint_dim)
contraction_c_dot: jax.Array | None = None   # (E, constraint_dim)
```

- [ ] **Step 1: Append fields to RolloutBatch** (default `None` preserves existing call sites). ✅ done.
- [ ] **Step 2: Run existing PPO tests — must still pass** (`uv run pytest tests/ -k ppo -v`). ✅ done.
- [ ] **Step 3: Commit**. ✅ done.

**Moved to Task 7 (natural boundary):** `StepData` extension + `collect_step` slicing + post-scan `RolloutBatch(...)` forwarding all require scan-carry threading of contraction state, which is also where metric_params threading happens. Keeping Task 4 minimal keeps baseline PPO fully unchanged.

---

## Task 5: Handstand env — emit contraction observations

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_bongo_handstand.py` at `:242` (the `self._obs_groups = {...}` line)

**Port decision:** Add third group `contraction_state` containing `[c, c_dot]` where `c` = deviation from handstand target (body gravity vector deviation from `[1, 0, 0]` — matches env's `target_gravity` at `:245`). This makes `constraint_dim = 3`. Small, physically-grounded.

- [ ] **Step 1: Add ObsTerms**

```python
contraction_terms = [
    ObsTerm("c_gravity_residual",
            lambda data, **kw: self.get_gravity(data) - jp.array([1.0, 0.0, 0.0])),
    ObsTerm("c_dot_gravity",
            lambda data, **kw: -jp.cross(self.get_gyro(data), self.get_gravity(data))),
]
self._obs_groups = {
    "state": state_terms,
    "privileged_state": privileged_terms,
    "contraction_state": contraction_terms,   # NEW
}
```

- [ ] **Step 2: Verify `compute_obs` handles new group transparently** — it iterates `groups` dict, so yes.
- [ ] **Step 3: Add env smoke test** — reset+step returns obs dict with `contraction_state` key of shape (6,).
- [ ] **Step 4: Commit**

**Packing convention:** `compute_obs` hstacks terms → `contraction_state` is a `(2*constraint_dim,)` vector. Downstream consumers (Task 4 `collect_step`, Task 6 metric loss, Task 7 reward aug) split as:
```python
c = contraction_state[..., :constraint_dim]
c_dot = contraction_state[..., constraint_dim:]
```
Constraint dim for this env = 3.

**Normalization routing:** `train_ppo_fast.py:176-181` extracts obs via hardcoded keys `obs["state"]` and `obs["privileged_state"]` — `contraction_state` is invisible to the normalizer by construction and passes through unnormalized (matches ref). No wrapper edits needed. Task 4 `collect_step` must explicitly read `env_state.obs["contraction_state"]` alongside the existing policy/critic slices.

---

## Task 6: Metric training step (separate optimizer)

**Files:**
- Create: `jax_rl/algos/ppo_contraction.py` — fork of `ppo.py` with metric training wired in.
- Modify: `jax_rl/algos/__init__.py` — export `PPOContraction`.

**Architectural decision:** baseline `ppo.py` stays untouched so ablations are bit-identical to pre-port. The contraction variant lives in its own file and can reuse PPO helpers (`compute_gae`, `RolloutBatch`, etc.) via import. Cost: two maintenance targets; any future PPO refactor must be mirrored. Acceptable given experimental stage.

**Reference:** `/tmp/cppo/contraction_ppo.py:357-459` (`update_contraction_metric`).

**Port decisions:**
- Extend `TrainingState` with `metric_params`, `metric_opt_state`. Keep `None` when contraction disabled to avoid affecting existing code paths.
- Metric update runs **per minibatch**, inside the existing `_minibatch_step` scan at `ppo.py:75` — matching ref where `update_contraction_metric` is called at `contraction_ppo.py:651` inside the PPO minibatch loop. With default config (num_epochs=4, num_minibatches=32) this yields 128 metric gradient steps per rollout. Worth monitoring; may overfit for larger `constraint_dim`.
- Loss: `mean(ReLU(V̇ + αV + ε))` computed per sample with `jax.vmap` over `jax.grad(V)` of single-sample V.
- Ref at `contraction_ppo.py:661` does NOT add contraction loss to PPO loss (it's commented out) — we match that.
- Capture `alpha`, `epsilon_contraction` as local vars in `PPO.__init__` (like existing pattern at `ppo.py:64-73`) before defining the JIT'd closure, to avoid retracing on config change.
- **Double-backward cost:** outer `jax.grad(metric_loss_fn, argnums=0)` differentiates through `jax.vmap(jax.grad(V_single))` — this is a 2nd-order / `create_graph=True`-style reverse-through-reverse. Faithful to ref (`contraction_ppo.py:320`). Expect ~2-3× cost of a regular metric forward+backward. If spectral-norm is later added as a `custom_vjp`, verify reverse-through-reverse remains well-defined.
- **No explicit `stop_gradient` on `c_batch`/`c_dot_batch` needed.** JAX restricts grad to `argnums`; unlike PyTorch, constants-from-the-optimizer-perspective don't need `.detach()`.

- [ ] **Step 1: Write identity test** — instantiate PPO twice on the same PRNG seed and dummy batch:
  - `ppo_a = PPO(config_no_contraction, ...)`
  - `ppo_b = PPO(config_with_contraction, ...)` where `config_with_contraction.contraction` has `penalty_coef=0` (or just disabled path, depending on branching)
  - Run one `_update` on identical `RolloutBatch`.
  - Assert `jax.tree.map(jnp.allclose, state_a.actor_params, state_b.actor_params)` all True, same for critic.
  - Rationale: directly verifies "contraction code path does not interfere with actor/critic." Stronger than jaxpr-string-match; doesn't rely on op-reorder stability.

- [ ] **Step 2: Extend TrainingState**

```python
@flax.struct.dataclass
class TrainingState:
    actor_params: Any
    critic_params: Any
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    metric_params: Any = None
    metric_opt_state: optax.OptState | None = None
```

- [ ] **Step 3: Instantiate metric + optimizer in `PPO.__init__`** — gated on `config.contraction is not None`.

- [ ] **Step 4: Add metric loss fn**

```python
# alpha, eps, constraint_coef captured as locals (see port decisions)
def metric_loss_fn(metric_params, c_batch, c_dot_batch):
    c_scaled = c_batch * constraint_coef           # ref: contraction_ppo.py:296
    c_dot_scaled = c_dot_batch * constraint_coef   # ref: contraction_ppo.py:324
    def V_single(c):
        M = metric.apply(metric_params, c[None])[0]
        return c @ M @ c
    V = jax.vmap(V_single)(c_scaled)
    grad_V = jax.vmap(jax.grad(V_single))(c_scaled)
    V_dot = jnp.sum(grad_V * c_dot_scaled, axis=-1)
    penalty = jax.nn.relu(V_dot + alpha * V + eps)
    return jnp.mean(penalty), {"penalty": penalty.mean(), "V_mean": V.mean(), "V_dot_mean": V_dot.mean()}
```

Optimizer: `optax.adam(metric_lr)` — matches ref at `/tmp/cppo/contraction_ppo.py:157`.

- [ ] **Step 5: Add optimizer update step** inside `_minibatch_step` at `ppo.py:75-122`. Compute:
  ```python
  (loss_val, aux), grads = jax.value_and_grad(metric_loss_fn, has_aux=True, argnums=0)(
      state.metric_params, c_mb, c_dot_mb
  )
  metric_updates, new_metric_opt_state = metric_optimizer.update(grads, state.metric_opt_state)
  new_metric_params = optax.apply_updates(state.metric_params, metric_updates)
  ```
  No explicit detach needed (see port decisions — JAX grad restricts by argnums).

- [ ] **Step 6: Commit**

---

## Task 7: Reward augmentation

**Files:**
- Modify: `jax_rl/envs/locomotion/go2_bongo_handstand.py` reward spec (near `:248`)

**Reference:** `/tmp/cppo/contraction_ppo.py:905-908`:
```python
contraction_rewards = (epsilon - penalty) * penalty_coef
```

**Port decision:** Ref computes this in `process_env_step` (`contraction_ppo.py:782-810`) during rollout collection — BEFORE the next `update()`. So the metric params used at rollout time are *stale* (from the previous update). Plan matches this: compute penalty at rollout time with current `state.metric_params` and add to per-step reward.

- [ ] **Step 1: Target is `train_ppo_fast.py`** (decision locked per Task 4). Handstand is MJX/Warp → fast path. `metric_params` must thread through `lax.scan` carry.

- [ ] **Step 2: Implement `compute_contraction_reward(metric_params, c, c_dot) -> (E,) reward_bonus`** as a single JIT-compatible fn. Uses `jax.vmap(jax.grad(V_single))` (same inner as Task 6 metric loss but no outer backward — forward eval only).

- [ ] **Step 3: Thread `metric_params` through scan carry.**
  - Add to `collect_step` carry tuple alongside existing actor params.
  - **Timing — CRITICAL:** in ref `process_env_step` reads contraction obs from `self.transition` stored during `act()` BEFORE the env step. Match this: capture `contraction_state = env_state.obs["contraction_state"]` from the PRE-step `env_state`, NOT from the post-step one. Otherwise reward `r_t` pairs with penalty at `s_{t+1}` (off-by-one vs ref).
  - Scale `c, c_dot` by `constraint_coef` before calling `compute_contraction_reward` — matches ref at `contraction_ppo.py:885`.
  - **Scaling order:** reward aug must be added AFTER `reward_scaling` at [train_ppo_fast.py:235](train_ppo_fast.py#L235), not before. Contraction bonus has its own scale via `penalty_coef` and should not be affected by env reward normalization.
  - Emit `penalty_rew = (ε - penalty) * penalty_coef` exactly per ref `:908`.

- [ ] **Step 4: Regression test** — with `config.contraction = None`, scan output rewards bit-identical to pre-port code (gated Python `if` around the augmentation works because config is compile-time constant for the closure).

- [ ] **Step 5: Commit**

---

## Task 8: End-to-end smoke training

**Files:**
- Modify: `train_ppo_fast.py` (decision from Task 7 Step 1) — wire `ContractionConfig`, populate `constraint_dim` at runtime.

- [ ] **Step 0: Runtime plumbing.** After `make_envs(...)` returns `env_state` but before `PPO(config, ...)`:
  ```python
  if cfg.ppo.contraction is not None:
      c_state = env_state.obs["contraction_state"]  # (num_envs, 2*constraint_dim)
      cfg.ppo.contraction.constraint_dim = c_state.shape[-1] // 2
      cfg.ppo.contraction.validate()  # asserts constraint_dim > 0
  ```
  Use the already-batched `env_state` returned from `make_envs`. Do NOT call a fresh `env.reset(PRNGKey(0))` — the wrapped env expects batched keys of shape `(num_envs, 2)`. All downstream consumers read from `cfg.ppo.contraction.constraint_dim`.

- [ ] **Step 1: Establish baseline first.** `go2_bongo_handstand` has NO entry in [env_presets.py:13-72](jax_rl/configs/env_presets.py#L13) — `get_preset()` returns default `PPOConfig` for unknown envs (not tuned). Before comparing, either (a) add a `Go2BongoHandstand` preset with tuned hyperparams, or (b) run the pre-port code once with current defaults and pin that as the baseline. Pick (b) for speed. Record baseline return + seed in the journal file. 500k or 1M steps is sufficient — this is a smoke reference, not a SOTA target.
- [ ] **Step 2: 1M-step run with contraction enabled** — concrete success criteria:
  - (a) mean episode return ≥ 80% of baseline at same step count
  - (b) `contraction_penalty` final mean < 0.5 × initial mean
  - (c) no NaN/Inf in any loss or metric
  - (d) `V_mean` and `L` Frobenius norm grow < 10× from init (if violated, add spectral-norm Lipschitz bound per Deviation 1)

- [ ] **Step 2.5: 1k-step canary** before the 1M-step run — verify (c) + (d) are satisfied early. If L-norm already blown, abort and add spectral norm first.

- [ ] **Step 2.6: Ablation** — `penalty_coef=0` run (metric still trains, reward aug disabled). Checks whether metric alone (via learning signal from penalty gradient) influences policy vs whether the reward shaping is the load-bearing piece. Isolates the two contribution pathways.
- [ ] **Step 3: Log `V_mean`, `V_dot_mean`, `contraction_penalty`, `||L||_F` to W&B.**
- [ ] **Step 4: Commit with smoke results in commit body.**

---

## Docs sync (per `CLAUDE.md`)

- [ ] Update `.context/TODO.md` — mark "evaluate ContractionPPO" done, open "tune α, ε on handstand"
- [ ] Add `.context/journals/2026-04-21.md` — port summary, deviations from paper
- [ ] `.context/LESSONS.md` — lesson: "contraction-theory RL sidesteps `∂f/∂x` by treating `ẋ` as env observation"
- [ ] NOT updating `deploy/` — deploy interface unchanged.

---

## Deviations from the paper (logged for reviewer)

1. **Spectral norm deferred.** Ref uses `torch.nn.utils.spectral_norm` with `spectral_norm_bound=3.0` (loose — 3³ = 27 per 2-hidden-layer MLP, suggesting "soft" regularization rather than tight Lipschitz control). JAX has no drop-in. MVP uses orthogonal init only. **Trigger for adding spectral norm:** if Task 8 criterion (d) fails (`V_mean` or `||L||_F` > 10× init), add power-iteration spectral norm.
2. **`x = c` only**, dropping the `e = state − policy_output` half of ref's formulation (ref also comments it out at `contraction_ppo.py:307-308, 327-328` — uses only `c`). Simpler, matches ref's actual behavior.
3. **No contraction term in PPO loss** (matches ref `:661` which commented it out). Only reward augmentation.
4. **Constraint choice for handstand:** 3-dim gravity residual. Ref handstand uses a different boat-deck constraint set (`/tmp/cppo/boatdeck_env_cfg.py`). Ours is simpler and appropriate for the task.
5. **No phase schedule.** Ref `contraction_ppo.py:465-475` docstring describes a phase-1 (metric-only) / phase-2 (integrated) schedule. Since ref's phase-2 loss coupling is itself commented out at line 661, the schedule collapses to "always metric learning, always reward augmentation" — we match that directly and skip the phase machinery.
6. **`contraction_logging_scale`** (ref `contraction_cfg.py:29`, value 1000.0) dropped — purely cosmetic for logs.
7. **Stochastic policy note + weakened theory.** Lyapunov theory assumes deterministic closed-loop dynamics; PPO samples actions. Because `x = c` only (no `e = state − action`), the sampled action enters `V̇` only indirectly via `ẋ = ġ_b = −ω × g_b` through next-step state evolution — same as ref. **Caveat:** by dropping `e`, our formulation is strictly weaker than ref's *stated* theory — it can only regularize w.r.t. state perturbations given a deterministic action, not w.r.t. policy perturbations. Since ref's own code also drops `e` (`:307-308, 327-328` commented), we match ref-as-implemented rather than ref-as-described.
8. **`ẋ` supplied as obs → `V̇` is a partial derivative.** The metric network sees `c` and `ċ` as independent inputs and computes `V̇ = (∂V/∂c) · ċ`. In reality `ċ = −ω × g` depends on `c` (since `g = c + target_gravity`), so the true total time derivative `dV/dt` has an implicit term we are discarding. Faithful to ref (which does the same via the `contraction_c_dot_batch` input), but academically this is a **partial-derivative Lyapunov condition**, not a full one. No correctness implications for training stability; just worth noting for anyone interpreting the learned metric.

---

## Checkpoint compatibility

Adding `metric_params: Any = None` and `metric_opt_state: optax.OptState | None = None` to `TrainingState` changes the PyTree structure. Orbax `StandardCheckpointer.restore(..., target=target)` at [checkpointing.py:162-185](jax_rl/training/checkpointing.py#L162-L185) is structure-sensitive.

**Decision for MVP:** fresh runs only. Document in Task 8 that old handstand baselines cannot be resumed under the new code; must re-train from scratch. If resume is needed later, add an Orbax tree-transform to inject `None` leaves. Out of scope here.
