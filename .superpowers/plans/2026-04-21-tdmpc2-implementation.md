# TD-MPC2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port TD-MPC2 (Hansen et al. 2024) to this JAX/Flax framework, landing at benchmarks `mppi_return ≥ 850` on DMC CheetahRun and `≥ 800` on DMC HumanoidRun.

**Architecture:** Pure-math algo module in `jax_rl/algos/tdmpc2.py` (networks, loss, MPPI planner, update fns). Standalone training script. Shared replay buffer extended with per-episode sequence sampling. Three small utility modules (SimNorm, two-hot, Q-scale EMA). All HPs verified against source `/tmp/tdmpc2/` across 5 review iterations.

**Tech Stack:** JAX, Flax (linen), Optax, distrax (existing repo uses), gymnasium DMC wrappers (via MuJoCo Playground registry).

**Spec:** `.superpowers/specs/2026-04-21-tdmpc2-design.md` — read it before starting. Every load-bearing HP and formula is grounded with `source file:line` citations there.

---

## API Contracts (read before any coding)

These naming conventions are referenced in every task below. Drift here causes "works on unit tests, fails on integration" bugs.

**Param trees:**
- `wm_params` = `{"encoder", "dynamics", "reward", "q_ensemble"}` — world-model params, updated by world-model optimizer.
- `target_params` = same structure as `wm_params`, EMA-updated copies.
- `policy_params` = PolicyPrior params, updated by separate policy optimizer. Threaded as its own argument to `world_model_loss` / `compute_td_target`; NEVER nested inside `wm_params` or `target_params` (keeps `jax.grad` clean).
- `plan_params` = `{"encoder", "dynamics", "reward", "q_ensemble", "policy"}` — bundle used by MPPI at inference-only (both collect and eval). Built via `plan_params = {**wm_params, "policy": policy_params}` right before calling `plan()`.

**Flax module instances:** all `Encoder.apply(params, ...)`-style calls in code samples below are SHORTHAND. In real implementation, instantiate modules ONCE at the top of the algo file:

```python
# In jax_rl/algos/tdmpc2.py, module-level (or inside a builder fn)
encoder = Encoder(enc_dim=..., num_layers=..., latent_dim=..., simnorm_dim=...)
dynamics = Dynamics(mlp_dim=..., latent_dim=..., simnorm_dim=...)
reward_net = Reward(mlp_dim=..., num_bins=...)
q_ensemble = QEnsemble(mlp_dim=..., num_bins=..., num_q=..., dropout=...)
policy_net = PolicyPrior(mlp_dim=..., action_dim=..., log_std_min=..., log_std_max=...)
```

Then call `encoder.apply(params, obs)` (instance method, not class method) everywhere `Encoder.apply(params, obs)` appears below. The hyperparameters come from `cfg`, so this is a `build_modules(cfg)` factory returning a `dataclasses` or dict of module instances.

**Optimizers live OUTSIDE `cfg`:** `optax.GradientTransformation` is not a JAX pytree leaf. Build optimizers in the training script and pass via `functools.partial` or `jax.jit(static_argnames=...)` — not as a `cfg` field.

**`action_dim` flow:** the env determines `action_dim`, not the config defaults. `make_tdmpc2_config(episode_length, action_dim=env.action_space.shape[0], ...)` — `action_dim` is a required arg at preset-load time, threads through to MPPI and policy loss.

**`cfg.discount`:** derived by `make_tdmpc2_config` from `episode_lengths[0]`. Do not instantiate `TDMPC2Config()` directly — always go through the factory.

---

## Preflight

- [ ] **Verify source clone exists at `/tmp/tdmpc2/`.** If missing:

```bash
cd /tmp && git clone --depth 1 https://github.com/nicklashansen/tdmpc2.git
```

- [ ] **Read the spec end-to-end.** It's ~450 lines. Every task below assumes you know the spec's loss composition, TD target path, MPPI mechanics, and truncation semantics.

- [ ] **Keep spec and source open in editor tabs.** You will check source line-by-line when implementing loss and MPPI.

- [ ] **Confirm project rules:** use `uv run python` (never bare `python`/`python3`), no `Co-Authored-By` in commits, off-policy obs-norm happens at sample time (N/A for TD-MPC2 — we don't normalize obs in MVP).

---

## Phase A — Utilities (foundations, standalone testable)

### Task A1: SimNorm activation

**Files:**
- Create: `jax_rl/utils/simnorm.py`
- Create: `tests/test_simnorm.py`

- [ ] **Step 1: Write failing test** (`tests/test_simnorm.py`)

```python
"""Tests for SimNorm activation (TD-MPC2 latent normalization)."""
import jax
import jax.numpy as jnp
import pytest

from jax_rl.utils.simnorm import simnorm


def test_simnorm_output_sums_to_d_over_v():
    # For latent_dim d with simplex-dim V, output has d/V chunks each summing to 1
    d, V = 512, 8
    x = jax.random.normal(jax.random.PRNGKey(0), (4, d))
    y = simnorm(x, V=V)
    assert y.shape == (4, d)
    # Each chunk of size V along last dim sums to 1 (per batch element)
    chunks = y.reshape(4, d // V, V)
    sums = chunks.sum(axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5)

def test_simnorm_handles_zero_input():
    # Zero input → each chunk is uniform 1/V (no NaN)
    d, V = 16, 4
    x = jnp.zeros((2, d))
    y = simnorm(x, V=V)
    assert jnp.all(jnp.isfinite(y))
    chunks = y.reshape(2, d // V, V)
    assert jnp.allclose(chunks, 1.0 / V, atol=1e-6)

def test_simnorm_gradient_flows():
    d, V = 16, 4
    def loss_fn(x):
        return simnorm(x, V=V).sum()
    x = jax.random.normal(jax.random.PRNGKey(1), (2, d))
    g = jax.grad(loss_fn)(x)
    assert jnp.all(jnp.isfinite(g))

def test_simnorm_requires_divisible():
    d, V = 16, 5  # 16 % 5 != 0
    x = jnp.zeros((2, d))
    with pytest.raises(AssertionError):
        simnorm(x, V=V)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_simnorm.py -v
```

Expected: `ImportError: cannot import name 'simnorm' from 'jax_rl.utils.simnorm'`.

- [ ] **Step 3: Write implementation** (`jax_rl/utils/simnorm.py`)

```python
"""SimNorm — TD-MPC2's latent normalization activation.

Reshapes the last dim into chunks of size V, applies softmax within each chunk,
then flattens back. Produces an L1-normalized, structurally non-collapsing latent.

Source: /tmp/tdmpc2/tdmpc2/common/layers.py:74-91
"""
import jax
import jax.numpy as jnp


def simnorm(x: jax.Array, V: int = 8) -> jax.Array:
    """Apply SimNorm to the last dimension of x.

    Args:
        x: (..., d) where d % V == 0
        V: simplex dimension (chunk size)

    Returns:
        (..., d) with each d/V chunk along last dim summing to 1.
    """
    d = x.shape[-1]
    assert d % V == 0, f"Last dim {d} must be divisible by V={V}"
    shape = x.shape[:-1] + (d // V, V)
    x_chunked = x.reshape(shape)
    y = jax.nn.softmax(x_chunked, axis=-1)
    return y.reshape(x.shape)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python -m pytest tests/test_simnorm.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/utils/simnorm.py tests/test_simnorm.py
git commit -m "feat(tdmpc2): add SimNorm latent activation"
```

---

### Task A2: Two-hot, symlog, symexp

**Files:**
- Create: `jax_rl/utils/twohot.py`
- Create: `tests/test_twohot.py`

- [ ] **Step 1: Write failing tests** (`tests/test_twohot.py`)

```python
"""Tests for two-hot encoding, symlog, symexp (TD-MPC2 reward/Q categorical targets)."""
import jax
import jax.numpy as jnp

from jax_rl.utils.twohot import symlog, symexp, two_hot, two_hot_inv


def test_symlog_symexp_roundtrip():
    x = jnp.array([-1e4, -10.0, -1.0, 0.0, 1.0, 10.0, 1e4])
    assert jnp.allclose(symexp(symlog(x)), x, atol=1e-3)

def test_symlog_zero_is_zero():
    assert float(symlog(jnp.array(0.0))) == 0.0

def test_two_hot_sums_to_one():
    x = jnp.array([[0.0], [5.0], [-3.0]])
    enc = two_hot(x, vmin=-10.0, vmax=10.0, num_bins=101)
    assert enc.shape == (3, 101)
    assert jnp.allclose(enc.sum(axis=-1), 1.0, atol=1e-5)

def test_two_hot_decode_roundtrip():
    # Encode post-symlog value, then decode, should recover input (+ symexp wrap)
    for x_val in [-5.0, 0.0, 2.5, 8.9]:
        x = jnp.array([[x_val]])
        enc = two_hot(x, vmin=-10.0, vmax=10.0, num_bins=101)  # x already in [vmin, vmax]
        dec = two_hot_inv(enc, vmin=-10.0, vmax=10.0, num_bins=101, apply_symexp=False)
        assert jnp.allclose(dec, x, atol=1e-4), f"Failed at {x_val}: got {float(dec)}"

def test_two_hot_inv_with_symexp():
    # Encode symlog(r), decode with symexp, recovers r
    r = jnp.array([[100.0], [-50.0], [0.0]])
    enc = two_hot(symlog(r), vmin=-10.0, vmax=10.0, num_bins=101)
    r_recovered = two_hot_inv(enc, vmin=-10.0, vmax=10.0, num_bins=101, apply_symexp=True)
    assert jnp.allclose(r_recovered, r, atol=1.0)  # ~1% error acceptable at large values

def test_two_hot_clamps_out_of_range():
    # Values outside [vmin, vmax] should clamp, no NaN
    x = jnp.array([[-100.0], [100.0]])
    enc = two_hot(x, vmin=-10.0, vmax=10.0, num_bins=101)
    assert jnp.all(jnp.isfinite(enc))
    # Heavy probability mass should land on boundary bins
    assert enc[0, 0] > 0.9  # bin 0 corresponds to vmin
    assert enc[1, -1] > 0.9  # last bin corresponds to vmax
```

- [ ] **Step 2: Run test, verify failure**

```bash
uv run python -m pytest tests/test_twohot.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implementation** (`jax_rl/utils/twohot.py`)

```python
"""Two-hot categorical encoding + symlog/symexp for TD-MPC2 reward/value targets.

Matches source /tmp/tdmpc2/tdmpc2/common/math.py.
"""
import jax
import jax.numpy as jnp


def symlog(x: jax.Array) -> jax.Array:
    """Signed log: sign(x) * log(|x| + 1). Monotone, symmetric about 0."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Inverse of symlog."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


def two_hot(x: jax.Array, vmin: float, vmax: float, num_bins: int) -> jax.Array:
    """Encode scalar(s) as two-hot distribution over num_bins.

    Args:
        x: (..., 1) scalar values (already symlog'd if applicable). Out-of-range values clamp.
        vmin, vmax: value range covered by bins.
        num_bins: number of bins (source uses 101).

    Returns:
        (..., num_bins) probability distribution (sums to 1, non-zero on at most 2 adjacent bins).
    """
    x = jnp.clip(x, vmin, vmax)
    bin_size = (vmax - vmin) / (num_bins - 1)
    # Position x in bin index space
    bin_pos = (x.squeeze(-1) - vmin) / bin_size  # (...,)
    lower_idx = jnp.floor(bin_pos).astype(jnp.int32)
    upper_idx = jnp.clip(lower_idx + 1, 0, num_bins - 1)
    lower_idx = jnp.clip(lower_idx, 0, num_bins - 1)
    upper_weight = bin_pos - lower_idx.astype(bin_pos.dtype)
    lower_weight = 1.0 - upper_weight

    # Scatter onto bins
    one_hot_lower = jax.nn.one_hot(lower_idx, num_bins)
    one_hot_upper = jax.nn.one_hot(upper_idx, num_bins)
    return lower_weight[..., None] * one_hot_lower + upper_weight[..., None] * one_hot_upper


def two_hot_inv(
    probs: jax.Array,
    vmin: float,
    vmax: float,
    num_bins: int,
    apply_symexp: bool = True,
) -> jax.Array:
    """Decode probability distribution back to scalar value.

    Args:
        probs: (..., num_bins) — softmax over logits, or a two-hot-encoded distribution.
        apply_symexp: if True, applies symexp to the decoded bin-centered value (matches source TD-MPC2 decoding path).

    Returns:
        (..., 1) scalar values.
    """
    bin_centers = jnp.linspace(vmin, vmax, num_bins)  # (num_bins,)
    value = (probs * bin_centers).sum(axis=-1, keepdims=True)
    if apply_symexp:
        value = symexp(value)
    return value


def two_hot_ce_loss(logits: jax.Array, target_value: jax.Array,
                    vmin: float, vmax: float, num_bins: int,
                    apply_symlog: bool = True) -> jax.Array:
    """Cross-entropy loss between logits and two-hot target.

    Args:
        logits: (..., num_bins)
        target_value: (..., 1) scalar values (raw; symlog will be applied if apply_symlog=True)

    Returns:
        (...,) per-sample CE loss.
    """
    if apply_symlog:
        target_value = symlog(target_value)
    target = two_hot(target_value, vmin, vmax, num_bins)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -(target * log_probs).sum(axis=-1)
```

- [ ] **Step 4: Run test, verify pass**

```bash
uv run python -m pytest tests/test_twohot.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/utils/twohot.py tests/test_twohot.py
git commit -m "feat(tdmpc2): two-hot + symlog/symexp utilities"
```

---

### Task A3: Q-scale EMA tracker

**Files:**
- Create: `jax_rl/utils/qscale.py`
- Create: `tests/test_qscale.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for Q-scale running percentile EMA (TD-MPC2 policy loss scaling)."""
import jax
import jax.numpy as jnp

from jax_rl.utils.qscale import QScaleState, qscale_init, qscale_update, qscale_apply


def test_qscale_init_is_one():
    state = qscale_init()
    assert float(state.range_ema) == 1.0

def test_qscale_clamps_min_1():
    # If p95 - p5 is tiny, EMA should clamp to 1.0 before the lerp
    state = qscale_init()
    # Fake batch with tiny spread
    q = jnp.linspace(0.499, 0.501, 100)
    new_state = qscale_update(state, q, tau=0.01)
    # After one update with range <<1, new range should still be ~1.0 (clamp before EMA)
    assert float(new_state.range_ema) >= 1.0 - 1e-5

def test_qscale_moves_toward_actual_range():
    state = qscale_init()
    # Large spread: p95 - p5 should be ~180 for uniform [-100, 100]
    q = jnp.linspace(-100.0, 100.0, 1000)
    for _ in range(200):
        state = qscale_update(state, q, tau=0.01)
    # EMA should have moved substantially from 1.0 toward ~180
    assert float(state.range_ema) > 50.0

def test_qscale_apply_divides():
    state = QScaleState(range_ema=jnp.array(10.0))
    q = jnp.array([5.0, 10.0, 20.0])
    scaled = qscale_apply(state, q)
    assert jnp.allclose(scaled, jnp.array([0.5, 1.0, 2.0]), atol=1e-5)
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run python -m pytest tests/test_qscale.py -v
```

- [ ] **Step 3: Implementation** (`jax_rl/utils/qscale.py`)

```python
"""Q-scale running percentile EMA tracker.

Tracks 5th/95th percentiles of Q outputs across updates, EMA'd with tau.
Scales Q values in the policy loss before adding the entropy bonus.
Source: /tmp/tdmpc2/tdmpc2/common/scale.py
"""
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class QScaleState:
    """Single scalar range (p95 - p5), EMA'd, clamped to min 1.0."""
    range_ema: jax.Array  # scalar


def qscale_init() -> QScaleState:
    return QScaleState(range_ema=jnp.array(1.0))


def qscale_update(state: QScaleState, qs: jax.Array, tau: float = 0.01) -> QScaleState:
    """Update range EMA from batch of Q values.

    Args:
        state: current QScaleState.
        qs: Q values, any shape. Percentiles computed across ALL elements.
            Source uses `qs[0]` = time-step-0 slice of avg-of-2 Q tensor, shape (B, 1).
            Caller is responsible for providing the right slice.
        tau: EMA rate.
    """
    p5 = jnp.percentile(qs, 5.0)
    p95 = jnp.percentile(qs, 95.0)
    new_range = jnp.maximum(p95 - p5, 1.0)  # load-bearing clamp (source scale.py:41)
    range_ema = state.range_ema + tau * (new_range - state.range_ema)
    return QScaleState(range_ema=range_ema)


def qscale_apply(state: QScaleState, qs: jax.Array) -> jax.Array:
    """Divide Q values by current range EMA."""
    return qs / (state.range_ema + 1e-8)
```

- [ ] **Step 4: Run, verify pass**

- [ ] **Step 5: Commit**

```bash
git add jax_rl/utils/qscale.py tests/test_qscale.py
git commit -m "feat(tdmpc2): Q-scale percentile EMA tracker with min=1 clamp"
```

---

## Phase B — Replay buffer extension

### Task B1: Track episode_id on add

**Files:**
- Modify: `jax_rl/buffers/jax_replay_buffer.py`
- Modify: `tests/test_replay_buffer.py` (or create if absent)

- [ ] **Step 1: Read current buffer** — identify the `add_batch` method and the field layout.

```bash
cat jax_rl/buffers/jax_replay_buffer.py | head -200
```

- [ ] **Step 2: Write failing test**

Create/extend `tests/test_replay_buffer.py`:

```python
def test_buffer_tracks_episode_id():
    """episode_id increments whenever prior transition had done OR truncated."""
    from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
    import numpy as np
    buf = JaxReplayBuffer(obs_dim=3, action_dim=2, max_size=100)
    # Add 5 transitions, with done=True at index 2
    for i in range(5):
        obs = np.ones(3, dtype=np.float32) * i
        action = np.ones(2, dtype=np.float32)
        reward = np.float32(0.0)
        done = (i == 2)
        trunc = False
        next_obs = np.ones(3, dtype=np.float32) * (i + 1)
        buf.add(obs, action, reward, next_obs, done, trunc)
    # Episode IDs: 0, 0, 0, 1, 1 (increment AFTER the done)
    ids = np.array(buf.episode_ids[:5])
    assert list(ids) == [0, 0, 0, 1, 1], f"Got {list(ids)}"
```

- [ ] **Step 3: Run, verify failure** (attribute/field not present)

- [ ] **Step 4: Implementation** — add `episode_ids` field + update on `add`.

In `jax_rl/buffers/jax_replay_buffer.py`, after the existing `self.truncations` allocation:

```python
# Episode ID for per-episode sequence sampling (TD-MPC2 sample_sequence).
# Incremented AFTER any transition with done or truncated == True.
self.episode_ids = jnp.zeros((max_size,), dtype=jnp.int32)
self._current_episode_id = jnp.array(0, dtype=jnp.int32)
```

Modify `add` (or `add_batch`) to increment `self._current_episode_id` when the previous transition's done/truncated was True. Store current `_current_episode_id` at the current ptr.

*Note: if `add` is `jit`-compiled, `_current_episode_id` must live as a jnp scalar and flow through the JIT boundary. If using a dataclass TrainState pattern, add `current_episode_id` to it. For simplicity if `add` is pure Python wrapping jit'd ops, `int`-valued `_current_episode_id` is fine — just update it in Python before the jit call.*

Follow whichever pattern matches existing `add`/`add_batch` style.

- [ ] **Step 5: Run test, verify pass**

- [ ] **Step 6: Commit**

```bash
git add jax_rl/buffers/jax_replay_buffer.py tests/test_replay_buffer.py
git commit -m "feat(buffer): track episode_id for per-episode sequence sampling"
```

---

### Task B2: `sample_sequence(batch, H)` with per-episode match

**Files:**
- Modify: `jax_rl/buffers/jax_replay_buffer.py`
- Modify: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write failing tests**

```python
def test_sample_sequence_shape():
    buf = JaxReplayBuffer(obs_dim=3, action_dim=2, max_size=100)
    for i in range(50):
        buf.add(np.ones(3, dtype=np.float32) * i, np.zeros(2, dtype=np.float32),
                np.float32(0.0), np.ones(3, dtype=np.float32) * (i+1),
                done=False, trunc=(i == 49))  # one episode of length 50
    seq = buf.sample_sequence(batch=8, H=3, key=jax.random.PRNGKey(0))
    assert seq["obs"].shape == (4, 8, 3), seq["obs"].shape  # (H+1, B, obs_dim)
    assert seq["actions"].shape == (3, 8, 2)
    assert seq["rewards"].shape == (3, 8, 1)

def test_sample_sequence_rejects_cross_episode():
    """No sampled window should span two episodes."""
    buf = JaxReplayBuffer(obs_dim=2, action_dim=1, max_size=100)
    # Episode 1: indices 0..4 (done at 4)
    for i in range(5):
        done = (i == 4)
        buf.add(np.ones(2) * i, np.zeros(1), 0.0, np.ones(2) * (i+1), done, False)
    # Episode 2: indices 5..9
    for i in range(5, 10):
        done = (i == 9)
        buf.add(np.ones(2) * i, np.zeros(1), 0.0, np.ones(2) * (i+1), done, False)
    # Sample with H=3 (window length 4); valid starts: [0,1,2] or [5,6,7]
    seq = buf.sample_sequence(batch=64, H=3, key=jax.random.PRNGKey(42))
    # Each window's obs[0,:,0] should equal a valid start index in {0,1,2,5,6,7}
    starts = np.array(seq["obs"][0, :, 0])  # obs[0] is first step's obs
    for s in starts:
        assert s in [0, 1, 2, 5, 6, 7], f"Invalid start {s}"

def test_sample_sequence_handles_small_buffer():
    buf = JaxReplayBuffer(obs_dim=2, action_dim=1, max_size=100)
    # Only 2 transitions — not enough for H=3 window (need H+1 = 4)
    for i in range(2):
        buf.add(np.ones(2), np.zeros(1), 0.0, np.ones(2), False, False)
    # Should raise or return empty (implementer choice; pick one and test it)
    with pytest.raises(ValueError):
        buf.sample_sequence(batch=4, H=3, key=jax.random.PRNGKey(0))
```

- [ ] **Step 2: Run, verify failure**

- [ ] **Step 3: Implementation**

Add to `JaxReplayBuffer`:

```python
def sample_sequence(self, batch: int, H: int, key: jax.Array) -> dict:
    """Sample `batch` contiguous H+1 sequence windows, all within a single episode.

    Returns dict with:
      obs        (H+1, batch, obs_dim)
      actions    (H,   batch, action_dim)
      rewards    (H,   batch, 1)
      dones      (H,   batch, 1)       — terminated flags
      truncations (H,  batch, 1)
    """
    if self.size < H + 1:
        raise ValueError(f"Buffer has {self.size} < {H+1} transitions")

    # Draw candidate starting indices in [0, size - (H+1)], verify episode_id constant
    # across all H+1 slots. Reject invalid; resample.
    # Rejection-sampling loop: bounded by a sensible max (e.g. 16 tries); track rejection
    # rate as a diagnostic metric.
    max_start = self.size - (H + 1)
    # Simple implementation: sample starts, compute per-candidate validity, use where-mask
    # + fallback to first valid. JIT-friendly version:
    def _one_sample(key):
        start_candidates = jax.random.randint(key, (batch * 4,), 0, max_start + 1)
        eids = self.episode_ids[start_candidates[:, None] + jnp.arange(H + 1)[None, :]]
        valid = jnp.all(eids == eids[:, :1], axis=1)
        # Take first `batch` valid
        idxs = jnp.where(valid, start_candidates, -1)
        # Sort so valid come first
        order = jnp.argsort(-valid.astype(jnp.int32))  # valid=True (1) first
        chosen = start_candidates[order][:batch]
        return chosen
    starts = _one_sample(key)  # (batch,)

    # Gather H+1 observations, H actions/rewards/dones/truncations
    obs_idx = starts[:, None] + jnp.arange(H + 1)[None, :]  # (batch, H+1)
    next_idx = starts[:, None] + jnp.arange(H)[None, :]     # (batch, H)

    obs = self.obs[obs_idx].transpose(1, 0, 2)              # (H+1, batch, obs_dim)
    actions = self.actions[next_idx].transpose(1, 0, 2)
    rewards = self.rewards[next_idx].transpose(1, 0, 2)
    dones = self.dones[next_idx].transpose(1, 0, 2)
    trunc = self.truncations[next_idx].transpose(1, 0, 2)

    return {"obs": obs, "actions": actions, "rewards": rewards,
            "dones": dones, "truncations": trunc}
```

*Implementation note:* the rejection logic above oversamples 4× then filters. If too many consecutive samples come up invalid (rare — requires buffer near-full of very short episodes), bump multiplier. Log `rejection_rate = 1 - valid.mean()` for diagnostics (wire into training metrics in Task H3).

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add jax_rl/buffers/jax_replay_buffer.py tests/test_replay_buffer.py
git commit -m "feat(buffer): sample_sequence with per-episode window match"
```

---

## Phase C — Config

### Task C1: `TDMPC2Config` dataclass

**Files:**
- Create: `jax_rl/configs/tdmpc2_config.py`
- Create: `tests/test_tdmpc2_config.py`

- [ ] **Step 1: Write failing test** (minimal — just instantiation + invariants)

```python
def test_tdmpc2_config_defaults_match_spec():
    from jax_rl.configs.tdmpc2_config import TDMPC2Config, compute_discount, make_tdmpc2_config
    c = make_tdmpc2_config(action_dim=6, episode_length=500)
    # Bin size derived, not hardcoded
    assert (c.vmax - c.vmin) / (c.num_bins - 1) == 0.2
    # action_dim wired through
    assert c.action_dim == 6
    # Discount heuristic
    assert compute_discount(500, 5, 0.95, 0.995) == 0.99
    assert compute_discount(1000, 5, 0.95, 0.995) == 0.995
    assert compute_discount(10, 5, 0.95, 0.995) == 0.95
    # Factory wires discount onto cfg
    assert make_tdmpc2_config(action_dim=6, episode_length=500).discount == 0.99
    assert make_tdmpc2_config(action_dim=6, episode_length=1000).discount == 0.995

def test_tdmpc2_config_rejects_zero_action_dim():
    import pytest
    from jax_rl.configs.tdmpc2_config import make_tdmpc2_config
    with pytest.raises(AssertionError):
        make_tdmpc2_config(action_dim=0)
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement config**

```python
"""TD-MPC2 configuration.

All HPs sourced from /tmp/tdmpc2/tdmpc2/config.yaml. Do NOT change defaults without
citing source lines.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)  # frozen → hashable → safe as `jax.jit(static_argnames="cfg")` arg
class TDMPC2Config:
    # Architecture
    latent_dim: int = 512
    mlp_dim: int = 512
    enc_dim: int = 256
    num_enc_layers: int = 2
    simnorm_dim: int = 8
    dropout: float = 0.01  # Q heads only
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    # Loss
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 1e-4
    rho: float = 0.5
    grad_clip_norm: float = 20.0

    # Optimization
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    pi_optim_eps: float = 1e-5
    tau: float = 0.01            # shared by target EMA and Q-scale EMA
    batch_size: int = 256
    horizon: int = 3

    # Policy prior bounds
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    # MPPI
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    mppi_iterations: int = 6
    mppi_temperature: float = 0.5
    mppi_min_std: float = 0.05
    mppi_max_std: float = 2.0

    # Training loop
    total_steps: int = 1_000_000
    seed_steps: int = 2500
    utd: int = 1
    collect_mode: str = "mppi"
    num_envs: int = 8
    num_eval_envs: int = 8
    eval_every: int = 50_000
    eval_episodes: int = 10
    buffer_size: int = 1_000_000

    # Discount heuristic
    discount_denom: int = 5
    discount_min: float = 0.95
    discount_max: float = 0.995
    discount: float = 0.99   # derived from episode_lengths[0] at preset load; see below

    # Env spec (set by make_tdmpc2_config from env — NOT a true default)
    action_dim: int = 0            # MUST be overridden at preset load; assertion elsewhere

    # Multi-task C-seams (B-mode defaults)
    num_tasks: int = 1
    task_names: tuple[str, ...] = ("single",)
    episode_lengths: tuple[int, ...] = (500,)


def compute_discount(episode_length: int, denom: int, dmin: float, dmax: float) -> float:
    """Source tdmpc2.py:58-71 discount heuristic."""
    frac = episode_length / denom
    d = max(0.0, (frac - 1) / frac) if frac > 0 else 0.0
    return float(max(dmin, min(d, dmax)))


def make_tdmpc2_config(
    action_dim: int,
    episode_length: int = 500,
    task_name: str = "single",
    **overrides,
) -> TDMPC2Config:
    """Factory: builds TDMPC2Config with `discount` derived from episode_length and
    `action_dim` pulled from the env spec. Both are required at preset-load time.

    Use this instead of calling TDMPC2Config() directly.
    """
    assert action_dim > 0, f"action_dim must be positive, got {action_dim}"
    base = TDMPC2Config(
        action_dim=action_dim,
        episode_lengths=(episode_length,),
        task_names=(task_name,),
    )
    d = compute_discount(episode_length, base.discount_denom, base.discount_min, base.discount_max)
    from dataclasses import replace
    return replace(base, discount=d, **overrides)
```

- [ ] **Step 4: Run test, verify pass**

- [ ] **Step 5: Commit**

---

### Task C2: DMC presets

**Files:**
- Modify: `jax_rl/configs/env_presets.py`

- [ ] **Step 1: Read current `env_presets.py`** — understand the preset pattern.

- [ ] **Step 2: Add TDMPC2 presets**

At the bottom of `env_presets.py`:

```python
from jax_rl.configs.tdmpc2_config import TDMPC2Config, compute_discount

def tdmpc2_dmc_preset(task_name: str, episode_length: int = 500) -> tuple:
    """TD-MPC2 preset for DM Control tasks. Returns (TrainConfig-like, TDMPC2Config)."""
    algo_cfg = TDMPC2Config(
        episode_lengths=(episode_length,),
        task_names=(task_name,),
    )
    # Compute discount from ep length
    algo_cfg = replace(algo_cfg, ...)  # Use dataclasses.replace to keep immutable flavor
    # Since TrainConfig/env loading uses existing patterns, return (None, algo_cfg) or
    # whatever the repo convention is — check existing presets for the exact tuple shape.
    return algo_cfg  # or (TrainConfig(...), algo_cfg) — match existing pattern
```

*This task is small; exact structure depends on existing `env_presets.py`. If presets return `(TrainConfig, AlgoConfig)` tuples, match that. Use `CheetahRun`, `HumanoidRun`, `AcrobotSwingup` as the three P1 targets.*

- [ ] **Step 3: Commit**

```bash
git add jax_rl/configs/env_presets.py jax_rl/configs/tdmpc2_config.py tests/test_tdmpc2_config.py
git commit -m "feat(tdmpc2): TDMPC2Config + DMC presets (CheetahRun, HumanoidRun, AcrobotSwingup)"
```

---

## Phase D — Networks

### Task D1: NormedLinear + Mish building block

**Files:**
- Create: `jax_rl/algos/tdmpc2.py` (start this file; add layers helper)
- Create: `tests/test_tdmpc2.py` (start file)

- [ ] **Step 1: Write failing test**

```python
# tests/test_tdmpc2.py
"""Tests for TD-MPC2 networks and losses."""
import jax
import jax.numpy as jnp
from flax import linen as nn

from jax_rl.algos.tdmpc2 import NormedLinear


def test_normed_linear_shape_and_activation():
    layer = NormedLinear(features=32)
    params = layer.init(jax.random.PRNGKey(0), jnp.zeros((4, 16)))
    y = layer.apply(params, jnp.ones((4, 16)))
    assert y.shape == (4, 32)
    assert jnp.all(jnp.isfinite(y))
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement** (`jax_rl/algos/tdmpc2.py` — start of file)

```python
"""TD-MPC2 algorithm.

Networks, loss, MPPI planner, update fns. Pure math — no env knowledge.

All HPs verified against /tmp/tdmpc2/. See .superpowers/specs/2026-04-21-tdmpc2-design.md
for the full paper-audit trail.
"""
from typing import Any, Optional
from dataclasses import field

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jax_rl.configs.tdmpc2_config import TDMPC2Config
from jax_rl.utils.simnorm import simnorm
from jax_rl.utils.twohot import two_hot, two_hot_inv, symlog, two_hot_ce_loss
from jax_rl.utils.qscale import QScaleState, qscale_init, qscale_update, qscale_apply


# ------------------ Building blocks ------------------

class NormedLinear(nn.Module):
    """Linear → LayerNorm → Mish. Matches source common/layers.py NormedLinear."""
    features: int
    dropout: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=self.features,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.activation.mish(x)
        if self.dropout > 0:
            x = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(x)
        return x
```

- [ ] **Step 4: Run, verify pass**

- [ ] **Step 5: Commit**

```bash
git add jax_rl/algos/tdmpc2.py tests/test_tdmpc2.py
git commit -m "feat(tdmpc2): NormedLinear building block (Mish + LayerNorm)"
```

---

### Task D2: Encoder

**Files:**
- Modify: `jax_rl/algos/tdmpc2.py`
- Modify: `tests/test_tdmpc2.py`

- [ ] **Step 1: Failing test**

```python
def test_encoder_output_shape_and_simnorm():
    from jax_rl.algos.tdmpc2 import Encoder
    from jax_rl.utils.simnorm import simnorm
    enc = Encoder(enc_dim=256, num_layers=2, latent_dim=512, simnorm_dim=8)
    params = enc.init(jax.random.PRNGKey(0), jnp.zeros((4, 48)))
    z = enc.apply(params, jnp.ones((4, 48)))
    assert z.shape == (4, 512)
    # Latent respects SimNorm (chunks sum to 1)
    chunks = z.reshape(4, 512 // 8, 8)
    assert jnp.allclose(chunks.sum(-1), 1.0, atol=1e-5)
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement**

```python
class Encoder(nn.Module):
    """h(obs) → z with SimNorm output.

    Arch: num_layers × NormedLinear(enc_dim) → Dense(latent_dim) → SimNorm.
    Source: common/layers.py enc(), config.yaml num_enc_layers=2, enc_dim=256.
    """
    enc_dim: int
    num_layers: int
    latent_dim: int
    simnorm_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(self.num_layers):
            x = NormedLinear(features=self.enc_dim)(x)
        x = nn.Dense(
            features=self.latent_dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        return simnorm(x, V=self.simnorm_dim)
```

- [ ] **Step 4: Run, verify pass**

- [ ] **Step 5: Commit**

---

### Task D3: Dynamics head

**Files:** `jax_rl/algos/tdmpc2.py`, `tests/test_tdmpc2.py`

- [ ] **Step 1: Failing test** — shape check + SimNorm output.
- [ ] **Step 2: Run, fail.**
- [ ] **Step 3: Implement**

```python
class Dynamics(nn.Module):
    """d(z, a) → z' with SimNorm output."""
    mlp_dim: int
    latent_dim: int
    simnorm_dim: int

    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], axis=-1)
        x = NormedLinear(features=self.mlp_dim)(x)
        x = NormedLinear(features=self.mlp_dim)(x)
        x = nn.Dense(
            features=self.latent_dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        return simnorm(x, V=self.simnorm_dim)
```

- [ ] **Step 4: Run, pass.**
- [ ] **Step 5: Commit.**

---

### Task D4: Reward head

**Files:** same.

- [ ] **Step 1: Failing test** — output shape `(batch, num_bins)`, finite.
- [ ] **Step 2: Fail.**
- [ ] **Step 3: Implement** — note zero-init output kernel.

```python
class Reward(nn.Module):
    """R(z, a) → reward logits over num_bins."""
    mlp_dim: int
    num_bins: int

    @nn.compact
    def __call__(self, z, a):
        x = jnp.concatenate([z, a], axis=-1)
        x = NormedLinear(features=self.mlp_dim)(x)
        x = NormedLinear(features=self.mlp_dim)(x)
        # Zero-init on output kernel (source common/world_model.py:31)
        return nn.Dense(
            features=self.num_bins,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task D5: Q ensemble

**Files:** same.

- [ ] **Step 1: Failing test** — check ensemble output shape `(num_q, batch, num_bins)`, zero-init on heads.

```python
def test_q_ensemble_shape_and_zero_init():
    from jax_rl.algos.tdmpc2 import QEnsemble
    q = QEnsemble(mlp_dim=512, num_bins=101, num_q=5, dropout=0.01)
    params = q.init(jax.random.PRNGKey(0), jnp.zeros((4, 512)), jnp.zeros((4, 6)),
                     deterministic=True)
    out = q.apply(params, jnp.ones((4, 512)), jnp.ones((4, 6)), deterministic=True)
    assert out.shape == (5, 4, 101)
    # Zero-init on final layer → output is ~0 at init
    assert jnp.abs(out).max() < 1e-3
```

- [ ] **Step 2: Fail.**
- [ ] **Step 3: Implement** — use `flax.linen.vmap` over a single Q-head module for param-stacking.

```python
class QHead(nn.Module):
    """Single Q head: 2×NormedLinear + Dense(num_bins). Dropout on first layer only."""
    mlp_dim: int
    num_bins: int
    dropout: float

    @nn.compact
    def __call__(self, z, a, deterministic: bool):
        x = jnp.concatenate([z, a], axis=-1)
        x = NormedLinear(features=self.mlp_dim, dropout=self.dropout,
                         deterministic=deterministic)(x)
        x = NormedLinear(features=self.mlp_dim)(x)  # no dropout here
        return nn.Dense(
            features=self.num_bins,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)


class QEnsemble(nn.Module):
    """5 Q heads via vmap-over-params (matches source nn.ParameterList semantics)."""
    mlp_dim: int
    num_bins: int
    num_q: int
    dropout: float

    @nn.compact
    def __call__(self, z, a, deterministic: bool):
        VmappedQ = nn.vmap(
            QHead,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            axis_size=self.num_q,
        )
        return VmappedQ(
            mlp_dim=self.mlp_dim,
            num_bins=self.num_bins,
            dropout=self.dropout,
        )(z, a, deterministic)
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task D6: Policy prior (tanh-bounded log_std + sampling + log-prob)

**Files:** same.

- [ ] **Step 1: Failing tests** — this is where the silent-failure bugs were; tests must be thorough.

```python
def test_policy_log_std_bounded():
    from jax_rl.algos.tdmpc2 import PolicyPrior, bound_log_std
    # bound_log_std maps raw → [log_std_min, log_std_max] via tanh
    raw = jnp.array([-10.0, 0.0, 10.0])
    bounded = bound_log_std(raw, log_std_min=-10.0, log_std_max=2.0)
    # At raw=-10, bounded ≈ log_std_min; at raw=+10, ≈ log_std_max; at raw=0, midpoint
    assert float(bounded[0]) < -9.5
    assert float(bounded[2]) > 1.5
    assert jnp.abs(bounded[1] - (-10.0 + 2.0) / 2) < 0.1

def test_policy_sample_shapes():
    from jax_rl.algos.tdmpc2 import PolicyPrior
    pol = PolicyPrior(mlp_dim=512, action_dim=6, log_std_min=-10.0, log_std_max=2.0)
    params = pol.init(jax.random.PRNGKey(0), jnp.zeros((4, 512)), jax.random.PRNGKey(1))
    action, extras = pol.apply(params, jnp.ones((4, 512)), jax.random.PRNGKey(2))
    assert action.shape == (4, 6)
    assert jnp.all(jnp.abs(action) <= 1.0)  # tanh-squashed
    assert "log_prob_pre" in extras
    assert "log_prob_post" in extras
    assert extras["log_prob_pre"].shape == (4,)

def test_squash_jacobian_numerical_safety():
    """At tanh saturation, naive 1-tanh² is 0; source uses relu(1-a²)+1e-6."""
    from jax_rl.algos.tdmpc2 import squash_log_prob_correction
    # Inputs with very large magnitude produce actions near ±1
    pre = jnp.array([20.0, -20.0])
    a = jnp.tanh(pre)
    corr = squash_log_prob_correction(a)
    # Finite, not -inf
    assert jnp.all(jnp.isfinite(corr))
```

- [ ] **Step 2: Fail.**
- [ ] **Step 3: Implement**

```python
def bound_log_std(raw: jax.Array, log_std_min: float, log_std_max: float) -> jax.Array:
    """Tanh-based mapping: low + 0.5·(high−low)·(tanh(x)+1).

    Source: common/math.py:13-14.
    """
    return log_std_min + 0.5 * (log_std_max - log_std_min) * (jnp.tanh(raw) + 1.0)


def squash_log_prob_correction(a: jax.Array) -> jax.Array:
    """Jacobian correction for tanh squash: -Σ log(relu(1 - a²) + 1e-6).

    Source: common/math.py squash(). The relu + 1e-6 floor is load-bearing numerical
    safety; naive 1 - tanh² NaNs at saturation.
    """
    return jnp.sum(jnp.log(jax.nn.relu(1.0 - a ** 2) + 1e-6), axis=-1)


def gaussian_log_prob(x: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Standard Gaussian log-prob, summed over last dim. Pre-squash x."""
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(
        ((x - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2 * jnp.pi),
        axis=-1,
    )


class PolicyPrior(nn.Module):
    """π(z) → tanh-squashed reparameterized Gaussian action.

    Returns (action, extras) where extras contains both pre- and post-squash log-probs.
    """
    mlp_dim: int
    action_dim: int
    log_std_min: float
    log_std_max: float

    @nn.compact
    def __call__(self, z, key):
        x = NormedLinear(features=self.mlp_dim)(z)
        x = NormedLinear(features=self.mlp_dim)(x)
        out = nn.Dense(
            features=2 * self.action_dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        mean, raw_log_std = jnp.split(out, 2, axis=-1)
        log_std = bound_log_std(raw_log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)
        eps = jax.random.normal(key, mean.shape)
        pre = mean + std * eps
        action = jnp.tanh(pre)
        log_prob_pre = gaussian_log_prob(pre, mean, log_std)
        log_prob_post = log_prob_pre - squash_log_prob_correction(action)
        return action, {
            "pre": pre, "mean": mean, "log_std": log_std,
            "log_prob_pre": log_prob_pre, "log_prob_post": log_prob_post,
        }
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

## Phase E — Losses

### Task E1: World model loss (consistency + reward + value, unmasked, per-H normalized)

**Files:** `jax_rl/algos/tdmpc2.py`, `tests/test_tdmpc2.py`.

- [ ] **Step 1: Failing tests** — load-bearing test 21 (loss normalization regression).

```python
def test_world_model_loss_per_h_normalization():
    """L_consistency and L_reward divided by H; L_value by (H*num_q)."""
    from jax_rl.algos.tdmpc2 import world_model_loss
    # Hand-crafted tensors with known values → compute analytically
    # Assert output matches formula with /H and /(H*num_q)
    # Full test body during implementation.
    pass  # placeholder; expand in implementation

def test_world_model_loss_no_terminal_mask():
    """Terminated=True at step h=1 should NOT mask losses at h=2."""
    from jax_rl.algos.tdmpc2 import world_model_loss
    # Construct batch with terminated at h=1; compute loss; compare to same batch
    # with terminated=False. Values should be identical (no mask).
    pass
```

- [ ] **Step 2: Fail.**
- [ ] **Step 3: Implement**

```python
def world_model_loss(
    params,          # online world-model params: encoder, dynamics, reward, q_ensemble
    target_params,   # target nets: encoder, dynamics, reward, q_ensemble (NO target policy)
    policy_params,   # online policy — passed SEPARATELY, stop-grad inside compute_td_target
    batch,           # dict from buffer.sample_sequence
    cfg: TDMPC2Config,
    key: jax.Array,
):
    """Compute world model loss. `policy_params` kept separate so jax.grad w.r.t. `params`
    does not produce spurious policy gradients.
    """
    """Compute L_world_total = consistency_coef·L_c + reward_coef·L_r + value_coef·L_v.

    All three losses are UNMASKED (source does not mask by terminated/truncated).
    Normalization: L_c and L_r divided by H; L_v divided by (H * num_q).
    Rho discount: rho^h applied per horizon step before summing.

    Returns (total_loss, metrics_dict).
    """
    H = cfg.horizon
    obs_seq = batch["obs"]          # (H+1, B, obs_dim)
    actions = batch["actions"]      # (H,   B, action_dim)
    rewards = batch["rewards"]      # (H,   B, 1)
    terminated = batch["dones"]     # (H,   B, 1); source calls this `terminated`

    # 1. Encode all observed steps with online encoder (stop-grad on target path).
    encode = lambda p, o: Encoder.apply_fn(p["encoder"], o)  # pseudo; real impl uses actual apply
    # NB: for simplicity here, use nn.apply form; actual impl will wire through the agent struct.

    z_targets = jax.lax.stop_gradient(jax.vmap(encode, in_axes=(None, 0))(params, obs_seq))
    # z_targets shape: (H+1, B, latent_dim)

    # 2. Forward-roll dynamics from z_0.
    z_0 = encode(params, obs_seq[0])  # gradient-carrying
    def scan_body(z, h):
        a = actions[h]
        z_next = Dynamics.apply(params["dynamics"], z, a)
        r_logits = Reward.apply(params["reward"], z, a)
        q_logits = QEnsemble.apply(params["q_ensemble"], z, a, deterministic=False, rngs={"dropout": key})
        return z_next, (z_next, r_logits, q_logits)
    _, (z_pred_seq, r_logits_seq, q_logits_seq) = jax.lax.scan(scan_body, z_0, jnp.arange(H))
    # z_pred_seq: (H, B, latent_dim) — predicted ẑ_{1..H}
    # r_logits_seq: (H, B, num_bins)
    # q_logits_seq: (H, num_q, B, num_bins)

    # 3. Consistency: MSE(ẑ_{h+1}, sg(z_targets_{h+1})) for h=0..H-1
    z_target_next = z_targets[1:]  # (H, B, latent_dim)
    consistency_per_h = jnp.mean((z_pred_seq - z_target_next) ** 2, axis=-1)  # (H, B)
    rho_powers = cfg.rho ** jnp.arange(H)  # (H,)
    L_consistency = (rho_powers[:, None] * consistency_per_h).mean(axis=-1).sum() / H

    # 4. Reward loss: CE(r̂_logits_h, twohot(symlog(r_h)))
    reward_ce_per_h = two_hot_ce_loss(
        r_logits_seq, rewards, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symlog=True
    )  # (H, B)
    L_reward = (rho_powers[:, None] * reward_ce_per_h).mean(axis=-1).sum() / H

    # 5. Value loss: compute target_q, CE over 5 heads summed, normalize by (H * num_q)
    # compute_td_target uses: encoder_online (params['encoder']) on obs[h+1], online policy
    # (params['policy']), target Q ensemble (target_params['q_ensemble']). (See Task E2.)
    target_q = compute_td_target(
        target_params=target_params,     # target Q ensemble
        online_wm_params=params,         # online encoder (stop-grad inside)
        policy_params=jax.lax.stop_gradient(policy_params),  # online pi, detached
        batch=batch, cfg=cfg, key=key,
    )  # (H, B, 1) — fully detached from world model grad graph
    # q_logits_seq has shape (H, num_q, B, num_bins); reshape so CE computes per head
    # Value CE sum over heads
    def value_ce_per_head(q_logits_for_head):
        # q_logits_for_head: (H, B, num_bins)
        return two_hot_ce_loss(q_logits_for_head, target_q, cfg.vmin, cfg.vmax, cfg.num_bins,
                                apply_symlog=True)  # (H, B)
    # vmap over num_q axis
    ce_all_heads = jax.vmap(value_ce_per_head, in_axes=1, out_axes=1)(q_logits_seq)  # (H, num_q, B)
    value_ce_summed = ce_all_heads.sum(axis=1)  # (H, B) — sum over heads
    L_value = (rho_powers[:, None] * value_ce_summed).mean(axis=-1).sum() / (H * cfg.num_q)

    L_total = (
        cfg.consistency_coef * L_consistency
        + cfg.reward_coef * L_reward
        + cfg.value_coef * L_value
    )
    return L_total, {
        "L_consistency_raw": L_consistency, "L_reward_raw": L_reward,
        "L_value_raw": L_value, "L_world_total": L_total,
    }
```

*Wire the actual network params properly — the pseudo-code above uses `Encoder.apply_fn` as shorthand; the real impl uses Flax's init/apply with proper `params` tree.*

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task E2: TD target — `encoder_online(obs[h+1])`, online pi, target-Q min-of-2 (decode-then-min)

**Files:** same.

- [ ] **Step 1: Failing test — Test 25 from spec (TD-target provenance)**

```python
def test_td_target_uses_encoder_online_not_dynamics():
    """Target path must use encoder_online(obs[h+1]), NOT dynamics(z_h, a_h)."""
    from jax_rl.algos.tdmpc2 import compute_td_target
    # Hand-craft batch where encoder(obs[h+1]) and dynamics(z_h, a_h) diverge.
    # Set dynamics to identity (params that make it identity) but encoder non-trivial.
    # Target should match encoder output, not dynamics output.
    # Full scaffolding below during implementation.
    pass

def test_td_target_decodes_per_q_then_mins():
    """Min is taken AFTER per-Q decoding, not on logits."""
    from jax_rl.algos.tdmpc2 import compute_td_target
    # Construct 2 Q-head logits where decoded values order differs from logit argmax ordering.
    # Assert target uses decoded min, not logit-min decoded.
    pass
```

- [ ] **Step 2: Fail.**
- [ ] **Step 3: Implement**

```python
def compute_td_target(
    target_params,     # target Q ensemble + (unused) target encoder/dynamics/reward
    online_wm_params,  # online encoder for obs[h+1] encoding
    policy_params,     # online policy (caller stop-grads before passing)
    batch,
    cfg: TDMPC2Config,
    key: jax.Array,
) -> jax.Array:
    """Source tdmpc2.py:253-264.

    Steps:
      1. next_z_h = encoder_online(obs[h+1])  for h in 0..H-1 (stop-grad)
      2. a_next_h = sample π_online(next_z_h) (stochastic, reparameterized)
      3. For each (next_z_h, a_next_h), evaluate ALL 5 target-Q heads → logits
      4. Randomly select 2 heads, decode each through two_hot_inv with symexp
      5. Elementwise min of those 2 decoded scalars
      6. target_q = r + γ·(1 − terminated)·min

    γ is taken from discount heuristic; passed via cfg (set externally).
    """
    H = cfg.horizon
    gamma = cfg.discount  # computed externally; stored on cfg at creation

    obs_next = batch["obs"][1:]            # (H, B, obs_dim)
    rewards = batch["rewards"]             # (H, B, 1)
    terminated = batch["dones"]            # (H, B, 1)

    # 1. Online encoder on real next obs (stop-grad — target is fully detached).
    # obs_next shape: (H, B, obs_dim). Encoder expects (batch, obs_dim) — LayerNorm is per-example.
    # Flatten (H, B) → (H*B,) for a single apply, then reshape back. Avoids vmap-over-time
    # which would produce correlated dropout/noise across the time axis when modules are stochastic.
    H, B = obs_next.shape[:2]
    next_z = encoder.apply(online_wm_params["encoder"], obs_next.reshape(H * B, -1))
    next_z = jax.lax.stop_gradient(next_z.reshape(H, B, -1))

    # 2. Online policy sample (policy_params already stop-gradded by caller).
    # One PRNGKey per (h, b) pair — split over both axes to avoid noise correlation across B.
    key_pi, key_q = jax.random.split(key, 2)
    keys_pi = jax.random.split(key_pi, H * B).reshape(H, B, 2)  # (H, B, 2) key array
    sample_pi = lambda z, k: policy_net.apply(policy_params, z[None], k)[0]  # (action_dim,)
    # Nested vmap over (H, B): independent PRNG noise per element
    a_next_fn = jax.vmap(jax.vmap(sample_pi, in_axes=(0, 0)), in_axes=(0, 0))
    # But PolicyPrior returns (action, extras) — just take action
    def _sample_action(z, k):
        a, _ = policy_net.apply(policy_params, z[None], k)
        return a[0]
    a_next = jax.vmap(jax.vmap(_sample_action, in_axes=(0, 0)), in_axes=(0, 0))(next_z, keys_pi)
    a_next = jax.lax.stop_gradient(a_next)  # (H, B, action_dim)

    # 3. Target Q ensemble logits, all 5 heads
    q_logits = jax.vmap(lambda z, a: QEnsemble.apply(
        target_params["q_ensemble"], z, a, deterministic=True,
    ))(next_z, a_next)
    # q_logits shape: (H, num_q, B, num_bins)

    # 4-5. Subsample 2 random heads per update-step, decode each, elementwise min
    perm = jax.random.permutation(key_q, cfg.num_q)[:2]
    selected = q_logits[:, perm, :, :]  # (H, 2, B, num_bins)
    # Softmax to probs, then two_hot_inv with symexp
    probs = jax.nn.softmax(selected, axis=-1)
    decoded = two_hot_inv(probs, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True)
    # decoded shape: (H, 2, B, 1)
    q_min = jnp.min(decoded, axis=1)  # (H, B, 1) — elementwise min across the 2 heads

    # 6. Target
    return rewards + gamma * (1.0 - terminated) * q_min
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task E3: Policy loss — correct sign + scaled_entropy pre-squash

**Files:** same.

- [ ] **Step 1: Failing test — Tests 24 + 26 from spec.**

```python
def test_policy_loss_sign():
    """L_policy = -(1/(H+1)) · Σ_h rho^h · (entropy_coef·scaled_entropy + qs_scaled).

    Outer negative, + inside. Source tdmpc2.py:227.
    """
    from jax_rl.algos.tdmpc2 import policy_loss
    # Construct known zs, verify scalar equals formula transcribed directly from source.
    pass

def test_scaled_entropy_uses_pre_squash_log_prob():
    """In single-task mode, scaled_entropy = -log_prob_pre · action_dim."""
    from jax_rl.algos.tdmpc2 import compute_scaled_entropy
    # Generate random (μ, log_σ, ε). Compute scaled_entropy two ways:
    # (a) our formula: -log_prob_pre · action_dim
    # (b) source ratio formula transcribed
    # Assert equality to 1e-6.
    pass
```

- [ ] **Step 2: Fail.**
- [ ] **Step 3: Implement**

```python
def compute_scaled_entropy(log_prob_pre: jax.Array, action_dim: int) -> jax.Array:
    """Single-task simplification of source's scaled_entropy formula.

    Source: common/world_model.py:176-183. Uses pre-squash log_prob × action_dim.
    """
    return -log_prob_pre * action_dim


def policy_loss(
    online_params,
    qscale_state: QScaleState,
    zs_detached: jax.Array,  # (H+1, B, latent_dim); detached from world model graph
    cfg: TDMPC2Config,
    key: jax.Array,
):
    """L_policy = −(1/(H+1)) · Σ_h rho^h · (entropy_coef · scaled_entropy + qs_scaled).

    Source: tdmpc2.py:219-227.
    """
    H_plus_1 = zs_detached.shape[0]
    rho_powers = cfg.rho ** jnp.arange(H_plus_1)

    # Sample actions from online policy at each latent step
    keys = jax.random.split(key, H_plus_1)
    sample_at_step = jax.vmap(lambda z, k: PolicyPrior.apply(online_params["policy"], z, k))
    a, extras = sample_at_step(zs_detached, keys)
    # extras["log_prob_pre"]: (H+1, B)

    scaled_entropy = compute_scaled_entropy(extras["log_prob_pre"], cfg.action_dim)

    # Evaluate detached Q-ensemble (avg-of-2 random heads) at these (z, a) pairs.
    # Use a stop-grad alias of online Q params (the `_detach_Qs` pattern).
    key_q = keys[0]  # same key reuse; source picks once per update
    perm = jax.random.permutation(key_q, cfg.num_q)[:2]
    q_logits_all = jax.vmap(lambda z, act: QEnsemble.apply(
        jax.lax.stop_gradient(online_params["q_ensemble"]), z, act, deterministic=True,
    ))(zs_detached, a)  # (H+1, num_q, B, num_bins)
    q_selected = q_logits_all[:, perm, :, :]  # (H+1, 2, B, num_bins)
    q_probs = jax.nn.softmax(q_selected, axis=-1)
    q_decoded = two_hot_inv(q_probs, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True)
    # (H+1, 2, B, 1)
    q_avg = q_decoded.mean(axis=1).squeeze(-1)  # (H+1, B)

    qs_scaled = qscale_apply(qscale_state, q_avg)

    # Source formula (tdmpc2.py:227):
    #   pi_loss = (-(entropy_coef * scaled_entropy + qs).mean(dim=(1,2)) * rho).mean()
    #
    # Step 1: mean over batch dim → (H+1,)
    # Step 2: multiply by rho^t → (H+1,) weighted
    # Step 3: mean over time (1/(H+1) normalization) → scalar
    # Step 4: outer negation → scalar
    #
    # CRITICAL: do NOT double-mean. The `.mean()` over axis=-1 (batch) is the ONLY
    # averaging done per-step; the time axis gets ONE `.mean()` at the end.
    per_step = cfg.entropy_coef * scaled_entropy + qs_scaled  # (H+1, B)
    per_step_mean_over_batch = per_step.mean(axis=-1)          # (H+1,) — batch mean only
    weighted = rho_powers * per_step_mean_over_batch           # (H+1,) — rho-weighted
    # One `.mean()` = 1/(H+1) normalization over time
    L_policy = -weighted.mean()

    return L_policy, {
        "scaled_entropy_mean": scaled_entropy.mean(),
        "q_avg_mean": q_avg.mean(),
        "L_policy": L_policy,
        "a_t0": jax.lax.stop_gradient(a[0]),  # (B, action_dim) — reused by Q-scale update in G2
    }
```

*Note: the `q_avg` path as-written takes `avg-of-2-decoded`. This matches source `return_type='avg'` which uses `.mean(0)` on the `(2, ...)` decoded tensor (source world_model.py:206-212).*

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task E4: `compute_all_latents` helper — roll dynamics forward to produce (H+1) latents

**Files:** `jax_rl/algos/tdmpc2.py`, `tests/test_tdmpc2.py`.

- [ ] **Step 1: Failing test** — encoder + H-step dynamics rollout produces correct shape.

```python
def test_compute_all_latents_shape():
    from jax_rl.algos.tdmpc2 import compute_all_latents
    # Build fake params, obs[0] shape (B, obs_dim), actions (H, B, action_dim)
    # Assert output zs shape = (H+1, B, latent_dim)
    pass
```

- [ ] **Step 2: Fail.**
- [ ] **Step 3: Implement**

```python
def compute_all_latents(wm_params, obs_0: jax.Array, actions: jax.Array,
                        cfg: TDMPC2Config) -> jax.Array:
    """Encode obs_0 then roll dynamics forward H steps.

    Returns zs of shape (H+1, B, latent_dim):
      zs[0] = encoder(obs_0)
      zs[h+1] = dynamics(zs[h], actions[h])  for h = 0..H-1
    """
    H = actions.shape[0]
    z_0 = Encoder.apply(wm_params["encoder"], obs_0)  # (B, latent_dim)

    def scan_body(z, a):
        z_next = Dynamics.apply(wm_params["dynamics"], z, a)
        return z_next, z_next

    _, zs_rest = jax.lax.scan(scan_body, z_0, actions)  # zs_rest: (H, B, latent_dim)
    zs = jnp.concatenate([z_0[None, :], zs_rest], axis=0)  # (H+1, B, latent_dim)
    return zs
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task E5: End-to-end loss parity against source (sanity guardrail)

**Files:** `tests/test_tdmpc2.py`.

**Purpose:** the only ironclad defense against iter-1..iter-5-style silent bugs: feed identical `(obs, actions, rewards, terminated)` and identical network params to both our JAX impl and the PyTorch source impl, compare scalar loss values.

- [ ] **Step 1: Write fixture that builds a tiny TDMPC2 model in both frameworks** with matching init (set seed, use same trunc_normal std, zero-init same output layers).

- [ ] **Step 2: Run source's `_update` once on fixed batch**, capture: `consistency_loss`, `reward_loss`, `value_loss`, `pi_loss` scalars.

- [ ] **Step 3: Run our `world_model_loss` + `policy_loss` once on same batch**, capture our scalars.

- [ ] **Step 4: Assert each pair within `1e-3` relative tolerance.** If any diverges, debug BEFORE wiring training loop.

*This test exists specifically because 5 spec-review iterations found critical bugs. If it passes, we're done with silent-failure-class bugs. If it fails, the test pinpoints which loss component is wrong.*

- [ ] **Step 5: Commit.**

```bash
git add tests/test_tdmpc2.py
git commit -m "test(tdmpc2): E5 end-to-end loss parity vs PyTorch source"
```

---

## Phase F — MPPI planner

### Task F1: MPPI core iteration (sample → rollout → score → elite select → mean/std update)

**Files:** `jax_rl/algos/tdmpc2.py`, `tests/test_tdmpc2.py`.

**Reference:** source `/tmp/tdmpc2/tdmpc2/tdmpc2.py:plan()` lines 140-207.

- [ ] **Step 1: Failing test**

```python
def test_mppi_converges_on_toy_landscape():
    """Toy landscape: reward = -||a - target||². After N MPPI iterations, mean ≈ target."""
    # Build "fake" world model where dynamics is identity and reward = -||a - target||²
    # Run plan() with fixed key; assert elite-weighted mean moves from 0 toward target.
    pass  # expand during implementation

def test_mppi_trajectory_shapes():
    """Gaussian samples have shape (horizon, num_samples - num_pi_trajs, action_dim)."""
    pass

def test_mppi_score_has_gamma_powers_and_terminal_Q():
    """score(τ) = Σ_h γ^h · r̂(z_h, a_h) + γ^H · Q_avg_of_2(z_H, π(z_H))."""
    pass
```

- [ ] **Step 2: Fail.**

- [ ] **Step 3: Implement MPPI as pure JAX function**

```python
def mppi_rollout(plan_params, z_0: jax.Array, actions_seq: jax.Array,
                 cfg: TDMPC2Config, key: jax.Array) -> jax.Array:
    """`plan_params` = {encoder, dynamics, reward, q_ensemble, policy} — see API Contracts."""
    """Roll dynamics forward H steps, accumulating discounted reward + terminal Q.

    Args:
        z_0: (N, latent_dim) — latent starting points (N = num_samples)
        actions_seq: (horizon, N, action_dim) — candidate actions per horizon step
    Returns:
        (N,) — predicted return per trajectory.
    """
    H = actions_seq.shape[0]
    gamma = cfg.discount

    def step(carry, inputs):
        z, discount_factor, G = carry
        a = inputs
        r_logits = reward_net.apply(plan_params["reward"], z, a)
        r_probs = jax.nn.softmax(r_logits, axis=-1)
        r_hat = two_hot_inv(r_probs, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True).squeeze(-1)
        G = G + discount_factor * r_hat
        z_next = dynamics.apply(plan_params["dynamics"], z, a)
        return (z_next, discount_factor * gamma, G), None

    (z_final, _, G_reward), _ = jax.lax.scan(
        step, (z_0, jnp.ones(z_0.shape[0]), jnp.zeros(z_0.shape[0])), actions_seq
    )

    # Terminal Q bootstrap: γ^H · Q_avg_of_2(z_final, π(z_final))
    key_pi, key_q = jax.random.split(key, 2)
    a_terminal, _ = policy_net.apply(plan_params["policy"], z_final, key_pi)
    q_logits = q_ensemble.apply(plan_params["q_ensemble"], z_final, a_terminal, deterministic=True)
    # Subsample 2 heads, decode, average (matches return_type='avg')
    perm = jax.random.permutation(key_q, cfg.num_q)[:2]
    q_selected = q_logits[perm]  # (2, N, num_bins)
    q_probs = jax.nn.softmax(q_selected, axis=-1)
    q_decoded = two_hot_inv(q_probs, cfg.vmin, cfg.vmax, cfg.num_bins, apply_symexp=True).squeeze(-1)
    q_terminal = q_decoded.mean(axis=0)  # (N,)

    return G_reward + (gamma ** H) * q_terminal


def mppi_iteration(mean, std, plan_params, z_0, pi_trajs, cfg, key):
    """One MPPI iteration: sample, score, update mean/std with elite-weighted stats."""
    key_sample, key_rollout = jax.random.split(key, 2)
    N_gauss = cfg.num_samples - cfg.num_pi_trajs
    # Gaussian samples: i.i.d. across (horizon, N_gauss, action_dim)
    eps = jax.random.normal(key_sample, (cfg.horizon, N_gauss, cfg.action_dim))
    gauss_actions = jnp.clip(mean[:, None, :] + std[:, None, :] * eps, -1.0, 1.0)
    # Concatenate pi_trajs (horizon, num_pi_trajs, action_dim) and gauss_actions
    actions = jnp.concatenate([pi_trajs, gauss_actions], axis=1)  # (H, num_samples, action_dim)

    # Score each trajectory by rolling from z_0
    z_0_broadcast = jnp.broadcast_to(z_0, (cfg.num_samples, z_0.shape[-1]))
    scores = mppi_rollout(plan_params, z_0_broadcast, actions, cfg, key_rollout)

    # Elite selection: top-K by score
    elite_idx = jax.lax.top_k(scores, cfg.num_elites)[1]  # (num_elites,)
    elite_scores = scores[elite_idx]
    elite_actions = actions[:, elite_idx, :]  # (H, num_elites, action_dim)

    # Elite weights: softmax with temperature
    max_score = elite_scores.max()
    exp_scores = jnp.exp((elite_scores - max_score) / cfg.mppi_temperature)
    weights = exp_scores / (exp_scores.sum() + 1e-9)  # (num_elites,)

    # Update mean/std (weighted empirical moments over elite actions)
    new_mean = (weights[None, :, None] * elite_actions).sum(axis=1)  # (H, action_dim)
    var = (weights[None, :, None] * (elite_actions - new_mean[:, None, :]) ** 2).sum(axis=1)
    new_std = jnp.clip(jnp.sqrt(var), cfg.mppi_min_std, cfg.mppi_max_std)

    return new_mean, new_std, elite_actions, weights
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

```bash
git commit -m "feat(tdmpc2): MPPI rollout + iteration core (score, elite select, mean/std update)"
```

---

### Task F2: `num_pi_trajs=24` policy seeding with `horizon` samples, `horizon-1` dynamics advances

**Files:** same.

**Reference:** source `/tmp/tdmpc2/tdmpc2/tdmpc2.py:155-165`.

- [ ] **Step 1: Failing test**

```python
def test_mppi_pi_seeds_shape():
    """Pi-seeded actions have shape (horizon, num_pi_trajs, action_dim)."""
    from jax_rl.algos.tdmpc2 import sample_pi_trajectories
    z_0 = jnp.zeros((cfg.latent_dim,))  # single-env latent
    key = jax.random.PRNGKey(0)
    pi_trajs = sample_pi_trajectories(wm_params, z_0, cfg, key)
    assert pi_trajs.shape == (cfg.horizon, cfg.num_pi_trajs, cfg.action_dim)

def test_mppi_pi_seeds_dynamics_advances_count():
    """horizon policy samples but only horizon-1 dynamics advances (latent at last step
    is z_{horizon-1}, policy sampled from it but no advance after)."""
    # Patch Dynamics.apply to count calls.
    pass
```

- [ ] **Step 2: Fail.**

- [ ] **Step 3: Implement**

```python
def sample_pi_trajectories(plan_params, z_0: jax.Array, cfg: TDMPC2Config,
                            key: jax.Array) -> jax.Array:
    """`plan_params` = {encoder, dynamics, reward, q_ensemble, policy} — see API Contracts."""
    """Roll policy prior forward through latent dynamics to produce seed trajectories.

    Source tdmpc2.py:155-165:
      - Broadcast z_0 to (num_pi_trajs, latent_dim)
      - For h in range(horizon): sample a_h ~ π(z_h), store a_h
        - For h in range(horizon - 1) only: advance z_{h+1} = dynamics(z_h, a_h)
      - Return stacked actions shape (horizon, num_pi_trajs, action_dim)
    """
    N = cfg.num_pi_trajs
    z = jnp.broadcast_to(z_0, (N,) + z_0.shape)  # (N, latent_dim)

    def step(carry, h_idx):
        z, key = carry
        key, subkey = jax.random.split(key)
        a, _ = policy_net.apply(plan_params["policy"], z, subkey)  # (N, action_dim)
        # Advance dynamics ONLY for h < horizon - 1 (last step: sample a but don't advance)
        do_advance = h_idx < cfg.horizon - 1
        z_next = jnp.where(
            do_advance,
            dynamics.apply(plan_params["dynamics"], z, a),
            z,  # don't advance at the final step
        )
        return (z_next, key), a

    _, actions = jax.lax.scan(step, (z, key), jnp.arange(cfg.horizon))
    # actions shape: (horizon, N, action_dim)
    return actions
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task F3: `_prev_mean` warm-start + `t0` reset (both per-env)

**Files:** same.

**Reference:** source `/tmp/tdmpc2/tdmpc2/tdmpc2.py:41, 167-168, 206`.

- [ ] **Step 1: Failing tests**

```python
def test_prev_mean_shift_on_t0_false():
    """mean[:-1] = prev_mean[1:]; mean[-1] = 0."""
    from jax_rl.algos.tdmpc2 import init_mppi_mean
    prev = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (horizon=3, action_dim=2)
    t0 = jnp.array(False)
    new = init_mppi_mean(prev, t0, horizon=3, action_dim=2)
    # new[0] = prev[1] = [3,4], new[1] = prev[2] = [5,6], new[2] = 0
    assert jnp.allclose(new, jnp.array([[3.0, 4.0], [5.0, 6.0], [0.0, 0.0]]))

def test_prev_mean_reset_on_t0_true():
    """On new episode, init_mppi_mean returns zeros."""
    from jax_rl.algos.tdmpc2 import init_mppi_mean
    prev = jnp.ones((3, 2))
    t0 = jnp.array(True)
    new = init_mppi_mean(prev, t0, horizon=3, action_dim=2)
    assert jnp.allclose(new, jnp.zeros((3, 2)))

def test_prev_mean_batched_env_independence():
    """Per-env shift + reset: each env's prev_mean handled independently."""
    from jax_rl.algos.tdmpc2 import init_mppi_mean_batched
    prev = jnp.tile(jnp.arange(6, dtype=jnp.float32).reshape(3, 2), (4, 1, 1))
    # (num_envs=4, horizon=3, action_dim=2)
    t0 = jnp.array([True, False, True, False])  # envs 0 and 2 just reset
    new = init_mppi_mean_batched(prev, t0, horizon=3, action_dim=2)
    # Envs 0, 2: zeros; envs 1, 3: shifted prev
    assert jnp.allclose(new[0], jnp.zeros((3, 2)))
    assert jnp.allclose(new[2], jnp.zeros((3, 2)))
    expected_shift = jnp.stack([prev[1, 1], prev[1, 2], jnp.zeros(2)])
    assert jnp.allclose(new[1], expected_shift)
```

- [ ] **Step 2: Fail.**

- [ ] **Step 3: Implement**

```python
def init_mppi_mean(prev_mean: jax.Array, t0: jax.Array, horizon: int, action_dim: int) -> jax.Array:
    """Warm-start the MPPI mean for a single env.

    If t0 is True, return zeros. Otherwise, shift prev_mean by one: new[:-1] = prev[1:], new[-1] = 0.
    """
    shifted = jnp.concatenate([prev_mean[1:], jnp.zeros((1, action_dim))], axis=0)
    return jnp.where(t0, jnp.zeros_like(shifted), shifted)


def init_mppi_mean_batched(prev_mean: jax.Array, t0: jax.Array, horizon: int,
                            action_dim: int) -> jax.Array:
    """Per-env warm-start. prev_mean: (num_envs, horizon, action_dim); t0: (num_envs,)."""
    return jax.vmap(lambda p, t: init_mppi_mean(p, t, horizon, action_dim))(prev_mean, t0)
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task F4: Single-elite Gumbel action + `std[0]`·ε noise (eval skips) + full `plan()` fn

**Files:** same.

**Reference:** source `/tmp/tdmpc2/tdmpc2/tdmpc2.py:201-207`.

- [ ] **Step 1: Failing tests**

```python
def test_mppi_gumbel_single_elite_sampling():
    """Source line 203-204: idx = sample from Categorical(weights); action = elite_actions[0, idx].

    With fixed key and fixed weights [0.9, 0.1, ...], Gumbel-softmax should pick idx=0 consistently.
    """
    from jax_rl.algos.tdmpc2 import gumbel_sample_elite
    weights = jnp.array([0.9, 0.05, 0.03, 0.02])
    elite_actions = jnp.array([[1.0], [2.0], [3.0], [4.0]])  # action_dim=1
    samples = [gumbel_sample_elite(jax.random.PRNGKey(i), weights, elite_actions) for i in range(20)]
    # Most samples should be 1.0 (the high-weight elite)
    from collections import Counter
    counts = Counter([float(s) for s in samples])
    assert counts[1.0] >= 15  # ~90% with noise tolerance

def test_mppi_noise_is_std_at_t0():
    """When eval_mode=False, action = elite_t0 + std[0]·ε.

    Verify noise magnitude matches std[0], not std of all elites."""
    pass

def test_mppi_eval_mode_skips_noise():
    """With eval_mode=True, planner returns elite_t0 without added noise."""
    pass
```

- [ ] **Step 2: Fail.**

- [ ] **Step 3: Implement Gumbel sampling + final `plan()` wrapper**

```python
def gumbel_sample_elite(key: jax.Array, weights: jax.Array, elite_actions: jax.Array) -> jax.Array:
    """Sample one elite via categorical on weights (Gumbel-softmax argmax trick).

    Args:
        weights: (num_elites,) — elite weights (should sum to 1).
        elite_actions: (horizon, num_elites, action_dim) — elite action sequences.
    Returns:
        action_t0: (action_dim,) — first action of the sampled elite trajectory.
    """
    logits = jnp.log(weights + 1e-9)
    gumbels = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-9) + 1e-9)
    idx = jnp.argmax(logits + gumbels)
    return elite_actions[0, idx]  # first action of sampled elite


def plan(plan_params, z_0: jax.Array, prev_mean: jax.Array, t0: jax.Array,
         cfg: TDMPC2Config, key: jax.Array, eval_mode: bool = False
         ) -> tuple[jax.Array, jax.Array]:
    """`plan_params` = {encoder, dynamics, reward, q_ensemble, policy} — see API Contracts.

    Caller (train script) builds it via `{**wm_params, "policy": policy_params}` before this call.
    """
    """Full MPPI planner for a single env (vmap externally for num_envs > 1).

    Args:
        z_0: (latent_dim,) — current latent.
        prev_mean: (horizon, action_dim) — previous optimized mean.
        t0: scalar bool — True if this is a new episode.
    Returns:
        (action, new_prev_mean):
          action: (action_dim,) — action to execute this step.
          new_prev_mean: (horizon, action_dim) — optimized mean for next step's warm-start.
    """
    key_init, key_pi, key_iter, key_action = jax.random.split(key, 4)

    # 1. Warm-start mean + init std
    mean = init_mppi_mean(prev_mean, t0, cfg.horizon, cfg.action_dim)
    std = jnp.full((cfg.horizon, cfg.action_dim), cfg.mppi_max_std)

    # 2. Sample 24 pi trajectories ONCE at the start (source samples once, reuses each iteration)
    pi_trajs = sample_pi_trajectories(plan_params, z_0, cfg, key_pi)

    # 3. MPPI iteration loop
    iterations = cfg.mppi_iterations + (2 if cfg.action_dim >= 20 else 0)

    def iter_body(carry, key_i):
        mean_c, std_c = carry
        new_mean, new_std, elite_actions, weights = mppi_iteration(
            mean_c, std_c, plan_params, z_0, pi_trajs, cfg, key_i
        )
        return (new_mean, new_std), (elite_actions, weights)

    iter_keys = jax.random.split(key_iter, iterations)
    (final_mean, final_std), (all_elites, all_weights) = jax.lax.scan(
        iter_body, (mean, std), iter_keys
    )
    # Use final iteration's elites + weights for action sampling
    elite_actions = all_elites[-1]  # (horizon, num_elites, action_dim)
    weights = all_weights[-1]       # (num_elites,)

    # 4. Sample single elite via Gumbel, take its t=0 action
    action = gumbel_sample_elite(key_action, weights, elite_actions)

    # 5. Add exploration noise (skip in eval_mode)
    noise = jax.random.normal(jax.random.fold_in(key_action, 1), (cfg.action_dim,)) * final_std[0]
    action = jnp.where(eval_mode, action, action + noise)
    action = jnp.clip(action, -1.0, 1.0)

    return action, final_mean  # new_prev_mean = final_mean


# Batched MPPI over envs:
plan_batched = jax.vmap(plan, in_axes=(None, 0, 0, 0, None, 0, None))
#   wm_params: shared across envs (None)
#   z_0, prev_mean, t0, key: per-env (0)
#   cfg, eval_mode: static (None)
```

- [ ] **Step 4: Pass all F1-F4 tests.**

```bash
uv run python -m pytest tests/test_tdmpc2.py -v -k mppi
```

- [ ] **Step 5: Commit.**

```bash
git commit -m "feat(tdmpc2): MPPI plan() with Gumbel elite sampling, std[0] noise, eval skip, vmap over envs"
```

---

## Phase G — Agent API / TrainState

### Task G1: TrainState

**Files:** `jax_rl/algos/tdmpc2.py`.

- [ ] **Step 1: Failing test** — TrainState instantiation + dtype checks.
- [ ] **Step 2: Fail.**
- [ ] **Step 3: Implement**

```python
@flax.struct.dataclass
class TDMPC2State:
    # Online params
    encoder_params: Any
    dynamics_params: Any
    reward_params: Any
    q_ensemble_params: Any
    policy_params: Any
    # Target params (for encoder, dynamics, reward, q — NOT policy)
    encoder_target_params: Any
    dynamics_target_params: Any
    reward_target_params: Any
    q_ensemble_target_params: Any
    # Optimizers
    world_model_opt_state: Any    # single Adam w/ multi_transform; world model + qs + task_emb
    policy_opt_state: Any         # separate Adam
    # Q-scale tracker
    qscale: QScaleState
    # MPPI state (per-env)
    prev_mean: jax.Array          # (num_envs, horizon, action_dim)
    # RNG
    key: jax.Array
    # Step counter
    step: jax.Array
```

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task G2: Update function (world model → policy → target EMA → Q-scale)

**Files:** same.

- [ ] **Step 1: Failing test — Test 11 smoke** — one update, all scalars finite.

```python
def test_update_smoke_no_nan():
    """Single update step: all losses finite, params changed, target moved by tau*delta."""
    pass
```

- [ ] **Step 2: Fail.**
- [ ] **Step 3: Implement**

```python
def make_update_step(cfg: TDMPC2Config, wm_optimizer, policy_optimizer):
    """Factory: returns a jit'd update_step closed over the (non-pytree) optimizers.

    Optax GradientTransformation is not a pytree leaf, so it cannot live on cfg/state.
    Caller builds optimizers once in the train script and calls this factory to get the
    jit'd step fn. cfg is `@dataclass(frozen=True)`, hashable, safe as a closure over @jit.

    IMPORTANT — INDENTATION: every line below up to `return update_step` must be nested
    inside `update_step`. Do NOT flatten when transcribing.
    """
    @jax.jit
    def update_step(state: TDMPC2State, batch: dict) -> tuple[TDMPC2State, dict]:
        """One gradient step. Order matters: world model → policy (on detached latents) → EMA → Q-scale."""
        # (all the body that follows must be indented 8 spaces — inside update_step)
    key_wm, key_pol, key_next = jax.random.split(state.key, 3)

    # 1. World model update
    wm_params = {
        "encoder": state.encoder_params, "dynamics": state.dynamics_params,
        "reward": state.reward_params, "q_ensemble": state.q_ensemble_params,
    }
    target_params = {
        "encoder": state.encoder_target_params, "dynamics": state.dynamics_target_params,
        "reward": state.reward_target_params, "q_ensemble": state.q_ensemble_target_params,
    }
    # world_model_loss receives policy_params SEPARATELY (not inside wm_params), so jax.grad
    # w.r.t. wm_params does not produce policy gradients.
    (wm_loss, wm_metrics), wm_grads = jax.value_and_grad(
        world_model_loss, has_aux=True, argnums=0  # grad w.r.t. params (argnum 0)
    )(wm_params, target_params, state.policy_params, batch, cfg, key_wm)
    # Apply grad (multi_transform optimizer with encoder at scaled LR)
    wm_updates, new_wm_opt_state = wm_optimizer.update(
        wm_grads, state.world_model_opt_state, wm_params
    )
    wm_params_new = optax.apply_updates(wm_params, wm_updates)

    # 2. Policy update (detached latents from updated world model)
    # Encode + roll dynamics once to get (H+1) latents, then stop-grad.
    zs_detached = jax.lax.stop_gradient(
        compute_all_latents(wm_params_new, batch["obs"][0], batch["actions"], cfg)
    )
    pol_params = {"policy": state.policy_params, "q_ensemble": wm_params_new["q_ensemble"]}
    (pol_loss, pol_metrics), pol_grads = jax.value_and_grad(
        policy_loss, has_aux=True
    )(pol_params, state.qscale, zs_detached, cfg, key_pol)
    pol_updates, new_pol_opt_state = policy_optimizer.update(
        pol_grads, state.policy_opt_state, pol_params
    )
    policy_params_new = optax.apply_updates(pol_params, pol_updates)["policy"]

    # 3. Q-scale update from t=0 avg-of-2 Q values on zs_detached.
    # Source tdmpc2.py:222: `self.scale.update(qs[0])` where qs is shape (H+1, B, 1) avg-of-2.
    # We extract q_avg at t=0 the same way policy_loss does (avg of 2 random heads decoded).
    key_qscale = jax.random.split(key_pol)[0]
    perm_qs = jax.random.permutation(key_qscale, cfg.num_q)[:2]
    q_logits_for_scale = q_ensemble.apply(
        jax.lax.stop_gradient(wm_params_new["q_ensemble"]),
        zs_detached[0],                    # t=0 latents, (B, latent_dim)
        jax.lax.stop_gradient(pol_metrics["a_t0"]),  # t=0 sampled action (added to pol_metrics)
        deterministic=True,
    )  # (num_q, B, num_bins)
    q_sel = q_logits_for_scale[perm_qs]   # (2, B, num_bins)
    q_dec = two_hot_inv(jax.nn.softmax(q_sel, -1), cfg.vmin, cfg.vmax, cfg.num_bins,
                        apply_symexp=True).squeeze(-1)  # (2, B)
    q_avg_t0 = q_dec.mean(axis=0)          # (B,)
    new_qscale = qscale_update(state.qscale, q_avg_t0, tau=cfg.tau)

    # 4. Target EMA
    def ema_tree(target, online, tau):
        return jax.tree_util.tree_map(lambda t, o: t + tau * (o - t), target, online)
    new_target_encoder = ema_tree(state.encoder_target_params, wm_params_new["encoder"], cfg.tau)
    new_target_dynamics = ema_tree(state.dynamics_target_params, wm_params_new["dynamics"], cfg.tau)
    new_target_reward = ema_tree(state.reward_target_params, wm_params_new["reward"], cfg.tau)
    new_target_q = ema_tree(state.q_ensemble_target_params, wm_params_new["q_ensemble"], cfg.tau)

    # 5. Pack new state
    new_state = state.replace(
        encoder_params=wm_params_new["encoder"],
        dynamics_params=wm_params_new["dynamics"],
        reward_params=wm_params_new["reward"],
        q_ensemble_params=wm_params_new["q_ensemble"],
        policy_params=policy_params_new,
        encoder_target_params=new_target_encoder,
        dynamics_target_params=new_target_dynamics,
        reward_target_params=new_target_reward,
        q_ensemble_target_params=new_target_q,
        world_model_opt_state=new_wm_opt_state,
        policy_opt_state=new_pol_opt_state,
        qscale=new_qscale,
        key=key_next,
        step=state.step + 1,
    )

    return new_state, {**wm_metrics, **pol_metrics}
    # ^^^ end of update_step body (all lines since `key_wm, key_pol, key_next = ...` must
    # be indented 8 spaces — inside update_step). The code samples above are written at
    # 4-space indent for readability; the transcriber MUST indent the whole block one level
    # deeper when copying into the real file.

    return update_step   # OUTSIDE update_step, inside make_update_step — returns the jit'd fn
```

*Note:* `policy_loss` must add `a_t0 = sampled_actions[0]` (the t=0 sampled actions, stop-gradded) to its returned metrics dict so Q-scale can reuse it without re-sampling. See Task E3 and add `"a_t0": jax.lax.stop_gradient(a[0])` to the metrics return.

- [ ] **Step 4: Pass.**
- [ ] **Step 5: Commit.**

---

### Task G3: Single Adam with `multi_transform` for encoder-scaled LR

**Files:** same.

- [ ] **Step 1: Failing test** — check that after one step, encoder params moved less than other params (consistent with lower LR).

- [ ] **Step 2-5:** Implement via Optax `optax.multi_transform`:

```python
def build_world_model_optimizer(cfg: TDMPC2Config):
    """Single Adam with per-param-group LR via multi_transform.

    Group "a" (scaled LR): encoder  → lr · enc_lr_scale = 9e-5
    Group "b" (default LR): dynamics, reward, q_ensemble (+ task_emb if present) → lr = 3e-4
    """
    tx_a = optax.adam(cfg.lr * cfg.enc_lr_scale)
    tx_b = optax.adam(cfg.lr)

    def label_fn(params):
        """Walk the params pytree and label each LEAF based on its path prefix.

        Returns a pytree with identical structure, leaves replaced by "a" or "b".
        The top-level dict should have keys {"encoder", "dynamics", "reward", "q_ensemble"[, "task_emb"]}.
        """
        from jax.tree_util import tree_map_with_path

        def _label(path, _leaf):
            # `path` is a tuple of keys like (DictKey("encoder"), DictKey("Dense_0"), ...)
            # Top-level key is path[0]. Use DictKey('key').key attribute.
            top_key = path[0].key if hasattr(path[0], "key") else str(path[0])
            return "a" if top_key == "encoder" else "b"

        return tree_map_with_path(_label, params)

    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.multi_transform({"a": tx_a, "b": tx_b}, label_fn),
    )
```

**Test it explicitly:**
```python
def test_multi_transform_label_fn():
    # Build fake params tree with encoder + dynamics top-level dicts
    params = {
        "encoder": {"Dense_0": {"kernel": jnp.zeros((4, 8)), "bias": jnp.zeros((8,))}},
        "dynamics": {"Dense_0": {"kernel": jnp.zeros((4, 8))}},
    }
    tx = build_world_model_optimizer(TDMPC2Config())
    state = tx.init(params)
    # Take a step with grads of 1.0 → encoder should move by ~9e-5, dynamics by ~3e-4
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, state, params)
    new = optax.apply_updates(params, updates)
    enc_delta = jnp.abs(new["encoder"]["Dense_0"]["kernel"]).mean()
    dyn_delta = jnp.abs(new["dynamics"]["Dense_0"]["kernel"]).mean()
    assert dyn_delta > enc_delta * 3.0  # default LR is > 3× encoder LR
```


- [ ] **Step 5: Commit.**

---

## Phase H — Training script

### Task H1: `train_tdmpc2.py` skeleton — CLI, env, buffer, train state init

**Files:**
- Create: `train_tdmpc2.py`

- [ ] **Step 1: Draft CLI and init sequence** — mirror `train_flashsac.py` or `train_fast_sac.py` for patterns.

```bash
head -100 train_flashsac.py  # study existing skeleton
```

- [ ] **Step 2: Implement** — CLI with `--env`, `--num-envs`, `--total-timesteps`, `--seed`, `--eval-every`, `--collect-mode {mppi,prior}`, `--ckpt-dir`.

- [ ] **Step 3: Init** — env (Playground registry), buffer (with episode_id tracking), train state (fresh params).

- [ ] **Step 4: Test** — run `uv run python train_tdmpc2.py --env CheetahRun --total-timesteps 0` and verify no crash at init.

- [ ] **Step 5: Commit.**

---

### Task H2: Warmup — `seed_steps` random actions + gradient burst at boundary

**Files:** `train_tdmpc2.py`.

- [ ] **Step 1: Implement warmup**

```python
# During seed_steps, collect with uniform random actions, no updates.
for step in range(cfg.seed_steps):
    a = np.random.uniform(-1, 1, (cfg.num_envs, action_dim))
    obs, r, done, trunc, _ = env.step(a)
    buffer.add_batch(prev_obs, a, r, obs, done, trunc)
    prev_obs = obs

# At seed_steps boundary: burst of seed_steps gradient updates.
for _ in range(cfg.seed_steps):
    batch = buffer.sample_sequence(cfg.batch_size, cfg.horizon, next_key())
    state, metrics = update_step(state, batch, cfg)
```

- [ ] **Step 2: Unit test** — run warmup with tiny `seed_steps=50`, assert state updated `50 + 50 = 100` times total.

- [ ] **Step 3-5:** Commit.

---

### Task H3: Main loop — collect (MPPI or prior) + update UTD=1

**Files:** `train_tdmpc2.py`.

- [ ] **Step 1: Implement main loop**

```python
for step in range(cfg.seed_steps, cfg.total_steps):
    if cfg.collect_mode == "mppi":
        a, new_prev_mean = mppi_plan(state, obs_current, t0=episode_reset_mask, eval_mode=False)
        state = state.replace(prev_mean=new_prev_mean)
    else:  # prior
        a, _ = sample_policy_prior(state, obs_current, eval_mode=False)

    obs_next, r, done, trunc, _ = env.step(a)
    buffer.add_batch(obs_current, a, r, obs_next, done, trunc)
    obs_current = obs_next

    # UTD=1 update
    batch = buffer.sample_sequence(cfg.batch_size, cfg.horizon, next_key())
    state, metrics = update_step(state, batch, cfg)

    if step % cfg.eval_every == 0:
        eval_metrics = run_eval(state, env, cfg)
        log_to_wandb(step, metrics, eval_metrics)
```

- [ ] **Step 2-5:** smoke test + commit.

---

### Task H4: Eval — both MPPI and prior modes per episode

**Files:** `train_tdmpc2.py`.

- [ ] **Step 1: Implement `run_eval`**

```python
def run_eval(state, env, cfg):
    """Run num_eval_envs episodes in both MPPI and prior mode."""
    # Separate _prev_mean for eval (don't touch collect's).
    eval_prev_mean = jnp.zeros((cfg.num_eval_envs, cfg.horizon, action_dim))
    prior_returns = []
    mppi_returns = []
    for ep in range(cfg.eval_episodes):
        # ... roll out with MPPI, get return
        # ... roll out with prior, get return
        pass
    return {
        "mppi_return": np.mean(mppi_returns),
        "prior_return": np.mean(prior_returns),
        "mppi_prior_gap": np.mean(mppi_returns) - np.mean(prior_returns),
    }
```

*Important:* eval `prev_mean` must be a local variable, NOT `state.prev_mean`, to avoid the Test 27 isolation bug.

- [ ] **Step 2: Unit test (Test 27)** — interleave collect, eval, collect; assert collect `prev_mean` unchanged by eval.

- [ ] **Step 3-5:** Commit.

---

### Task H5: Checkpointing + metrics logging

**Files:** `train_tdmpc2.py`.

- [ ] **Step 1:** mirror existing checkpoint pattern (`checkpoint_mgr.save(step, state)`). Save both `actor_params.npy` (policy prior only) and `world_model_params.npy` (full world model for MPPI).

- [ ] **Step 2:** write `metrics.csv` + wandb init.

- [ ] **Step 3:** test resume from checkpoint — load, continue training 1 step, verify no crash.

- [ ] **Step 4-5:** commit.

---

## Phase I — Probe tests (the ones we've been referring to)

### Task I1: Failure probes (tests 17-19)

**Files:** `tests/test_tdmpc2.py`.

- [ ] **Step 1: Implement test 17 (NaN injection)** — inject NaN into encoder output, assert MPPI skips and doesn't crash.

- [ ] **Step 2: Implement test 18 (reward saturation)** — env with reward=1e4, verify symlog path, saturation warning.

- [ ] **Step 3: Implement test 19 (truncation semantics)** — `terminated=True` at h=1 in H=3 window. Assert reward/value/consistency losses at h=2 ARE computed (not masked); TD target bootstrap at h=1 IS zeroed.

- [ ] **Step 4:** run all three, verify pass.

- [ ] **Step 5: Commit.**

---

### Task I2: Structural tests (20-23)

**Files:** `tests/test_tdmpc2.py`.

- [ ] **Step 1: Test 20 (cross-episode buffer)** — already partly covered in Phase B tests; extend with rejection-rate diagnostic.

- [ ] **Step 2: Test 21 (loss normalization regression)** — hand-crafted batch, assert `/H` and `/(H·num_q)` factors present.

- [ ] **Step 3: Test 22 (Q-scale range clamp)** — `p95-p5 = 0.01` input, assert clamp to 1.0.

- [ ] **Step 4: Test 23 (truncation vs terminated)** — construct batch with mixed `terminated`/`truncated`/normal; verify losses don't mask, TD target only multiplies by `(1-terminated)`.

- [ ] **Step 5: Commit.**

---

### Task I3: Critical-bug regression tests (24-27)

**Files:** `tests/test_tdmpc2.py`.

- [ ] **Step 1: Test 24 (policy loss sign)** — transcribe source formula, compare our scalar to 1e-6.

- [ ] **Step 2: Test 25 (TD-target provenance)** — hand-craft where `encoder(obs[h+1]) ≠ dynamics(z_h, a_h)`, assert target uses encoder path.

- [ ] **Step 3: Test 26 (scaled_entropy equality)** — verify `-log_prob_pre·action_dim` matches source's ratio formula.

- [ ] **Step 4: Test 27 (eval/collect `_prev_mean` isolation)** — interleave collect/eval/collect, assert collect `_prev_mean` untouched.

- [ ] **Step 5: Commit.**

---

## Phase J — Smoke + benchmark

### Task J1: Integration smoke (Test 11)

**Files:** `tests/test_tdmpc2.py`.

- [ ] **Step 1:** 1 collect step, 1 buffer add, 1 sample, 1 update. Assert all losses finite, all state fields updated.

- [ ] **Step 2-5:** commit.

---

### Task J2: DMC CartpoleSwingup toy (Test 12)

**Files:** none new (use `train_tdmpc2.py`).

- [ ] **Step 1:** run `uv run python train_tdmpc2.py --env CartpoleSwingup --total-timesteps 50000 --seed 0`.

- [ ] **Step 2:** assert final `mppi_return > 200` (random-policy is ~80; any learning signal crosses 200).

- [ ] **Step 3:** document result in `.context/journals/2026-04-XX.md`.

- [ ] **Step 4: Commit journal.**

---

### Task J3: DMC CheetahRun benchmark (blocking before claiming success)

**Files:** journal + `.context/AGENT_HANDOFF.md` benchmark table.

- [ ] **Step 1:** launch 1M-step run in background:

```bash
nohup uv run python train_tdmpc2.py --env CheetahRun --total-timesteps 1000000 --seed 42 \
  > /tmp/tdmpc2-cheetah.log 2>&1 &
```

- [ ] **Step 2:** monitor periodically with `grep 'EVAL' /tmp/tdmpc2-cheetah.log | tail -20`.

- [ ] **Step 3:** when complete, record:
  - Peak `mppi_return`
  - Peak `prior_return`
  - Gap at end of training
  - Final `L_world_total` components
  - Wall-clock time

- [ ] **Step 4:** bar: `mppi_return ≥ 850`. If fails, diagnose per §10 failure modes. If passes, update `.context/AGENT_HANDOFF.md` benchmark table.

- [ ] **Step 5: Commit journal + handoff updates.**

---

### Task J4: DMC HumanoidRun benchmark (blocking)

**Files:** same as J3.

- [ ] **Step 1-5:** repeat J3 for HumanoidRun. Bar: `mppi_return ≥ 800`.

---

## Post-implementation

- [ ] **Doc sync checkpoint** (per project CLAUDE.md):
  - Update `.context/TODO.md` — mark TD-MPC2 P1 complete
  - Write `.context/journals/2026-04-XX.md` with benchmark results
  - Add TD-MPC2 row to algo quick-reference table in `.context/AGENT_HANDOFF.md`
  - Check if `docs/scripts/gen_cli_reference.py` and `gen_env_presets.py` need reruns

- [ ] **Retire spec** (optional) — move to `.superpowers/specs/archive/` once implementation matches.

---

## Gotchas to remember

1. `uv run python`, never bare python.
2. No `Co-Authored-By` in commits.
3. Do NOT pipe background tasks through `| head -N` or `| tail -N` — kills the process.
4. JIT compile takes 1-3 min at start of training — empty output is normal.
5. If NaN appears: check (a) SimNorm clip on encoder pre-activation, (b) two-hot saturation warnings, (c) policy `log_std_min` bound.
6. Check lessons files (`.context/lessons/*.md`) BEFORE debugging — especially `jax_performance.md`, `offpolicy.md`, `distributional.md`.

---

## Reference skills & files

- `@/home/stevenman/Desktop/Work/Research/jax-learning/.superpowers/specs/2026-04-21-tdmpc2-design.md` — the full design spec
- Source: `/tmp/tdmpc2/` — keep open while implementing
- `jax_rl/algos/sac.py`, `jax_rl/algos/fast_sac.py` — pattern references
- `train_flashsac.py`, `train_fast_sac.py` — training script references
- `.context/lessons/distributional.md` — C51 / HL-Gauss precedent lessons
- `.context/lessons/offpolicy.md` — truncation handling, exploration HPs
