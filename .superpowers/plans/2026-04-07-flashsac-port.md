# FlashSAC JAX Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port FlashSAC from PyTorch to JAX/Flax as a standalone algorithm + training script, for A/B benchmarking against FastSAC.

**Architecture:** Bottom-up build: reward scaling utils → network blocks → algo class → training script. Each layer is independently testable. The algo uses Flax BatchNorm with mutable state, weight normalization as a post-step projection, and a cross-batch (2B) forward pass pattern for shared BN statistics.

**Tech Stack:** JAX, Flax (linen), optax, existing `jax_rl` infrastructure (replay buffer, env factory, distributional utils)

**Spec:** `.superpowers/specs/2026-04-07-flashsac-port-design.md`
**Reference PyTorch impl:** `/tmp/FlashSAC/` (clone of https://github.com/Holiday-Robot/FlashSAC)

---

## File Map

| File | Responsibility | Dependencies |
|------|---------------|-------------|
| `jax_rl/utils/reward_scaling.py` | RewardNormState, update/scale functions | flax, jax (no internal deps) |
| `jax_rl/configs/flash_sac_config.py` | FlashSACConfig dataclass | None |
| `jax_rl/networks/flash_blocks.py` | FlashSACEmbedder, FlashSACBlock, UnitRMSNorm, FlashSACActor, FlashSACCritic | flax.linen |
| `jax_rl/algos/flash_sac.py` | FlashSAC class (TrainingState, NoiseState, update, init, select_action, weight norm) | flash_blocks, flash_sac_config, distributional utils |
| `train_flashsac.py` | Standalone training script | flash_sac algo, env factory, replay buffer |
| `tests/test_flash_blocks.py` | Network architecture tests | flash_blocks |
| `tests/test_reward_scaling.py` | Reward normalizer tests | reward_scaling |
| `tests/test_flash_sac.py` | Algo integration tests | flash_sac |

---

## Task 1: Reward Scaling Utility

**Files:**
- Create: `jax_rl/utils/reward_scaling.py`
- Create: `tests/test_reward_scaling.py`

- [ ] **Step 1: Write failing tests for RewardNormState**

```python
# tests/test_reward_scaling.py
import jax
import jax.numpy as jnp
import pytest
from jax_rl.utils.reward_scaling import (
    RewardNormState, init_reward_norm, update_reward_stats, scale_reward,
)

def test_init_creates_correct_shapes():
    state = init_reward_norm(num_envs=4)
    assert state.G_r.shape == (4,)
    assert state.G_r_max.shape == ()
    assert state.G_mean.shape == ()
    assert state.G_var.shape == ()
    assert state.G_count.shape == ()
    assert state.G_var == 1.0  # init to 1, not 0

def test_update_resets_on_done():
    state = init_reward_norm(num_envs=2)
    reward = jnp.array([1.0, 2.0])
    terminated = jnp.array([0.0, 1.0])
    truncated = jnp.array([0.0, 0.0])
    new_state = update_reward_stats(state, reward, terminated, truncated, gamma=0.99)
    # Env 0: G_r = 0.99 * 0.0 + 1.0 = 1.0
    # Env 1: done=1, so G_r = 0.99 * (1-1) * 0.0 + 2.0 = 2.0
    assert jnp.allclose(new_state.G_r, jnp.array([1.0, 2.0]))

def test_scale_reward_bounds_output():
    state = init_reward_norm(num_envs=1)
    # Manually set variance high so denominator is sqrt(var)
    state = state.replace(G_var=jnp.array(100.0), G_r_max=jnp.array(50.0))
    reward = jnp.array([10.0])
    scaled = scale_reward(state, reward, G_max=5.0)
    # denominator = max(sqrt(100 + 1e-8), 50/5) = max(10, 10) = 10
    assert jnp.allclose(scaled, jnp.array([1.0]))

def test_scale_reward_uses_G_r_max_floor():
    state = init_reward_norm(num_envs=1)
    # Low variance but high G_r_max → floor kicks in
    state = state.replace(G_var=jnp.array(0.01), G_r_max=jnp.array(25.0))
    reward = jnp.array([5.0])
    scaled = scale_reward(state, reward, G_max=5.0)
    # denominator = max(sqrt(0.01 + 1e-8), 25/5) = max(0.1, 5.0) = 5.0
    assert jnp.allclose(scaled, jnp.array([1.0]))

def test_welford_update_accumulates():
    state = init_reward_norm(num_envs=4)
    key = jax.random.PRNGKey(0)
    for i in range(10):
        reward = jax.random.normal(key, (4,))
        key, _ = jax.random.split(key)
        terminated = jnp.zeros(4)
        truncated = jnp.zeros(4)
        state = update_reward_stats(state, reward, terminated, truncated, gamma=0.99)
    assert state.G_count > 0
    assert state.G_var > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_reward_scaling.py -v
```
Expected: `ModuleNotFoundError` — module doesn't exist yet.

- [ ] **Step 3: Implement reward_scaling.py**

Create `jax_rl/utils/reward_scaling.py` with:
- `RewardNormState` — `@flax.struct.dataclass` with fields: `G_r`, `G_r_max`, `G_mean`, `G_var`, `G_count`
- `init_reward_norm(num_envs)` → `RewardNormState`
- `update_reward_stats(state, reward, terminated, truncated, gamma)` → new state
  - `done = jnp.maximum(terminated, truncated)` (both reset the return)
  - `G_r = gamma * (1 - done) * G_r + reward`
  - `G_r_max = max(G_r_max, max(|G_r|))`
  - Welford update on G_r (use epsilon=1e-4 in `m_a = running_var * (running_count + 1e-4)`)
- `scale_reward(state, reward, G_max=5.0, eps=1e-8)` → scaled reward
  - `r / max(sqrt(G_var + eps), G_r_max / G_max)`

Reference: `/tmp/FlashSAC/flash_rl/agents/utils/reward_normalization.py`

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_reward_scaling.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/utils/reward_scaling.py tests/test_reward_scaling.py
git commit -m "feat: add adaptive reward scaling utility for FlashSAC"
```

---

## Task 2: FlashSAC Config

**Files:**
- Create: `jax_rl/configs/flash_sac_config.py`

- [ ] **Step 1: Create config dataclass**

Create `jax_rl/configs/flash_sac_config.py` with `FlashSACConfig` dataclass. All defaults from spec Section 8 — copy values exactly from the spec. Follow the pattern in `jax_rl/configs/fast_sac_config.py` for style (standalone dataclass, no inheritance, docstring explaining provenance).

Fields:
```python
@dataclass
class FlashSACConfig:
    # Architecture
    num_blocks: int = 2
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    expansion: int = 4
    num_atoms: int = 101
    v_min: float = -5.0
    v_max: float = 5.0

    # Training
    tau: float = 0.01
    policy_delay: int = 2
    batch_size: int = 2048
    buffer_size: int = 1_000_000
    min_buffer_size: int = 10_000
    grad_updates_per_step: int = 1
    gamma: float = 0.99
    n_step: int = 1

    # Temperature
    alpha_init: float = 0.01
    sigma_target: float = 0.15

    # Actor regularization
    bc_alpha: float = 0.0

    # Reward scaling
    normalize_reward: bool = True
    G_max: float = 5.0

    # Exploration
    noise_zeta_mu: float = 2.0
    noise_zeta_max: int = 16

    # LR schedule (warmup → cosine decay) — shared by actor, critic, AND temperature
    lr_init: float = 3e-4
    lr_peak: float = 3e-4
    lr_end: float = 1.5e-4
    lr_warmup_frac: float = 1e-6
    lr_decay_frac: float = 1.0

    # Weight norm
    weight_norm: bool = True
```

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "from jax_rl.configs.flash_sac_config import FlashSACConfig; print(FlashSACConfig())"
```
Expected: prints config with all defaults.

- [ ] **Step 3: Commit**

```bash
git add jax_rl/configs/flash_sac_config.py
git commit -m "feat: add FlashSACConfig dataclass"
```

---

## Task 3: Network Blocks

**Files:**
- Create: `jax_rl/networks/flash_blocks.py`
- Create: `tests/test_flash_blocks.py`

- [ ] **Step 1: Write failing tests for FlashSACBlock**

```python
# tests/test_flash_blocks.py
import jax
import jax.numpy as jnp
import pytest
from jax_rl.networks.flash_blocks import (
    FlashSACEmbedder, FlashSACBlock, UnitRMSNorm,
    FlashSACActor, FlashSACCritic,
)

HIDDEN_DIM = 32  # small for tests
OBS_DIM = 12
ACTION_DIM = 4
BATCH = 8
KEY = jax.random.PRNGKey(0)

def test_embedder_shapes():
    model = FlashSACEmbedder(hidden_dim=HIDDEN_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    assert 'params' in variables
    assert 'batch_stats' in variables
    out = model.apply(variables, jnp.ones((BATCH, OBS_DIM)), train=False)
    assert out.shape == (BATCH, HIDDEN_DIM)

def test_embedder_bn_updates_on_train():
    model = FlashSACEmbedder(hidden_dim=HIDDEN_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    x = jax.random.normal(KEY, (BATCH, OBS_DIM))
    out, updates = model.apply(variables, x, train=True, mutable=['batch_stats'])
    assert 'batch_stats' in updates
    # Running mean should have shifted from zeros
    old_mean = variables['batch_stats']['BatchNorm_0']['mean']
    new_mean = updates['batch_stats']['BatchNorm_0']['mean']
    assert not jnp.allclose(old_mean, new_mean)

def test_block_residual_connection():
    model = FlashSACBlock(hidden_dim=HIDDEN_DIM, expansion=4)
    x = jnp.ones((BATCH, HIDDEN_DIM))
    variables = model.init(KEY, x, train=False)
    out = model.apply(variables, x, train=False)
    assert out.shape == (BATCH, HIDDEN_DIM)
    # Output should differ from input (non-identity) but not be zero
    assert not jnp.allclose(out, x)
    assert not jnp.allclose(out, jnp.zeros_like(out))

def test_rms_norm_output_shape():
    model = UnitRMSNorm()
    variables = model.init(KEY, jnp.zeros((BATCH, HIDDEN_DIM)))
    out = model.apply(variables, jnp.ones((BATCH, HIDDEN_DIM)))
    assert out.shape == (BATCH, HIDDEN_DIM)

def test_actor_output_shapes():
    model = FlashSACActor(
        hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4,
        action_dim=ACTION_DIM,
    )
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    (mean, log_std) = model.apply(variables, jnp.ones((BATCH, OBS_DIM)), train=False)
    assert mean.shape == (BATCH, ACTION_DIM)
    assert log_std.shape == (BATCH, ACTION_DIM)

def test_critic_output_shapes():
    model = FlashSACCritic(
        hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4,
        num_atoms=51,
    )
    obs = jnp.zeros((BATCH, OBS_DIM))
    act = jnp.zeros((BATCH, ACTION_DIM))
    variables = model.init(KEY, obs, act, train=False)
    logits = model.apply(variables, obs, act, train=False)
    assert logits.shape == (BATCH, 51)

def test_actor_bn_mutable_forward():
    model = FlashSACActor(
        hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4,
        action_dim=ACTION_DIM,
    )
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    x = jax.random.normal(KEY, (BATCH, OBS_DIM))
    (mean, log_std), updates = model.apply(
        variables, x, train=True, mutable=['batch_stats'])
    assert 'batch_stats' in updates
    assert mean.shape == (BATCH, ACTION_DIM)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_flash_blocks.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement flash_blocks.py**

Create `jax_rl/networks/flash_blocks.py` with these Flax modules:

**`FlashSACEmbedder(nn.Module)`:**
- Attributes: `hidden_dim: int`
- `__call__(self, x, train)`:
  - `x = nn.BatchNorm(momentum=0.01, epsilon=1e-5, use_running_average=not train)(x)`
  - `x = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=nn.initializers.orthogonal())(x)`
  - No activation (intentional — first block provides nonlinearity)

**`FlashSACBlock(nn.Module)`:**
- Attributes: `hidden_dim: int`, `expansion: int = 4`
- `__call__(self, x, train)`:
  - `residual = x`
  - `x = nn.Dense(self.hidden_dim * self.expansion, use_bias=False, kernel_init=nn.initializers.orthogonal())(x)`
  - `x = nn.BatchNorm(momentum=0.01, epsilon=1e-5, use_running_average=not train)(x)`
  - `x = nn.relu(x)`
  - `x = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=nn.initializers.orthogonal())(x)`
  - `x = nn.BatchNorm(momentum=0.01, epsilon=1e-5, use_running_average=not train)(x)`
  - `x = nn.relu(x)`
  - `x = x + residual`

**`UnitRMSNorm(nn.Module)`:**
- `__call__(self, x)`:
  - `scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))`
  - `rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)`
  - `return (x / rms) * scale`

**`FlashSACActor(nn.Module)`:**
- Attributes: `hidden_dim`, `num_blocks`, `expansion`, `action_dim`
- `__call__(self, obs, train)`:
  - Embedder → blocks → RMSNorm → mean/logstd heads (UnitDense + free bias)
  - `log_std = -10 + 12 * 0.5 * (1 + tanh(raw_log_std))` (maps [-10, 2])
  - Returns `(mean, log_std)` tuple

**`FlashSACCritic(nn.Module)`:**
- Attributes: `hidden_dim`, `num_blocks`, `expansion`, `num_atoms`
- `__call__(self, obs, action, train)`:
  - `x = concat([obs, action])`
  - Embedder → blocks → RMSNorm → UnitDense(num_atoms) + free bias
  - Returns logits `(batch, num_atoms)`

Reference: `/tmp/FlashSAC/flash_rl/agents/flashSAC/layer.py`

**Important implementation notes:**
- All `Dense` layers use `use_bias=False` and `kernel_init=nn.initializers.orthogonal()`
- The mean/logstd heads in the actor have separate `self.param('mean_bias', ...)` and `self.param('logstd_bias', ...)` free parameters
- The critic value head has a `self.param('value_bias', ...)` free parameter
- `nn.BatchNorm` in Flax uses `use_running_average` (True for eval, False for training) — this is the inverse of PyTorch's `training` flag

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_flash_blocks.py -v
```
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/networks/flash_blocks.py tests/test_flash_blocks.py
git commit -m "feat: add FlashSAC network blocks (embedder, residual block, actor, critic)"
```

---

## Task 4: Weight Normalization

**Files:**
- Modify: `jax_rl/networks/flash_blocks.py` (add `normalize_weights` function)
- Create: `tests/test_weight_norm.py`

- [ ] **Step 1: Write failing tests for normalize_weights**

```python
# tests/test_weight_norm.py
import jax
import jax.numpy as jnp
import pytest
from jax_rl.networks.flash_blocks import (
    FlashSACActor, FlashSACCritic, normalize_weights,
)

KEY = jax.random.PRNGKey(0)
BATCH = 8
OBS_DIM = 12
ACTION_DIM = 4

def test_kernel_columns_become_unit_norm():
    model = FlashSACActor(hidden_dim=32, num_blocks=1, expansion=4, action_dim=ACTION_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    normed_params = normalize_weights(variables['params'])
    # Check a Dense kernel — each column should have unit L2 norm
    # Find any kernel in the param tree
    leaves = jax.tree_util.tree_leaves_with_path(normed_params)
    for path, leaf in leaves:
        path_str = '/'.join(str(p) for p in path)
        if 'kernel' in path_str and leaf.ndim == 2:
            col_norms = jnp.linalg.norm(leaf, axis=0)
            assert jnp.allclose(col_norms, 1.0, atol=1e-6), f"Kernel at {path_str} not unit norm: {col_norms}"

def test_batchnorm_scale_bias_joint_norm_sqrt_d():
    model = FlashSACActor(hidden_dim=HIDDEN_DIM, num_blocks=1, expansion=4, action_dim=ACTION_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    normed_params = normalize_weights(variables['params'])
    # Find all BatchNorm modules and verify joint (scale, bias) norm = sqrt(D)
    flat = jax.tree_util.tree_leaves_with_path(normed_params)
    bn_groups = {}
    for path, leaf in flat:
        path_str = '/'.join(str(p) for p in path)
        if 'BatchNorm' in path_str and ('scale' in path_str or 'bias' in path_str):
            parent = path_str.rsplit('/', 1)[0]
            if parent not in bn_groups:
                bn_groups[parent] = {}
            key_name = 'scale' if 'scale' in path_str else 'bias'
            bn_groups[parent][key_name] = leaf
    for parent, params in bn_groups.items():
        if 'scale' in params and 'bias' in params:
            scale, bias = params['scale'], params['bias']
            d = scale.shape[-1]
            joint_norm = jnp.sqrt(jnp.sum(scale**2 + bias**2))
            assert jnp.allclose(joint_norm, jnp.sqrt(d), atol=1e-5), \
                f"BN at {parent}: joint norm {joint_norm:.4f} != sqrt({d})={jnp.sqrt(d):.4f}"

def test_normalize_is_idempotent():
    model = FlashSACActor(hidden_dim=32, num_blocks=1, expansion=4, action_dim=ACTION_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    normed_once = normalize_weights(variables['params'])
    normed_twice = normalize_weights(normed_once)
    leaves_once = jax.tree_util.tree_leaves(normed_once)
    leaves_twice = jax.tree_util.tree_leaves(normed_twice)
    for l1, l2 in zip(leaves_once, leaves_twice):
        assert jnp.allclose(l1, l2, atol=1e-6), "Weight norm is not idempotent"

def test_free_bias_params_unchanged():
    model = FlashSACActor(hidden_dim=32, num_blocks=1, expansion=4, action_dim=ACTION_DIM)
    variables = model.init(KEY, jnp.zeros((BATCH, OBS_DIM)), train=False)
    # Set free biases to known values
    original_params = variables['params']
    normed_params = normalize_weights(original_params)
    # mean_bias and logstd_bias should be unchanged
    leaves_orig = jax.tree_util.tree_leaves_with_path(original_params)
    leaves_norm = jax.tree_util.tree_leaves_with_path(normed_params)
    for (path_o, leaf_o), (path_n, leaf_n) in zip(leaves_orig, leaves_norm):
        path_str = '/'.join(str(p) for p in path_o)
        if 'mean_bias' in path_str or 'logstd_bias' in path_str or 'value_bias' in path_str:
            assert jnp.allclose(leaf_o, leaf_n), f"Free bias at {path_str} was modified"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_weight_norm.py -v
```
Expected: `ImportError` on `normalize_weights`.

- [ ] **Step 3: Implement normalize_weights in flash_blocks.py**

Add to `jax_rl/networks/flash_blocks.py`:

```python
def normalize_weights(params):
    """Project network weights to constraint set (unit-norm kernels, sqrt(D)-norm BN params).
    
    Applied after init() and after each optimizer step. Uses tree_map_with_path
    to identify parameter types by their path in the Flax param tree.
    """
    def _normalize_leaf(path, leaf):
        path_str = '/'.join(str(p) for p in path)
        
        if leaf.ndim < 2:
            # Check for BatchNorm scale — needs joint normalization with bias
            # Handled separately below
            if 'BatchNorm' in path_str and 'scale' in path_str:
                return leaf  # placeholder — handled by _normalize_bn_pair
            if 'UnitRMSNorm' in path_str and 'scale' in path_str:
                d = leaf.shape[-1]
                sqsum = jnp.sum(leaf * leaf)
                factor = jnp.sqrt(d) / jnp.sqrt(sqsum + 1e-8)
                return leaf * factor
            return leaf  # free bias params, etc.
        
        if 'kernel' in path_str:
            # Normalize each column (output neuron) to unit L2 norm
            # Flax kernel shape: (input_dim, output_dim)
            col_norms = jnp.linalg.norm(leaf, axis=0, keepdims=True)
            return leaf / jnp.maximum(col_norms, 1e-8)
        
        return leaf
    
    # First pass: normalize kernels and RMSNorm scale
    params = jax.tree_util.tree_map_with_path(_normalize_leaf, params)
    
    # Second pass: jointly normalize BatchNorm (scale, bias) pairs to ||·||₂ = sqrt(D)
    # ... implementation that walks the param tree and finds BatchNorm modules
    
    return params
```

The BatchNorm joint normalization requires finding paired `scale` and `bias` leaves under the same `BatchNorm` parent. Implement by flattening the param tree, grouping by parent path, and normalizing pairs together.

Reference: `/tmp/FlashSAC/flash_rl/agents/flashSAC/layer.py` lines 52-58 (`UnitBatchNorm.normalize_parameters`)

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_weight_norm.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/networks/flash_blocks.py tests/test_weight_norm.py
git commit -m "feat: add weight normalization for FlashSAC (kernel unit-norm, BN sqrt-d)"
```

---

## Task 5: FlashSAC Algorithm

This is the largest task. The algo class contains: TrainingState, init, update (with cross-batch BN, policy delay, weight norm), select_action, and the FlashSAC-specific C51 projection.

**Files:**
- Create: `jax_rl/algos/flash_sac.py`
- Create: `tests/test_flash_sac.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_flash_sac.py
import jax
import jax.numpy as jnp
import optax
import pytest
from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.algos.flash_sac import FlashSAC

OBS_DIM = 12
ACTION_DIM = 4
BATCH_SIZE = 16
KEY = jax.random.PRNGKey(42)

def _make_batch(key):
    """Match existing buffer key convention: 'done' (= terminated), 'truncation'."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    obs = jax.random.normal(k1, (BATCH_SIZE, OBS_DIM))
    next_obs = jax.random.normal(k4, (BATCH_SIZE, OBS_DIM))
    return {
        "obs": obs,
        "action": jax.random.uniform(k2, (BATCH_SIZE, ACTION_DIM), minval=-1, maxval=1),
        "reward": jnp.zeros((BATCH_SIZE, 1)),
        "next_obs": next_obs,
        "done": jnp.zeros((BATCH_SIZE, 1)),          # = terminated (buffer convention)
        "truncation": jnp.zeros((BATCH_SIZE, 1)),
        "critic_obs": obs,
        "critic_next_obs": next_obs,
    }

def _make_flash_sac():
    cfg = FlashSACConfig(
        num_blocks=1, actor_hidden_dim=32, critic_hidden_dim=32,
        num_atoms=21, batch_size=BATCH_SIZE,
    )
    # Pass optimizer externally (matching FastSAC pattern for codebase consistency)
    opt = optax.adam(3e-4)
    alpha_opt = optax.adam(3e-4)
    return FlashSAC(config=cfg, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                    optimizer=opt, alpha_optimizer=alpha_opt)

def test_init_produces_valid_state():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    assert state.actor_params is not None
    assert state.q1_params is not None
    assert state.q2_params is not None
    assert state.actor_batch_stats is not None
    assert state.q1_batch_stats is not None
    assert state.target_q1_batch_stats is not None
    # reward_norm_state lives outside TrainingState (in training script)
    assert state.noise_state is not None

def test_select_action_shape():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    obs = jnp.ones((1, OBS_DIM))
    # select_action matches existing codebase signature: (actor_params, obs, key, deterministic)
    # batch_stats captured by closure (same pattern as FastSAC captures network refs)
    action = algo.select_action(state.actor_params, obs, KEY)
    assert action.shape == (1, ACTION_DIM)
    assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)

def test_update_returns_metrics():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    batch = _make_batch(KEY)
    new_state, metrics = algo.update(state, batch)
    assert "q1_loss" in metrics or "critic_loss" in metrics
    assert new_state.update_count == 1

def test_update_modifies_params():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    batch = _make_batch(KEY)
    new_state, _ = algo.update(state, batch)
    # Critic params should change (critic updates every step)
    q1_diff = jax.tree_util.tree_map(lambda a, b: jnp.sum(jnp.abs(a - b)),
                                      state.q1_params, new_state.q1_params)
    total_diff = sum(jax.tree_util.tree_leaves(q1_diff))
    assert total_diff > 0, "Critic params should change after update"

def test_deterministic_action_consistency():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    obs = jnp.ones((1, OBS_DIM))
    a1 = algo.select_action(state.actor_params, obs, KEY, deterministic=True)
    a2 = algo.select_action(state.actor_params, obs, KEY, deterministic=True)
    assert jnp.allclose(a1, a2)

def test_get_q_value():
    algo = _make_flash_sac()
    state = algo.init(KEY)
    obs = jnp.ones((1, OBS_DIM))
    action = jnp.zeros((1, ACTION_DIM))
    q_val = algo.get_q_value(state, obs, action)
    assert q_val.shape == (1,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_flash_sac.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement flash_sac.py — TrainingState and init()**

Create `jax_rl/algos/flash_sac.py`. Start with imports, TrainingState dataclass, and `init()` method. Follow the pattern from `jax_rl/algos/fast_sac.py`:

**Constructor signature** (matches FastSAC pattern for codebase consistency):
```python
def __init__(self, config, obs_dim, action_dim, optimizer, alpha_optimizer,
             gamma=0.99, handle_truncation=True, critic_obs_dim=None,
             num_envs=1):
```
- `optimizer` / `alpha_optimizer` passed externally (training script builds warmup+cosine schedule)
- `num_envs` needed for NoiseState and RewardNormState initialization
- `gamma` is a constructor param (read from config in the training script, passed here)

**TrainingState** includes:
- BatchNorm state fields (`actor_batch_stats`, `q1_batch_stats`, etc.)
- `noise_state` — `NoiseState` dataclass defined in this file (noise, count, repeat_n)
- **NOT** `reward_norm_state` — kept outside TrainingState in the training script (simpler flow, matches how obs_norm_state is handled in `train_offpolicy.py`)

**`NoiseState`** — defined in `flash_sac.py`:
```python
@flax.struct.dataclass
class NoiseState:
    noise: jnp.ndarray      # (num_envs, action_dim)
    count: jnp.ndarray      # (num_envs,)
    repeat_n: jnp.ndarray   # (num_envs,)
```

**`init()`** must call `normalize_weights()` on freshly-initialized params.

**`select_action` signature** matches existing codebase convention:
```python
select_action(actor_params, obs, key, deterministic=False) -> action
```
Batch_stats are captured by the closure (same as FastSAC captures network refs). The training script must update the closure's batch_stats reference when they change — achieved by having `select_action` read from the TrainingState directly, or by rebuilding the closure. Simplest approach: `select_action` takes the full state or a tuple.

**Batch key convention:** The algo reads `batch["done"]` (= terminated) and `batch["truncation"]` from the buffer. For C51 bootstrap, use `batch["done"]` only (NOT done | truncation). This matches how the existing buffer stores transitions.

Key differences from FastSAC's init:
- Networks are `FlashSACActor` and `FlashSACCritic` (not `Actor` + `DistributionalQHead`)
- Variables contain `{'params': ..., 'batch_stats': ...}` from `model.init()`
- Entropy target uses unified formula: `0.5 * action_dim * log(2πe * σ²)`

- [ ] **Step 4: Implement flash_sac.py — select_action()**

JIT'd closure for action selection:
- Deterministic: `tanh(mean)` using `model.apply({params, batch_stats}, obs, train=False)`
- Stochastic: `sample_gaussian(mean, log_std, key, squash=True)` from `jax_rl/networks/distributions.py`
- Signature: `select_action(actor_params, actor_batch_stats, obs, key, deterministic=False) -> action`

- [ ] **Step 5: Implement flash_sac.py — update() closure**

The most complex part. Follow spec Section 4 exactly:

1. **Actor update** (gated by `update_count % policy_delay == 0`):
   - Concat `[obs; next_obs]` → 2B batch through actor with `train=True, mutable=['batch_stats']`
   - Take first half for loss
   - Critic forward on `critic_obs` with `train=False` — `min(Q1, Q2)` scalar values
   - `loss = mean(alpha * log_prob - min_q)`
   - BC regularization if `bc_alpha > 0`: `loss += bc_alpha * stop_gradient(|Q|.mean()) * MSE`
   - Apply optimizer, then `normalize_weights()` on updated actor params

2. **Temperature update** (same gating):
   - `entropy = -mean(log_prob)`
   - `loss = exp(log_alpha) * (entropy - target_entropy)`

3. **Critic update** (every step):
   - Sample `next_action` from freshly-updated actor (no grad, `train=False`)
   - Concat `obs_all`, `act_all` (2B)
   - Target critic forward on `obs_all` (`train=True, mutable`) → split, take second half
   - **Min-Q selection**: compute expected Q from both targets, select full log_prob distribution from lower
   - FlashSAC C51 projection: `target_bin = reward + gamma^n * (bins - alpha*log_prob) * (1-terminated)`
   - Online critic forward on `obs_all` (`train=True, mutable`) → split, take first half
   - Cross-entropy loss with log_softmax clamped to -30.0
   - Apply optimizer, then `normalize_weights()`

4. **Target EMA** (after critic):
   - Polyak on params only (not batch_stats)
   - Target batch_stats come from the target critic's own `train=True` forward in step 3

5. **Policy delay via `jax.lax.cond`**:
   - Both branches must return identical pytree structures (including batch_stats)
   - Skip branch: pass through existing params/batch_stats unchanged

Reference: `/tmp/FlashSAC/flash_rl/agents/flashSAC/update.py`

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_flash_sac.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add jax_rl/algos/flash_sac.py tests/test_flash_sac.py
git commit -m "feat: add FlashSAC algorithm (cross-batch BN, weight norm, adaptive reward scaling)"
```

---

## Task 6: Training Script

**Files:**
- Create: `train_flashsac.py`

- [ ] **Step 1: Copy boilerplate from train_offpolicy.py**

Copy `train_offpolicy.py` to `train_flashsac.py`. Strip out all algo-specific code (ALGO_REGISTRY, _make_algo, the existing algo dispatch). Keep:
- Argparse (--env, --seed, --total-timesteps, --num-envs, --eval-freq, --wandb, --log-dir)
- Env creation via `make_env`
- Replay buffer setup
- Eval loop
- Logging (wandb/tensorboard)
- Checkpointing

- [ ] **Step 2: Add FlashSAC-specific collection loop**

Replace the generic collection loop with FlashSAC's:
1. `select_action_with_noise(state, obs, key)` → uses noise repetition from NoiseState
2. After `env.step()`: call `update_reward_stats()` on the reward_norm_state
3. Pass `terminated` AND `truncated` to reward stats update

The noise repetition logic lives in the training script (not the algo class) since it's a collection-time operation:
```python
# Precompute Zeta CDF once
zeta_cdf = _make_zeta_cdf(mu=config.noise_zeta_mu, max_n=config.noise_zeta_max)

@jax.jit
def select_action_with_noise(actor_params, actor_batch_stats, obs, noise_state, key):
    mean, log_std = actor.apply({'params': actor_params, 'batch_stats': actor_batch_stats},
                                 obs, train=False)
    std = jnp.exp(log_std)
    # Noise repetition
    reinit = (noise_state.count == 0) | (noise_state.count >= noise_state.repeat_n)
    k1, k2 = jax.random.split(key)
    new_noise = jax.random.normal(k1, mean.shape)
    u = jax.random.uniform(k2, (mean.shape[0],))
    new_n = jnp.searchsorted(zeta_cdf, u) + 1
    noise = jnp.where(reinit[:, None], new_noise, noise_state.noise)
    repeat_n = jnp.where(reinit, new_n, noise_state.repeat_n)
    count = jnp.where(reinit, jnp.ones_like(noise_state.count), noise_state.count + 1)
    action = jnp.tanh(mean + std * noise)
    new_noise_state = NoiseState(noise=noise, count=count, repeat_n=repeat_n)
    return action, new_noise_state
```

- [ ] **Step 3: Add FlashSAC-specific gradient loop**

```python
# reward_norm_state lives OUTSIDE TrainingState (simpler flow, no JIT boundary issues)
# It's updated in the collection loop and read in the gradient loop.

for _ in range(config.grad_updates_per_step):
    batch = buffer.sample(config.batch_size)
    if config.normalize_reward:
        batch['reward'] = scale_reward(reward_norm_state, batch['reward'], G_max=config.G_max)
    training_state, metrics = algo.update(training_state, batch)
```

- [ ] **Step 4: Smoke test on CPU**

```bash
uv run python train_flashsac.py --env CartpoleBalance --total-timesteps 50000 --num-envs 4 --seed 0
```

Expected: Runs without error, prints training metrics every eval interval, weight norms remain bounded.

- [ ] **Step 5: Commit**

```bash
git add train_flashsac.py
git commit -m "feat: add standalone FlashSAC training script"
```

---

## Task 7: Integration Smoke Test

**Files:** None created — this is a validation step.

- [ ] **Step 1: Run full test suite to verify no regressions**

```bash
uv run python -m pytest tests/ -v --ignore=tests/archive
```
Expected: All existing tests still pass + new FlashSAC tests pass.

- [ ] **Step 2: Run FlashSAC training smoke test**

```bash
uv run python train_flashsac.py --env CartpoleBalance --total-timesteps 100000 --num-envs 8 --seed 42
```

Verify:
- Training loop runs without error
- Metrics logged: critic_loss, actor_loss, entropy, alpha, eval_return
- Weight norms stay bounded (should see ~1.0 for kernels after normalization)
- Reward scaling active (reward_scale metric changes over time)
- Eval return improves (CartpoleBalance should reach ~800+ in 100K steps if algo works)

- [ ] **Step 3: Commit any fixes**

If smoke test reveals issues, fix and commit incrementally.

- [ ] **Step 4: Update docs**

Per CLAUDE.md doc sync rules:
- `.context/TODO.md` — add FlashSAC port as completed, add A/B benchmark as next step
- `.context/AGENT_HANDOFF.md` — add FlashSAC to algo list, note training script

```bash
git add .context/TODO.md .context/AGENT_HANDOFF.md
git commit -m "docs: update project docs with FlashSAC port"
```
