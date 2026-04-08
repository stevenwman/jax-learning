# Split train_offpolicy.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `train_offpolicy.py` (498 lines, 4 algos via branching) with 4 standalone per-algo training scripts.

**Architecture:** Each script is self-contained (~180 lines), duplicates ~60 lines of shared loop boilerplate, but reads top-to-bottom with zero branching. Shared utilities in `jax_rl/training/` (make_envs, EpisodeTracker, CheckpointManager, logging, eval) remain unchanged.

**Tech Stack:** JAX, optax, argparse. Existing `jax_rl/training/` utilities.

---

### Task 1: Create train_sac.py

**Files:**
- Create: `train_sac.py`
- Reference: `train_offpolicy.py` (source template)

- [ ] **Step 1: Create train_sac.py from train_offpolicy.py**

Copy `train_offpolicy.py` to `train_sac.py`. Then make these changes:

1. **Docstring** — update usage example:
```python
"""SAC training script.

Usage:
    uv run python train_sac.py --env Go2WarpJoystickFlat
    uv run python train_sac.py --env WalkerWalk --obs-norm
    uv run python train_sac.py --env Go2WarpJoystickFlat --reset-mode per_step --wandb
"""
```

2. **Remove** the algo registry (lines 45-57), `_register()`, and `_make_algo()` (lines 50-108). Replace with inline SAC setup.

3. **Imports** — replace preset imports with just SAC:
```python
from jax_rl.configs.env_presets import get_sac_preset
```

4. **train() function** — remove `algo_name` parameter, hardcode `algo_name = "sac"`. Remove `family` variable. Inline the SAC algo setup:
```python
from jax_rl.algos.sac import SAC
if algo_cfg.grad_clip_norm is not None:
    optimizer = optax.chain(optax.clip_by_global_norm(algo_cfg.grad_clip_norm), optax.adam(cfg.lr))
else:
    optimizer = optax.adam(cfg.lr)
alpha_optimizer = optax.adam(algo_cfg.alpha_lr)
algo = SAC(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
           optimizer=optimizer, alpha_optimizer=alpha_optimizer,
           gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
           critic_obs_dim=critic_obs_dim)
```

5. **Exploration** — inline SAC explore (remove the `if family ==` branch):
```python
def explore(actor_params, obs, key):
    return algo.select_action(actor_params, obs, key)
log_extra_fields = [("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")]
log_extra_keys = ["entropy", "alpha", "alpha_loss"]
```

6. **Metrics** — remove TD3 policy_delay branch (lines 340-344). Keep only:
```python
last_metrics = step_metrics
```

7. **CLI** — remove `--algo` argument, remove `--exploration-noise` (SAC uses entropy). Remove `ALGO_REGISTRY` lookup. Hardcode preset:
```python
cfg, algo_cfg = get_sac_preset(args.env)
```

8. **Remove** `--batch-size`, `--grad-updates-per-step`, `--buffer-size` args (vanilla SAC uses defaults, not tunable at CLI like Fast variants). Keep `--target-entropy-scale`.

- [ ] **Step 2: Smoke test train_sac.py**

Run: `uv run python train_sac.py --env CartpoleBalance --total-timesteps 50000 --num-envs 32 2>&1 | tail -5`
Expected: Training runs, shows SPS and return, exits cleanly.

- [ ] **Step 3: Commit**

```bash
git add train_sac.py
git commit -m "feat: standalone train_sac.py (split from train_offpolicy.py)"
```

---

### Task 2: Create train_td3.py

**Files:**
- Create: `train_td3.py`
- Reference: `train_sac.py` (just created, use as template)

- [ ] **Step 1: Create train_td3.py from train_sac.py**

Copy `train_sac.py` to `train_td3.py`. Then make these changes:

1. **Docstring** — update:
```python
"""TD3 training script.

Usage:
    uv run python train_td3.py --env CheetahRun
    uv run python train_td3.py --env WalkerWalk --exploration-noise 0.15
"""
```

2. **Imports** — swap SAC for TD3:
```python
from jax_rl.algos.td3 import TD3
from jax_rl.configs.env_presets import get_td3_preset
```

3. **Algo setup** — replace SAC with TD3:
```python
optimizer = optax.adam(cfg.lr)
algo = TD3(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
           actor_optimizer=optimizer, critic_optimizer=optax.adam(cfg.lr),
           gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
           critic_obs_dim=critic_obs_dim)
```

4. **Exploration** — replace SAC entropy-based with TD3 noise-based:
```python
exploration_noise_std = getattr(algo_cfg, 'exploration_noise_std', 0.1)
noise_min = getattr(algo_cfg, 'noise_min', None)
noise_max = getattr(algo_cfg, 'noise_max', None)

def explore(actor_params, obs, key):
    key, noise_key = jax.random.split(key)
    if noise_min is not None:
        noise_std = jax.random.uniform(noise_key, (), minval=noise_min, maxval=noise_max)
    else:
        noise_std = exploration_noise_std
    return algo.select_action(actor_params, obs, key, deterministic=False, exploration_noise=noise_std)
log_extra_fields = []
log_extra_keys = []
```

5. **Metrics** — add TD3 policy_delay handling:
```python
if float(step_metrics.get("actor_loss", 0.0)) != 0.0:
    last_metrics = step_metrics
else:
    last_metrics = {**step_metrics, "actor_loss": last_metrics.get("actor_loss", 0.0)}
```

6. **CLI** — swap `--target-entropy-scale` for `--exploration-noise`. Hardcode `get_td3_preset`. Set `algo_name = "td3"`.

- [ ] **Step 2: Smoke test train_td3.py**

Run: `uv run python train_td3.py --env CartpoleBalance --total-timesteps 50000 --num-envs 32 2>&1 | tail -5`
Expected: Training runs, no entropy/alpha in output, exits cleanly.

- [ ] **Step 3: Commit**

```bash
git add train_td3.py
git commit -m "feat: standalone train_td3.py (split from train_offpolicy.py)"
```

---

### Task 3: Create train_fast_sac.py

**Files:**
- Create: `train_fast_sac.py`
- Reference: `train_sac.py` (SAC family, same exploration/metrics)

- [ ] **Step 1: Create train_fast_sac.py from train_sac.py**

Copy `train_sac.py` to `train_fast_sac.py`. Then make these changes:

1. **Docstring** — update:
```python
"""FastSAC training script (C51 distributional + SAC).

Usage:
    uv run python train_fast_sac.py --env Go2WarpJoystickFlat
    uv run python train_fast_sac.py --env HumanoidRun --obs-norm
    uv run python train_fast_sac.py --env Go2WarpJoystickFlat --reset-mode per_step --wandb
"""
```

2. **Imports** — swap for FastSAC:
```python
from jax_rl.algos.fast_sac import FastSAC
from jax_rl.configs.env_presets import get_fast_sac_preset
```

3. **Algo setup** — replace SAC adam with FastSAC adamw+cosine:
```python
warmup_steps = algo_cfg.min_buffer_size // cfg.num_envs
train_iters = (cfg.total_timesteps // cfg.num_envs) - warmup_steps
total_grad_est = train_iters * algo_cfg.grad_updates_per_step
lr_schedule = optax.cosine_decay_schedule(cfg.lr, total_grad_est, alpha=algo_cfg.lr_end / cfg.lr) if algo_cfg.lr_end < cfg.lr else cfg.lr
optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
alpha_optimizer = optax.adam(algo_cfg.alpha_lr)
algo = FastSAC(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
               optimizer=optimizer, alpha_optimizer=alpha_optimizer,
               gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
               critic_obs_dim=critic_obs_dim)
```

4. **CLI** — add `--batch-size`, `--grad-updates-per-step`, `--buffer-size` args (Fast variants expose these). Hardcode `get_fast_sac_preset`. Set `algo_name = "fast_sac"`.

5. **Exploration + metrics** — same as SAC (no changes needed from train_sac.py).

- [ ] **Step 2: Smoke test train_fast_sac.py**

Run: `uv run python train_fast_sac.py --env CartpoleBalance --total-timesteps 50000 --num-envs 32 2>&1 | tail -5`
Expected: Training runs with entropy/alpha logging, exits cleanly.

- [ ] **Step 3: Commit**

```bash
git add train_fast_sac.py
git commit -m "feat: standalone train_fast_sac.py (split from train_offpolicy.py)"
```

---

### Task 4: Create train_fast_td3.py

**Files:**
- Create: `train_fast_td3.py`
- Reference: `train_td3.py` (TD3 family, same exploration/metrics)

- [ ] **Step 1: Create train_fast_td3.py from train_td3.py**

Copy `train_td3.py` to `train_fast_td3.py`. Then make these changes:

1. **Docstring** — update:
```python
"""FastTD3 training script (C51 distributional + TD3).

Usage:
    uv run python train_fast_td3.py --env CheetahRun
    uv run python train_fast_td3.py --env CheetahRun --exploration-noise 0.15
"""
```

2. **Imports** — swap for FastTD3:
```python
from jax_rl.algos.fast_td3 import FastTD3
from jax_rl.configs.env_presets import get_fast_td3_preset
```

3. **Algo setup** — replace TD3 adam with FastTD3 adamw+cosine:
```python
warmup_steps = algo_cfg.min_buffer_size // cfg.num_envs
train_iters = (cfg.total_timesteps // cfg.num_envs) - warmup_steps
total_grad_est = train_iters * algo_cfg.grad_updates_per_step
lr_schedule = optax.cosine_decay_schedule(cfg.lr, total_grad_est, alpha=algo_cfg.lr_end / cfg.lr) if algo_cfg.lr_end < cfg.lr else cfg.lr
actor_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
critic_optimizer = optax.adamw(lr_schedule, b2=0.95, weight_decay=0.001)
algo = FastTD3(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
               actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
               gamma=cfg.gamma, handle_truncation=cfg.handle_truncation,
               critic_obs_dim=critic_obs_dim)
```

4. **CLI** — add `--batch-size`, `--grad-updates-per-step`, `--buffer-size` args. Hardcode `get_fast_td3_preset`. Set `algo_name = "fast_td3"`.

5. **Exploration + metrics** — same as TD3 (no changes needed from train_td3.py).

- [ ] **Step 2: Smoke test train_fast_td3.py**

Run: `uv run python train_fast_td3.py --env CartpoleBalance --total-timesteps 50000 --num-envs 32 2>&1 | tail -5`
Expected: Training runs, no entropy/alpha in output, exits cleanly.

- [ ] **Step 3: Commit**

```bash
git add train_fast_td3.py
git commit -m "feat: standalone train_fast_td3.py (split from train_offpolicy.py)"
```

---

### Task 5: Delete train_offpolicy.py and update docs

**Files:**
- Delete: `train_offpolicy.py`
- Modify: `.context/AGENT_HANDOFF.md`
- Modify: `.context/TODO.md`

- [ ] **Step 1: Delete train_offpolicy.py**

```bash
git rm train_offpolicy.py
```

- [ ] **Step 2: Update AGENT_HANDOFF.md**

In the "Key entry points" section, replace:
```
├── train_offpolicy.py        # SAC/TD3/FastTD3/FastSAC via --algo flag
```
with:
```
├── train_sac.py              # SAC (vanilla, entropy-regularized)
├── train_td3.py              # TD3 (deterministic, delayed updates)
├── train_fast_sac.py         # FastSAC (C51 distributional + SAC)
├── train_fast_td3.py         # FastTD3 (C51 distributional + TD3)
├── train_flashsac.py         # FlashSAC (inverted residual + BatchNorm + adaptive reward)
```

In the "Algorithm quick reference" table, update the "Training script" column:
- SAC: `train_sac.py`
- TD3: `train_td3.py`
- FastTD3: `train_fast_td3.py`
- FastSAC: `train_fast_sac.py`

In the "Quick Reference > Commands" section, update example commands:
```bash
uv run python train_fast_sac.py --env Go2WarpJoystickFlat --num-envs 1024 --total-timesteps 20000000 --domain-rand
```

- [ ] **Step 3: Update any other docs referencing train_offpolicy.py**

Search for `train_offpolicy` in `.context/`, `docs/`, and update references to point to the appropriate per-algo script. Common pattern: replace `train_offpolicy.py --algo fast_sac` with `train_fast_sac.py`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: delete train_offpolicy.py, update docs for per-algo scripts"
```

---

### Task 6: Regression test all 4 scripts

- [ ] **Step 1: Run all 4 smoke tests sequentially**

```bash
uv run python train_sac.py --env CartpoleBalance --total-timesteps 50000 --num-envs 32 2>&1 | tail -3
uv run python train_td3.py --env CartpoleBalance --total-timesteps 50000 --num-envs 32 2>&1 | tail -3
uv run python train_fast_sac.py --env CartpoleBalance --total-timesteps 50000 --num-envs 32 2>&1 | tail -3
uv run python train_fast_td3.py --env CartpoleBalance --total-timesteps 50000 --num-envs 32 2>&1 | tail -3
```

Expected: All 4 print SPS + return, exit cleanly with no errors.

- [ ] **Step 2: Test Go2 with reset-mode per_step on FastSAC**

```bash
uv run python train_fast_sac.py --env Go2WarpJoystickFlat --total-timesteps 50000 --num-envs 32 --reset-mode per_step 2>&1 | tail -3
```

Expected: Runs with DomainRandWrapper, dict obs detected, exits cleanly.

- [ ] **Step 3: Test wandb flag doesn't crash (dry run without actual wandb)**

```bash
uv run python train_fast_sac.py --env CartpoleBalance --total-timesteps 20000 --num-envs 16 --wandb 2>&1 | head -10
```

Expected: Initializes wandb (or warns if not configured), starts training.

- [ ] **Step 4: Run existing test suite**

```bash
uv run python -m pytest tests/ -v --timeout=120 2>&1 | tail -10
```

Expected: All existing tests pass. No tests reference `train_offpolicy.py` directly.
