# Asymmetric Off-Policy Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow off-policy algos (SAC, TD3, FastSAC, FastTD3) to use a privileged critic — actor sees `obs["state"]` (48d), critic sees `obs["privileged_state"]` (122d). Directly benefits Go2 Warp training and is a novel contribution (no published SAC + privileged critic + legged locomotion).

**Architecture:** Add optional `critic_obs_dim` parameter to all off-policy algo constructors. When set, Q networks are initialized with `critic_obs_dim` input dim instead of `obs_dim`. The training loop passes `batch["critic_obs"]` and `batch["critic_next_obs"]` alongside `batch["obs"]` and `batch["next_obs"]`. Each algo's loss functions read the appropriate key. When `critic_obs_dim` is None, `critic_obs == obs` (backward compatible).

**Tech Stack:** JAX, Flax, existing QHead/Actor/DistributionalQHead networks

---

## The Subtle Part

In `_actor_loss`, the actor proposes an action using `batch["obs"]` (48d), but the Q network must evaluate that action using `batch["critic_obs"]` (122d). This means the actor loss has TWO obs inputs:

```python
def _actor_loss(actor_params, q1_params, q2_params, log_alpha, batch, key):
    obs = batch["obs"]              # 48d — actor input
    critic_obs = batch["critic_obs"]  # 122d — Q evaluation input

    action, log_prob = _actor_forward(actor_params, obs, key)        # actor sees 48d
    q1_val = q1.apply(q1_params, critic_obs, action)                 # critic sees 122d
    q2_val = q2.apply(q2_params, critic_obs, action)                 # critic sees 122d
    ...
```

This is NOT just "pass different obs to critic" — it's also changing the Q evaluation inside the actor loss. Getting this wrong (e.g., passing 48d to a 122d Q network) crashes silently or produces garbage gradients.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `jax_rl/algos/sac.py` | Modify | Add `critic_obs_dim`, use `batch["critic_obs"]` in losses |
| `jax_rl/algos/td3.py` | Modify | Same |
| `jax_rl/algos/fast_sac.py` | Modify | Same |
| `jax_rl/algos/fast_td3.py` | Modify | Same |
| `train_offpolicy.py` | Modify | Detect dict obs, pass `critic_obs`/`critic_next_obs` in batch |
| `jax_rl/utils/eval.py` | Modify | Pass `critic_obs` to `q_fn` for Q diagnostics |
| `tests/test_asymmetric_critic.py` | Create | Targeted tests per algo |
| `tests/test_offpolicy_algos.py` | Modify | Verify existing tests still pass |

**Files NOT touched:**
- QHead, DistributionalQHead — Flax lazy init handles different input dims automatically
- Actor, builders.py — actor network is unchanged
- configs — no new config fields needed (critic_obs_dim derived from env at runtime)
- env code — dict obs already exists, no changes

---

### Task 1: Modify SAC to support asymmetric critic

**Files:**
- Modify: `jax_rl/algos/sac.py`

- [ ] **Step 1: Add `critic_obs_dim` constructor parameter**

In `SAC.__init__`, add `critic_obs_dim: int | None = None` after `action_dim`. Store it:

```python
self.critic_obs_dim = critic_obs_dim or obs_dim
```

- [ ] **Step 2: Use `critic_obs_dim` in `init()` for Q network initialization**

In `init()`, change:
```python
dummy_obs = jnp.zeros((1, self.obs_dim))
...
q1_params = self.q1.init(k3, dummy_obs, dummy_action)
q2_params = self.q2.init(k4, dummy_obs, dummy_action)
```
To:
```python
dummy_obs = jnp.zeros((1, self.obs_dim))
dummy_critic_obs = jnp.zeros((1, self.critic_obs_dim))
dummy_action = jnp.zeros((1, self.action_dim))

actor_params = self.actor.init(k1, dummy_obs)
q1_params = self.q1.init(k3, dummy_critic_obs, dummy_action)
q2_params = self.q2.init(k4, dummy_critic_obs, dummy_action)
```

- [ ] **Step 3: Update `_critic_loss` to use `batch["critic_obs"]`**

Change:
```python
obs = batch["obs"]
...
next_obs = batch["next_obs"]
```
To:
```python
obs = batch["critic_obs"]
...
next_obs = batch["critic_next_obs"]
```

The actor forward call inside critic loss (`_actor_forward(actor_params, next_obs, key)`) must use actor-compatible obs. Since SAC's critic loss computes `next_action` from the policy for the entropy-adjusted target, this needs to use `batch["next_obs"]` (48d) for the actor but `batch["critic_next_obs"]` (122d) for the Q target. Update:

```python
def _critic_loss(q_params, actor_params, target_q1_params, target_q2_params,
                 log_alpha, batch, key):
    q1_params_, q2_params_ = q_params
    critic_obs = batch["critic_obs"]
    action = batch["action"]
    reward = batch["reward"].squeeze(-1)
    critic_next_obs = batch["critic_next_obs"]
    next_obs = batch["next_obs"]  # for actor forward (48d)
    done = batch["done"].squeeze(-1)
    truncation = batch["truncation"].squeeze(-1)

    alpha = jnp.exp(log_alpha)

    # Next action from current policy — uses ACTOR obs (48d)
    next_action, next_log_prob = _actor_forward(actor_params, next_obs, key)

    # Target Q — uses CRITIC obs (122d)
    tq1_val = q1.apply(target_q1_params, critic_next_obs, next_action)
    tq2_val = q2.apply(target_q2_params, critic_next_obs, next_action)
    ...

    # Online Q — uses CRITIC obs (122d)
    q1_val = q1.apply(q1_params_, critic_obs, action)
    q2_val = q2.apply(q2_params_, critic_obs, action)
    ...
```

- [ ] **Step 4: Update `_actor_loss` to use `batch["critic_obs"]` for Q evaluation**

```python
def _actor_loss(actor_params, q1_params_, q2_params_, log_alpha, batch, key):
    obs = batch["obs"]              # 48d — actor input
    critic_obs = batch["critic_obs"]  # 122d — Q evaluation

    action, log_prob = _actor_forward(actor_params, obs, key)
    q1_val = q1.apply(q1_params_, critic_obs, action)
    q2_val = q2.apply(q2_params_, critic_obs, action)
    ...
```

- [ ] **Step 5: Update `_alpha_loss`**

Alpha loss only uses actor obs — no change needed (already uses `batch["obs"]`).

- [ ] **Step 6: Update `get_q_value`**

This takes raw `obs` for Q diagnostics. It should accept `critic_obs` when asymmetric:

```python
def get_q_value(self, state: TrainingState, obs: jax.Array, action: jax.Array,
                critic_obs: jax.Array | None = None) -> jax.Array:
    q_obs = critic_obs if critic_obs is not None else obs
    return self.q1.apply(state.q1_params, q_obs, action)
```

- [ ] **Step 7: Commit**

```bash
git add jax_rl/algos/sac.py
git commit -m "feat: asymmetric critic support in SAC — actor/critic see different obs"
```

---

### Task 2: Modify TD3, FastSAC, FastTD3 (same pattern)

**Files:**
- Modify: `jax_rl/algos/td3.py`
- Modify: `jax_rl/algos/fast_sac.py`
- Modify: `jax_rl/algos/fast_td3.py`

Apply the same changes as Task 1 to each algo:
1. Add `critic_obs_dim` constructor param
2. Use `critic_obs_dim` for Q network init
3. `_critic_loss`: use `batch["critic_obs"]` / `batch["critic_next_obs"]` for Q, `batch["next_obs"]` for actor forward
4. `_actor_loss`: use `batch["obs"]` for actor, `batch["critic_obs"]` for Q
5. `get_q_value`: accept optional `critic_obs`

**TD3-specific note:** TD3's critic loss calls `_actor_forward(target_actor_params, next_obs)` for target policy smoothing — this must use `batch["next_obs"]` (actor obs), not critic obs.

**FastTD3/FastSAC-specific note:** Same pattern but Q is distributional (logits, not scalars). The `q.apply()` calls are identical — just pass `critic_obs` instead of `obs`. The distribution math is unchanged.

- [ ] **Step 1: Modify TD3**
- [ ] **Step 2: Modify FastSAC**
- [ ] **Step 3: Modify FastTD3**
- [ ] **Step 4: Commit**

```bash
git add jax_rl/algos/td3.py jax_rl/algos/fast_sac.py jax_rl/algos/fast_td3.py
git commit -m "feat: asymmetric critic support in TD3, FastSAC, FastTD3"
```

---

### Task 3: Modify training loop to pass critic obs

**Files:**
- Modify: `train_offpolicy.py`

- [ ] **Step 1: Detect dict obs and extract both state and privileged_state**

Find the existing `_get_obs` function and the dict obs detection. Add `critic_obs` extraction:

```python
dict_obs = isinstance(env_state.obs, dict)
if dict_obs:
    obs_dim = env_state.obs["state"].shape[-1]
    critic_obs_dim = env_state.obs.get("privileged_state", env_state.obs["state"]).shape[-1]
    has_privileged = "privileged_state" in env_state.obs
    print(f"  Dict obs detected: actor={obs_dim}d, critic={critic_obs_dim}d")
else:
    critic_obs_dim = None
    has_privileged = False
```

- [ ] **Step 2: Pass `critic_obs_dim` to algo constructor**

In `_make_algo`, pass `critic_obs_dim`:

```python
return SAC(config=algo_cfg, obs_dim=obs_dim, action_dim=action_dim,
           critic_obs_dim=critic_obs_dim, ...)
```

Same for TD3, FastSAC, FastTD3.

- [ ] **Step 3: Add `critic_obs` to replay buffer batch**

The replay buffer needs to store `critic_obs` and `critic_next_obs`. Two approaches:

**Approach A (simple):** Add `critic_obs` and `critic_next_obs` as separate buffer arrays. Memory cost: `2 * buffer_size * 122 * 4 bytes` = ~3.9 GB at 4M. Total with obs: 5.4 GB. Tight but fits.

**Approach B (efficient):** Since `privileged_state` contains `state` as its first 48d (it's `jp.hstack([state, ...extra_sensors...])`), we could store only `privileged_state` (122d) and derive `state = privileged_state[:, :48]`. One buffer, no redundancy.

**Recommendation:** Approach A for now — separate arrays, clear and simple. Optimize later if memory is tight.

In the training loop's buffer add:
```python
def _get_critic_obs(obs):
    if has_privileged:
        return obs["privileged_state"] if dict_obs else obs
    return _get_obs(obs)

# In the step loop:
raw_obs = _get_obs(env_state.obs)
critic_raw_obs = _get_critic_obs(env_state.obs)
...
next_raw_obs = _get_obs(env_state.obs)
next_critic_raw_obs = _get_critic_obs(env_state.obs)

buffer.add_batch(obs=raw_obs, action=action, reward=...,
                 next_obs=next_raw_obs, done=..., truncation=...)
# Store critic obs in separate buffer or add to batch
```

**Actually — simpler approach:** Just add `critic_obs` and `critic_next_obs` to the batch dict that goes to `algo.update()`. The buffer doesn't need to know about them — construct the critic batch keys at sample time from a separate buffer or from the env obs directly.

Wait — the buffer samples random indices. We need critic obs at the same indices. So they must be stored in the buffer.

**Simplest correct approach:** Create a second buffer for critic obs, or extend `JaxReplayBuffer` to accept extra fields. For now, just use two parallel buffers:

```python
buffer = JaxReplayBuffer(obs_dim, action_dim, max_size=algo_cfg.buffer_size, ...)
if has_privileged:
    critic_buffer = JaxReplayBuffer(critic_obs_dim, action_dim, max_size=algo_cfg.buffer_size)
```

At sample time, sample with the same key to get matching indices... except our buffer uses random indices internally. This won't work with two separate buffers.

**Correct approach:** Add `critic_obs` and `critic_next_obs` as optional extra arrays in `JaxReplayBuffer`. The buffer stores them alongside obs, and `sample()` returns them in the batch dict.

- [ ] **Step 4: Extend JaxReplayBuffer with optional extra obs fields**

Add optional `extra_obs_dims: dict[str, int] | None = None` to `JaxReplayBuffer.__init__`. When set, allocates extra buffers and includes them in `add_batch()` / `sample()`.

```python
# In __init__:
self._extra_obs = {}
if extra_obs_dims:
    for name, dim in extra_obs_dims.items():
        self._extra_obs[name] = jnp.zeros((max_size, dim), dtype=jnp.float32)
        self._extra_obs[f"next_{name}"] = jnp.zeros((max_size, dim), dtype=jnp.float32)
```

In `add_batch`, accept `**extra_obs` kwargs. In `sample`, return them in the dict.

This is the cleanest extension — one buffer, same indices, type-safe.

- [ ] **Step 5: Wire in training loop**

```python
extra_dims = {"critic_obs": critic_obs_dim} if has_privileged else None
buffer = JaxReplayBuffer(obs_dim, action_dim, max_size=..., extra_obs_dims=extra_dims)

# In step loop:
buffer.add_batch(obs=raw_obs, action=action, ...,
                 critic_obs=critic_raw_obs, critic_next_obs=next_critic_raw_obs)

# sample() returns batch with "critic_obs" and "critic_next_obs" keys
```

When `has_privileged` is False, `extra_dims=None`, buffer works exactly as before. No regression.

- [ ] **Step 6: Commit**

```bash
git add train_offpolicy.py jax_rl/buffers/jax_replay_buffer.py
git commit -m "feat: pass privileged obs to critic in training loop"
```

---

### Task 4: Write tests

**Files:**
- Create: `tests/test_asymmetric_critic.py`

These tests are CRITICAL — they verify the obs routing is correct.

- [ ] **Step 1: Test Q network gets critic_obs_dim input**

```python
"""Tests for asymmetric off-policy critic."""
import jax
import jax.numpy as jnp
import pytest
import optax

OBS_DIM = 48
CRITIC_OBS_DIM = 122
ACTION_DIM = 12


class TestSACAsymmetric:
    def test_q_network_uses_critic_obs_dim(self):
        """Q network should be initialized with critic_obs_dim, not obs_dim."""
        from jax_rl.algos.sac import SAC
        from jax_rl.configs.sac_config import SACConfig
        cfg = SACConfig(hidden_dim=(32, 32))
        sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3),
                  critic_obs_dim=CRITIC_OBS_DIM)
        state = sac.init(jax.random.PRNGKey(0))

        # Q network should accept 122d obs
        q_val = sac.q1.apply(state.q1_params, jnp.zeros((1, CRITIC_OBS_DIM)),
                             jnp.zeros((1, ACTION_DIM)))
        assert q_val.shape == (1,)

        # Q network should FAIL with 48d obs (wrong input dim)
        with pytest.raises(Exception):
            sac.q1.apply(state.q1_params, jnp.zeros((1, OBS_DIM)),
                         jnp.zeros((1, ACTION_DIM)))

    def test_actor_uses_obs_dim(self):
        """Actor should still use obs_dim (48d), not critic_obs_dim."""
        from jax_rl.algos.sac import SAC
        from jax_rl.configs.sac_config import SACConfig
        cfg = SACConfig(hidden_dim=(32, 32))
        sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3),
                  critic_obs_dim=CRITIC_OBS_DIM)
        state = sac.init(jax.random.PRNGKey(0))

        # Actor should work with 48d
        action = sac.select_action(state.actor_params, jnp.zeros(OBS_DIM),
                                   jax.random.PRNGKey(1))
        assert action.shape == (ACTION_DIM,)

    def test_update_with_asymmetric_batch(self):
        """Full update should work with critic_obs/critic_next_obs in batch."""
        from jax_rl.algos.sac import SAC
        from jax_rl.configs.sac_config import SACConfig
        cfg = SACConfig(hidden_dim=(32, 32))
        sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3),
                  critic_obs_dim=CRITIC_OBS_DIM)
        state = sac.init(jax.random.PRNGKey(0))

        batch = {
            "obs": jnp.zeros((32, OBS_DIM)),
            "critic_obs": jnp.zeros((32, CRITIC_OBS_DIM)),
            "action": jnp.zeros((32, ACTION_DIM)),
            "reward": jnp.zeros((32, 1)),
            "next_obs": jnp.zeros((32, OBS_DIM)),
            "critic_next_obs": jnp.zeros((32, CRITIC_OBS_DIM)),
            "done": jnp.zeros((32, 1)),
            "truncation": jnp.zeros((32, 1)),
        }
        new_state, metrics = sac.update(state, batch)
        assert "q1_loss" in metrics
        assert "actor_loss" in metrics
        assert jnp.isfinite(metrics["q1_loss"])
        assert jnp.isfinite(metrics["actor_loss"])

    def test_backward_compatible_without_critic_obs(self):
        """When critic_obs_dim is None, batch["critic_obs"] should still work
        (training loop sets critic_obs = obs)."""
        from jax_rl.algos.sac import SAC
        from jax_rl.configs.sac_config import SACConfig
        cfg = SACConfig(hidden_dim=(32, 32))
        sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3))
        state = sac.init(jax.random.PRNGKey(0))

        batch = {
            "obs": jnp.ones((32, OBS_DIM)),
            "critic_obs": jnp.ones((32, OBS_DIM)),  # same as obs
            "action": jnp.zeros((32, ACTION_DIM)),
            "reward": jnp.zeros((32, 1)),
            "next_obs": jnp.ones((32, OBS_DIM)),
            "critic_next_obs": jnp.ones((32, OBS_DIM)),
            "done": jnp.zeros((32, 1)),
            "truncation": jnp.zeros((32, 1)),
        }
        new_state, metrics = sac.update(state, batch)
        assert jnp.isfinite(metrics["q1_loss"])

    def test_get_q_value_with_critic_obs(self):
        """get_q_value should use critic_obs when provided."""
        from jax_rl.algos.sac import SAC
        from jax_rl.configs.sac_config import SACConfig
        cfg = SACConfig(hidden_dim=(32, 32))
        sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3),
                  critic_obs_dim=CRITIC_OBS_DIM)
        state = sac.init(jax.random.PRNGKey(0))

        q = sac.get_q_value(state, jnp.zeros(OBS_DIM), jnp.zeros(ACTION_DIM),
                            critic_obs=jnp.zeros(CRITIC_OBS_DIM))
        assert jnp.isfinite(q)
```

- [ ] **Step 2: Same test pattern for TD3, FastSAC, FastTD3**

Create `TestTD3Asymmetric`, `TestFastSACAsymmetric`, `TestFastTD3Asymmetric` with the same 5 tests each, adapted for each algo's constructor and config.

- [ ] **Step 3: Test that actor NEVER sees critic_obs_dim**

```python
def test_actor_gradient_uses_obs_not_critic_obs():
    """Actor gradients should flow through obs (48d), not critic_obs (122d).
    This catches the subtle bug where _actor_loss accidentally passes
    critic_obs to the actor forward pass."""
    from jax_rl.algos.sac import SAC
    from jax_rl.configs.sac_config import SACConfig
    cfg = SACConfig(hidden_dim=(32, 32))
    sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3),
              critic_obs_dim=CRITIC_OBS_DIM)
    state = sac.init(jax.random.PRNGKey(0))

    # Create batch where obs and critic_obs are DIFFERENT
    batch = {
        "obs": jnp.ones((32, OBS_DIM)) * 1.0,
        "critic_obs": jnp.ones((32, CRITIC_OBS_DIM)) * 2.0,
        "action": jnp.zeros((32, ACTION_DIM)),
        "reward": jnp.zeros((32, 1)),
        "next_obs": jnp.ones((32, OBS_DIM)) * 1.0,
        "critic_next_obs": jnp.ones((32, CRITIC_OBS_DIM)) * 2.0,
        "done": jnp.zeros((32, 1)),
        "truncation": jnp.zeros((32, 1)),
    }
    # Should not crash — actor forward uses 48d, Q uses 122d
    new_state, metrics = sac.update(state, batch)
    assert jnp.isfinite(metrics["actor_loss"])

    # Verify actor params changed (gradient flowed)
    actor_diff = jax.tree.map(lambda a, b: jnp.sum(jnp.abs(a - b)),
                              state.actor_params, new_state.actor_params)
    total_diff = sum(jax.tree.leaves(actor_diff))
    assert total_diff > 0, "Actor params should change after update"
```

- [ ] **Step 4: Run all tests**

Run: `JAX_PLATFORMS=cpu uv run python -m pytest tests/test_asymmetric_critic.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_asymmetric_critic.py
git commit -m "test: comprehensive asymmetric critic tests for all off-policy algos"
```

---

### Task 5: Update eval Q diagnostics

**Files:**
- Modify: `jax_rl/utils/eval.py`
- Modify: `train_offpolicy.py` (q_fn lambda)

- [ ] **Step 1: Update q_fn in training loop**

The `q_fn` lambda passed to `evaluate()` needs to use critic obs. Currently:
```python
q_fn=lambda obs, action: algo.get_q_value(_ts, obs, action)
```

For dict obs envs, `obs` from the eval env is a dict. The lambda needs to extract both:
```python
q_fn=lambda obs, action: algo.get_q_value(
    _ts,
    obs["state"] if isinstance(obs, dict) else obs,
    action,
    critic_obs=obs.get("privileged_state") if isinstance(obs, dict) else None,
)
```

- [ ] **Step 2: Update eval.py to pass obs correctly**

In `evaluate()`, the `q_fn` receives raw env obs. For dict obs, this is already the full dict. The lambda above handles extraction. No changes needed to `eval.py` itself.

- [ ] **Step 3: Commit**

```bash
git add train_offpolicy.py
git commit -m "feat: eval Q diagnostics use privileged obs for asymmetric critic"
```

---

### Task 6: Full test suite + integration smoke test

- [ ] **Step 1: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests pass (158+ existing + ~25 new asymmetric tests).

- [ ] **Step 2: Smoke test Go2 with asymmetric critic**

```bash
XLA_CLIENT_MEM_FRACTION=0.5 uv run python train_offpolicy.py \
    --algo fast_sac --env Go2WarpJoystickFlat \
    --num-envs 64 --total-timesteps 100000
```
Expected: Runs without crash. Critic uses 122d, actor uses 48d.

- [ ] **Step 3: Update docs**

Update `.context/TODO.md` — mark asymmetric off-policy critic as done.
Update `.context/journals/` — add notes.
Update `.context/AGENT_HANDOFF.md` — update Go2 section to note asymmetric critic is available.

- [ ] **Step 4: Commit**

```bash
git add .context/
git commit -m "docs: update for asymmetric off-policy critic"
```
