# Lessons Learned - JAX RL Framework

A running log of debugging victories, JAX gotchas, and hard-won insights from building this RL framework.

---

## Session 1: Foundation & Debugging (Feb 2026)

### 🐛 **Bug #1: The atanh(±1) Singularity**

**What happened:**
- PPO update was returning NaN for policy loss and entropy
- Actions sampled from the policy worked fine, but recomputing log_prob gave NaN

**Root cause:**
- When using `squash=True` (tanh transform), actions are bounded to [-1, 1]
- Computing log_prob requires inverse tanh: `atanh(x) = 0.5 * log((1+x)/(1-x))`
- At x = ±1: `atanh(±1) = ±∞` → NaN propagation

**The fix:**
```python
# In distributions.py
ATANH_EPSILON = 1e-6

if squash:
    # Clip actions away from ±1 to avoid atanh singularities
    action = jnp.clip(action, -1.0 + ATANH_EPSILON, 1.0 - ATANH_EPSILON)
    dist = distrax.Transformed(base_dist, distrax.Tanh())
```

**Lesson:** Numerical singularities are common in RL. Always check boundary conditions!

---

### 🎯 **JAX Fundamentals**

#### **1. Tracer Errors**
**Problem:** Can't use regular Python `print()` inside functions passed to `jax.grad()` or `jax.jit()`.

**Why:** JAX traces functions to build computation graphs. During tracing, values are abstract "tracers", not concrete numbers.

**Solution:**
```python
# ❌ Wrong
print(f"value: {x:.2f}")  # TypeError: unsupported format string passed to Tracer

# ✅ Correct
jax.debug.print("value: {}", x)  # Works with tracers
```

**Lesson:** Separate **trace time** (Python code analysis) from **execution time** (JAX computation).

---

#### **2. PRNG Key Management**
**Problem:** Using the same key for multiple random operations gives correlated samples.

**Wrong:**
```python
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, shape=(10,))
b = jax.random.normal(key, shape=(10,))  # Same as a!
```

**Right:**
```python
key = jax.random.PRNGKey(0)
key, subkey1, subkey2 = jax.random.split(key, 3)
a = jax.random.normal(subkey1, shape=(10,))
b = jax.random.normal(subkey2, shape=(10,))  # Independent!
```

**Lesson:** JAX RNG is **pure functional** - always split keys for independent randomness.

---

#### **3. Array Reshaping with `-1`**
`-1` means "infer this dimension from the rest."

```python
x.reshape(-1)                    # Flatten everything to 1D: (32, 4) → (128,)
x.reshape(-1, x.shape[-1])      # Flatten all dims except last: (32, 4, 17) → (128, 17)
```

**Common pattern in PPO:** Flatten (num_steps, num_envs, ...) → (num_steps * num_envs, ...) before SGD updates. PPO doesn't care about temporal order during updates — all transitions are independent samples.

```python
obs = batch.obs.reshape(-1, batch.obs.shape[-1])   # (32, 4, 17) → (128, 17)
advantages = batch.advantages.reshape(-1)            # (32, 4) → (128,)
```

---

### 🏗️ **Architecture Insights**

#### **1. When to Use Squashing**
**Common misconception:** Squashing is for off-policy algorithms (SAC), not on-policy (PPO).

**Truth:** Squashing is about **action space bounds**, not on/off-policy!
- Bounded actions (robot joints: [-1, 1]) → Use squashing OR bounded distribution (Beta)
- Unbounded actions → Unbounded Gaussian is fine

PPO can use either depending on the environment.

---

#### **2. Separate Actor/Critic Optimizers**
**Why separate?**
- No need for `value_coef` hyperparameter
- Cleaner gradient flow
- Independent learning rates
- Easier to debug (losses don't mix)

```python
# ✅ Our approach (optax + TrainingState)
actor_optimizer = optax.adam(lr_schedule)
critic_optimizer = optax.adam(lr_schedule)

# PPO takes them as constructor args
ppo = PPO(config, obs_dim, action_dim, actor_optimizer, critic_optimizer)

# Internally: separate grad/update calls per network
actor_updates, new_actor_opt = actor_optimizer.update(actor_grads, state.actor_opt_state)
value_updates, new_value_opt = critic_optimizer.update(value_grads, state.critic_opt_state)
```

---

#### **3. Jit-Friendly vs Not — Know the Difference**
**Not everything needs to be jitted.** Two collection strategies:

- **Jittable env** (MJX/Brax): `jax.lax.scan` for collection → fully on GPU
- **Non-jittable env** (Gymnasium, real robot): Python loop + mutable buffer → fine!

Both produce the same `RolloutBatch` → same `ppo.update(batch)` call.

**Rule:** Profile first, optimize second. Buffer writes are not the bottleneck — gradient computation and env stepping are.

---

### 🔧 **Debugging Techniques**

#### **1. Systematic NaN Hunting**
1. **Reproduce** - Isolate the minimal case that triggers NaN
2. **Instrument** - Add `jax.debug.print()` at intermediate steps
3. **Trace backwards** - Find where NaN first appears
4. **Fix root cause** - Don't mask with try/except!

**Example from this session:**
```
gaussian_log_prob - mean: OK ✓
gaussian_log_prob - log_std: OK ✓
gaussian_log_prob - action: OK ✓
gaussian_log_prob - log_prob RESULT: NaN ❌  ← Found it!
```

### 📦 **Package Structure**

**Lesson:** Use `__init__.py` files to create clean public APIs.

**Without:**
```python
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.networks.builders import Actor
```

**With:**
```python
from jax_rl.configs import PPOConfig
from jax_rl.networks import Actor
```

**Tip:** Export high-level components, keep implementation details internal.

---

### 🎓 **Key Takeaways**

1. **JAX is functional** - No hidden state, explicit randomness, pure functions
2. **Numerical stability matters** - Always check boundaries and edge cases
3. **Debug systematically** - Instrument, trace, fix root cause
4. **Design for modularity** - Protocols, builders, separation of concerns
5. **Test incrementally** - Verify each component before building on it

---

## Session 4: CheetahRun, Performance, & Tooling (Mar 2026)

### **`jax.lax.scan` vs Python Loops — 542x Speedup**

Benchmarked three PPO implementations on the same update workload:

| Variant | Per-update time | Speedup |
|---------|----------------|---------|
| `ppo.py` (Python loops, no JIT) | 2.712s | 1x |
| `ppo_jit.py` (JIT closures, Python epoch loops) | 0.089s | 30x |
| `ppo_scan.py` (JIT + `jax.lax.scan` for epochs) | 0.005s | 542x |

**Why scan matters:** For small MLPs (64×64), 95% of each call was Python↔XLA dispatch overhead. `jax.lax.scan` compiles the entire epoch loop into a single XLA program — no Python roundtrips. The JIT-only version still had Python loops for epochs/minibatches, causing repeated dispatch.

**Lesson:** If your model is small and your loop body is fast, Python loop overhead dominates. `scan` eliminates it entirely.

---

### **Entropy Coefficient Tuning — When Exploration Hurts**

**What happened:** CheetahRun with `entropy_coef=0.001` — entropy grew unboundedly (9→16+), returns plateaued at ~248.

**Root cause:** With unconstrained `log_std` (clipped to [-5, 2]) and a positive entropy bonus, the optimizer pushes `log_std` toward the upper bound to maximize entropy. The policy becomes too noisy to learn anything useful.

**The fix:** Set `entropy_coef=0.0` for CheetahRun. Returns jumped to 503 at 10M steps.

**Diagnostic signals:**
- Entropy growing monotonically = entropy bonus is dominating task reward
- High KL + high clip fraction = policy changing too aggressively
- PLoss near 0 at end = policy gradient exhausted (needs LR annealing)

**Lesson:** Entropy bonus is environment-dependent. Simple tasks (CartpoleBalance) benefit from exploration (`entropy_coef=0.01`). Complex continuous control (CheetahRun) often needs `entropy_coef=0.0` to avoid the entropy-maximization trap.

---

### **LR Annealing for Continuous Control**

**Problem:** With constant LR (3e-4), policy loss converges to ~0 and clip fraction stays high (0.27–0.35) late in training. The policy keeps making large updates even when it should be fine-tuning.

**Fix:** Linear LR schedule annealing to 0 over total gradient steps:
```python
total_gradient_steps = num_iterations * num_updates_per_batch * num_epochs * num_minibatches
lr_schedule = optax.linear_schedule(lr, 0.0, total_gradient_steps)
optimizer = optax.adam(lr_schedule)  # optionally chain with clip_by_global_norm
```

**Lesson:** LR annealing lets the policy explore broadly early and fine-tune late. Standard practice for PPO on continuous control — Brax does this too.

---

### **Optimizer Decoupling — Keep Algorithms Pure**

PPO shouldn't own optimizer construction. Optimizers (LR schedule, grad clipping, future exotic optimizers like Muon) are external concerns.

**Before:** PPO internally created optimizers from config fields (`actor_lr`, `anneal_lr`, etc.)
**After:** PPO takes `actor_optimizer` and `critic_optimizer` as constructor args. Train script builds them.

```python
# train.py constructs optimizers — grad clipping is optional
if max_grad_norm is not None:
    actor_optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_schedule))
else:
    actor_optimizer = optax.adam(lr_schedule)
ppo = PPO(config, obs_dim, action_dim, actor_optimizer, critic_optimizer)
```

**Lesson:** Same principle as dependency injection — algorithms define *what* they optimize, not *how*. Makes swapping optimizers trivial (Muon, Shampoo, etc.) without touching algorithm code.

---

### **Orbax Checkpointing**

- `ocp.StandardCheckpointer()` saves/restores arbitrary pytrees (works with Linen or NNX)
- **Must call `checkpointer.wait_until_finished()`** after save — otherwise process exit kills the async write thread ("cannot schedule new futures after shutdown")
- Save `meta.json` alongside checkpoint for config reconstruction (env name, hidden dims, obs/action dims)
- Checkpoint + metrics CSV + meta.json in timestamped directories: `checkpoints/{timestamp}_{env}_seed{seed}/`

---

### **Video Recording — Two-Phase Approach**

MuJoCo Playground's `env.render()` is CPU-side and can't be JIT'd. Solution:

1. **Phase 1 (GPU):** `jax.lax.scan` the rollout — fast, collects all states
2. **Phase 2 (CPU):** Render frames from saved states — slow but unavoidable

```python
# Phase 1: JIT-compiled rollout
(_, _, _), trajectory = jax.lax.scan(rollout_step, init_carry, None, length=max_steps)

# Phase 2: CPU rendering
states = [env_state] + [jax.tree.map(lambda x: x[i], trajectory) for i in range(num_frames)]
frames = env.render(states, camera=camera)
```

**Lesson:** Separate compute from rendering. The rollout itself is fast (~0.3s for 1000 steps including JIT). Rendering is the bottleneck (~50ms/frame).

---

### **CheetahRun Hyperparameter Journey**

| Config | Return | Notes |
|--------|--------|-------|
| entropy_coef=0.001, 10M steps | ~248 | Entropy grew to 16+, policy too noisy |
| entropy_coef=0.0, 10M steps | ~503 | Entropy controlled, policy learning |
| entropy_coef=0.0, 20M steps | ~666 | More steps helped, approaching target |
| + LR annealing, 20M steps | ~615 | Annealed too aggressively, policy froze late |
| Preset HPs (2048 envs, lr=1e-3, 16 epochs, reward_scaling=10) | **826** | Passed target at 20M steps |

Target was ≥700 at 60M steps (MuJoCo Playground paper). Hit **826 at 20M** with proper HPs. Key changes: 2048 envs, shorter rollouts (30 steps), higher LR (1e-3), more epochs (16), reward scaling (10x).

---

## Session 5: HumanoidRun Debugging & Pipeline Robustness (Mar 2026)

### **State-Independent vs State-Dependent log_std**

**Problem:** HumanoidRun (21-dim actions) NaN'd repeatedly with state-dependent std (`nn.Dense` mapping features → log_std).

**Root cause:** Dense layer produces unpredictable initial log_std values. Outlier observations spike log_std → entropy explodes → NaN. With 21 action dims, the entropy is already ~30 nats from dimensionality alone (`0.5 * 21 * (1 + log(2π)) ≈ 29.8`).

**Fix:** State-independent log_std (`nn.Param`), matching Brax PPO:
```python
# State-independent: single learned vector, same for all observations
log_std = self.param('log_std', nn.initializers.constant(jnp.log(init_noise_std)), (action_dim,))
log_std = jnp.broadcast_to(log_std, mean.shape)

# MUST clip in both modes — without this, entropy bonus pushes log_std → ∞
log_std = jnp.clip(log_std, log_std_min, log_std_max)
```

**Lesson:** For PPO, state-independent std is more stable. Save state-dependent std for SAC where maximum entropy is the objective. Always clip log_std regardless of mode.

---

### **Minibatch Count, Not Size, Is the Right Parameterization**

**Problem:** Hardcoded `minibatch_size = min(2048, samples_per_iter)`. When we increased `num_steps` from 30 to 480 (16x more data), `samples_per_iter` went from 61k to 983k. But minibatch_size stayed at 2048, so `num_minibatches` went from 30 to 480. With 16 epochs, that's 7,680 gradient steps per iteration — Brax uses 512.

**The coupling:** `grad_steps = num_minibatches × num_epochs`. Fixing `minibatch_size` means `num_minibatches` scales with data volume. Fixing `num_minibatches` keeps gradient steps constant regardless of data volume.

**Fix:** Parameterize by `num_minibatches` (default 32, matching Brax). Derive `minibatch_size = samples_per_iter // num_minibatches`.

**Lesson:** When comparing against a reference implementation, match the *structure* (what's fixed vs derived), not just the numbers. Brax fixes `num_minibatches=32` and `batch_size=1024`; we were fixing `minibatch_size=2048` which is a completely different parameterization.

---

### **Batch-Level vs Per-Minibatch Advantage Normalization**

Brax normalizes advantages over the entire batch (~983k samples) before splitting into minibatches. We were normalizing per-minibatch (~2k samples), giving noisy mean/std estimates.

With 30k+ samples per minibatch (after the fix), the difference is small. But for smaller setups or early in training, batch-level normalization is strictly better.

---

### **Reshape Crash on Non-Divisible Batches**

`obs[perm].reshape(num_minibatches, minibatch_size, -1)` silently assumes `N == num_minibatches * minibatch_size`. If not (e.g., 5000 samples with minibatch_size=1024 → 4 batches of 1024 = 4096 ≠ 5000), it crashes.

**Fix:** Truncate the permutation: `perm = perm[:num_minibatches * minibatch_size]`.

**Lesson:** Always handle the case where data doesn't divide evenly. It works by coincidence for tuned presets but breaks for arbitrary configs.

---

### **HumanoidRun Diagnostic Signals**

Added `log_std_mean`, `log_std_min`, `log_std_max` to PPO metrics. Key observations:

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| Clip fraction | 0.05–0.20 | 0.50–0.80 (policy overshooting) |
| Entropy | Slowly decreasing | Growing monotonically (entropy bonus dominating) |
| VLoss | Stable, tracks returns | Collapsing to 0.00 (overfitting to batch) |
| KL | < 0.1 | Spikes > 1.0 (imminent NaN) |
| log_std | Slowly decreasing from ~0 | Growing or stuck at clip bounds |

---

### **Interleaved Collect→Update (num_updates_per_batch)**

**Problem:** With one long rollout (480 steps) followed by 4 epochs of updates, the policy used for collection diverges significantly from the policy being optimized. With short rollouts (20 steps) + 4 epochs, we massively overtrain on tiny batches — policy collapsed (entropy 25 → -1.25, clip fraction → 0.000).

**Brax's approach:** Instead of `collect(long) → update(many epochs)`, Brax does:
```
for _ in range(num_updates_per_batch):  # 16 cycles
    data = collect(20 steps)             # fresh data each time
    sgd_update(data, num_epochs=1)       # just 1 epoch
```

**Why it matters:**
- Policy stays close to the data-collecting policy (lower staleness)
- Episode resets naturally stagger across cycles (fixes VLoss oscillation from synchronized resets)
- Same total gradient steps (16 × 1 × 32 = 512) but each uses on-policy data
- `num_updates_per_batch=1, num_epochs=4` recovers our old behavior

**Key numbers (HumanoidRun):**

| Config | Grad steps/iter | Data freshness | Result |
|--------|----------------|----------------|--------|
| num_steps=480, epochs=4, updates=1 | 128 | 480 steps stale | Returns 7.6 (slow) |
| num_steps=20, epochs=4, updates=1 | 128 | Fresh but overtrained | Collapsed (entropy → -1.25) |
| num_steps=20, epochs=1, updates=16 | 512 | Fresh each cycle | Brax reference config |

**Lesson:** The right granularity is many short collect→update cycles, not fewer long ones. `num_updates_per_batch` generalizes our pipeline — setting it to 1 recovers the old behavior.

---

### **VLoss Oscillation from Synchronized Episode Resets**

**Problem:** VLoss alternated between ~0.35 and ~61.5 every other iteration. Even iterations (episode boundaries) had huge VLoss; odd iterations had near-zero VLoss.

**Root cause:** `wrap_for_brax_training` resets all envs at the same timestep (episode_length=1000). With 4096 envs all resetting simultaneously, the value function sees completely different return distributions on boundary iterations vs mid-episode iterations. It overfits to mid-episode predictions, then gets blindsided at boundaries.

**How Brax avoids this:** With `num_updates_per_batch=16` and `num_steps=20`, updates happen every 20 steps. After the first episode completes, the 16 sequential collection phases within an iteration mean envs are at different points in their episodes across update cycles.

**Lesson:** Synchronized resets + long rollouts = VLoss instability. Interleaved short rollouts naturally desynchronize episode phases.

---

### **Truncation Bootstrap Bug in Auto-Reset Environments**

**Problem:** HumanoidRun returns stuck at ~7.6 at 60M steps (reference: 50-200). Multiple HP tuning attempts couldn't close the gap. Turns out it was a correctness bug, not a hyperparameter issue.

**Root cause:** `wrap_for_brax_training` auto-resets envs on episode timeout. After reset, `env_state.obs` is the **reset observation** (humanoid standing still), not the terminal observation (humanoid running). Our GAE used `values[t+1] = V(reset_obs)` as the bootstrap for truncated transitions. Since V(reset) << V(running), this created a systematic negative bias — good trajectories were undervalued at every episode boundary.

**Wrong fix (reward adjustment):** Adding `gamma * V(s_t) * truncation` to the reward creates massive synchronized reward spikes (all envs truncate at once) → NaN at iter 9.

**Correct fix (GAE-level):**
```python
# In compute_gae: replace V(reset_obs) with V(s_t) at truncation steps
next_values = jnp.where(truncations, values, next_values)
effective_dones = dones * (1.0 - truncations)  # bootstrap through truncations
```

No reward modification. The truncation flag flows through the buffer into GAE, where the bootstrap value is corrected without creating spikes.

**Lesson:** Auto-reset environments silently corrupt value estimates at episode boundaries. Always check what observation the value function sees after a reset — if it's the wrong state, the entire value function learns wrong targets. Fix in GAE (where the bootstrap happens), not in the reward signal.

---

## Future Topics to Explore

- [x] `jax.jit` compilation and when to use it
- [x] `jax.lax.scan` vs Python loops (efficiency) — 542x speedup on PPO update
- [x] Checkpointing with Orbax
- [ ] Performance profiling with JAX
- [ ] Pytrees and how Linen models work as pytrees
- [ ] Device placement (CPU vs GPU)
- [ ] Layer normalization in JAX/Linen (needed for FastTD3)

---

*"The best way to learn is to break things, then fix them systematically."*
