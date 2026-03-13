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

#### **3. Vectorization with `jax.vmap`**
**Problem:** Need to apply a function over a batch dimension without a Python loop.

**Example:**
```python
# Sample actions for all timesteps
batched_action_select = jax.vmap(ppo.select_action, in_axes=(0, 0, None))
actions, log_probs, values = batched_action_select(obs, keys, False)
#                                                    ^    ^     ^
#                                              vmap over obs and keys, broadcast False
```

**Lesson:** `vmap` is JAX's way to vectorize. Specify which axes to map over with `in_axes`.

---

#### **4. Array Reshaping with `-1`**
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
# ✅ Our approach
actor_optimizer = nnx.Optimizer(self.actor, actor_tx, wrt=nnx.Param)
critic_optimizer = nnx.Optimizer(self.critic, critic_tx, wrt=nnx.Param)

# Update separately
actor_optimizer.update(self.actor, actor_grads)
critic_optimizer.update(self.critic, critic_grads)
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

#### **2. NNX 0.11+ API Changes**
```python
# New API requires wrt parameter
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# Update requires both model and grads
optimizer.update(model, grads)
```

---

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
total_gradient_steps = num_iterations * num_epochs * num_minibatches
lr_schedule = optax.linear_schedule(lr, 0.0, total_gradient_steps)
optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule))
```

**Lesson:** LR annealing lets the policy explore broadly early and fine-tune late. Standard practice for PPO on continuous control — Brax does this too.

---

### **Optimizer Decoupling — Keep Algorithms Pure**

PPO shouldn't own optimizer construction. Optimizers (LR schedule, grad clipping, future exotic optimizers like Muon) are external concerns.

**Before:** PPO internally created optimizers from config fields (`actor_lr`, `anneal_lr`, etc.)
**After:** PPO takes `actor_optimizer` and `critic_optimizer` as constructor args. Train script builds them.

```python
# train.py constructs optimizers
actor_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule))
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

### **Brax Wrapper Episode Tracking**

**Observation:** Episode returns appear "frozen" for ~16 iterations, then jump. This is NOT a bug.

**Why:** `wrap_for_brax_training` synchronizes episode resets. With 256 envs and episode_length=1000, all envs complete at the same time (every `1000 / 64 ≈ 16` iterations). Between completions, the running average doesn't update because no new episodes finish.

**Lesson:** Don't confuse synchronized episode completion with training stagnation. The policy is still learning between episode boundaries — you just can't see it in the return metric until episodes complete.

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
