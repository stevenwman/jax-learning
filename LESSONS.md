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

## Future Topics to Explore

- [ ] `jax.jit` compilation and when to use it
- [ ] Performance profiling with JAX
- [ ] `jax.lax.scan` vs Python loops (efficiency)
- [ ] Pytrees and how NNX models work as pytrees
- [ ] Device placement (CPU vs GPU)
- [ ] Batch normalization in JAX/NNX
- [ ] Checkpointing with Orbax

---

*"The best way to learn is to break things, then fix them systematically."*
