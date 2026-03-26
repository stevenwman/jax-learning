# Learner Lessons — JAX & RL Fundamentals

Personal reference for JAX patterns, Flax/Linen mental models, and RL concepts learned while building this framework.

---

## JAX Fundamentals

### Tracer Errors
Can't use regular Python `print()` inside functions passed to `jax.grad()` or `jax.jit()`.

JAX traces functions to build computation graphs. During tracing, values are abstract "tracers", not concrete numbers.

```python
# Wrong
print(f"value: {x:.2f}")  # TypeError: unsupported format string passed to Tracer

# Correct
jax.debug.print("value: {}", x)  # Works with tracers
```

**Lesson:** Separate **trace time** (Python code analysis) from **execution time** (JAX computation).

---

### PRNG Key Management
Using the same key for multiple random operations gives correlated samples.

```python
# Wrong
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, shape=(10,))
b = jax.random.normal(key, shape=(10,))  # Same as a!

# Right
key = jax.random.PRNGKey(0)
key, subkey1, subkey2 = jax.random.split(key, 3)
a = jax.random.normal(subkey1, shape=(10,))
b = jax.random.normal(subkey2, shape=(10,))  # Independent!
```

JAX RNG is **pure functional** — always split keys for independent randomness.

---

### Array Reshaping with `-1`
`-1` means "infer this dimension from the rest."

```python
x.reshape(-1)                    # Flatten everything to 1D: (32, 4) → (128,)
x.reshape(-1, x.shape[-1])      # Flatten all dims except last: (32, 4, 17) → (128, 17)
```

Common pattern in PPO: Flatten (num_steps, num_envs, ...) → (num_steps * num_envs, ...) before SGD updates.

---

### `jax.block_until_ready()` — Required for Accurate Timing
JAX operations are **asynchronous** — `env.step()` returns immediately, queueing work on the GPU. Without `block_until_ready()`, you're timing the Python dispatch, not the actual computation.

---

### Python `if` vs `jax.lax.cond`
Python `if` on a traced value crashes. Use `jax.lax.cond` for branching inside `jit`, `scan`, `vmap`.

```python
# Fails inside scan/jit with traced bool
if deterministic:
    return jnp.tanh(mean)

# Works everywhere
return jax.lax.cond(deterministic, lambda: jnp.tanh(mean), lambda: action)
```

Rule: if a function might ever be called inside `scan`/`vmap`, use `jax.lax.cond`.

---

## Flax Linen Mental Model

### Blueprint vs State
**Linen:** Module is a blueprint. Parameters live separately.
```python
model = nn.Dense(4)                       # just a template, no params
params = model.init(key, jnp.zeros(3))    # params created here
y = model.apply(params, x)               # params passed in explicitly
```

Why this matters for RL: `update()` takes params in and returns new params out. No mutation, fully jittable, easy to checkpoint.

### @nn.compact — Layers Without __init__
```python
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)    # created + registered on first call (init)
        x = jax.nn.relu(x)      # reused on every subsequent call (apply)
        return nn.Dense(4)(x)
```

`nn.Dense(256)` only specifies output features. Input features are inferred during `model.init(key, dummy_input)`.

### TrainingState: The Functional State Container
All mutable state lives in one pytree:
```python
@flax.struct.dataclass
class TrainingState:
    actor_params: Params
    critic_params: Params
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
```

State flows through functions, not objects:
```python
state = ppo.init(key)
state, metrics = ppo.update(state, batch, key)  # state in → new state out
```

### jax.value_and_grad with Closures
Differentiate w.r.t. params. Loss function closes over data:
```python
def actor_loss_fn(actor_params):
    mean, log_std = self.actor.apply(actor_params, mb_obs)
    ...
    return loss, metrics

(loss, metrics), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor_params)
```

`has_aux=True`: function returns `(loss, extra_stuff)`. Differentiates only the first element.

### Optax: The Explicit Optimizer Pattern
```python
optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
opt_state = optimizer.init(params)

# In update — nothing is mutated
updates, new_opt_state = optimizer.update(grads, opt_state)
new_params = optax.apply_updates(params, updates)
```

---

## RL Architecture Insights

### When to Use Squashing
Squashing is about **action space bounds**, not on/off-policy:
- Bounded actions (robot joints: [-1, 1]) → Use tanh squashing
- Unbounded actions → Unbounded Gaussian is fine

PPO, SAC, and TD3 all use tanh squashing in this codebase.

### Separate Actor/Critic Optimizers
- No need for `value_coef` hyperparameter
- Independent learning rates
- Cleaner gradient flow, easier to debug

### Jit-Friendly vs Not
Two collection strategies:
- **Jittable env** (MJX/Brax): `jax.lax.scan` for collection → fully on GPU
- **Non-jittable env** (Gymnasium, real robot): Python loop + mutable buffer → fine!

Both produce the same batch → same `update(batch)` call.

---

## Debugging Techniques

### Systematic NaN Hunting
1. **Reproduce** — Isolate the minimal case that triggers NaN
2. **Instrument** — Add `jax.debug.print()` at intermediate steps
3. **Trace backwards** — Find where NaN first appears
4. **Fix root cause** — Don't mask with try/except!

### The atanh(±1) Singularity
Tanh-squashed actions bounded to [-1, 1]. Computing log_prob requires `atanh(x)` which diverges at ±1.

Fix: clip actions away from boundaries:
```python
action = jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6)
```

### JAX/XLA GPU Memory Model

JAX pre-allocates a fixed GPU memory pool on first use (`XLA_CLIENT_MEM_FRACTION`, default 75%). All buffers come from this pool — no dynamic OS allocation at runtime.

**Compilation (JIT):** Runs on CPU. XLA plans a deterministic memory schedule — exactly when each buffer is allocated, used, and freed. The schedule must fit the pool.

**When pool is tight:** XLA compensates by:
- **Rematerialization** — recompute values instead of keeping them in memory (same result, slower)
- Different **tiling/layout strategies** for matmuls
- **Autotuning** tests fewer kernel variants (less scratch space available)

**Runtime:** Completely deterministic. No dynamic allocation. Compiled kernels run within the fixed pool.

**Practical impact:** More VRAM → less rematerialization → ~5-10% faster. Restarting a training run with more free VRAM only helps if you recompile (JIT again). Once compiled, the memory schedule is baked in.

### Package Structure
Use `__init__.py` for clean public APIs:
```python
# Instead of: from jax_rl.configs.ppo_config import PPOConfig
# Do:         from jax_rl.configs import PPOConfig
```
