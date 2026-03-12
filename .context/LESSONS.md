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
# Linen + optax approach (current)
actor_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
critic_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))

# In update: explicit three-step process
updates, new_opt_state = actor_optimizer.update(grads, state.actor_opt_state)
new_params = optax.apply_updates(state.actor_params, updates)
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

#### **2. Linen Init Needs Dummy Input**
```python
# Linen infers input shapes from data — init needs a dummy
model = nn.Dense(256)
params = model.init(key, jnp.zeros(obs_dim))  # jnp.zeros(obs_dim) tells Linen the input shape
```
If you forget the dummy input, you get a confusing error about missing arguments.

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

1. **JAX is functional** — No hidden state, explicit randomness, pure functions
2. **Numerical stability matters** — Always check boundaries and edge cases
3. **Debug systematically** — Instrument, trace, fix root cause
4. **Design for modularity** — Separate concerns: modules for computation, functions for everything else
5. **Test incrementally** — Verify each component before building on it

---

## Session 2: NNX → Linen Conversion (Mar 2026)

### 🔄 **Why We Switched from NNX to Linen**

NNX is the newer, more Pythonic Flax API. But for high-throughput RL with small MLPs, Linen is better:

1. **Performance**: `nnx.jit` traverses the Python object graph on every call — ~3x overhead on small MLPs. Linen's params are plain pytrees with zero overhead inside jit.
2. **Ecosystem**: Brax, FastTD3, MuJoCo Playground all use Linen / functional params. We can lift patterns directly.
3. **Immutability = safety**: Impossible to accidentally mutate shared state across parallel envs or between actor/critic.
4. **Learning**: Forces you to understand pytrees, pure functions, and JIT boundaries — NNX hides this behind sugar.

---

### 🧠 **The Core Mental Model: Blueprint vs State**

**NNX (old):** Module owns its parameters. It's an object.
```python
# NNX: module IS the model — params live inside
layer = nnx.Linear(3, 4, rngs=rngs)
y = layer(x)                          # uses internal params
```

**Linen (new):** Module is a blueprint. Parameters live separately.
```python
# Linen: module DESCRIBES the model — params live outside
model = nn.Dense(4)                                        # just a template, no params
params = model.init(key, jnp.zeros(3))                     # params created here
y = model.apply(params, x)                                 # params passed in explicitly
```

**Why this matters for RL:** In a training loop, `update()` takes params in and returns new params out. No mutation, fully jittable, easy to checkpoint.

---

### 📐 **@nn.compact — Layers Without __init__**

In NNX, you build layers in `__init__` and use them in `__call__`. In Linen with `@nn.compact`, you do both in `__call__`:

```python
# NNX
class MLP(nnx.Module):
    def __init__(self, rngs):
        self.layer1 = nnx.Linear(3, 256, rngs=rngs)
        self.layer2 = nnx.Linear(256, 4, rngs=rngs)
    def __call__(self, x):
        return self.layer2(nnx.relu(self.layer1(x)))

# Linen
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)    # created + registered on first call (init)
        x = jax.nn.relu(x)      # reused on every subsequent call (apply)
        return nn.Dense(4)(x)
```

Linen tracks submodules by order of creation. First `model.init()` creates them; every `model.apply()` reuses them.

**Key difference:** `nn.Dense(256)` only specifies output features. Input features are inferred from the data during `model.init(key, dummy_input)`. That's why init needs a dummy input.

---

### 🏗️ **TrainingState: The Functional State Container**

In NNX, the algorithm object holds everything:
```python
# NNX — state scattered across mutable objects
self.actor          # params inside
self.critic         # params inside
self.actor_optimizer  # optimizer state inside
```

In Linen, all mutable state lives in one pytree:
```python
# Linen — all state in one place, explicitly threaded
@flax.struct.dataclass
class TrainingState:
    actor_params: Params
    critic_params: Params
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
```

The algorithm class only holds blueprints (modules) and config — things that never change:
```python
class PPO:
    def __init__(self, config, obs_dim, action_dim):
        self.actor = Actor(...)          # blueprint, no params
        self.critic = Critic(...)        # blueprint, no params
        self.actor_optimizer = optax.chain(...)  # optimizer definition, no state
```

**State flows through functions, not objects:**
```python
state = ppo.init(key)                      # creates initial TrainingState
state, metrics = ppo.update(state, batch, key)  # state in → new state out
state, metrics = ppo.update(state, batch, key)  # same pattern every step
```

---

### ⚡ **jax.value_and_grad with Closures**

In NNX, you differentiate with respect to the model:
```python
grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
(loss, metrics), grads = grad_fn(self.actor)
```

In Linen, you differentiate with respect to params. The loss function closes over the data:
```python
def actor_loss_fn(actor_params):
    # actor_params is what we differentiate w.r.t.
    # mb_obs, mb_actions, etc. are captured from the enclosing scope
    mean, log_std = self.actor.apply(actor_params, mb_obs)
    log_probs = gaussian_log_prob(mean, log_std, mb_actions)
    ...
    return loss, metrics

(loss, metrics), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor_params)
```

**`has_aux=True`:** Tells JAX the function returns `(loss, extra_stuff)`. It differentiates only the first element and passes the second through untouched.

**Why closures work here:** Define the loss function inside the minibatch loop. It captures the current minibatch slices (`mb_obs`, `mb_actions`, etc.) from the enclosing scope. The only argument is params — exactly what `value_and_grad` needs.

---

### 🔧 **Optax: The Explicit Optimizer Pattern**

NNX wraps optimizer state internally:
```python
optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
optimizer.update(model, grads)  # mutates model in place
```

With Linen, you use optax directly — three separate steps:
```python
# Define (in __init__)
optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))

# Initialize (in init)
opt_state = optimizer.init(params)

# Update (in update) — nothing is mutated
updates, new_opt_state = optimizer.update(grads, opt_state)
new_params = optax.apply_updates(params, updates)
```

More verbose but fully explicit. You always know what state is changing and when.

---

### 🚫 **Methods That Disappeared**

NNX `Actor` had three methods: `__call__`, `sample`, `log_prob`. In Linen, `Actor` only has `__call__` — it returns `(mean, log_std)`. Sampling and log_prob use standalone functions that already existed:

```python
# NNX — methods on the module
mean, log_std = actor(obs)
action, log_prob = actor.sample(obs, key)
log_prob = actor.log_prob(obs, action)

# Linen — module + standalone functions
mean, log_std = actor.apply(params, obs)
action, log_prob = sample_gaussian(mean, log_std, key)
log_prob = gaussian_log_prob(mean, log_std, action)
```

**Lesson:** Modules define computation graphs. Everything else is just a function. `sample_gaussian` and `gaussian_log_prob` were always standalone in `distributions.py` — the NNX methods were just wrappers.

---

### 🐛 **Bug: nn.Params Doesn't Exist**

```python
actor_params: nn.Params  # AttributeError: module 'flax.linen' has no attribute 'Params'
```

Params in Linen are just nested dicts — there's no special type. Use:
```python
from typing import Any
Params = Any
```

---

### 🎓 **Key Takeaways**

1. **Linen modules are blueprints** — they describe computation, they don't hold state
2. **Params are just pytrees** — dicts of arrays, nothing magic
3. **TrainingState is the single source of truth** — all mutable state in one place
4. **Functions > methods** — if it doesn't need the module's structure, make it a function
5. **Closures are the standard pattern** — loss functions close over data, take params as the only argument
6. **The math doesn't change** — PPO loss, GAE, distributions are identical; only the plumbing changed

---

## Future Topics to Explore

- [ ] `jax.jit` — wrapping `update` for compilation speedup
- [ ] `jax.lax.scan` vs Python loops inside jitted functions
- [ ] Performance profiling with JAX (compilation time vs execution time)
- [ ] Pytrees — how Linen params are structured, `jax.tree_util`
- [ ] Observation normalization (running statistics)
- [ ] Connecting to MuJoCo Playground environments
- [ ] The collect loop: buffer filling + GAE in practice
- [ ] Device placement (CPU vs GPU)
- [ ] Checkpointing with Orbax
- [ ] `flax.struct.dataclass` vs regular dataclass vs NamedTuple

---

*"The best way to learn is to break things, then fix them systematically."*