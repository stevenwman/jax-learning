# Lessons Learned - JAX RL Framework

A running log of debugging victories, JAX gotchas, and hard-won insights from building this RL framework.

---

## Session 1: Foundation & Debugging (Feb 2026)

### Bug #1: The atanh(±1) Singularity

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

### JAX Fundamentals

#### 1. Tracer Errors
**Problem:** Can't use regular Python `print()` inside functions passed to `jax.grad()` or `jax.jit()`.

**Why:** JAX traces functions to build computation graphs. During tracing, values are abstract "tracers", not concrete numbers.

**Solution:**
```python
# Wrong
print(f"value: {x:.2f}")  # TypeError: unsupported format string passed to Tracer

# Correct
jax.debug.print("value: {}", x)  # Works with tracers
```

**Lesson:** Separate **trace time** (Python code analysis) from **execution time** (JAX computation).

---

#### 2. PRNG Key Management
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

#### 3. Array Reshaping with `-1`
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

### Architecture Insights

#### 1. When to Use Squashing
**Common misconception:** Squashing is for off-policy algorithms (SAC), not on-policy (PPO).

**Truth:** Squashing is about **action space bounds**, not on/off-policy!
- Bounded actions (robot joints: [-1, 1]) → Use squashing OR bounded distribution (Beta)
- Unbounded actions → Unbounded Gaussian is fine

PPO can use either depending on the environment.

---

#### 2. Separate Actor/Critic Optimizers
**Why separate?**
- No need for `value_coef` hyperparameter
- Cleaner gradient flow
- Independent learning rates
- Easier to debug (losses don't mix)

```python
# Our approach (optax + TrainingState)
actor_optimizer = optax.adam(lr_schedule)
critic_optimizer = optax.adam(lr_schedule)

# PPO takes them as constructor args
ppo = PPO(config, obs_dim, action_dim, actor_optimizer, critic_optimizer)

# Internally: separate grad/update calls per network
actor_updates, new_actor_opt = actor_optimizer.update(actor_grads, state.actor_opt_state)
value_updates, new_value_opt = critic_optimizer.update(value_grads, state.critic_opt_state)
```

---

#### 3. Jit-Friendly vs Not — Know the Difference
**Not everything needs to be jitted.** Two collection strategies:

- **Jittable env** (MJX/Brax): `jax.lax.scan` for collection → fully on GPU
- **Non-jittable env** (Gymnasium, real robot): Python loop + mutable buffer → fine!

Both produce the same `RolloutBatch` → same `ppo.update(batch)` call.

**Rule:** Profile first, optimize second. Buffer writes are not the bottleneck — gradient computation and env stepping are.

---

### Debugging Techniques

#### 1. Systematic NaN Hunting
1. **Reproduce** - Isolate the minimal case that triggers NaN
2. **Instrument** - Add `jax.debug.print()` at intermediate steps
3. **Trace backwards** - Find where NaN first appears
4. **Fix root cause** - Don't mask with try/except!

**Example from this session:**
```
gaussian_log_prob - mean: OK
gaussian_log_prob - log_std: OK
gaussian_log_prob - action: OK
gaussian_log_prob - log_prob RESULT: NaN  ← Found it!
```

### Package Structure

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

### Key Takeaways

1. **JAX is functional** - No hidden state, explicit randomness, pure functions
2. **Numerical stability matters** - Always check boundaries and edge cases
3. **Debug systematically** - Instrument, trace, fix root cause
4. **Design for modularity** - Protocols, builders, separation of concerns
5. **Test incrementally** - Verify each component before building on it

---

## Session 2: NNX → Linen Conversion (Mar 2026)

### Why We Switched from NNX to Linen

NNX is the newer, more Pythonic Flax API. But for high-throughput RL with small MLPs, Linen is better:

1. **Performance**: `nnx.jit` traverses the Python object graph on every call — ~3x overhead on small MLPs. Linen's params are plain pytrees with zero overhead inside jit.
2. **Ecosystem**: Brax, FastTD3, MuJoCo Playground all use Linen / functional params. We can lift patterns directly.
3. **Immutability = safety**: Impossible to accidentally mutate shared state across parallel envs or between actor/critic.
4. **Learning**: Forces you to understand pytrees, pure functions, and JIT boundaries — NNX hides this behind sugar.

---

### The Core Mental Model: Blueprint vs State

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

### @nn.compact — Layers Without __init__

In NNX, you build layers in `__init__` and use them in `__call__`. In Linen with `@nn.compact`, you do both in `__call__`:

```python
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

The algorithm class only holds blueprints (modules) and config — things that never change. **State flows through functions, not objects:**
```python
state = ppo.init(key)                      # creates initial TrainingState
state, metrics = ppo.update(state, batch, key)  # state in → new state out
```

---

### jax.value_and_grad with Closures

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

---

### Optax: The Explicit Optimizer Pattern

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

### Key Takeaways

1. **Linen modules are blueprints** — they describe computation, they don't hold state
2. **Params are just pytrees** — dicts of arrays, nothing magic
3. **TrainingState is the single source of truth** — all mutable state in one place
4. **Functions > methods** — if it doesn't need the module's structure, make it a function
5. **Closures are the standard pattern** — loss functions close over data, take params as the only argument
6. **The math doesn't change** — PPO loss, GAE, distributions are identical; only the plumbing changed

---

## Session 3: JIT Performance & First Training Curve (Mar 2026)

### The 600x Speedup: `jax.jit(env.step)`

**What happened:**
- PPO training on CartpoleBalance took ~184s per iteration (64 envs x 64 steps)
- GPU usage was 0%. Suspected JIT retracing in PPO update.

**Root cause:**
- MuJoCo Playground's `env.step` dispatches ~200 MJX physics primitives **eagerly** when called from a Python loop
- Each `env.step` call took **2.8s** for 64 envs — the physics kernel wasn't fused
- 64 steps x 2.8s = ~180s per iteration. The PPO update was fine (~0.05s). **99% of time was in env.step.**

**The fix — one line:**
```python
env_step = jax.jit(env.step)  # Fuse ~200 MJX primitives into one GPU kernel
# Then use env_step(env_state, action) instead of env.step(env_state, action)
```

**Result:** 184s/iter → 0.3s/iter. Training completed 1M steps in ~85s. Avg return: 995.4 (target >= 950).

**Why Playground doesn't auto-JIT:** Brax training scripts typically `jax.lax.scan` the entire training loop — the outer JIT covers env.step. Our Python collect loop calls env.step from Python, so each call is dispatched eagerly. The wrapper doesn't add its own JIT boundary.

**Lesson:** If you're calling a JAX-based env from a Python loop, **always** wrap with `jax.jit`. If GPU utilization is 0%, the computation is probably dispatching hundreds of tiny kernels instead of one fused one.

---

### Diagnostic Methodology: Isolate, Don't Guess

**Bad instinct:** "It's slow → must be retracing in `value_and_grad`." We created `ppo_jit.py` with JIT closures to fix "retracing." Didn't help — because the bottleneck was elsewhere.

**Good method:** Created `diagnose_speed.py` that tests each operation in isolation:

```
Test 1: env.step           → 2.822s/call  ← THE BOTTLENECK
Test 2: select_action      → 0.000s/call
Test 3: norm_update        → 0.029s/call
Test 4: full collect loop  → 183.23s (= 64 x 2.8s, confirms env.step)
Test 5: _minibatch_step    → 0.001s/call
```

**Lesson:** When something is slow in JAX, **time each operation in isolation** before optimizing. The bottleneck is rarely where you think it is.

---

### jax.lax.scan — Eliminating Python Dispatch Overhead

Three versions of the same PPO update, benchmarked head-to-head:

| Version | `update()` per call | Speedup vs eager |
|---|---:|---:|
| `ppo.py` — Python loops, no JIT | 2.712s | 1x |
| `ppo_jit.py` — JIT closures, Python loops | 0.089s | 30x |
| `ppo_scan.py` — `jax.lax.scan` (fully compiled) | 0.005s | 542x |

**Why scan matters:** For small MLPs (64x64 hidden), ~95% of each call was Python-XLA dispatch overhead. `jax.lax.scan` compiles the entire epoch loop into a single XLA program — no Python roundtrips.

**Rule of thumb:** If you're calling a JIT'd function >10x in a loop, `jax.lax.scan` it.

---

### GAE Truncation Handling

MuJoCo Playground signals episode timeout via `env_state.info["truncation"]`. At truncation, the env auto-resets, so the next obs is the RESET state — V(reset_obs) is wrong for bootstrapping.

**Our approach (matching Brax):** Zero out the entire TD error at truncation steps. This means:

| Scenario | `done` | `truncation` | Delta | Propagation |
|----------|--------|-------------|-------|-------------|
| True terminal (fell over) | 1 | 0 | r - V(s) | Stop |
| Timeout (hit episode_length) | 1 | 1 | **0** (zeroed) | Stop |
| Mid-episode | 0 | 0 | r + γV(s') - V(s) | Continue |

Loses one transition of signal per episode (~0.1%), but avoids biased bootstrap values. See Session 5 for the full debugging journey.

---

### `jax.block_until_ready()` — Required for Accurate Timing

JAX operations are **asynchronous** — `env.step()` returns immediately, queueing work on the GPU. Without `block_until_ready()`, you're timing the Python dispatch, not the actual computation.

---

### Key Takeaways

1. **Profile before optimizing** — the bottleneck is rarely where you think
2. **`jax.jit(env.step)` is mandatory** for MJX envs called from Python loops
3. **GPU at 0% = eager dispatch** — hundreds of tiny kernels instead of one fused one
4. **Separate warmup from timing** — first call includes JIT compilation
5. **`block_until_ready()`** — without it, JAX timing is meaningless
6. **Truncation != termination** — bootstrap through timeouts, zero out true terminals

---

## Session 4: CheetahRun, Performance, & Tooling (Mar 2026)

### `jax.lax.scan` vs Python Loops — 542x Speedup

Benchmarked three PPO implementations on the same update workload:

| Variant | Per-update time | Speedup |
|---------|----------------|---------|
| `ppo.py` (Python loops, no JIT) | 2.712s | 1x |
| `ppo_jit.py` (JIT closures, Python epoch loops) | 0.089s | 30x |
| `ppo_scan.py` (JIT + `jax.lax.scan` for epochs) | 0.005s | 542x |

**Why scan matters:** For small MLPs (64x64), 95% of each call was Python-XLA dispatch overhead. `jax.lax.scan` compiles the entire epoch loop into a single XLA program — no Python roundtrips. The JIT-only version still had Python loops for epochs/minibatches, causing repeated dispatch.

**Lesson:** If your model is small and your loop body is fast, Python loop overhead dominates. `scan` eliminates it entirely.

---

### Entropy Coefficient Tuning — When Exploration Hurts

**What happened:** CheetahRun with `entropy_coef=0.001` — entropy grew unboundedly (9→16+), returns plateaued at ~248.

**Root cause:** With unconstrained `log_std` (clipped to [-5, 2]) and a positive entropy bonus, the optimizer pushes `log_std` toward the upper bound to maximize entropy. The policy becomes too noisy to learn anything useful.

**The fix:** Set `entropy_coef=0.0` for CheetahRun. Returns jumped to 503 at 10M steps.

**Diagnostic signals:**
- Entropy growing monotonically = entropy bonus is dominating task reward
- High KL + high clip fraction = policy changing too aggressively
- PLoss near 0 at end = policy gradient exhausted (needs LR annealing)

**Lesson:** Entropy bonus is environment-dependent. Simple tasks (CartpoleBalance) benefit from exploration (`entropy_coef=0.01`). Complex continuous control (CheetahRun) often needs `entropy_coef=0.0` to avoid the entropy-maximization trap.

---

### LR Annealing for Continuous Control

**Problem:** With constant LR (3e-4), policy loss converges to ~0 and clip fraction stays high (0.27-0.35) late in training. The policy keeps making large updates even when it should be fine-tuning.

**Fix:** Linear LR schedule annealing to 0 over total gradient steps:
```python
total_gradient_steps = num_iterations * num_updates_per_batch * num_epochs * num_minibatches
lr_schedule = optax.linear_schedule(lr, 0.0, total_gradient_steps)
optimizer = optax.adam(lr_schedule)  # optionally chain with clip_by_global_norm
```

**Lesson:** LR annealing lets the policy explore broadly early and fine-tune late. Standard practice for PPO on continuous control — Brax does this too.

---

### Optimizer Decoupling — Keep Algorithms Pure

PPO shouldn't own optimizer construction. Optimizers (LR schedule, grad clipping, future exotic optimizers like Muon) are external concerns.

**Before:** PPO internally created optimizers from config fields (`actor_lr`, `anneal_lr`, etc.)
**After:** PPO takes `actor_optimizer` and `critic_optimizer` as constructor args. Train script builds them.

**Lesson:** Same principle as dependency injection — algorithms define *what* they optimize, not *how*. Makes swapping optimizers trivial (Muon, Shampoo, etc.) without touching algorithm code.

---

### Orbax Checkpointing

- `ocp.StandardCheckpointer()` saves/restores arbitrary pytrees (works with Linen or NNX)
- **Must call `checkpointer.wait_until_finished()`** after save — otherwise process exit kills the async write thread ("cannot schedule new futures after shutdown")
- Save `meta.json` alongside checkpoint for config reconstruction (env name, hidden dims, obs/action dims)
- Checkpoint + metrics CSV + meta.json in timestamped directories: `checkpoints/{timestamp}_{env}_seed{seed}/`

---

### Video Recording — Two-Phase Approach

MuJoCo Playground's `env.render()` is CPU-side and can't be JIT'd. Solution:

1. **Phase 1 (GPU):** `jax.lax.scan` the rollout — fast, collects all states
2. **Phase 2 (CPU):** Render frames from saved states — slow but unavoidable

**Lesson:** Separate compute from rendering. The rollout itself is fast (~0.3s for 1000 steps including JIT). Rendering is the bottleneck (~50ms/frame).

---

### CheetahRun Hyperparameter Journey

| Config | Return | Notes |
|--------|--------|-------|
| entropy_coef=0.001, 10M steps | ~248 | Entropy grew to 16+, policy too noisy |
| entropy_coef=0.0, 10M steps | ~503 | Entropy controlled, policy learning |
| entropy_coef=0.0, 20M steps | ~666 | More steps helped, approaching target |
| + LR annealing, 20M steps | ~615 | Annealed too aggressively, policy froze late |
| Preset HPs (2048 envs, lr=1e-3, 16 epochs, reward_scaling=10) | **826** | Passed target at 20M steps |

Target was >=700 at 60M steps (MuJoCo Playground paper). Hit **826 at 20M** with proper HPs. Key changes: 2048 envs, shorter rollouts (30 steps), higher LR (1e-3), more epochs (16), reward scaling (10x).

---

## Session 5: HumanoidRun Debugging & Pipeline Robustness (Mar 2026)

### State-Independent vs State-Dependent log_std

**Problem:** HumanoidRun (21-dim actions) NaN'd repeatedly with state-dependent std (`nn.Dense` mapping features → log_std).

**Root cause:** Dense layer produces unpredictable initial log_std values. Outlier observations spike log_std → entropy explodes → NaN. With 21 action dims, the entropy is already ~30 nats from dimensionality alone (`0.5 * 21 * (1 + log(2pi)) ≈ 29.8`).

**Fix:** State-independent log_std (`nn.Param`), matching Brax PPO:
```python
# State-independent: single learned vector, same for all observations
log_std = self.param('log_std', nn.initializers.constant(jnp.log(init_noise_std)), (action_dim,))
log_std = jnp.broadcast_to(log_std, mean.shape)

# MUST clip in both modes — without this, entropy bonus pushes log_std → infinity
log_std = jnp.clip(log_std, log_std_min, log_std_max)
```

**Lesson:** For PPO, state-independent std is more stable. Save state-dependent std for SAC where maximum entropy is the objective. Always clip log_std regardless of mode.

---

### Minibatch Count, Not Size, Is the Right Parameterization

**Problem:** Hardcoded `minibatch_size = min(2048, samples_per_iter)`. When we increased `num_steps` from 30 to 480 (16x more data), `samples_per_iter` went from 61k to 983k. But minibatch_size stayed at 2048, so `num_minibatches` went from 30 to 480. With 16 epochs, that's 7,680 gradient steps per iteration — Brax uses 512.

**The coupling:** `grad_steps = num_minibatches x num_epochs`. Fixing `minibatch_size` means `num_minibatches` scales with data volume. Fixing `num_minibatches` keeps gradient steps constant regardless of data volume.

**Fix:** Parameterize by `num_minibatches` (default 32, matching Brax). Derive `minibatch_size = samples_per_iter // num_minibatches`.

**Lesson:** When comparing against a reference implementation, match the *structure* (what's fixed vs derived), not just the numbers. Brax fixes `num_minibatches=32` and `batch_size=1024`; we were fixing `minibatch_size=2048` which is a completely different parameterization.

---

### Batch-Level vs Per-Minibatch Advantage Normalization

Brax normalizes advantages over the entire batch (~983k samples) before splitting into minibatches. We were normalizing per-minibatch (~2k samples), giving noisy mean/std estimates.

With 30k+ samples per minibatch (after the fix), the difference is small. But for smaller setups or early in training, batch-level normalization is strictly better.

---

### Reshape Crash on Non-Divisible Batches

`obs[perm].reshape(num_minibatches, minibatch_size, -1)` silently assumes `N == num_minibatches * minibatch_size`. If not (e.g., 5000 samples with minibatch_size=1024 → 4 batches of 1024 = 4096 != 5000), it crashes.

**Fix:** Truncate the permutation: `perm = perm[:num_minibatches * minibatch_size]`.

**Lesson:** Always handle the case where data doesn't divide evenly. It works by coincidence for tuned presets but breaks for arbitrary configs.

---

### HumanoidRun Diagnostic Signals

Added `log_std_mean`, `log_std_min`, `log_std_max` to PPO metrics. Key observations:

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| Clip fraction | 0.05-0.20 | 0.50-0.80 (policy overshooting) |
| Entropy | Slowly decreasing | Growing monotonically (entropy bonus dominating) |
| VLoss | Stable, tracks returns | Collapsing to 0.00 (overfitting to batch) |
| KL | < 0.1 | Spikes > 1.0 (imminent NaN) |
| log_std | Slowly decreasing from ~0 | Growing or stuck at clip bounds |

---

### Interleaved Collect→Update (num_updates_per_batch)

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
- Same total gradient steps (16 x 1 x 32 = 512) but each uses on-policy data
- `num_updates_per_batch=1, num_epochs=4` recovers our old behavior

**Key numbers (HumanoidRun):**

| Config | Grad steps/iter | Data freshness | Result |
|--------|----------------|----------------|--------|
| num_steps=480, epochs=4, updates=1 | 128 | 480 steps stale | Returns 7.6 (slow) |
| num_steps=20, epochs=4, updates=1 | 128 | Fresh but overtrained | Collapsed (entropy → -1.25) |
| num_steps=20, epochs=1, updates=16 | 512 | Fresh each cycle | Brax reference config |

**Lesson:** The right granularity is many short collect→update cycles, not fewer long ones. `num_updates_per_batch` generalizes our pipeline — setting it to 1 recovers the old behavior.

---

### VLoss Oscillation from Synchronized Episode Resets

**Problem:** VLoss alternated between ~0.35 and ~61.5 every other iteration. Even iterations (episode boundaries) had huge VLoss; odd iterations had near-zero VLoss.

**Root cause:** `wrap_for_brax_training` resets all envs at the same timestep (episode_length=1000). With 4096 envs all resetting simultaneously, the value function sees completely different return distributions on boundary iterations vs mid-episode iterations. It overfits to mid-episode predictions, then gets blindsided at boundaries.

**How Brax avoids this:** With `num_updates_per_batch=16` and `num_steps=20`, updates happen every 20 steps. After the first episode completes, the 16 sequential collection phases within an iteration mean envs are at different points in their episodes across update cycles.

**Lesson:** Synchronized resets + long rollouts = VLoss instability. Interleaved short rollouts naturally desynchronize episode phases.

---

### Truncation Handling in Auto-Reset Environments (Matching Brax)

**Problem:** HumanoidRun returns stuck at ~7.6 at 60M steps (reference: 50-200). Multiple HP tuning attempts couldn't close the gap. Turned out to be a GAE correctness bug, not a hyperparameter issue.

**Root cause:** `wrap_for_brax_training` auto-resets envs on episode timeout. After reset, `env_state.obs` is the **reset observation** (humanoid standing still), not the terminal observation (humanoid running). Our GAE used `values[t+1] = V(reset_obs)` as the bootstrap for truncated transitions. Since V(reset) << V(running), this created a systematic negative bias.

**Failed attempts:**
1. **Reward adjustment** (`gamma * V(s_t) * truncation`): massive synchronized reward spikes → NaN at iter 9
2. **Bootstrap correction + two-flag propagation** (`next_values = where(trunc, values, next_values)`): The V(s_t) bootstrap approximation grows biased as V improves, and two separate done flags for bootstrap vs propagation is error-prone

**Final fix (matching `brax.training.agents.ppo.losses.compute_gae`):**
```python
truncation_mask = 1.0 - truncations  # 0 at truncation, 1 otherwise
termination = dones * (1.0 - truncations)  # 1 only at true terminals

# TD errors: zero out entirely at truncation steps
deltas = rewards + gamma * (1 - termination) * next_values - values
deltas = deltas * truncation_mask  # zero at truncation

# Propagation: stop at both truncation and true terminal
gae = deltas[t] + gamma * gae_lambda * (1 - termination[t]) * truncation_mask[t] * gae
```

**What this achieves:**
- **Advantage at truncation = 0**: no gradient signal for policy at that step
- **Value target = V(s_t)**: return = A + V = 0 + V(s_t), so no critic gradient either
- **Propagation stops**: next episode's advantages don't leak in

You lose one transition of training signal per episode (0.1% with 1000-step episodes), but the value function stays unbiased. This is strictly better than trying to approximate V(terminal_obs) with V(s_t).

**Lesson:** When auto-reset environments corrupt the post-terminal observation, the cleanest fix is to zero out the entire TD error at that step rather than trying to correct the bootstrap value. Matches Brax's approach and avoids subtle bugs from splitting done flags.

---

## Session 6: Brax Parity Audit & Tanh Squashing (Mar 2026)

### Matching a Reference Implementation Requires Structural Parity, Not Just HPs

**Problem:** Copied Brax's hyperparameters (num_envs, num_steps, epochs, etc.) but training still failed or underperformed. Multiple runs NaN'd or plateaued at returns ~5 vs Brax's 50-200.

**Root cause:** Hyperparameters only work correctly within the structural context they were tuned for. Key structural differences that made the same HPs behave differently:

| What we thought was equivalent | What was actually different |
|-------------------------------|---------------------------|
| `num_updates_per_batch=16, num_epochs=1` | Brax: 1 collection → 16 epochs. Ours: 16 fresh collections → 1 epoch each |
| `num_steps=30` with 2048 envs (61k samples) | Brax collects `batch_size × num_minibatches × unroll = 1024 × 32 × 30 = 983k` samples |
| `entropy_coef=1e-2` with raw Gaussian entropy | Brax uses tanh-corrected entropy (~14 nats). Raw Gaussian entropy was ~30 nats — 2x the bonus |
| `squash=False` + `jnp.clip(action, -1, 1)` | Brax uses `tanh_normal` with proper Jacobian correction in log_prob and entropy |

**Lesson:** When matching a reference implementation, audit the *structure* (data flow, loss computation, distribution math), not just the *numbers* (learning rate, batch size, epochs). The same HP value can have completely different effects in a different structural context.

---

### Tanh Squashing Requires Matched Entropy Computation

**Problem:** Enabled `squash=True` to match Brax's `tanh_normal`. Immediate blowup — KL 296k, entropy exploding from 33 to 71, clip fraction 1.0.

**Root cause:** `entropy_gaussian(log_std)` computed raw Gaussian entropy: `0.5 * (log(2π) + 1 + 2*log_std)` summed over 21 dims = ~30 nats at `log_std=0`. But the actual entropy of the tanh-squashed distribution is ~14 nats (tanh compresses the distribution). The `entropy_coef=1e-2` bonus on the inflated value pushed `log_std` to the clip maximum.

**Fix — single-sample entropy estimate (matching Brax):**
```python
def entropy_gaussian(log_std, mean=None, key=None, squash=False):
    if squash:
        std = jnp.exp(log_std)
        dist = distrax.Normal(loc=mean, scale=std)
        raw_actions = dist.sample(seed=key)
        log_prob = dist.log_prob(raw_actions)
        # Tanh Jacobian correction
        log_prob -= jnp.log(1 - jnp.tanh(raw_actions) ** 2 + 1e-6)
        return -log_prob.sum(axis=-1)  # entropy = -E[log p(x)]
    else:
        # Closed-form Gaussian entropy
        return (0.5 * (jnp.log(2 * jnp.pi) + 1 + 2 * log_std)).sum(axis=-1)
```

**Lesson:** Entropy and log_prob must always be computed in the same space. If actions are tanh-squashed, entropy must include the Jacobian correction `log(1 - tanh(x)²)`. Using pre-transform entropy with post-transform actions is a guaranteed blowup.

---

### Batch Volume Controls How Many SGD Epochs Are Safe

**Problem:** With `num_steps=30, num_envs=2048` (61k samples) and `num_epochs=16`, entropy exploded and clip fraction hit 1.0.

**Root cause:** 16 epochs of SGD over 61k samples = massive overtraining. The policy changed so much that importance sampling ratios blew up. Brax uses 16 epochs too, but over 983k samples — 16x more data per epoch.

**Fix:** `num_steps=480` → `2048 × 480 = 983k` samples, matching Brax's collection volume. `minibatch_size = 983k / 32 = 30,720` also matches.

**Rule of thumb:** More SGD epochs require proportionally more data. If you increase epochs, increase batch size. If you can't increase batch size, reduce epochs.

---

### Remaining Performance Gap

Run 15 (all parity fixes) achieved stable training but only 5.3 avg return at 60M steps (target: 50-200). Remaining structural differences from Brax:

1. **Per-minibatch GAE recomputation** — Brax re-evaluates V(s) from current params inside each minibatch loss. Our GAE uses stored values from collection time, which go stale over 16 epochs.
2. **State-dependent std** — Brax's `tanh_normal` predicts both mean and scale from the network. Ours uses a state-independent learned `log_std` param.

---

## Session 7: PPO Validation & Algorithm Limits (Mar 2026)

### Know When PPO Is the Wrong Tool

**Problem:** HumanoidRun returns stuck at ~7-10 after 60M steps despite full Brax structural parity. Spent multiple sessions tuning HPs and fixing structural differences.

**Resolution:** Wrote `bench_brax_ppo.py` to run Brax's own PPO with the exact MuJoCo Playground config. Result: ~8-10 return — identical to ours. Checked the Playground paper (Figure 7): PPO is flat at ~10-20 through 100M steps. SAC reaches ~200 by 5M steps.

**Literature confirms this is expected:**
- dm_control paper (Tassa 2018): A3C scored 1.0/1000 on humanoid_run
- Brax paper: humanoid "does not find successful policies" with PPO even at 500M steps
- D4PG (off-policy, distributed) only reached ~464/1000 — the best reported score
- FastTD3 (2025) is current SOTA for humanoid locomotion

**Lesson:** Before spending days tuning an algorithm, check whether the algorithm is known to work on the task at all. A 5-minute literature search or a reference implementation sanity check can save days of HP tuning. PPO on humanoid_run is a known failure case — the signal was there in the original dm_control paper from 2018.

**When to suspect algorithm limits vs implementation bugs:**
- Your metrics look healthy (stable KL, reasonable clip fraction, no NaN) but returns are flat → likely algorithm limits
- Reference implementation gets the same score → definitely algorithm limits
- Metrics are unhealthy (KL spikes, entropy explosion, NaN) → implementation bug

---

### PPO Validation Summary

| Environment | Our Result | Target | Status |
|-------------|-----------|--------|--------|
| CartpoleBalance | >=995 @ 1M | >=950 @ 10M | PASS |
| CheetahRun | 826 @ 20M | >=700 @ 60M | PASS |
| HumanoidRun | ~7-10 @ 60M | matches Brax reference (~8-10) | PASS (algorithm limit, not bug) |

PPO implementation is validated. HumanoidRun requires off-policy methods (SAC, TD3, FastTD3).

---

## Session 8: Checkpoint Design & Inference Pipeline (Mar 2026)

### Orbax Restore Requires Exact Pytree Structure Match

**Problem:** `record_video.py` crashed restoring a checkpoint saved with `optax.chain(clip_by_global_norm, adam)` while the restore target was initialized with bare `optax.adam`. Error: `EmptyState vs dict`.

**Why it happens:** Orbax serializes the pytree structure of the optimizer state (not just the arrays). `optax.chain(A, B)` produces a tuple `(StateA, StateB)`. Bare `optax.adam` produces just `ScaleByAdamState`. These are structurally incompatible — orbax refuses to restore.

**Things that don't help:**
- `strict=False` on `StandardCheckpointer.restore()` — only for shape mismatches (array padding/truncation), not tree structure
- `partial_restore=True` — only available on the lower-level `PyTreeRestore` API, not `StandardCheckpointer`

**Targetless restore (emergency escape hatch):**
```python
# No target = restores as raw nested dict (UNSAFE warning, but works)
raw = ocp.StandardCheckpointer().restore(os.path.abspath(ckpt_dir))
actor_params = raw['training_state']['actor_params']
```
Use this when you need to extract specific leaves and don't have the exact pytree structure.

**Correct design:**
```python
# In train.py — save inference artifact separately from training artifact
np.save(os.path.join(ckpt_dir, "actor_params.npy"), {
    "actor_params": jax.device_get(training_state.actor_params),
    "norm_mean": jax.device_get(norm_state.mean),
    "norm_mean_of_squares": jax.device_get(norm_state.mean_of_squares),
    "norm_count": int(jax.device_get(norm_state.count)),
}, allow_pickle=True)

# In record_video.py — load without orbax
saved = np.load(params_path, allow_pickle=True).item()
training_state = training_state.replace(actor_params=saved["actor_params"])
```

**Lesson:** Separate inference artifacts from training artifacts. The orbax checkpoint is for resuming training (must match optimizer structure). A plain numpy file is for inference (no structure to match, immune to optimizer changes).

---

### Checkpoint Should Be Fully Self-Describing

**Problem:** `meta.json` stored only the network arch (hidden dims, activation, squash). `record_video.py` couldn't reconstruct the optimizer chain because `max_grad_norm` wasn't saved.

**Better approach:** Store the full training config as `dataclasses.asdict(cfg)` in meta.json. All hyperparameters, network config, and training setup in one place — sufficient to reproduce or resume any run without digging through code history.

```python
meta = {
    "obs_dim": obs_dim,
    "action_dim": action_dim,
    "train_config": dataclasses.asdict(cfg),  # full TrainConfig + nested PPOConfig
}
```

**Lesson:** A checkpoint that requires external knowledge to load is incomplete. The meta.json should answer: "given only this directory, can I fully reconstruct what was trained and how?"

---

### Synchronized Episode Resets Create Periodic VLoss Spikes

**Pattern observed in WalkerWalk training (episode_length=1000, num_steps=30, num_envs=2048):**
- Every ~33 iterations (1000/30 ≈ 33.3), all 2048 envs reset simultaneously
- VLoss spikes from ~1 to ~14,000, entropy collapses to -6.5
- Self-corrects within ~5 iterations

**Why entropy goes negative:** With tanh-squash + single-sample entropy estimate (`entropy = -log_prob`): if a large gradient update briefly collapses std, the sample falls near the mode where log_prob is very high (positive). Negative entropy is the artifact, not a sign of real divergence.

**Why it self-corrects:** The spike is a distribution shift (reset obs vs mid-episode obs), not a runaway update. Once the critic refits to the mixed distribution, everything stabilizes.

**Didn't prevent convergence** for WalkerWalk (833 avg return). But HumanoidRun NaN'd at these boundaries — higher-dim action space amplified the instability.

**Mitigation:** Brax's `num_updates_per_batch=16, num_steps=20` interleaves collections so episode phases are desynchronized across update cycles. After the first episode completes, subsequent update cycles have envs at different episode phases → no synchronized resets.

---

## Session 9: SAC Implementation & Off-Policy Pitfalls (Mar 2026)

### Online Obs Normalization Is Incompatible with Off-Policy Replay

**Problem:** SAC on WalkerWalk diverged immediately — Q values to -15,000, alpha to 19+, returns stuck at 32.

**Root cause:** At env reset, all 128 envs have near-identical observations (standing pose). The running variance estimate for some obs dimensions is ≈ 0. Normalizing `next_obs` (which differs slightly after one random step) divides by `sqrt(0) + 1e-8`, producing values of ±2 billion stored in the replay buffer.

**Why PPO is immune:** PPO is on-policy — obs are normalized and consumed in the same iteration. The norm stats match the data. SAC's replay buffer stores normalized obs that become stale as the running statistics evolve. Old transitions were normalized with statistics from 100k steps ago.

**The diagnostic trail:**
1. Q values diverging monotonically → something wrong with TD targets
2. TD target = -47 million on first gradient step → next_obs is corrupted
3. `next_obs` range in buffer: ±2 billion → normalization is exploding
4. `norm_state` variance ≈ 0 for some dims after identical reset obs → division by near-zero

**Fix:** Removed obs normalization entirely for SAC. The Q-network's LayerNorm provides implicit input normalization. Initialized norm_state as identity (mean=0, var=1) for checkpoint compatibility with record_video.py.

**Brax reference confirms:** Brax SAC does not normalize observations.

**Lesson:** On-policy normalization (update stats → normalize → use immediately) is safe. Off-policy normalization (update stats → normalize → store → sample much later) is fundamentally broken because the stored normalized values become stale as statistics drift. For off-policy methods: either normalize at sample time with current stats (complex), or skip normalization and let LayerNorm handle it (simple, recommended).

---

### SAC Validation Results

| Environment | SAC Result | PPO Result | SAC Steps | PPO Steps | Speedup |
|-------------|-----------|------------|-----------|-----------|---------|
| WalkerWalk | 975 avg, 995 max | 833 avg, 992 max | 5M | 60M | 12x more sample-efficient, higher score |
| HumanoidRun | ~23+ at 1M (in progress) | ~10 at 60M | 5M target | 60M | Already 2x PPO's best at 20% of steps |

SAC matches the MuJoCo Playground paper's Figure 7 trajectory. WalkerWalk converged to >950 by 800k steps.

---

### Python `if` vs `jax.lax.cond` Inside Traced Functions

**Problem:** `sac.select_action` used `if deterministic:` to branch between `tanh(mean)` and sampling. Worked fine during training (called from Python loop), crashed inside `jax.lax.scan` for video recording:
```
TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[]
```

**Why it worked during training:** `@jax.jit` traces `select_action` once per unique signature. When called from a Python loop with `deterministic=False` (a concrete Python bool), JIT traces the `False` branch and compiles it. The `if` is resolved at trace time, not runtime.

**Why it broke in scan:** `jax.lax.scan` traces its body function with abstract values. All arguments become tracers — including `deterministic`. A Python `if` on a tracer is illegal because JAX can't evaluate it at trace time.

**Fix:**
```python
# Before (Python control flow — fails inside scan)
if deterministic:
    return jnp.tanh(mean)
action, _ = sample_gaussian(mean, log_std, key, squash=True)
return action

# After (JAX control flow — works everywhere)
action, _ = sample_gaussian(mean, log_std, key, squash=True)
return jax.lax.cond(deterministic, lambda: jnp.tanh(mean), lambda: action)
```

**Rule of thumb:** If a function might ever be called inside `scan`, `vmap`, or another traced context, use `jax.lax.cond` for branching, not Python `if`. Functions that are ONLY called from Python loops can use `if` safely — but `lax.cond` is always safe.

---

## Future Topics to Explore

- [x] `jax.jit` compilation and when to use it
- [x] `jax.lax.scan` vs Python loops (efficiency) — 542x speedup on PPO update
- [x] Checkpointing with Orbax
- [ ] Performance profiling with JAX
- [ ] Pytrees and how Linen models work as pytrees
- [ ] Device placement (CPU vs GPU)
- [ ] Layer normalization in JAX/Linen (needed for FastTD3)
- [ ] Deterministic eval rollouts (separate from training returns)

---

*"The best way to learn is to break things, then fix them systematically."*
