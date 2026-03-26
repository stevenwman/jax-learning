# JAX Performance Lessons

---

## Python Collect Loops Kill JAX Throughput — Use lax.scan (2026-03-24)

**What happened:** PPO training 5x slower than Brax (47 min vs 10 min for 50M steps at 512 envs).

**Root cause:** Python-level rollout loop forces GPU sync at every step.

```
Pure env.step (512 envs, 20 steps): 0.111s → 92,000 sps
Python collect loop (same):         9.729s → 1,053 sps
Overhead ratio: 87x
```

**Lesson:** Every Python-level operation between JIT'd calls is a sync point. Put the entire loop body inside `lax.scan` or `lax.while_loop`.

---

## `jax.lax.scan` vs Python Loops — 542x Speedup

| Variant | Per-update time | Speedup |
|---------|----------------|---------|
| Python loops, no JIT | 2.712s | 1x |
| JIT closures, Python epoch loops | 0.089s | 30x |
| JIT + `jax.lax.scan` for epochs | 0.005s | 542x |

For small MLPs (64x64), 95% of each call is Python-XLA dispatch overhead. `scan` compiles the entire loop into a single XLA program.

---

## JIT Closure Recompilation in Eval Loops (2026-03-25)

**What happened:** Training projected at 2.5 hours instead of 15 min. Each eval recompiled from scratch (~15s).

**Root cause:** `_make_eval_action()` created a new closure capturing `frozen_norm` each call. New function object = JIT cache miss.

**Fix:** Define eval function once, pass `norm_state` as an argument. Same function object = cache hit.

**Lesson:** Never create `@jax.jit`-decorated functions inside loops. If a value changes between calls, pass it as an argument.

---

## lax.scan Episode Return Tracking Requires Persistent Carry (2026-03-25)

**Problem:** Training return showed `0.0` for entire runs.

**Root cause:** `ep_return` reset to zeros at each `_collect()` call. Episodes span ~50 collect calls.

**Fix:** `running_ep_return` persists across collect calls as part of the outer training loop state.

**Lesson:** Any state that spans episode boundaries must live OUTSIDE the scan init.

---

## Python `if` vs `jax.lax.cond` Inside Traced Functions

**Problem:** `if deterministic:` in `select_action` worked in Python loops, crashed inside `lax.scan`:
```
TracerBoolConversionError: Attempted boolean conversion of traced array
```

**Why:** `scan` traces with abstract values — all arguments become tracers. Python `if` on a tracer is illegal.

**Fix:**
```python
# Before: if deterministic: return jnp.tanh(mean)
# After:
return jax.lax.cond(deterministic, lambda: jnp.tanh(mean), lambda: action)
```

**Rule:** If a function might ever be called inside `scan`, `vmap`, or another traced context, use `jax.lax.cond`.

---

## JIT the Hot Path — Non-JIT'd JAX Can Be Slower Than Numpy

**Problem:** JAX replay buffer was **5x slower** than numpy (1.85ms vs 0.34ms at batch=512).

**Root cause:** `jax_array[random_indices]` without JIT dispatches a Python-level gather op.

| Batch | Numpy | JAX (no JIT) | JAX (JIT'd) |
|-------|-------|-------------|-------------|
| 512 | 0.33ms | 1.85ms | 0.12ms (2.7x faster) |
| 32,768 | 1.74ms | — | 0.13ms (13.3x faster) |

**Lesson:** "Put it in a jax.Array" is not enough — JIT the operations on it too.

---

## jax.lax.scan Carry Cost — Large Buffer Arrays Kill Throughput

**Problem:** Scanning the inner gradient loop with 4M-entry buffer arrays in carry was 30% slower than Python loop.

- 100K buffer: scan 1.17x faster (carry is cheap)
- 4M buffer: scan 0.70x slower (carry overhead dominates)

**Lesson:** `jax.lax.scan` isn't free — carry size matters. Don't put the buffer in carry. Accept Python loop when carry would be large.

---

## A Faster Component Doesn't Mean Faster Training

JAX buffer showed 4.8x faster sampling, but end-to-end throughput only improved 1.5%.

**Root cause:** Gradient steps dominate wall-clock. Buffer sampling is <10% of total step time.

**Lesson:** Profile the full pipeline before optimizing a component. A 10x speedup on 5% of runtime = 0.5% end-to-end.
