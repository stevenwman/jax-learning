# Outdated Lessons

Lessons that were valid at the time but no longer apply to the current codebase. Kept for historical reference.

---

## NNX 0.11+ API Changes (from Session 1)

*Outdated because: We migrated from Flax NNX to Flax Linen + raw optax. NNX optimizers are no longer used.*

```python
# New API requires wrt parameter
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# Update requires both model and grads
optimizer.update(model, grads)
```

---

## Vectorization with `jax.vmap` for Action Selection (from Session 1)

*Outdated because: We don't vmap `select_action`. Actions are selected per-step in a Python collection loop, and the batched env already provides vectorized obs across all envs.*

```python
# Sample actions for all timesteps
batched_action_select = jax.vmap(ppo.select_action, in_axes=(0, 0, None))
actions, log_probs, values = batched_action_select(obs, keys, False)
```

**Note:** `jax.vmap` itself is still a valid JAX pattern — it's just not how we do action selection in this codebase. Our `select_action` already handles batched obs via Linen's built-in batch handling.

---

## Brax Wrapper Episode Tracking — Frozen Returns (from Session 4)

*Outdated because: With `num_updates_per_batch=16` and interleaved collect→update cycles, episode resets naturally stagger after the first episode. The "frozen for ~16 iterations" pattern was specific to the old single-rollout pipeline with small env counts.*

**Original observation:** Episode returns appear "frozen" for ~16 iterations, then jump. This is NOT a bug.

**Why (old explanation):** `wrap_for_brax_training` synchronizes episode resets. With 256 envs and episode_length=1000, all envs complete at the same time (every `1000 / 64 ≈ 16` iterations).

**Current behavior:** With 4096 envs and 16 update cycles of 20 steps each (320 steps/iter), episodes complete within ~3 iterations. The staggering from interleaved updates means returns update more smoothly.
