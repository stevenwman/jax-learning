# Shared Buffer vs Per-Env Buffer: Speed & Memory Analysis

## Memory Comparison

### Current Design (Per-Env Buffer)
```
Shape: (num_envs=2048, capacity, dim)
Memory for capacity=500:  ~450 MB
Memory for capacity=5000: ~4.5 GB
Memory for capacity=50k:  ~45 GB ❌ (impossible)
```

### Shared Buffer Design
```
Shape: (capacity, dim)
Memory for capacity=500k: ~220 MB ✓
Memory for capacity=1M:   ~440 MB ✓
Memory for capacity=5M:   ~2.2 GB ✓ (still reasonable!)
```

**Memory savings: ~2048× reduction!**

## Speed Analysis

### Current Design (Per-Env)

**Adding:**
- Uses `vmap` over envs → parallel writes per env
- Each env writes to its own buffer slot
- **Speed:** Very fast (parallelized per env)

**Sampling:**
- Step 1: Random env selection (1 random op)
- Step 2: Random time selection per env (1 random op with bounds)
- Step 3: Gather with 2D indexing `arr[env_idx, time_idx]`
- **Speed:** Fast, but requires 2 random operations + gather

### Shared Buffer Design

**Adding:**
- Flatten all env transitions → single batch
- Use `scatter` (`.at[].set()`) to update buffer
- JAX can parallelize scatter operations
- **Speed:** Should be similar or slightly slower (scatter vs vmap)

**Sampling:**
- Single random operation: `randint(0, count)`
- Simple 1D indexing: `arr[idx]`
- **Speed:** FASTER (1 random op vs 2, simpler indexing)

## Performance Concerns & Solutions

### Concern 1: "Scatter is slower than vmap"
**Reality:** 
- Scatter operations in JAX are well-optimized
- For 2048 transitions, scatter should be fast
- The memory savings (2048×) often outweigh small speed differences

### Concern 2: "Sequential writes vs parallel"
**Reality:**
- Current: 2048 parallel writes (one per env)
- Shared: 2048 scatter operations (also parallelized by JAX)
- Both are parallelized, just different patterns

### Concern 3: "Wraparound complexity"
**Solution:** 
- Use modulo arithmetic: `indices = (arange + ptr) % capacity`
- JAX handles this efficiently
- Scatter with modulo indices is well-optimized

## Benchmarking Recommendation

The speed difference is likely **negligible** in practice:
- Adding: ~same speed (both parallelized)
- Sampling: **Shared is faster** (1 random op vs 2)
- Memory: **Shared is 2048× better**

## When to Use Each

### Use Per-Env Buffer if:
- You need to track which env produced each transition
- You want to sample proportionally from each env
- Memory is not a concern
- You have very few envs (< 100)

### Use Shared Buffer if:
- Memory is a constraint (most cases!)
- You want larger buffer capacity
- You don't care which env produced transitions (off-policy RL)
- You have many envs (100+)

## Implementation Notes

The shared buffer implementation uses:
- `jnp.arange(num_new) + ptr) % capacity` for wraparound indices
- `.at[indices].set(values)` for scatter updates
- Single random operation for sampling

This is the standard approach used in most RL libraries (Stable-Baselines3, etc.)

## Recommendation

**Use shared buffer** - The memory savings are massive and speed should be comparable or better. The only downside is losing per-env tracking, which you don't need for off-policy RL anyway.

