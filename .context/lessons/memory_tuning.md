# GPU Memory Tuning Strategy

**Decision: Removed hardcoded `XLA_CLIENT_MEM_FRACTION` from all training scripts (2026-04-15).** Users now tune via shell environment when needed.

## Why?

Different training configs have fundamentally different memory requirements:
- Larger batch sizes, bigger networks, more envs → more JAX heap needed
- Hardcoding a single value (0.55, 0.7) either leaves memory on the table or constrains future experiments

The fraction is a **territory knob**, not a **fragmentation knob**. It controls how much VRAM JAX pre-claims, period. Changing it doesn't improve fragmentation—only `XLA_PYTHON_CLIENT_PREALLOCATE=true` does that.

## The Mechanism

### Preallocate (does the real work)

`XLA_PYTHON_CLIENT_PREALLOCATE=true` (default in JAX):
- Grabs a single contiguous slab of VRAM at startup
- Internal fragmentation handled by BFC allocator (Best-Fit with Coalescing)
- Prevents external fragmentation (the driver-heap kind you can't fix)
- **Cost:** All-or-nothing with other GPU tenants (Warp, other processes). If they need VRAM, JAX's slab-grab fails or evicts them.

### MEM_FRACTION (territory only)

`XLA_CLIENT_MEM_FRACTION=0.7` means JAX pre-grabs 70% of total GPU VRAM.
- Only matters if `PREALLOCATE=true`
- Higher fraction = more JAX heap, less room for other tenants
- Smaller fraction = less JAX heap, more room for others, but JAX hits OOM sooner
- Does **not** improve fragmentation—slab fragmentation is handled by BFC coalescing

## When to Tune

Hit an OOM during training? The signal depends on what runs out:

### JAX OOM (policy loss, env step):
```
RuntimeError: ... no memory available to allocate ...
```
→ JAX's slab is too small relative to your config.

**Action:** Increase MEM_FRACTION. Default to 0.7; try 0.75 or 0.8 if you have room.

```bash
XLA_CLIENT_MEM_FRACTION=0.75 uv run python scripts/train_fast_sac.py --env Go2WarpJoystickFlat
```

### Warp OOM (environment physics, graph capture):
```
CUDA out of memory. ... try setting XLA_PYTHON_CLIENT_PREALLOCATE=false ...
```
→ Warp doesn't have room to allocate its graph capture buffer.

**Action:** Either (a) decrease JAX's fraction, or (b) disable preallocation.

```bash
# Option A: reduce JAX's slab
XLA_CLIENT_MEM_FRACTION=0.55 uv run python scripts/train_fast_sac.py --env Go2WarpJoystickFlat

# Option B: let JAX allocate on-demand (loses fragmentation protection)
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/train_fast_sac.py --env Go2WarpJoystickFlat
```

### `record_video.py` OOM:
Uses `XLA_PYTHON_CLIENT_PREALLOCATE=false` by default to coexist with concurrent training on the same GPU. This trades fragmentation protection for sharing—acceptable for a one-off video record.

## Workflow

1. **First run:** Don't set MEM_FRACTION. Let JAX use the driver's default (≈90% on most systems).
2. **Hit JAX OOM?** Try `0.75`, then `0.8` (if Warp allows).
3. **Hit Warp OOM?** Try `0.55` or `0.6`.
4. **Found a sweet spot?** Note it in your shell history or a `train_go2.sh` wrapper.

No need to hardcode—configs evolve, and the right fraction moves with them.

## Edge Case: Contested GPU (multiple users / concurrent training)

If you're running training + recording simultaneously:

```bash
# Terminal 1: training, grabs 0.7 of VRAM
XLA_CLIENT_MEM_FRACTION=0.65 uv run python scripts/train_fast_sac.py --env Go2WarpJoystickFlat

# Terminal 2: video, uses 0.3 of VRAM  (preallocate=false by default in record_video.py)
MUJOCO_GL=egl uv run python scripts/record_video.py --checkpoint checkpoints/<run>/best
```

Both fit if fractions don't exceed 1.0 and neither is pinned.

## Key Insight

Before this investigation (April 2026), the code had hardcoded MEM_FRACTION values to "play it safe." But "safe" is a guess—it either underutilizes VRAM or constrains future experiments. The real safety comes from `PREALLOCATE=true` (which all our scripts use). The fraction is just a knob to move OOM boundaries; tune it per-config, not once and forever.
