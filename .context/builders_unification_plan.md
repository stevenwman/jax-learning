# Builders Unification Plan

**Status:** Planned (not started)
**When to do:** Before adding a second encoder type (CNN, SimbaV2, Transformer)

## Problem

PPO uses `builders.py` for network construction (encoder + head composed as a Linen module). SAC, TD3, FastTD3, and FastSAC all build encoder + heads inline in their algo `__init__`. This means adding a new encoder requires touching every algo file.

## Current State

| Algo | Actor | Critic | Construction |
|------|-------|--------|-------------|
| PPO | `Actor` (MlpEncoder + GaussianHead) | `Critic` (MlpEncoder + ValueHead) | via builders.py |
| SAC | MlpEncoder + GaussianHead | 2x QHead | inline in algo |
| TD3 | MlpEncoder + DeterministicHead | 2x QHead | inline in algo |
| FastTD3 | MlpEncoder + DeterministicHead | 2x DistributionalQHead | inline in algo |
| FastSAC | MlpEncoder + GaussianHead | 2x DistributionalQHead | inline in algo |

## Solution: Expand builders.py (Option A)

Add explicit composed modules for each actor/critic variant. Each is ~10 lines — boring and obvious.

```python
# builders.py — one composed module per network role

class Actor(nn.Module):              # encoder + GaussianHead (PPO, SAC, FastSAC)
class DeterministicActor(nn.Module): # encoder + DeterministicHead (TD3, FastTD3)
class VCritic(nn.Module):            # encoder + ValueHead (PPO)
class QCritic(nn.Module):            # encoder + QHead (SAC, TD3)
class DistributionalQCritic(nn.Module):  # encoder + DistributionalQHead (FastTD3, FastSAC)
```

### Why not a generic builder?

Q heads need `concat(obs, action)` before the encoder, which breaks a single generic `NetworkModule(encoder, head)` pattern. You'd need conditionals, and then it's not actually generic. Five small classes > one class with special cases.

### What changes per algo

- **PPO**: Already uses `Actor`/`Critic` — rename `Critic` → `VCritic` for clarity
- **SAC**: Replace inline encoder+head with `Actor` + 2x `QCritic`
- **TD3**: Replace inline encoder+head with `DeterministicActor` + 2x `QCritic`
- **FastTD3**: Replace inline with `DeterministicActor` + 2x `DistributionalQCritic`
- **FastSAC**: Replace inline with `Actor` + 2x `DistributionalQCritic`

### Encoder swapping

Once all algos use builders, swapping MLP for CNN means changing the builder internals (or adding an `encoder_type` field to `EncoderConfig`). No algo file touches required.

## Follow-on: Unified off-policy train script

Once builders unification is done, the 4 off-policy train scripts (SAC, TD3, FastTD3, FastSAC) can merge into one `train_offpolicy.py --algo sac|td3|fast_td3|fast_sac`.

Prerequisites beyond builders:
1. `select_action` handles exploration noise internally (TD3/FastTD3 currently add noise outside)
2. Algo registry: `ALGOS = {"sac": SAC, "td3": TD3, ...}` + matching preset loaders
3. Each algo returns its own metric keys — script logs whatever the algo returns without knowing the keys

Eliminates ~300 lines of duplication. The loop is identical; only instantiation differs.

**Not blocking Go2 or vision RL** — this is a cleanup once the interface is stable.

## What NOT to do

- No factory/registry pattern for encoders — YAGNI until we have 3+ encoder types
- No generic `NetworkModule` with conditionals
- No `BaseAlgorithm` ABC — algos stay as standalone classes
