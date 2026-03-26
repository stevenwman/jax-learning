# JAX RL Framework — Design Document

> **Status**: This document captures the original design rationale and key technical decisions.
> For current project state, see: `README.md` (overview), `TODO.md` (priorities),
> `AGENT_HANDOFF.md` (working patterns), `go2/sim_to_real_plan.md` (Go2 plan).
> Detailed architecture specs, phase checklists, and interface definitions have been
> removed — the code itself is now the source of truth.

## Overview

A modular, JAX-native reinforcement learning library built on Flax Linen, designed to make **efficient robot learning accessible**.

This framework is both a **learning vehicle** and a **lab tool**. Building it is an opportunity to develop deep fluency with JAX, Flax Linen, and RL algorithms by implementing them from scratch. The resulting codebase will be shared across the lab, so every design decision prioritizes interpretability — a new lab member should be able to read a single algorithm file and understand what's happening without chasing abstractions across ten modules.

### Why this framework?
- **Learnable**: Every algorithm is implemented from fundamentals, heavily commented, with clear mappings to the papers. No magic, no "just trust the base class."
- **Accessible**: Clean code the whole lab can read, modify, and extend — not research spaghetti
- **Efficient**: JAX-native with FastTD3/SAC for training in minutes, not hours
- **Robot-first**: Built for the sim-to-real pipeline — train fast in sim, deploy on hardware

### Core design principles
1. **Fast code, clear explanations** — use every JAX trick that improves performance, but comment the non-obvious ones so the lab can learn from them. The code should be efficient *and* educational.
2. Understand the code → modify it → run experiments → deploy on robot
3. Modular encoder+head architecture — swap components without rewriting algorithms
4. Sim-to-real as first-class citizen, not afterthought
5. **Self-contained envs** — env.step() returns fully processed (obs, reward, done). Training scripts never touch obs construction, frame stacking, or reward terms. New robot = new env class, zero train script changes.
6. **Brax-style shared utilities, not SB3-style class hierarchy** — algos own their training loops, shared infra is stateless functions. No Trainer base class, no BaseAlgorithm ABC. See LESSONS.md for the research behind this decision (compared Brax, SB3, Tianshou, CleanRL, Isaac Lab, MuJoCo Playground, MJLab).

### Target workflow
```
Simulation (MuJoCo Playground)
    → Train with PPO/SAC/FastTD3
    → Domain randomization + curriculum
    → Export policy (ONNX for Jetson, or numpy weights)
    → Validate in unitree_mujoco (same DDS interface as real robot)
    → Deploy on real robot via Unitree SDK2
```

---

## Technical Decisions

### Stack
- **JAX** for autodiff, jit, vmap, scan
- **Flax Linen** for neural networks (not NNX — see rationale below)
- **Optax** for optimizers
- **Orbax** for checkpointing
- **MuJoCo Playground** for environments (MJX backend, GPU-parallelized)

### Why Flax Linen over NNX?

NNX is the newer, more Pythonic Flax API and is recommended for new projects in general. However, **Linen is the better choice for high-throughput RL** for several specific reasons:

1. **Performance on small models**: `nnx.jit` traverses the object graph in pure Python on every call, adding overhead that primarily affects small-to-medium models — exactly the 256-512 hidden dim MLPs we use. Benchmarks show ~3x overhead for NNX on MLP workloads vs. Equinox/Linen.

2. **Ecosystem alignment**: Brax training (the reference PPO/SAC used by MuJoCo Playground) uses Flax Linen. FastTD3, CleanRL-JAX, and RSL-RL all use functional params-as-pytrees patterns.

3. **RL-native pattern**: Linen's `model.apply(params, obs)` maps cleanly onto RL's "params-in, metrics-out" training loop. Params are plain pytrees that flow through `jax.jit`, `jax.grad`, `jax.vmap` with zero Python overhead.

4. **Immutability = safety**: Linen's stateless design makes it impossible to accidentally mutate shared state across parallel environments or between actor/critic networks.

5. **Learning benefit**: Linen's explicit functional pattern (init → params → apply) forces you to understand how JAX actually works — pytrees, pure functions, JIT compilation boundaries. For a codebase meant to teach the lab JAX fundamentals, the explicitness is a feature.

### Config System
- **Dataclasses** for type safety and IDE support
- Per-algo configs (PPOConfig, SACConfig, etc.) with shared TrainConfig
- Per-environment presets in `env_presets.py`
- No framework dependency (no Hydra)

### Replay Buffer
- **JAX-native, GPU-resident** — entire collect→sample→update loop stays on-device
- Struct-of-arrays layout (separate arrays for obs, action, reward, etc.)
- Circular buffer with uniform sampling, fully jittable
- On-policy uses separate rollout buffer with GAE computation

### Environment Interface
- **Primary**: MuJoCo Playground (MJX/MJWarp, JAX-native, 1024+ parallel envs)
- **Future**: Gymnasium adapter for ManiSkill, custom envs
- **Sim-to-real**: Domain rand as robot-agnostic wrapper, validated via unitree_mujoco

### Closure-as-Method Pattern
All algos define JIT'd functions as closures inside `__init__` and assign them to self (e.g., `self._update = update`). This is because JAX's JIT traces Python functions and captures closed-over values — regular methods would try to trace `self`, which is a mutable Python object. This pattern is standard across JAX RL codebases (Brax, PureJaxRL). See `sac.py` class docstring for the full explanation.

---

## Key Lessons Learned

See `LESSONS.md` for the full list. The most impactful:

- **Always read source code, not just the paper** — FastDSAC paper says "Gaussian NLL" but code uses Huber loss. Cost us 2 days.
- **MJX produces both NaN AND Inf** — guard must check both. Inf comes from velocity overflow, different from NaN (solver failure).
- **PPO value loss scaling matters** — Brax uses 0.25x (0.5 MSE × 0.5 PPO coeff). Without this, critic gradients are 4x too large.
- **gamma=0.97 for locomotion** — the FastTD3/FastSAC paper uses 0.97, not 0.99. This alone was worth +140 eval points.
- **Reward weight ratios matter more than individual weights** — Go2's tracking/pose ratio needed to be >0.5 for actual locomotion vs crouching.
- **Don't kill training runs prematurely** — dip-then-recover is normal when physics change.
