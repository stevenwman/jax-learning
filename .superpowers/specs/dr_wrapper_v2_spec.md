# Domain Randomization Wrapper Specification

> **Note:** This spec was written during development as "DRv2". The wrapper has been renamed to `DomainRandWrapper` in `jax_rl/envs/wrappers/domain_rand.py`. All references to "DRv2Wrapper" below refer to `DomainRandWrapper`.

## Overview

`DomainRandWrapper` is a **unified episode boundary manager** that replaces both `AutoResetWrapper` and `DomainRandomizationVmapWrapper`. It handles:
1. Done detection + conditional state selection (where_done)
2. Per-episode initial condition re-randomization
3. Per-episode domain randomization (when spec provided)
4. Clean state.info reset on episode boundary
5. Correct obs via forward()

**Key property:** With no DR spec, it behaves as an improved AutoResetWrapper. With a DR spec, it adds per-episode physics randomization on top.

### What it replaces

| Old | New |
|-----|-----|
| `AutoResetWrapper(full_reset=False)` — cached replay, no per-episode variation | `DRv2Wrapper(env)` — fresh IC + clean state.info every episode |
| `AutoResetWrapper(full_reset=True)` — inconsistent perf across envs | `DRv2Wrapper(env)` — consistent, benchmarked |
| `DomainRandomizationVmapWrapper` — per-env fixed DR | `DRv2Wrapper(env)` with DR spec — per-episode DR |
| Separate `go2_randomize.py` / `bongo_randomize.py` files | DR spec declared in env class |

### Key changes from v1

| Aspect | v1 | v2 |
|--------|----|----|
| **Episode reset** | Cached replay (full_reset=False) or full reset every step (full_reset=True) | Inline reset in step() via where_done |
| **ICs** | 256 fixed (frozen at init) | Fresh per episode (default pose + noise) |
| **Model DR** | Fixed per env (vmapped once at init) | Per-episode re-randomization |
| **Runtime DR** | Hardcoded in env reset() | Declared via spec, applied by wrapper |
| **state.info** | Carries over between episodes | Properly reset on done |
| **Obs at boundary** | Cached first_obs (correct but repetitive) | Fresh obs via forward() (correct and varied) |

---

## DRSpec Definition

```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class DRSpec:
    """Single domain randomization parameter specification."""
    
    name: str
    """Human-readable name (e.g., "friction", "motor_strength")."""
    
    type: Literal["model", "runtime"]
    """Where randomization is applied.
    - "model": modifies mjx.Model fields (geom_friction, dof_damping, etc.)
    - "runtime": produces values stored in state.info[name], read by env in step()
    """
    
    # ── Model DR fields (type="model" only) ─────────────────────
    field: Optional[str] = None
    """MJX model field name (e.g., "geom_friction", "dof_damping")."""
    
    column: Optional[int] = None
    """If field is 2D, which column to randomize. None = entire field."""
    
    indices: Optional[tuple] = None
    """Row indices to randomize. None = all rows.
    Example: (6, 18) for joint DOFs only (slice notation)."""
    
    operation: Literal["set", "multiply", "add"] = "multiply"
    """How to apply: multiply (scale), set (absolute), add (offset)."""
    
    # ── Range ───────────────────────────────────────────────────
    min: float = 0.0
    max: float = 1.0
    
    per_element: bool = False
    """False: one sample broadcast to all elements.
    True: independent sample per element."""
    
    # ── Runtime DR only (type="runtime") ────────────────────────
    shape: Optional[tuple] = None
    """Output shape stored in state.info[name]. None = scalar ()."""
    
    description: Optional[str] = None
```

---

## Wrapper Architecture

```python
class DRv2Wrapper(Wrapper):
    """Unified episode boundary manager with optional domain randomization.
    
    Replaces AutoResetWrapper + DomainRandomizationVmapWrapper.
    
    With no DR spec: improved AutoResetWrapper (fresh ICs, clean state.info).
    With DR spec: adds per-episode physics randomization.
    """
    
    def __init__(self, env, episode_length: int = 1000):
        super().__init__(env)
        self.episode_length = episode_length
        
        # Get DR spec from env (empty list if env doesn't declare one)
        if hasattr(env, 'get_domain_randomization_spec'):
            self.dr_specs = env.get_domain_randomization_spec()
        else:
            self.dr_specs = []
        
        self._model_specs = [s for s in self.dr_specs if s.type == "model"]
        self._runtime_specs = [s for s in self.dr_specs if s.type == "runtime"]
    
    def reset(self, rng: jax.Array) -> State:
        """Initial reset. Called once at training start."""
        # Vmap over envs
        state = jax.vmap(self._single_reset)(rng)
        state.info['drv2_rng'] = rng
        state.info['drv2_steps'] = jp.zeros(rng.shape[0])
        return state
    
    def _single_reset(self, rng) -> State:
        """Reset one env: IC + DR + forward + obs."""
        # Apply model DR if specs exist
        # ... (sample and apply to model via tree_replace)
        
        # Env's own reset (samples IC, runs make_data + forward, builds info)
        state = self.env.reset(rng)
        
        # Apply runtime DR to state.info
        for spec in self._runtime_specs:
            rng, key = jax.random.split(rng)
            state.info[spec.name] = self._sample_spec(spec, key)
        
        return state
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step all envs. On done: reset IC + re-randomize DR inline."""
        
        # 1. Step all envs normally
        state = jax.vmap(self.env.step)(state, action)
        
        # 2. Track episode length, compute done
        steps = state.info['drv2_steps'] + 1
        truncation = steps >= self.episode_length
        done = state.done | truncation
        
        # 3. Prepare reset state for ALL envs (JAX computes both paths)
        rng_keys = jax.vmap(jax.random.split)(state.info['drv2_rng'])
        reset_rng, next_rng = rng_keys[:, 0], rng_keys[:, 1]
        reset_state = jax.vmap(self._single_reset)(reset_rng)
        
        # 4. where_done: select reset state for done envs, keep stepped for others
        def select(reset_val, step_val):
            d = done.reshape([done.shape[0]] + [1] * (len(step_val.shape) - 1))
            return jp.where(d, reset_val, step_val)
        
        out_data = jax.tree.map(select, reset_state.data, state.data)
        out_obs = jax.tree.map(select, reset_state.obs, state.obs)
        out_info = jax.tree.map(select, reset_state.info, state.info)
        
        # 5. Preserve cross-episode state
        out_info['drv2_rng'] = next_rng
        out_info['drv2_steps'] = jp.where(done, jp.zeros_like(steps), steps)
        
        return state.replace(data=out_data, obs=out_obs, done=done, info=out_info)
```

### Key design decisions

1. **reset_state computed every step** — same O(N) cost as full_reset=True, but we own the graph. Benchmark will tell us if XLA fuses it better than AutoResetWrapper's approach.

2. **Env's own reset() handles IC generation** — we don't duplicate the pose+noise logic. The env already knows how to reset itself. We just call it and layer DR on top.

3. **state.info fully replaced on done** — no stale values. Clean episode boundaries.

4. **DR applied in _single_reset** — model DR modifies the mjx model before env.reset(), runtime DR writes to state.info after.

---

## Integration: Env declares DR spec

```python
class Go2WarpJoystick(Env):
    def get_domain_randomization_spec(self) -> list[DRSpec]:
        return [
            DRSpec(name="friction", type="model", field="geom_friction",
                   column=0, min=0.3, max=1.5, operation="set"),
            DRSpec(name="motor_strength", type="model", field="actuator_gainprm",
                   column=0, min=0.9, max=1.1, per_element=True),
            DRSpec(name="dof_damping", type="model", field="dof_damping",
                   indices=(6, 18), min=0.7, max=2.0, per_element=True),
            DRSpec(name="dof_armature", type="model", field="dof_armature",
                   indices=(6, 18), min=0.9, max=1.3, per_element=True),
            DRSpec(name="dof_frictionloss", type="model", field="dof_frictionloss",
                   indices=(6, 18), min=0.7, max=1.5, per_element=True),
            DRSpec(name="body_mass", type="model", field="body_mass",
                   min=0.8, max=1.2, per_element=True),
            DRSpec(name="torso_com_jitter", type="model", field="body_ipos",
                   indices=(1, 2), min=-0.08, max=0.08, operation="add",
                   per_element=True),
            DRSpec(name="kp_scale", type="runtime", min=0.8, max=1.3),
            DRSpec(name="kd_scale", type="runtime", min=0.5, max=1.5),
        ]
```

---

## Wrapper stack

Old:
```python
env = VmapWrapper(env)                           # or DomainRandomizationVmapWrapper
env = EpisodeWrapper(env, episode_length)
env = AutoResetWrapper(env)
```

New:
```python
env = DRv2Wrapper(env, episode_length=1000)      # handles vmap, episodes, reset, DR
```

One wrapper. Done.

---

## Benchmark plan

Test across all envs (Go2 Warp, Go2 Bongo, Cartpole, CheetahRun, WalkerWalk, HumanoidRun):

| Config | What it tests |
|--------|---------------|
| `AutoResetWrapper(full_reset=False)` | Current baseline |
| `AutoResetWrapper(full_reset=True)` | Full reset comparison |
| `DRv2Wrapper(env)` — no DR spec | Autoreset replacement quality |
| `DRv2Wrapper(env)` — with DR spec | Full DR cost |

Success criteria:
- DRv2 (no spec) within 10% of AutoReset(full_reset=False) throughput
- DRv2 (with spec) provides per-episode DR with acceptable overhead
- No env has catastrophic slowdown (looking at you, CheetahRun)

---

## Migration

1. Build DRv2Wrapper alongside existing wrappers (no breaking changes)
2. Benchmark head-to-head
3. If DRv2 wins: update wrap_for_training() to use it, delete old wrappers
4. Delete go2_randomize.py, bongo_randomize.py
5. Move DR specs into env classes

## Open questions

1. **Model DR application:** DRv2 needs to modify the mjx model per-env. Currently DomainRandomizationVmapWrapper does this via tree_replace at init. DRv2 needs to do it per-episode inside _single_reset(). How does this interact with vmap? The model is shared — we need per-env model copies.

2. **EpisodeWrapper consolidation:** Should DRv2 also absorb EpisodeWrapper's truncation/metrics logic, or keep that separate?

3. **VmapWrapper:** DRv2 vmaps internally. Should it replace VmapWrapper too, or wrap a VmapWrapper?
