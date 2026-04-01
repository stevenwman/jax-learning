# MJLab Deep Comparison Audit

**Date:** 2026-04-01
**Context:** jax-learning vs [MJLab](https://github.com/mujocolab/mjlab) (mujocolab/mjlab)
**Our state:** FastSAC on Go2 Warp (eval 276.5), PPO, frame stacking, domain randomization, sim2sim validated.
**North star:** DIAYN → METRA → USD (skill discovery on real Unitree Go2)

---

## 1. Environment Architecture

**Current gap:** Our env is a monolithic 554-line class with inline obs/reward/reset; MJLab decomposes into ~8 managers where the env is a thin orchestrator.

**Us:** `WarpJoystick` contains all domain logic inline — 16 reward terms, 48d+122d obs construction, command sampling, contact detection. Adding a new concern (skill vectors, vision obs, curriculum) means editing the env class.

**MJLab:** `ManagerBasedRlEnv` (~530 lines) contains almost zero domain logic. Managers called in fixed sequence: event → command → action → observation → termination → reward → curriculum → metrics. Each task defined entirely by `ManagerBasedRlEnvCfg` dataclass. Adding DIAYN = adding an obs term + swapping reward config. No env code changes.

**Impact on north star: HIGH**
- DIAYN needs: (1) skill vector z in obs, (2) reward = discriminator output, (3) curriculum over skill difficulty. Monolithic pattern requires forking/heavy modification. Manager pattern is config composition.
- Vision RL needs: CNN obs group alongside proprioceptive. Manager pattern handles as second obs group.

**Recommendation: ADAPT — before DIAYN**
Don't adopt MJLab's full manager system (PyTorch, 8 managers, significant complexity). Adopt the *principle*: extract reward and obs computation into composable, config-driven functions. Refactor `_get_reward()` into a `RewardSpec` — list of `(name, weight, fn)` tuples the env iterates. DIAYN then swaps reward by replacing the spec, not forking the env. ~2 hours, ~80 lines.

---

## 2. Domain Randomization

**Current gap:** We randomize 8 parameters at env init only; MJLab has 4 trigger modes (startup/reset/interval/step) with per-env timers and deferred recomputation.

**Us:** `go2_randomize.py` applies 8 DR params once via `@jax.vmap` at env creation. Each env gets fixed physics for its lifetime. Kp/Kd scaling per-episode in `reset()`, velocity kicks periodic in `step()`. Core physics (friction, mass, COM) fixed per-env.

**MJLab:** `EventManager` supports startup, reset (with throttling), interval (per-env or global timers with sampled trigger times), and step modes. `@requires_model_fields` decorator tracks which fields need per-world expansion. Deferred `sim.recompute_constants()` at strongest needed level.

**Impact on north star: MEDIUM**
- Mid-episode randomization matters for sim2real (real disturbances don't wait for episode boundaries). Our velocity kicks partially cover this.
- For DIAYN: low impact. For real robot deployment: medium-high.
- Current DR is already aggressive (8 params, ranges validated against sim2sim transfer).

**Recommendation: DEFER — until after first real robot deployment attempt**
When we attempt real transfer and identify failure modes, add interval-mode DR for specific failing parameters. Don't pre-build EventManager infrastructure.

---

## 3. Curriculum Learning

**Current gap:** We have nothing; MJLab has `CurriculumManager` that adjusts difficulty per-env during reset.

**Us:** Zero curriculum support. TODO mentions "Wider DR ranges (Kp/Kd scaling, action delay — may need curriculum)" as active.

**MJLab:** `CurriculumManager.compute(env_ids)` called during `_reset_idx()` before sim/scene reset. Each term receives `(env, env_ids)`, returns arbitrary state. Logged under `Curriculum/` prefix. Simple: ~135 lines + `NullCurriculumManager` for envs without curriculum.

**Impact on north star: MEDIUM-HIGH**
- Kp/Kd DR widening: TODO explicitly notes "may need curriculum." Without it, wide ranges cause learning collapse.
- Terrain curriculum: needed for rough terrain, not needed for flat ground DIAYN.
- Command difficulty: gradual velocity range widening would improve early training stability.

**Recommendation: ADAPT — short-term, for Kp/Kd DR widening**
Add a `curriculum_fn` optional parameter to `reset()` that, given `(env_ids, episode_returns)`, returns updated DR range multipliers. Linear threshold: widen Kp/Kd range when avg return > target. ~1 hour, ~50 lines.

---

## 4. Training Infrastructure

**Current gap:** We depend on Brax/Playground for vectorization, auto-reset, wrapping; MJLab builds its own.

| Aspect | Ours | MJLab |
|--------|------|-------|
| Vectorization | jax.vmap over single-env | Native batched Warp arrays |
| Auto-reset | Cached initial state replay | Selective `_reset_idx(env_ids)` |
| Wrapping | ~~Playground~~ vendored wrappers | Built-in |
| Framework coupling | `env_setup.py` (1 file) | No external framework |

**Impact on north star: LOW-MEDIUM**
- Coupling already well-isolated. vmap approach is idiomatic JAX.
- Selective reset more efficient at scale but cached replay is simpler.
- We already vendored training wrappers (commit in TODO).

**Recommendation: IGNORE**
Coupling is clean and functional. Extract env factory interface from `env_setup.py` when adding ManiSkill, not preemptively.

---

## 5. Observation & Reward Systems

**Current gap:** Inline obs/reward in env class; MJLab has declarative managers with processing pipelines (noise, clip, scale, delay, history) and NaN policies.

**Us:**
- Obs: `_get_obs()` constructs dict with "state" (48d) and "privileged_state" (122d). Noise applied inline with per-sensor scales. No delay simulation, no declarative pipeline.
- Rewards: `_get_reward()` calls 16 methods, each returning a scalar. Weights from `config.reward_config.scales`.

**MJLab:**
- `ObservationManager`: per-term pipeline (compute → noise → clip → scale → NaN check → delay buffer → history buffer). Sophisticated delay simulation (min/max lag, hold probability, update period). History via circular buffer. NaN policy per-group.
- `RewardManager`: weighted aggregation, zero-weight skip, NaN sanitization, dt scaling, episode-average logging. ~173 lines.

**Impact on north star: MEDIUM**
- DIAYN: need clean reward swap. Inline approach requires env surgery; composable spec makes it trivial.
- Vision RL: dict obs already supports adding "pixels" group.
- Obs delay: valuable for sim2real, not blocking initial deployment.

**Recommendation: ADAPT — reward composability before DIAYN**
Same refactor as Axis 1: extract reward terms into composable spec. Obs pipeline is lower priority (dict obs + frame stacking already covers near-term needs).

---

## 6. Sensor & Actuator Abstraction

**Current gap:** We read sensors inline; MJLab has `sensor/` and `actuator/` modules.

**Us:** Sensor reads and actuator logic (PD control, joint→actuator remap) inline in env. Direct mapping to unitree SDK interface.

**MJLab:** Abstracted sensor/actuator modules for multi-robot support.

**Impact on north star: LOW**
Single robot (Go2). Inline reads map directly to unitree SDK (less translation = fewer bugs).

**Recommendation: IGNORE**
Premature generalization for single-robot project.

---

## 7. Scene & Terrain

**Current gap:** Flat ground only; MJLab has `terrains/` module for procedural terrain.

**Impact on north star: LOW**
DIAYN targets flat ground. Terrain is Phase 7+. First real robot target is flat indoor surfaces.

**Recommendation: DEFER**
Not relevant until after DIAYN and initial flat-ground deployment succeed.

---

## 8. Multi-GPU & Scaling

**Current gap:** Single-GPU; MJLab supports `--gpu-ids` multi-GPU.

**Impact on north star: LOW**
Single RTX 5080 is not a bottleneck. DIAYN adds discriminator but doesn't fundamentally change compute needs. Vision RL with CNN likely fits on single GPU for Go2 resolution.

**Recommendation: IGNORE**

---

## Summary

| Axis | Gap | Impact | Action |
|------|-----|--------|--------|
| Env Architecture | Monolithic vs composable | **HIGH** | ADAPT — extract reward/obs specs |
| Domain Rand | Init-only vs 4-mode events | MEDIUM | DEFER — add interval DR after real deployment |
| Curriculum | Nothing vs CurriculumManager | **MED-HIGH** | ADAPT — curriculum callback for Kp/Kd |
| Training Infra | Vendored wrappers vs built-in | LOW-MED | IGNORE |
| Obs & Reward | Inline vs declarative pipeline | **MEDIUM** | ADAPT — reward composability for DIAYN |
| Sensor/Actuator | Inline vs abstracted | LOW | IGNORE |
| Scene/Terrain | Flat only vs procedural | LOW | DEFER |
| Multi-GPU | Single vs multi | LOW | IGNORE |

---

## Top 3: If You Could Only Do 3 Things

### 1. Extract reward terms into composable `RewardSpec` (before DIAYN)
Refactor `_get_reward()` from 16 inline methods into a list of `(name, weight, fn)` tuples the env iterates. DIAYN swaps in `reward = discriminator(obs, z)` by replacing the spec, not forking the env. Highest-leverage change — unblocks the entire skill discovery roadmap. ~2 hours, ~80 lines.

### 2. Add curriculum callback for DR range scaling (for Kp/Kd widening)
Optional `curriculum_fn(env_ids, episode_returns) → dr_range_multipliers` in `reset()`. Linear threshold: widen Kp/Kd range when avg return exceeds target. Unblocks active TODO "Wider DR ranges — may need curriculum." ~1 hour, ~50 lines.

### 3. Make observation construction config-driven (before vision RL)
Refactor `_get_obs()` into `ObsSpec` — list of `(name, fn, noise_cfg)` tuples grouped by "policy" and "critic". Adding vision = config change, not env surgery. Enables clean DIAYN skill vector injection into obs. ~2 hours, ~100 lines.

**Bottom line:** MJLab's manager pattern is overbuilt for a single-robot research project, but its *principles* — composable rewards, config-driven observations, curriculum hooks — directly unblock our north star. Adopt the principles, not the framework. Three targeted refactors give 80% of the composability at 10% of the complexity.
