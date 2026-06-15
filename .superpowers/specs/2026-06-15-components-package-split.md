# Spec — Split `go2_warp_components.py` into a single-concern package

**Branch:** `linen_refactor`  ·  **Redesign move:** #1 (stand up the controller home + isolate mud, Bleed 1)
**Status:** approved-to-proceed (user waived review; writing is for shared context)

## Problem

`jax_rl/envs/locomotion/go2_warp_components.py` is 656 lines holding **four
unrelated concerns**:

| Concern | Symbols |
|---|---|
| Actuation | `Actuation, TorqueOnly, MotorModel, actuation_from_config` |
| Terrain | `_norm01, _fractal_perlin_noise_2d, _make_heightfield, Terrain, Flat, RoughHF, terrain_from_config` |
| Controllers | `_N_STIFFNESS, log_action_scale, lin_action_scale, var_action_size, impedance_gains, Controller, JointPD, OSC, VarImpedance, VarImpedanceMass, controller_from_config` |
| Force fields (mud) | `mud_foot_force, ForceField, NoField, MudField, field_from_config` |

Two redesign problems live here: (a) the **mud force field is research code
(Bleed 1) sitting in a core env module**, undifferentiated from the generic
controllers; (b) there is **no dedicated home for controllers** — new ones would
pile into this god-file.

Verified during exploration: the four concerns have **zero cross-module
dependencies** — actuation→`go2_warp_base` helpers, terrain→numpy/scipy/mujoco,
controllers→`go2_osc`, force_fields→jax/numpy. So they separate cleanly.

## Goal

Move each concern into its own module under a `components/` package, **with no
logic change** (verbatim move), behind a back-compat shim so no caller breaks.
This isolates mud and gives controllers a home — the seed of an eventual
`jax_rl/control/` — without claiming an abstraction (env-agnostic control) that
doesn't exist yet.

## Decisions (pinned)

1. **Location** = `jax_rl/envs/locomotion/components/`, *not* `jax_rl/control/`.
   The controllers read ~14 `env._*` internals (`_kp`, `_act_to_joint`,
   `_feet_site_id`, `_torso_body_id`, `_stall_torque`, `_force_field`,
   `_foot_linvel_sensor_adr`, `_apply_torque_speed_limit`, …) — they are
   go2-env-coupled. Promotion to a top-level env-agnostic `jax_rl/control/`
   requires decoupling first (a typed controller contract). That is a later,
   separately-spec'd move.

2. **Granularity** = full 4-way split now (`actuation.py`, `terrain.py`,
   `controllers.py`, `force_fields.py`). The concerns are independent; splitting
   once avoids a second churn. Mud-only extraction would leave a still-bloated
   ~550-line file and re-touch it later.

3. **Shim lifetime** = `go2_warp_components.py` becomes a thin re-export shim
   THIS commit (every existing `from ...go2_warp_components import X` keeps
   working). Repointing the ~9 internal importers to `components` and slimming
   the shim is a SEPARATE follow-up commit (small, bisectable). Shim is NOT
   deleted tonight (tests + external refs).

## Constraints

- **Behavior-preserving**: code moved verbatim; no edits to logic, gains, or
  control flow. Oracle = `scripts/smoke_refactor.py` (6 envs, signatures match
  to ~1e-3) + the green pytest subset (32 tests) staying green.
- **No circular import**: `components/actuation.py` imports `go2_warp_base` at
  top (as the original did); `go2_warp_base` keeps its *lazy* (`__init__`-time)
  imports of `*_from_config`, now resolved through the unchanged shim. Sequence
  verified safe — base's module-level helpers are defined before any env
  `__init__` runs, so actuation's top-level `from go2_warp_base import …`
  resolves.
- **Re-export completeness**: package `__init__` must export every public symbol
  AND the one private a test imports (`_make_heightfield`, in
  `tests/test_terrain_component.py`). Shim does `from ...components import *`
  plus explicit private re-exports.

## Out of scope (explicitly deferred)

- Repointing importers / deleting the shim (follow-up commit).
- Promoting controllers to `jax_rl/control/` (needs decoupling spec).
- Moving mud to `projects/` (needs the plugin-registration mechanism; would
  otherwise make core import projects = the very bleed the lint forbids).
- Any logic/refactor of the controller math.

## Acceptance

1. `uv run python scripts/smoke_refactor.py` → all 6 envs pass, signatures within
   ~1e-3 of the pre-split baseline:
   `flat-jointpd 12/170`, `flat-varimp 36/218`, `flat-mass 48/242`,
   `mud-slowfirm 36/218`, `mud-mass 48/242`, `mud-jointpd 12/170`.
2. The 32-test controller subset stays green on GPU.
3. `tests/test_layer_deps.py` still green.
4. `git diff` shows only: 5 new files under `components/` + `go2_warp_components.py`
   reduced to a shim. No other file touched.
