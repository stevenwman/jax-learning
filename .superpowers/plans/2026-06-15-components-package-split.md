# Plan — Components package split (executes the spec)

Spec: `.superpowers/specs/2026-06-15-components-package-split.md`
Oracle: `scripts/smoke_refactor.py` + 32-test GPU subset. Baseline already green.

## Symbol → module map (verbatim move, exact line ranges in original)

| New file | Symbols (from go2_warp_components.py) | Imports it needs |
|---|---|---|
| `components/actuation.py` | `Actuation, TorqueOnly, MotorModel, actuation_from_config` | `from jax_rl.envs.locomotion.go2_warp_base import torque_speed_clip, physical_armature` |
| `components/terrain.py` | `_norm01, _fractal_perlin_noise_2d, _make_heightfield, Terrain, Flat, RoughHF, terrain_from_config` | `numpy as np`, `mujoco`, `from scipy import ndimage` |
| `components/controllers.py` | `_N_STIFFNESS, log_action_scale, lin_action_scale, var_action_size, impedance_gains, Controller, JointPD, OSC, VarImpedance, VarImpedanceMass, controller_from_config` | `jax`, `jax.numpy as jp`, `numpy as np`, `mujoco`, `from mujoco import mjx`, `from jax_rl.envs.locomotion import go2_osc` |
| `components/force_fields.py` | `mud_foot_force, ForceField, NoField, MudField, field_from_config` | `jax`, `jax.numpy as jp`, `numpy as np` |
| `components/__init__.py` | re-export all of the above (incl. `_make_heightfield`) | — |

`go2_warp_components.py` → shim: `from jax_rl.envs.locomotion.components import *`
+ explicit re-export of privates (`_N_STIFFNESS, _norm01, _fractal_perlin_noise_2d, _make_heightfield`).

## Steps

1. Create `components/` dir + the 4 concern modules, pasting each symbol block
   **verbatim** from the original (preserve docstrings, comments, the
   STABILITY-CRITICAL notes, `from __future__ import annotations` where used).
2. Write `components/__init__.py` with explicit re-exports + `__all__`.
3. Replace `go2_warp_components.py` body with the shim (keep a docstring pointing
   to the package).
4. **Verify** (gate — do not commit until all green):
   a. `uv run python scripts/smoke_refactor.py` → 6/6, signatures ~match.
   b. GPU subset: `pytest tests/test_go2_osc.py tests/test_go2_osc_env.py
      tests/test_go2_warp_variants.py tests/test_terrain_component.py
      tests/test_go2_warp_env.py tests/test_var_impedance_damping.py`.
   c. `pytest tests/test_layer_deps.py`.
5. Commit (single commit, move + shim only). Message:
   `refactor(go2): split go2_warp_components into single-concern components/ package (shim kept)`.
6. If any gate fails → diagnose; if not cleanly fixable, `git checkout` the
   working tree (revert the split), leave harness+lint+spec+plan in place, note
   the blocker in the journal. Safe failure mode.

## Follow-up (separate commit, only if step 5 green)

- Repoint the ~9 internal importers (`go2_warp_base`, `go2_warp_joystick`,
  `go2_warp_curriculum`, tests) from `go2_warp_components` → `components`.
- Slim the shim to a deprecation re-export (keep for external/test back-compat).

## Journal

Append result to `.context/journals/2026-06-15.md` + flip the redesign-doc
footer status line.
