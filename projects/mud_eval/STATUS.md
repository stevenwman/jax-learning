# mud_eval — status

## ✅ M0 RESOLVED (2026-06-10)

**Working version combo** (newton 0.1.3 needs the **pre-batching** mujoco-warp):

```
warp-lang==1.12.0   mujoco==3.7.0   mujoco-warp==0.0.2   torch==2.11+cu128
+ trimesh + pycollada + numpy + pyyaml   (newton 0.1.3 vendored on sys.path)
```

`m0_smoke.py` builds the scene (Go2 13 bodies/18 dofs + 155k MPM particles across
thin/medium/thick mud), steps headless, all state finite. The earlier 3.x
mujoco-warp wall (2D `geom_*` arrays) is gone with **0.0.2** (1D, pre-per-world-
batching). Pins in `requirements.txt`.

> History of the blocker below kept for reference.

---

## Where we are (history)

Dedicated venv (`.venv/`) is up and the **physics stack is ~95% standing**:
- `warp-lang 1.12.0`, `torch 2.11+cu128` (works on the RTX 5080 / Blackwell),
  `trimesh`+`pycollada` (URDF/`.dae` mesh load), vendored `newton 0.1.3`.
- `m0_smoke.py` gets all the way through **scene build** — robot URDF loads, the
  three mud regions (thin/medium/thick) spawn as MPM particles, `SolverImplicitMPM`
  is created — and then fails when `SolverMuJoCo` initializes the rigid solver.

## The blocker — newton 0.1.3 ⟷ mujoco-warp version incompatibility

`newton.solvers.SolverMuJoCo` (rigid-body solver for the robot) calls into
`mujoco_warp`. Vendored **newton 0.1.3** was written against an **old, pre-
per-world-batching** mujoco-warp where the `mjw_model.geom_*` arrays are **1D**
`(ngeom,)`. Every **installable** mujoco-warp (3.7.0.1, 3.8.0, 3.8.1, 3.9.0.1)
makes those arrays **2D** `(nworld, ngeom)`. newton 0.1.3's
`update_geom_properties_kernel` (and the rest of its MuJoCo solver) declare the
1D signatures, so the launch dies:

```
RuntimeError: kernel 'update_geom_properties_kernel', argument 'geom_dataid'
expects an array with 1 dimension(s) but the passed array has 2 dimension(s).
```

This is **pervasive** (the whole solver assumes the 1D layout), not a one-line
patch. Version matrix tried:

| warp | mujoco | mujoco-warp | result |
|---|---|---|---|
| 1.14 | 3.9 | 3.9.0.1 | newton import breaks (`warp.context` removed in 1.14) |
| 1.12 | 3.9 | 3.8.1 / 3.8.0 | scene builds; `SolverMuJoCo` 2D `geom_*` mismatch |
| 1.12 | 3.7 | 3.7.0.1 | same 2D mismatch |
| 1.12 | 3.7 | (3.7 needs mujoco 3.7; 3.9 lacks `mjENBL_MULTICCD`) | — |

→ newton 0.1.3 needs a **0.x mujoco-warp** (pre-batching) + its matching mujoco +
warp. Newton_stuff has no lockfile, so the exact quadruple is unknown.

## Decision needed (asked on Slack)

1. **Provide the working versions** for the Newton triple-mud example —
   `warp` / `mujoco` / `mujoco-warp` (and python) from the env where it ran — or
   point me at a lockfile / conda env / the original machine to mirror. Then M0
   runs as-is. *(cleanest)*
2. **OR** authorize bumping to the published **newton-physics 1.0.0** (modern,
   pairs with modern mujoco-warp) and porting the `mpm_go2_multi` example from
   the 0.1.3 API to 1.0.0 — more work, but on the supported path.

## What's ready regardless of the above

- Checkpoints located (the 2026-06-09 physical-motor retrains — see README).
- Project scaffold + vendored newton (self-sufficient) + the M0 harness.
- M1 plan (policy loader + JAX obs adapter + controller hook) can be written
  against `vendor/newton/examples/robot/example_robot_go2.py::compute_obs`
  (the Newton state→obs reference) while the env stays blocked.
