# mud_eval — test jax-learning Go2 policies in Newton's triple-mud MPM sim

Self-contained eval harness: drop our **JAX-trained Go2 policies** (joint-PD /
OSC / variable-impedance) into the **Newton MPM** two-way-coupled *triple-mud*
environment (thin / medium / thick mud) and see how they cope. Eval/rollout
only — **not** a training backend (Newton physics isn't JAX-traceable; see below).

## Why a separate copy (not wired into the jax-rl pipeline)

- Newton's `solver.step` is a **Warp** call — it can't run inside `jax.jit`, so we
  can't reuse jax-learning's jitted `env.step`. This is a **hand-written Python
  step loop** that calls Newton for physics and runs the policy as inference.
- Everything lives here and is **self-sufficient**: the Newton lib is vendored
  under `vendor/newton/` (the example's 399 MB of recordings/policies are *not*
  copied), and a dedicated venv (`.venv/`) keeps jax-learning's training env clean.
  If it proves out we may merge the deps into the root `pyproject` later.

## Layout

```
vendor/newton/        vendored Newton 0.1.3 physics lib + the mpm_go2_multi example
                      (forked from Newton_stuff; recordings/policies excluded)
m0_smoke.py           M0: headless physics smoke (hold-pose, no policy)
.venv/                dedicated venv (warp 1.12 + mujoco + torch-cu128 + jax later)
```

## Runtime

Dedicated venv built off jax-learning's proven Warp-1.12-on-RTX-5080 stack:
`warp-lang==1.12.0`, `mujoco`, `torch` (cu128, Blackwell), `numpy`, `pyyaml`,
+ `jax`/`jax_rl` (added at M1 for policy inference). Newton is vendored (on
`sys.path`, no pip install). Requires the GPU.

## Milestones

- **M0** — headless physics smoke (robot + 3-mud MPM + coupling, hold pose). ← here
- **M1** — joint-PD JAX policy through an obs adapter + the owned `step()`.
- **M2** — OSC controller: foot-target action → Jᵀ·Λ·F torque from **mjData**
  Jacobian + `mj_fullM`, injected as joint forces.
- **M3** — variable-impedance (decode the stiffness tail).

## Policies to test (jax-learning worktree `checkpoints/`, 2026-06-09 physical-motor retrains)

These are the compliance-helps-on-rough winners — natural to re-test on graded mud:

| controller | checkpoint dir |
|---|---|
| joint-PD | `20260609_134104_fast_sac_go2warpjoystickflatphysical_seed0` |
| OSC fixed-soft | `20260609_140251_fast_sac_go2warposcflatsoftphysical_seed0` |
| var per-foot | `20260609_142511_fast_sac_go2warposcvarflatphysical_seed0` |
| var per-axis | `20260609_144837_fast_sac_go2warposcvaraxisflatphysical_seed0` |
| +damping (foot) | `20260609_165254_fast_sac_go2warposcvardampingflatphysical_seed0` |
| +damping (axis) | `20260609_190852_fast_sac_go2warposcvardampingaxisflatphysical_seed0` |

(in `/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/go2-osc-impedance/checkpoints/`)

## Caveat

Zero-shot sim2sim: Newton's Go2 model + MPM mud ≠ the MJX-Warp training model +
analytic contact. This tests robustness/transfer, not trained-in numbers.
