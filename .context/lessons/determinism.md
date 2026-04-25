# Determinism — JAX/XLA + GPU Physics

## TL;DR

- **JAX/XLA layer (algo)** can be made bit-identical across processes by setting
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. We empirically verified this on
  TD-MPC2: 50 `update_step` calls on a fixed fake batch produced identical
  param hashes across two processes when the flag was set, and diverged when
  it was not.
- **mujoco_warp (GPU physics)** is **not** bit-identical across processes. The
  XLA flag has no effect on Warp CUDA kernels. Only `wp.set_device("cpu")`
  gives bit-exact reproducibility today (slow). Fix is in flight upstream
  (NVIDIA/warp#1355, mujoco_warp#1281, target Warp 1.14, ~2026-06).
- Therefore **end-to-end training trajectories cannot be byte-identical
  across two GPU processes**. Same-process re-runs (where Warp graph is
  cached and replayed identically) appear to be deterministic.

## Empirical evidence (TD-MPC2 J3 debug, 2026-04-25)

Wrote `scripts/check_tdmpc2_determinism.py` with three subchecks: `init`,
`update`, `env`. Ran each twice in separate processes, diffed the printed
SHA-256 hashes of the relevant pytree.

| Check | XLA flag off | XLA flag on |
|---|---|---|
| `init` (network init) | bit-ID | bit-ID |
| `update` (50 update_steps on fixed fake batch) | DIVERGES | bit-ID |
| `env` (50 env_steps after reset, fixed actions) | DIVERGES | DIVERGES |

`env` diverges from the very first `reset` call — raw Cheetah obs hash
differs between processes (`475c91df...` vs `d4d59fe0...`) even with the
XLA flag set. `info.AutoResetWrapper_rng` and other JAX-side state DO
match — only the actual physics output (obs from MuJoCo simulation)
differs. Confirms the divergence is in Warp, not in our jax wrapper.

## Root cause (sources)

- **JAX/XLA non-det**: GPU reductions (`sum`, `mean`, scatter) use atomic
  adds. Float add is not associative — `(a+b)+c ≠ a+(b+c)` due to rounding —
  so different warp execution order produces different results. The XLA
  flag forces serial reductions and sticky kernel choice.
- **Warp non-det**: ~100s of `wp.atomic_add` calls in narrowphase contact
  resolution (kevinzakka, mar-yan24 in mujoco_warp#562). Same atomic ordering
  problem.

## Official acknowledgements

> "Is MJWarp on GPU deterministic? **No.** There may be ordering or small
> numerical differences between results computed by different executions of
> the same code. This is characteristic of non-deterministic atomic
> operations on GPU. Set device to CPU with `wp.set_device('cpu')` for
> deterministic results."
> — https://mujoco.readthedocs.io/en/latest/mjwarp/index.html

GitHub issue mujoco_warp#562 has DeepMind/NVIDIA confirming + Erwin
Coumans noting "non-deterministic RL is a nightmare to debug, surprised
this wasn't a requirement from the start."

NVIDIA/warp#1355 (Miles Macklin, Apr 2026) introduces opt-in
`wp.config.deterministic` flag rerouting `atomic_add/sub/min/max` through
"scatter-sort-reduce". Targets Warp 1.14, ~2026-06-01. Even with that,
mujoco_warp needs additional work (`atomic_add` inside `@wp.func` helpers
not yet supported by the new mode).

## Implications for our repo

- **Repeatability claims**: Cross-process bit-ID training is impossible
  on GPU today. Same-process re-runs APPEAR deterministic (Warp graph is
  cached and replayed). For experimental rigor, report seed-averaged
  results, not single-run numbers.
- **A/B testing two code paths**: Can't rely on byte-identical trajectories
  to isolate the effect of a code change. Use multi-seed runs or
  before/after metrics on the same process.
- **Eval-key isolation** (TD-MPC2 fix from 2026-04-25): can't be verified
  by diffing two-process runs. Verify by within-process inspection (log
  training key BEFORE and AFTER eval branch — if isolated, unchanged).
- **Reviewers**: if asked "why aren't your results bit-reproducible?",
  point at this lesson + the upstream issues.

## Workarounds (today)

- `XLA_FLAGS=--xla_gpu_deterministic_ops=true` — enables JAX/XLA layer
  determinism. Use for any test that doesn't depend on env physics.
  ~10–30% slowdown.
- `wp.set_device("cpu")` — only path to bit-exact end-to-end. Slow for
  full training; viable for unit tests that need physics determinism.
- For RL trajectory determinism: not currently possible on GPU. Wait for
  Warp 1.14 (~Jun 2026).

## Future fix path

1. Track NVIDIA/warp#1355 (Warp 1.14 release).
2. Track mujoco_warp#1281, #1300 (mujoco_warp deterministic mode).
3. When both land, retest by re-running `scripts/check_tdmpc2_determinism.py
   --check env` with `wp.config.deterministic = True` + `opt.deterministic =
   True` — should produce bit-ID env trajectories.
4. Document expected slowdown and update CI tests if appropriate.
