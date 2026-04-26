# TDMPC2 Refactor — Design Spec

**Date:** 2026-04-26
**Status:** Design (awaiting approval before plan)
**Author:** Claude (with Steven)

## Goal

Improve readability of `jax_rl/algos/tdmpc2.py` (1138 LOC monofile) without
breaking external API. Primary motivation is hand-off to other students;
modularity for vision (CNN encoder) is secondary; efficiency gains are
opportunistic.

## Non-goals

- Build a CNN encoder for image obs (out of scope; insert TODO marker only).
- Convert to class-based agent (other algos use that pattern, but
  functional factories are JAX-idiomatic and read like algorithm pseudocode;
  staying functional avoids touching well-tested JIT'd code).
- Refactor other algos (PPO/SAC/TD3/FlashSAC) to match — TDMPC2 is the
  outlier; the implicit "1 file per algo" convention works for algos that
  fit in <600 LOC.

## Approach

### Sub-package layout

Convert `jax_rl/algos/tdmpc2.py` into `jax_rl/algos/tdmpc2/` package
preserving the import path `from jax_rl.algos.tdmpc2 import ...` via
`__init__.py` re-exports.

```
jax_rl/algos/tdmpc2/
├── __init__.py    ~30  LOC — re-exports for back-compat
├── networks.py    ~300 LOC — Encoder, Dynamics, Reward, QHead, QEnsemble,
│                              PolicyPrior + helpers (NormedLinear, simnorm,
│                              mish, bound_log_std, squash_log_prob_correction,
│                              gaussian_log_prob, compute_scaled_entropy)
├── losses.py      ~250 LOC — compute_td_target, world_model_loss, policy_loss,
│                              compute_all_latents
├── mppi.py        ~350 LOC — mppi_rollout, mppi_iteration,
│                              sample_pi_trajectories, init_mppi_mean,
│                              init_mppi_mean_batched, gumbel_sample_elite,
│                              plan, make_plan_batched
└── agent.py       ~250 LOC — TDMPC2State, build_world_model_optimizer,
                               build_policy_optimizer, make_update_step
```

### Files NOT moving

- `jax_rl/algos/tdmpc2_runtime.py` — separate concern (shared init + eval,
  used by both train and eval scripts). Stays at current path.
- `jax_rl/utils/qscale.py` — already separate.
- `jax_rl/utils/two_hot.py` (or wherever it lives) — already separate.

### Vision encoder hook

`networks.py` Encoder section gets:

```python
class Encoder(nn.Module):
    """h(obs) → z. State obs only.

    For pixel obs (TD-MPC2 paper Section 3.1, Appendix C),
    swap with a CNN-bodied subclass: NormedConv2D chain → flatten →
    Dense(latent_dim) → SimNorm. Source impl in
    nicklashansen/tdmpc2 common/world_model.py:enc_pixels.
    Not implemented here — see TODO above class.
    """
```

No code change beyond the docstring + a `# TODO(vision):` comment block
above the class explaining what to add.

### Bonus cleanups (in-scope, surgical)

1. **Drop unused target params** from `TDMPC2State`:
   `encoder_target_params`, `dynamics_target_params`, `reward_target_params`.
   These are EMA'd but never read (audit at 2026-04-25 evening flagged this
   — only `q_ensemble_target_params` is used by `compute_td_target`).
   Saves ~25% of state memory. Modify `init_train_state` (in
   `tdmpc2_runtime.py`) to stop creating them, modify `make_update_step` to
   stop EMA'ing them.

2. **Strip rotted comments**: subagent-dev-era markers like "Module H4: F4",
   "iter-3 no-mask fix", "iter-4 sign fix" — these reference an
   implementation history that no longer matters and confuses readers.

3. **Inline single-use helpers**: if a helper is only called once and is
   <5 LOC, inline it with a comment explaining what it does. Reduces
   navigation hops without losing clarity.

## Validation

After refactor:

1. `git log --stat` shows file moves + small edits, total LOC delta near zero
   (modulo the dropped target params cleanup ~30 LOC removed).
2. **Numerical smoke**: 3k Cheetah smoke (`--total-timesteps 3000 --seed 0
   --num-envs 8 --eval-every 100000`) reproduces pre-refactor mppi value at
   step 3004 within float-32 tolerance under
   `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (per
   `lessons/determinism.md`).
3. **All scripts work unchanged**: `train_tdmpc2.py`, `eval_tdmpc2.py`,
   `record_video_tdmpc2.py`, `check_tdmpc2_determinism.py` all run end-to-end
   without import edits.
4. **No new test failures**: existing TDMPC2 tests still pass.

## Tradeoffs (surfaced at brainstorm)

- **File count vs cognitive friction**: 5 files (4 + __init__) vs 1.
  Chose 5 because each file has a single conceptual focus that maps to a
  learning order: nets → losses → planner → agent assembly. Imports are
  hidden behind the package's `__init__.py`.
- **Source-faithful vs JAX-idiomatic**: `nicklashansen/tdmpc2` splits into
  ~7 files (`world_model`, `layers`, `math`, `scale`, `init`, `buffer`,
  `online_trainer`). We chose 4 instead because (a) we already have separate
  `qscale.py`, `two_hot.py` utils — `math.py`/`scale.py` equivalents exist;
  (b) splitting `layers.py` from `world_model.py` made sense in PyTorch
  where layers had own state, but in Flax the building blocks (NormedLinear,
  simnorm) are tiny pure functions/classes that live naturally beside the
  networks that use them.
- **Class agent vs functional**: stayed functional. Class would require
  refactoring well-tested JIT'd `make_update_step` closure into methods,
  with risk of subtle JIT bugs. Functional reads like the paper's
  ψ_θ(s,a,r,s') pseudocode and is JAX-idiomatic.

## Out-of-scope cleanups (defer)

- Splitting `train_tdmpc2.py` (554 LOC) further — already separate from algo.
- Refactoring `tdmpc2_runtime.py` — well-bounded, recently extracted.
- Touching `record_video_tdmpc2.py` / `eval_tdmpc2.py` — small focused.
- CNN encoder implementation.
- Converting to class-based agent.

## Risk

- **Merge conflict with env-backend-refactor branch**: that branch doesn't
  edit `jax_rl/algos/tdmpc2.py` (verified: it touches env_setup,
  train_config, train_ppo, record_video). Refactor is safe to land.
- **Breaking external import**: mitigated by `__init__.py` re-exports.
  Validated by running all 4 TDMPC2 scripts unchanged.
- **JIT recompile cost**: package import doesn't change JIT behavior
  (closures still close over `cfg` the same way). No perf regression
  expected.
