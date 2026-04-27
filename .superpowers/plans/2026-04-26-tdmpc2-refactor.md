# TDMPC2 Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `jax_rl/algos/tdmpc2.py` (1138 LOC monofile) into `jax_rl/algos/tdmpc2/` sub-package with 4 focused modules, preserving all import paths via `__init__.py` re-exports.

**Architecture:** Pure file-split refactor. Code stays functional (no class-based agent). Each task moves one section of code from a temporary `tdmpc2_old.py` into a focused module under the new package, validating numerical equivalence after each move via deterministic 3k smoke test.

**Tech Stack:** Python, JAX, Flax. No new dependencies.

---

## Pre-flight notes

**Spec:** `.superpowers/specs/2026-04-26-tdmpc2-refactor-design.md` — read first.

**Files that import from `jax_rl.algos.tdmpc2` (must keep working unchanged):**

```
scripts/train_tdmpc2.py:24-27       imports: TDMPC2State, make_plan_batched, make_update_step
scripts/eval_tdmpc2.py:22           imports: make_plan_batched
scripts/check_tdmpc2_determinism.py:29  imports: make_update_step
scripts/record_video_tdmpc2.py:27   imports: make_plan_batched
jax_rl/algos/tdmpc2_runtime.py:15-19   imports: Encoder, Dynamics, Reward, QEnsemble, PolicyPrior, TDMPC2State, build_world_model_optimizer, build_policy_optimizer
tests/test_tdmpc2.py                imports many internals (see __all__ below)
```

**Required `__all__` in `tdmpc2/__init__.py` (covers production code + tests):**

The test suite (`tests/test_tdmpc2.py`) imports MANY internal symbols (`NormedLinear`, `mppi_rollout`, `compute_all_latents`, `bound_log_std`, etc.) — they're treated as part of the package's public surface for the duration of this refactor. List them all explicitly here so a `from jax_rl.algos.tdmpc2 import <internal>` keeps working without code changes outside the package:

```python
__all__ = [
    # Production-facing API
    "TDMPC2State",
    "make_plan_batched",
    "make_update_step",
    "Encoder", "Dynamics", "Reward", "QEnsemble", "PolicyPrior",
    "build_world_model_optimizer", "build_policy_optimizer",
    # Internal symbols imported by tests/test_tdmpc2.py — re-exported for back-compat
    "NormedLinear",
    "bound_log_std", "squash_log_prob_correction", "gaussian_log_prob",
    "compute_scaled_entropy",
    "compute_all_latents", "compute_td_target",
    "world_model_loss", "policy_loss",
    "mppi_rollout", "mppi_iteration", "sample_pi_trajectories",
    "init_mppi_mean", "init_mppi_mean_batched",
    "gumbel_sample_elite", "plan",
]
```

**Audit before plan execution**: re-verify the test import list with:
```bash
grep -oE "from jax_rl.algos.tdmpc2 import [A-Za-z_, ]+" tests/test_tdmpc2.py \
  | sed 's/from jax_rl.algos.tdmpc2 import //' | tr ',' '\n' | tr -d ' ' | sort -u
```
If new symbols appear (test file was edited since plan was written), add them to `__all__` before continuing.

**Code section line ranges in current `jax_rl/algos/tdmpc2.py`:**

| Section | Lines | Goes into |
|---|---|---|
| imports + activations (mish) | 1-26 | every module imports `mish` from networks (or all into networks.py) |
| NormedLinear + Encoder + Dynamics + Reward + QHead + QEnsemble | 28-176 | `networks.py` |
| Policy helpers (bound_log_std, squash_log_prob_correction, gaussian_log_prob) | 177-209 | `networks.py` |
| PolicyPrior | 211-254 | `networks.py` |
| compute_all_latents | 255-289 | `losses.py` |
| compute_td_target | 290-374 | `losses.py` |
| world_model_loss | 376-503 | `losses.py` |
| compute_scaled_entropy + policy_loss | 505-599 | `losses.py` |
| mppi_rollout, mppi_iteration, sample_pi_trajectories, init_mppi_mean(_batched) | 601-795 | `mppi.py` |
| gumbel_sample_elite, plan, make_plan_batched | 797-908 | `mppi.py` |
| TDMPC2State, build_world_model_optimizer, build_policy_optimizer | 910-975 | `agent.py` |
| make_update_step | 977-end | `agent.py` |

**Validation gate** (used after every code-move task):
```bash
PYTHONPATH=$PWD XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/train_tdmpc2.py --env CheetahRun --total-timesteps 3000 \
  --seed 0 --num-envs 8 --eval-every 100000 --ckpt-dir .temp/refactor_smoke
```

Then diff `step=N L_world=X L_policy=Y` lines against a baseline captured pre-refactor (Task 0). Tolerance: `atol=1e-5` for L_world / L_policy values.

---

## Task 0: Capture baseline + create scratch dir

**Files:**
- Read: `jax_rl/algos/tdmpc2.py`
- Create: `.temp/refactor_baseline/`

- [ ] **Step 1: Run baseline smoke before any code changes**

```bash
rm -rf .temp/refactor_baseline && mkdir -p .temp/refactor_baseline
PYTHONPATH=$PWD XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/train_tdmpc2.py --env CheetahRun --total-timesteps 3000 \
  --seed 0 --num-envs 8 --eval-every 100000 --ckpt-dir .temp/refactor_baseline \
  2>&1 | tee .temp/refactor_baseline/run.log
```

Expected: completes; final line shows `[tdmpc2] step=3_004 EVAL mppi=...`. Capture all `step=N L_world=... L_policy=...` lines into `.temp/refactor_baseline/loss_lines.txt`:

```bash
grep -E "^\[tdmpc2\] step=[0-9_]+ L_world" .temp/refactor_baseline/run.log > .temp/refactor_baseline/loss_lines.txt
wc -l .temp/refactor_baseline/loss_lines.txt    # ~100 lines
```

- [ ] **Step 2: Verify smoke output is reasonable**

Check the EVAL line at the end shows mppi > 50 (not zero/NaN). If suspicious, abort and investigate before proceeding.

- [ ] **Step 3: Commit baseline log (gitignored .temp/ — actually skip commit)**

`.temp/` is gitignored. Don't commit. Just keep the baseline file local for the rest of the refactor.

---

## Task 0.5: Capture baseline test-pass list

Cheap insurance against silent test breakage between refactor steps. Cost: ~30s.

**Files:**
- Read: `tests/test_tdmpc2.py`
- Output: `.temp/refactor_baseline/test_pass.txt`

- [ ] **Step 1: Run pytest, capture pass list**

```bash
uv run pytest tests/test_tdmpc2.py -v --no-header 2>&1 \
  | tee .temp/refactor_baseline/test_run.log \
  | grep -E "PASSED|FAILED|ERROR" \
  | sort > .temp/refactor_baseline/test_pass.txt
wc -l .temp/refactor_baseline/test_pass.txt
```

Capture the pass count. If any tests FAIL or ERROR pre-refactor, document them — the refactor is not responsible for fixing them, but they should remain in the same status post-refactor.

- [ ] **Step 2: After every code-move task (Tasks 2-5)**

Re-run the same command. If the new pass list diverges (a test that was passing now fails / errors at import), the refactor introduced a regression — usually a missing symbol in `__init__.py`. Fix immediately before committing.

```bash
uv run pytest tests/test_tdmpc2.py -v --no-header 2>&1 \
  | grep -E "PASSED|FAILED|ERROR" \
  | sort > .temp/refactor_smoke/test_pass.txt
diff .temp/refactor_baseline/test_pass.txt .temp/refactor_smoke/test_pass.txt
```

Expected: `diff` empty. (Don't include in every task body to keep them tight; just remember the rule: no test regressions.)

---

## Task 1: Set up package directory + alias module

Goal: convert `tdmpc2.py` to `tdmpc2_old.py` (a hidden alias) and create `tdmpc2/__init__.py` that re-exports from it. Net effect: zero behavior change, but now we have a sub-package to incrementally fill.

**Files:**
- Rename: `jax_rl/algos/tdmpc2.py` → `jax_rl/algos/tdmpc2_old.py`
- Create: `jax_rl/algos/tdmpc2/__init__.py`

- [ ] **Step 1: Rename via git mv (preserve history)**

```bash
git mv jax_rl/algos/tdmpc2.py jax_rl/algos/tdmpc2_old.py
```

- [ ] **Step 2: Create `tdmpc2/__init__.py` with re-exports**

```python
"""TDMPC2 sub-package. See .superpowers/specs/2026-04-26-tdmpc2-refactor-design.md.

Re-exports the public API from the legacy implementation file. As the refactor
progresses, individual symbols move from `tdmpc2_old.py` into focused modules
(`networks.py`, `losses.py`, `mppi.py`, `agent.py`); this `__init__.py` keeps
the import path `from jax_rl.algos.tdmpc2 import X` stable throughout.
"""

from jax_rl.algos.tdmpc2_old import (
    TDMPC2State,
    make_plan_batched,
    make_update_step,
    Encoder,
    Dynamics,
    Reward,
    QEnsemble,
    PolicyPrior,
    build_world_model_optimizer,
    build_policy_optimizer,
)

__all__ = [
    "TDMPC2State",
    "make_plan_batched",
    "make_update_step",
    "Encoder",
    "Dynamics",
    "Reward",
    "QEnsemble",
    "PolicyPrior",
    "build_world_model_optimizer",
    "build_policy_optimizer",
]
```

- [ ] **Step 3: Run validation smoke + diff**

```bash
rm -rf .temp/refactor_smoke && mkdir -p .temp/refactor_smoke
PYTHONPATH=$PWD XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/train_tdmpc2.py --env CheetahRun --total-timesteps 3000 \
  --seed 0 --num-envs 8 --eval-every 100000 --ckpt-dir .temp/refactor_smoke \
  2>&1 | tee .temp/refactor_smoke/run.log

grep -E "^\[tdmpc2\] step=[0-9_]+ L_world" .temp/refactor_smoke/run.log > .temp/refactor_smoke/loss_lines.txt
diff .temp/refactor_baseline/loss_lines.txt .temp/refactor_smoke/loss_lines.txt
```

Expected: `diff` exits 0 (byte-identical). If diff non-empty, abort: investigate import-order or module-identity issue.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/algos/tdmpc2_old.py jax_rl/algos/tdmpc2/__init__.py
git commit -m "refactor(tdmpc2): create sub-package skeleton with re-exports

Renamed tdmpc2.py → tdmpc2_old.py. New tdmpc2/__init__.py re-exports
the 10 public symbols. Subsequent commits move code section-by-section
out of tdmpc2_old.py into focused modules (networks/losses/mppi/agent).

Validated: 3k Cheetah smoke under deterministic XLA flag is byte-ID."
```

---

## Task 2: Move networks → `networks.py`

**Files:**
- Create: `jax_rl/algos/tdmpc2/networks.py`
- Modify: `jax_rl/algos/tdmpc2_old.py` (delete moved sections)
- Modify: `jax_rl/algos/tdmpc2/__init__.py` (re-route imports)

- [ ] **Step 1: Create `networks.py` with the moved code**

Copy from `tdmpc2_old.py` lines 1-254 (including imports + activations + 5 networks + policy helpers + PolicyPrior). Specifically:

- File header docstring (write fresh: "Networks: Encoder, Dynamics, Reward, QHead, QEnsemble, PolicyPrior. Plus building blocks (NormedLinear, simnorm, mish) and policy helpers (bound_log_std, squash_log_prob_correction, gaussian_log_prob, compute_scaled_entropy).")
- All imports needed (jax, jax.numpy as jnp, flax.linen as nn). Check what's actually used.
- `mish` (line 23)
- `NormedLinear` (line 30) — uses `nn.Dense, nn.LayerNorm, nn.Dropout, mish`
- `simnorm` helper — check if it's defined in tdmpc2.py or imported. If imported from `jax_rl.utils.simnorm`, just import it.
- `Encoder` (line 53) — add `# TODO(vision): For pixel obs, swap with a CNN-bodied subclass. Source impl: nicklashansen/tdmpc2 common/world_model.py:enc_pixels.` comment block ABOVE the class.
- `Dynamics` (line 77)
- `Reward` (line 99)
- `QHead` (line 121)
- `QEnsemble` (line 148)
- `bound_log_std` (line 179)
- `squash_log_prob_correction` (line 187)
- `gaussian_log_prob` (line 200)
- `PolicyPrior` (line 213)
- Move `compute_scaled_entropy` here too (line 507) — it's a policy helper.

Add `__all__ = ["mish", "NormedLinear", "Encoder", "Dynamics", "Reward", "QHead", "QEnsemble", "PolicyPrior", "bound_log_std", "squash_log_prob_correction", "gaussian_log_prob", "compute_scaled_entropy"]`.

- [ ] **Step 2: Delete moved sections from `tdmpc2_old.py`**

Remove lines 21-254 (everything from `# Activations` through end of `PolicyPrior` class) AND the `compute_scaled_entropy` function at line 507.

Add `from jax_rl.algos.tdmpc2.networks import (...)` at top of `tdmpc2_old.py` for any symbols still referenced internally — verify by grepping for their usages within the remaining `tdmpc2_old.py` content.

- [ ] **Step 3: Update `tdmpc2/__init__.py` to import networks from new location**

Re-route ALL network-related symbols (production + test-internal) from `networks` instead of `tdmpc2_old`:

```python
# top of __init__.py — supersedes Task 1's body
from jax_rl.algos.tdmpc2.networks import (
    NormedLinear,
    Encoder, Dynamics, Reward, QHead, QEnsemble, PolicyPrior,
    bound_log_std, squash_log_prob_correction, gaussian_log_prob,
    compute_scaled_entropy,
)
from jax_rl.algos.tdmpc2_old import (
    TDMPC2State,
    make_plan_batched,
    make_update_step,
    build_world_model_optimizer,
    build_policy_optimizer,
    # internal symbols still in tdmpc2_old (move out in later tasks):
    compute_all_latents, compute_td_target,
    world_model_loss, policy_loss,
    mppi_rollout, mppi_iteration, sample_pi_trajectories,
    init_mppi_mean, init_mppi_mean_batched,
    gumbel_sample_elite, plan,
)
```

Keep `__all__` from Task 1 unchanged (it's the full list — covers symbols regardless of which file they currently live in).

- [ ] **Step 4: Run validation smoke + diff**

```bash
rm -rf .temp/refactor_smoke && mkdir -p .temp/refactor_smoke
PYTHONPATH=$PWD XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/train_tdmpc2.py --env CheetahRun --total-timesteps 3000 \
  --seed 0 --num-envs 8 --eval-every 100000 --ckpt-dir .temp/refactor_smoke \
  2>&1 | tee .temp/refactor_smoke/run.log

grep -E "^\[tdmpc2\] step=[0-9_]+ L_world" .temp/refactor_smoke/run.log > .temp/refactor_smoke/loss_lines.txt
diff .temp/refactor_baseline/loss_lines.txt .temp/refactor_smoke/loss_lines.txt
```

Expected: `diff` exits 0.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/algos/tdmpc2/networks.py jax_rl/algos/tdmpc2/__init__.py jax_rl/algos/tdmpc2_old.py
git commit -m "refactor(tdmpc2): move networks → tdmpc2/networks.py"
```

---

## Task 3: Move losses → `losses.py`

**Files:**
- Create: `jax_rl/algos/tdmpc2/losses.py`
- Modify: `jax_rl/algos/tdmpc2_old.py`
- Modify: `jax_rl/algos/tdmpc2/__init__.py` (only if losses are exported — they're internal, so likely no __init__ change)

- [ ] **Step 1: Create `losses.py` with moved code**

File header docstring: "TD target + world-model + policy losses for TD-MPC2."

Move:
- `compute_all_latents` (line 257)
- `compute_td_target` (line 292) — apply Bug A fix already in (`terminated = jnp.clip(batch["dones"] - batch["truncations"], 0.0, 1.0)`)
- `world_model_loss` (line 378) — uses dropout key splitting (Bug 2 fix), keep that
- `policy_loss` (line 517)

Imports needed: `jax, jax.numpy as jnp`, `from jax_rl.utils.twohot import two_hot_inv, two_hot_ce_loss`, `from jax_rl.utils.qscale import qscale_apply, QScaleState`, AND `from jax_rl.algos.tdmpc2.networks import compute_scaled_entropy` (used by `policy_loss`).

Add `__all__` only for symbols imported externally (`compute_td_target`, `world_model_loss`, `policy_loss`, `compute_all_latents` — check who imports them; if only `agent.py` then __all__ optional but helps doc).

- [ ] **Step 2: Delete moved sections from `tdmpc2_old.py`**

Remove lines 255-599 (compute_all_latents, compute_td_target, world_model_loss, compute_scaled_entropy, policy_loss). compute_scaled_entropy is already in networks.py from Task 2 — verify it's no longer in tdmpc2_old.py.

Add `from jax_rl.algos.tdmpc2.losses import compute_td_target, world_model_loss, policy_loss, compute_all_latents` at top of tdmpc2_old.py if needed (check if `make_update_step` references them — it does).

- [ ] **Step 3: Update `tdmpc2/__init__.py` — route losses from new location**

Replace the lines re-exporting `compute_all_latents`, `compute_td_target`, `world_model_loss`, `policy_loss` (currently from `tdmpc2_old`) with imports from `tdmpc2.losses`:

```python
from jax_rl.algos.tdmpc2.losses import (
    compute_all_latents, compute_td_target,
    world_model_loss, policy_loss,
)
```

Remove those four names from the `from jax_rl.algos.tdmpc2_old import (...)` block.

- [ ] **Step 4: Run validation smoke + diff**

(Same as Task 2 Step 4.) Expected: `diff` exits 0.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/algos/tdmpc2/losses.py jax_rl/algos/tdmpc2/__init__.py jax_rl/algos/tdmpc2_old.py
git commit -m "refactor(tdmpc2): move losses → tdmpc2/losses.py"
```

---

## Task 4: Move MPPI → `mppi.py`

**Files:**
- Create: `jax_rl/algos/tdmpc2/mppi.py`
- Modify: `jax_rl/algos/tdmpc2_old.py`
- Modify: `jax_rl/algos/tdmpc2/__init__.py` (re-route `make_plan_batched`)

- [ ] **Step 1: Create `mppi.py` with moved code**

File header docstring: "MPPI planner: trajectory rollout, elite weighting, action selection, batched plan_fn."

Move:
- `mppi_rollout` (line 604)
- `mppi_iteration` (line 660)
- `sample_pi_trajectories` (line 727)
- `init_mppi_mean` (line 767)
- `init_mppi_mean_batched` (line 782)
- `gumbel_sample_elite` (line 800)
- `plan` (line 818)
- `make_plan_batched` (line 881)

Strip the rotted comment "# F4: plan() + gumbel_sample_elite + plan_batched" — replace with substantive section comment if needed.

Imports: `jax, jax.numpy as jnp`, `from jax_rl.utils.twohot import two_hot_inv`. Networks (Dynamics, Reward, QEnsemble, PolicyPrior) are passed in as arguments so no import.

Add `__all__ = ["make_plan_batched"]` (only public export). Internal funcs not in __all__.

- [ ] **Step 2: Delete moved sections from `tdmpc2_old.py`**

Remove lines 601-908. Add `from jax_rl.algos.tdmpc2.mppi import make_plan_batched` if `tdmpc2_old.py` still references it (it might not — `make_plan_batched` is exported by `__init__.py` and used by external code, not by remaining content).

- [ ] **Step 3: Update `tdmpc2/__init__.py` — route mppi from new location**

Add MPPI internals (used by tests):

```python
from jax_rl.algos.tdmpc2.mppi import (
    make_plan_batched,
    mppi_rollout, mppi_iteration, sample_pi_trajectories,
    init_mppi_mean, init_mppi_mean_batched,
    gumbel_sample_elite, plan,
)
```

Remove the corresponding 7 names from the `from jax_rl.algos.tdmpc2_old import (...)` block. Networks + losses imports unchanged from prior tasks.

- [ ] **Step 4: Run validation smoke + diff**

(Same as Task 2 Step 4.) Expected: `diff` exits 0.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/algos/tdmpc2/mppi.py jax_rl/algos/tdmpc2/__init__.py jax_rl/algos/tdmpc2_old.py
git commit -m "refactor(tdmpc2): move MPPI planner → tdmpc2/mppi.py"
```

---

## Task 5: Move agent (state + optimizers + update_step) → `agent.py`

**Files:**
- Create: `jax_rl/algos/tdmpc2/agent.py`
- Modify: `jax_rl/algos/tdmpc2_old.py` (becomes empty — delete in next task)
- Modify: `jax_rl/algos/tdmpc2/__init__.py`

- [ ] **Step 1: Create `agent.py` with moved code**

File header docstring: "TD-MPC2 training state + optimizers + update_step factory."

Move:
- `TDMPC2State` dataclass (line 912)
- `build_world_model_optimizer` (line 940)
- `build_policy_optimizer` (line 966)
- `make_update_step` (line 979)

Imports needed:
```python
import jax
import jax.numpy as jnp
import flax
import optax
from jax_rl.algos.tdmpc2.losses import world_model_loss, policy_loss
from jax_rl.utils.qscale import QScaleState, qscale_update
from jax_rl.utils.twohot import two_hot_inv  # used for qscale recompute Q decode
```

Networks (`Encoder`, `Dynamics`, etc.) are passed as arguments to `make_update_step`, so no network import needed.

Add `__all__ = ["TDMPC2State", "make_update_step", "build_world_model_optimizer", "build_policy_optimizer"]`.

- [ ] **Step 2: Verify `tdmpc2_old.py` is empty (or only has unused imports)**

```bash
wc -l jax_rl/algos/tdmpc2_old.py
```

Expected: very small (just imports + maybe a comment). If non-trivial code remains, identify what was missed and route accordingly.

- [ ] **Step 3: Update `tdmpc2/__init__.py` — drop tdmpc2_old import**

```python
"""TDMPC2 — model-based RL with learned world model + MPPI planner.

Re-exports the public API from focused submodules:
- networks: Encoder, Dynamics, Reward, QEnsemble, PolicyPrior
- mppi: make_plan_batched (the planner factory)
- agent: TDMPC2State, make_update_step, build_*_optimizer
"""

from jax_rl.algos.tdmpc2.networks import (
    Encoder, Dynamics, Reward, QEnsemble, PolicyPrior,
)
from jax_rl.algos.tdmpc2.mppi import make_plan_batched
from jax_rl.algos.tdmpc2.agent import (
    TDMPC2State, make_update_step,
    build_world_model_optimizer, build_policy_optimizer,
)

__all__ = [
    "TDMPC2State",
    "make_plan_batched",
    "make_update_step",
    "Encoder", "Dynamics", "Reward", "QEnsemble", "PolicyPrior",
    "build_world_model_optimizer", "build_policy_optimizer",
]
```

- [ ] **Step 4: Run validation smoke + diff**

(Same as Task 2 Step 4.) Expected: `diff` exits 0.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/algos/tdmpc2/agent.py jax_rl/algos/tdmpc2/__init__.py jax_rl/algos/tdmpc2_old.py
git commit -m "refactor(tdmpc2): move state + optimizers + update_step → tdmpc2/agent.py"
```

---

## Task 6: Delete `tdmpc2_old.py`

**Files:**
- Delete: `jax_rl/algos/tdmpc2_old.py`

- [ ] **Step 1: Verify `tdmpc2_old.py` is empty / has nothing referenced externally**

```bash
cat jax_rl/algos/tdmpc2_old.py
grep -rn "tdmpc2_old" --include="*.py" .
```

Expected output of the grep: nothing (no external referrers). The previous task (Task 5 Step 3) removed the last `from jax_rl.algos.tdmpc2_old import ...` line from `tdmpc2/__init__.py`. If grep shows hits, `__init__.py` was not fully migrated — go back to Task 5.

- [ ] **Step 2: Delete the file**

```bash
git rm jax_rl/algos/tdmpc2_old.py
```

- [ ] **Step 3: Run validation smoke + diff**

(Same as Task 2 Step 4.) Expected: `diff` exits 0.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor(tdmpc2): drop legacy tdmpc2_old.py — migration complete

jax_rl/algos/tdmpc2/ sub-package now holds all code:
- networks.py: 5 networks + helpers
- losses.py: TD target + world-model + policy losses
- mppi.py: planner
- agent.py: state + optimizers + update_step factory

External import paths unchanged via __init__.py re-exports."
```

---

## Task 7: Drop unused target params (encoder/dynamics/reward)

Per the spec: only `q_ensemble_target_params` is read by `compute_td_target`. The other three target params are EMA'd but never used. Removing saves ~25% of state memory.

**Files:**
- Modify: `jax_rl/algos/tdmpc2/agent.py` (TDMPC2State, make_update_step)
- Modify: `jax_rl/algos/tdmpc2_runtime.py` (init_train_state)

- [ ] **Step 1: Audit — confirm encoder/dynamics/reward target params are unread**

```bash
grep -n "encoder_target_params\|dynamics_target_params\|reward_target_params" jax_rl/algos/tdmpc2/ jax_rl/algos/tdmpc2_runtime.py scripts/
```

Expected: only assignments + EMA updates show up; no reads (no `state.encoder_target_params` access expression on the right-hand side of anything).

If any external script reads them, abort and surface to spec author — the audit's claim was wrong.

- [ ] **Step 2: Modify `TDMPC2State` in `agent.py`**

Remove these three fields:
```python
encoder_target_params: ...
dynamics_target_params: ...
reward_target_params: ...
```

Keep `q_ensemble_target_params` (it IS used by compute_td_target).

- [ ] **Step 3: Modify `make_update_step` in `agent.py`**

Remove the EMA tree-map calls for the three dropped target params. Keep the q_ensemble target EMA. Search for the EMA block (it was originally around line 1076-1079 of the old monofile).

In the `state.replace(...)` call at the end, remove the three `<x>_target_params=...` keyword args.

- [ ] **Step 4: Modify `init_train_state` in `tdmpc2_runtime.py`**

Remove the three `target_<x> = jax.tree_util.tree_map(lambda x: x, <x>_params)` lines. Remove the three `<x>_target_params=target_<x>,` kwargs in the `TDMPC2State(...)` constructor call. Keep `q_ensemble_target_params=target_q,`.

- [ ] **Step 4.5: ALSO modify `load_params_into_state` in `tdmpc2_runtime.py`**

This is a separate function in the same file (around line 207-244 pre-edit). It also constructs a `TDMPC2State` and currently sets:
```python
encoder_target_params=wm_loaded["encoder"],
dynamics_target_params=wm_loaded["dynamics"],
reward_target_params=wm_loaded["reward"],
q_ensemble_target_params=wm_loaded["q_ensemble"],
```

Remove the first three. Keep the q_ensemble line. If left in place, the `state.replace(...)` call (or `TDMPC2State(...)` constructor) will receive unknown kwargs and crash any script that loads a checkpoint (`scripts/eval_tdmpc2.py`, `scripts/record_video_tdmpc2.py`).

Verify by grepping for `_target_params=` in the file after edits — only `q_ensemble_target_params=` should remain.

- [ ] **Step 4.6: Sanity-check `_save_checkpoint` does NOT read these fields**

```bash
grep -n "encoder_target\|dynamics_target\|reward_target\|q_ensemble_target" scripts/train_tdmpc2.py
```

Expected: no matches. Save side flattens only online params (`policy_params` + `encoder/dynamics/reward/q_ensemble_params`), so checkpoint format is unchanged. If grep does return hits, escalate — checkpoint compat needs handling.

- [ ] **Step 5: Run validation smoke**

This change is NOT byte-identical (different state shape → different tree-flatten order → JIT recompile → potential different fp accumulation). Smoke must still complete and produce reasonable mppi value.

```bash
rm -rf .temp/refactor_smoke && mkdir -p .temp/refactor_smoke
PYTHONPATH=$PWD XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/train_tdmpc2.py --env CheetahRun --total-timesteps 3000 \
  --seed 0 --num-envs 8 --eval-every 100000 --ckpt-dir .temp/refactor_smoke \
  2>&1 | tee .temp/refactor_smoke/run.log

tail -3 .temp/refactor_smoke/run.log
```

Expected: completes without error; final EVAL line shows mppi > 50.

- [ ] **Step 6: Commit**

```bash
git add jax_rl/algos/tdmpc2/agent.py jax_rl/algos/tdmpc2_runtime.py
git commit -m "refactor(tdmpc2): drop unused encoder/dynamics/reward target params

Source maintains a target only for the Q ensemble (world_model.py:46);
encoder/dynamics/reward targets were never read by compute_td_target
(audit at 2026-04-25 confirmed). Remove the three unused fields from
TDMPC2State, the EMA updates from make_update_step, and the init paths
from tdmpc2_runtime.init_train_state.

Saves ~25% of state memory. Behavior unchanged on smoke (different
JIT cache vs pre-refactor; not byte-ID but mppi within band)."
```

---

## Task 8: Strip rotted comments

Subagent-dev-era comments referencing implementation iterations no longer exist as code — they confuse new readers. Remove specific phrases.

**Files:**
- Modify: `jax_rl/algos/tdmpc2/{networks,losses,mppi,agent}.py`

- [ ] **Step 1: Find rotted comments**

```bash
grep -n "iter-3\|iter-4\|Module H4\|Module F\|F2:\|F4:\|Phase H\|TODO(claude)\|H4: F4\|spec review" jax_rl/algos/tdmpc2/
```

- [ ] **Step 2: Delete or rewrite each match**

For each match:
- If comment references "iter-N" / "Phase H" / "Module H4" → delete the comment (the reference is meaningless).
- If comment is substantive ("# this fixes X") and the iteration phrase is incidental → keep the substance, drop the phrase.

Don't delete comments that explain why code does something non-obvious (e.g., "OUTER NEGATIVE wraps both entropy bonus AND Q term" — keep).

- [ ] **Step 3: Run validation smoke**

(Comment-only changes — should be byte-identical.)

```bash
rm -rf .temp/refactor_smoke && mkdir -p .temp/refactor_smoke
# ... same smoke command ...
diff .temp/refactor_baseline/loss_lines.txt .temp/refactor_smoke/loss_lines.txt
```

Expected: `diff` exits 0.

(NOTE: byte-identity expected here ONLY if Task 7 was NOT yet applied. If Task 7 done first, baseline is already drifted — relax to "smoke completes; mppi reasonable".)

- [ ] **Step 4: Commit**

```bash
git add jax_rl/algos/tdmpc2/
git commit -m "refactor(tdmpc2): strip subagent-dev-era comments (iter-N, Phase H, etc)"
```

---

## Task 9: Inline pre-listed single-use helpers

Pre-listed candidates (do NOT expand the list mid-task to prevent scope creep):

- `_pipe_obs` in `tdmpc2_runtime.py` — used 3+ times across train + eval + record_video — KEEP (do not inline).
- `compute_scaled_entropy` in `networks.py` — used once in `policy_loss` — INLINE candidate, but it has a docstring explaining the formula → KEEP (clarity).
- `init_mppi_mean` in `mppi.py` — used by `init_mppi_mean_batched` (1 call) — KEEP (helper has its own clear purpose).

Outcome: **no inlining is justified after pre-listing**. The candidates either have multiple users or carry explanatory weight. Skip this task.

- [ ] **Step 1: Verify the pre-listed candidates are truly the only single-use helpers**

```bash
# For each function in networks/losses/mppi/agent, count call sites:
grep -E "^def |^class " jax_rl/algos/tdmpc2/{networks,losses,mppi,agent}.py | wc -l
```

If this audit reveals new single-use helpers not anticipated in the spec, do NOT inline them in this task. Surface to spec author for re-scoping.

- [ ] **Step 2: Document the decision in commit message (no code change)**

Skip the commit. This task closes as "no action needed" — the inline-helpers cleanup item from the spec resolved to "none after audit."

---

## Task 10: Final validation + cleanup

**Files:**
- Read: all
- Run: `tests/` if any TDMPC2 tests exist
- Run: full smoke battery

- [ ] **Step 1: Verify final file structure**

```bash
ls jax_rl/algos/tdmpc2/
wc -l jax_rl/algos/tdmpc2/*.py
ls jax_rl/algos/tdmpc2_runtime.py    # should still exist
ls jax_rl/algos/tdmpc2.py            # should NOT exist
ls jax_rl/algos/tdmpc2_old.py        # should NOT exist
```

Expected:
```
jax_rl/algos/tdmpc2/__init__.py    ~30 LOC
jax_rl/algos/tdmpc2/networks.py    ~280 LOC
jax_rl/algos/tdmpc2/losses.py      ~250 LOC
jax_rl/algos/tdmpc2/mppi.py        ~310 LOC
jax_rl/algos/tdmpc2/agent.py       ~220 LOC
```

(Numbers approximate; actual will depend on docstrings, blank-line normalization.)

- [ ] **Step 2: Run all four scripts end-to-end smoke**

```bash
# train smoke (3k)
PYTHONPATH=$PWD XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/train_tdmpc2.py --env CheetahRun --total-timesteps 3000 \
  --seed 0 --num-envs 8 --eval-every 100000 --ckpt-dir .temp/refactor_final 2>&1 | tail -5

# eval (uses the just-saved ckpt)
PYTHONPATH=$PWD XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/eval_tdmpc2.py --env CheetahRun \
  --load-ckpt .temp/refactor_final --num-evals 1 --seed 100 2>&1 | tail -5

# determinism check
PYTHONPATH=$PWD XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 \
  uv run python scripts/check_tdmpc2_determinism.py --check init 2>&1 | tail -10

# record_video
PYTHONPATH=$PWD XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  uv run python scripts/record_video_tdmpc2.py --env CheetahRun \
  --load-ckpt .temp/refactor_final --mode mppi --num-steps 100 --seed 100 2>&1 | tail -5
```

All four must complete without error.

- [ ] **Step 3: Run any existing TDMPC2 tests**

```bash
uv run pytest tests/ -k tdmpc -v 2>&1 | tail -20
```

Expected: all tests pass (or, if no tests match, output shows "no tests ran" — that's acceptable, this refactor is type-driven not test-driven).

- [ ] **Step 4: Final commit if any cleanup needed**

If the final file LOC numbers diverge from spec (~280 vs 300) or there are tiny nits, commit those now. Otherwise this task is the read-and-verify gate; no code commit needed.

```bash
# only if needed:
git add ...
git commit -m "refactor(tdmpc2): final cleanup post-validation"
```

- [ ] **Step 5: Update doc — point readers at the new layout**

In `.context/lessons/tdmpc2.md`, find any references to specific line numbers in `jax_rl/algos/tdmpc2.py` and update to module:function references. Also add a one-line note at the top: "Code split into `jax_rl/algos/tdmpc2/{networks,losses,mppi,agent}.py` as of 2026-04-26."

```bash
git add .context/lessons/tdmpc2.md
git commit -m "docs(lessons): point tdmpc2 lesson at new sub-package paths"
```

---

## Risk register

- **Import circular**: networks.py → no internal imports. losses.py → networks (compute_scaled_entropy). mppi.py → no internal imports (modules passed as args). agent.py → losses. No cycles.
- **JIT cache miss after Task 7**: TDMPC2State shape changes when target fields drop → JIT must recompile. Acceptable; smoke will run hot-cache for the new shape.
- **`tdmpc2_old.py` accidentally left in**: Task 6 explicitly deletes; smoke runs against the new paths. If forgotten, `git status` at end of session will show stale file.
- **Reviewer-flagged scope creep**: Task 9 pre-locks the inline list; if new candidates appear during audit, surface rather than execute.

## Success criteria

1. `jax_rl/algos/tdmpc2/` directory exists with 5 files (`__init__.py` + 4 modules).
2. `jax_rl/algos/tdmpc2.py` and `tdmpc2_old.py` no longer exist.
3. All 4 scripts (train, eval, record_video, check_determinism) run end-to-end without import or runtime errors.
4. Pre-Task-7 commits are byte-ID with baseline loss lines.
5. Post-Task-7 smoke produces a reasonable mppi value (>50 at step 3004 for CheetahRun seed 0).
6. `tdmpc2_runtime.py` updated coherently with the agent state shape change.
7. `.context/lessons/tdmpc2.md` references new paths.
