# Test-Suite Cleanup — Execution Playbook

> Companion to `.context/audits/2026-04-27_test_suite_audit.md`. The audit
> is the *what + why*; this is the *exact diff + verification command*
> for an executing agent.
>
> **Read the audit first.** The playbook assumes you've internalized the
> marker taxonomy and the categorization table.

---

## How to use this doc

Each task is self-contained and independently mergeable. Recommended
order is the numbered list below — later tasks assume earlier ones are
done. Within a task:

- **Files:** literal paths + line numbers
- **Diff:** old → new, exact strings
- **Verify with:** command that proves done
- **Done when:** acceptance criteria
- **Gotchas:** things that will trip you, ranked by surprise

**Hard rules carried from `.context/lessons/algo_port_protocol.md`:**

- `uv run python <cmd>`, never bare `python` / `python3`
- Don't push to origin without owner approval
- Don't break the docs-drift canaries
  (`tests/test_docs_drift.py`, `tests/test_docs_code_blocks.py`) —
  default-lane pass count must not drop after each task
- Within-run equivalence checks only on GPU; never gate on cross-run
  bit-identity

---

## Phase ordering and dependencies

```
Task 1 (markers)            ─────────────────► Task 9 (CI YAML)
       │                                        ▲
       └──► Task 6 (off-policy loop param) ─────┘
       └──► Task 5 (synthetic policy_runner)  ──┤
       └──► Task 7 (DR persistence regression) ─┤
       └──► Task 8 (resume warmup behavioral) ──┘

Task 2 (move PRNGKey to fixtures)   — independent, low priority
Task 3 (delete h{1..5}.py)          — independent
Task 4 (rewrite or delete i3 isolation) — independent
Task 10 (docs/contributing.md)      — after Task 1
```

Tasks 1, 3, 4 are pure cleanup. Do them first (single small commit
each), verify default-lane pass count holds, then tackle 5-8.

---

## Task 1 — Add marker taxonomy + retarget default lane

**Cost:** 1 hr (most of it is per-file marker stamping).

### Files

- `pyproject.toml` (1 edit)
- 9 test files get module-level `pytestmark` (see table below)
- `tests/test_deploy_e2e.py` gets per-test marks (2 tests)

### Diff 1: `pyproject.toml` lines 71-77

**Old (verified — file is short, lines 71-77 are the entire pytest
config):**
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests", "deploy"]
markers = [
    "slow: marks tests as slow / GPU-bound (deselected by default; run with -m slow)",
]
addopts = "-m 'not slow'"
```

**New:**
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests", "deploy"]
markers = [
    "gpu: test needs CUDA (MJX physics, JAX device arrays beyond CPU)",
    "warp: test needs warp-lang library + GPU (implies gpu)",
    "go2: test needs Go2 Warp env (implies warp + gpu)",
    "deploy: test exercises deploy/ runtime",
    "network: test needs internet (e.g. arxiv, HF datasets)",
    "slow: test takes >30s wall-clock (can be CPU or GPU)",
]
addopts = "-m 'not (gpu or warp or go2 or network or slow)'"
```

### Diff 2: stamp module-level `pytestmark` on 9 GPU files

For each file in this table, insert `pytestmark` immediately after the
last `import` line at module scope (typically near line 8-15). If
multiple marks apply, use a list.

| File | New top-of-file addition |
|---|---|
| `tests/test_env_bundle.py` | `import pytest`<br>`pytestmark = pytest.mark.gpu` |
| `tests/test_offpolicy_loop.py` | already imports pytest at line 15<br>`pytestmark = pytest.mark.gpu`  ← add after line 15 |
| `tests/test_frame_stack_wrapper.py` | already imports pytest at line 4<br>`pytestmark = [pytest.mark.gpu, pytest.mark.warp]` |
| `tests/test_training_wrappers.py` | already imports pytest<br>`pytestmark = pytest.mark.gpu` |
| `tests/test_go2_warp_env.py` | `pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]` |
| `tests/test_go2_warp_curriculum_env.py` | same |
| `tests/test_go2_bongo_env.py` | same |
| `tests/test_go2_bongo_contraction_obs.py` | same |
| `tests/test_terrain_curriculum_dr_wrapper.py` | same |

Canonical insertion (example for `test_go2_warp_env.py`):

```python
"""..."""
import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]

# ... rest of file unchanged
```

If a file does not yet `import pytest`, add the import. Keep imports
alphabetized as the file already was.

### Diff 3: per-test marks in `tests/test_deploy_e2e.py`

Two tests need GPU+Warp+Go2 and exercise deploy contract. The
`pytest.importorskip("warp")` lines already in those tests are not
enough — they catch missing module, not missing CUDA. Add four marks
above each (matches audit §2 categorization):

**Lines 96 and 124** — prepend four marks each:

```python
@pytest.mark.gpu
@pytest.mark.warp
@pytest.mark.go2
@pytest.mark.deploy
def test_bongo_metadata_uses_handstand_pose_not_home():
    ...

@pytest.mark.gpu
@pytest.mark.warp
@pytest.mark.go2
@pytest.mark.deploy
def test_env_metadata_matches_deploy_constants():
    ...
```

Don't use module-level `pytestmark` for this file — the other 9 tests
in `test_deploy_e2e.py` are CPU-clean and should stay in the default
lane.

**Companion change for the `deploy` marker to be useful:** stamp
`pytestmark = pytest.mark.deploy` at module scope on
`deploy/test_policy_runner.py` (after its `import pytest` line). All 9
tests in that file are deploy-runtime tests by intent. Without this,
`pytest -m deploy` selects only the 2 above, missing the 9
hermetic-synthetic deploy tests that should also be in the lane.

### Diff 4: split `slow` from `network` in `tests/test_docs_drift.py`

Find the existing `@pytest.mark.slow` on `test_arxiv_ids_resolve` (one
of two `@pytest.mark.slow` decorators in the file — the arxiv one).
Change it to `@pytest.mark.network` so it stays excluded from default
but can be run separately.

```bash
grep -nB2 "test_arxiv_ids_resolve\|@pytest.mark.slow" tests/test_docs_drift.py
```

### Verify with

```bash
# 1. Markers registered
uv run python -m pytest --markers | grep -E "gpu|warp|go2|deploy|network|slow"
# Expect: 6 lines

# 2. Default lane: no GPU/Warp/Go2 tests
uv run python -m pytest --collect-only -q | tail -3
# Expect: ~600 collected, ~200 deselected (was 804/15)

# 3. CPU-clean default lane PASSES end-to-end
uv run python -m pytest -q
# Expect: ~600 passed, 0 failed
# (test_offpolicy_loop is now `gpu`-marked → out of default)

# 4. GPU lane is selectable
uv run python -m pytest --collect-only -q -m gpu | tail -3
# Expect: ~200 tests

# 5. Pass count of CPU-clean tests must not drop vs before
uv run python -m pytest -q tests/test_docs_drift.py tests/test_docs_code_blocks.py
# Expect: same count as pre-task
```

### Done when

- [x] Default `pytest -q` runs ~600 tests in 1-2 min, all green.
- [x] `pytest -m gpu` selects roughly 200 tests; `pytest -m "warp or go2"` selects ~70.
- [x] No drop in `test_docs_*` pass count.
- [x] `pytest --markers` lists the 6 markers with descriptions.

### Gotchas

- **`pytest.importorskip("warp")` is a red herring.** It only catches
  *missing module*, not missing CUDA. `warp-lang>=1.12.0` is in
  `pyproject.toml` so the module is always present in the venv —
  `importorskip` never fires. The `gpu` marker is the actual gate.
- **`pytestmark` interacts with parametrize.** If a file uses
  `@pytest.mark.parametrize`, module-level `pytestmark` still applies.
  No conflict; just double-check `test_docs_code_blocks.py` (260
  parametrized cases) is NOT marked gpu — it's CPU-clean.
- **Order of marks in a list doesn't matter** — `[gpu, warp, go2]` and
  `[go2, warp, gpu]` are equivalent for selection purposes.
- **Don't try to make `warp` imply `gpu` automatically.** pytest's
  marker system has no inheritance. Always tag the full set
  (`[gpu, warp]` or `[gpu, warp, go2]`) so the `-m gpu` selector
  catches it.

### Commit

One commit, message:
```
test: add gpu/warp/go2/deploy/network markers; CPU-only default lane

Default pytest lane was running 804 tests including Warp/Go2/MJX (9m39s).
Add module-level pytestmark to 9 GPU-required files; per-test mark on the
2 Warp tests in test_deploy_e2e.py; split slow vs network in
test_docs_drift. Default lane now ~600 hermetic tests in 1-2 min.

Codex P1 (test marker taxonomy underbuilt) closed.
```

---

## Task 2 — Move 9 module-level `KEY = jax.random.PRNGKey(...)` (DEMOTED to P3)

**Cost:** 30 min (mechanical, but pervasive within each file).

**Demoted because:** every offending file passes on CPU and these
violations only matter on GPU machines under simultaneous collection
of all 9 files. After Task 1, default lane excludes most CUDA-touching
tests, so the OOM cascade scenario barely exists. Fix only if a future
session hits collection-time OOM.

### If you do it

The mechanical pattern: replace module-level `KEY` with a fixture and
add `key` parameter to every test that used it.

**Don't try to be clever.** Each file uses `KEY` 5-40 times. The
cleanest mechanical change:

1. Delete module-level `KEY = jax.random.PRNGKey(N)` line.
2. Add at top of file (after imports):
   ```python
   @pytest.fixture
   def key():
       return jax.random.PRNGKey(N)  # use original seed
   ```
3. For each `def test_foo():` that uses `KEY`, change signature to
   `def test_foo(key):`.
4. Inside body: `KEY` → `key`. Use `replace_all` in Edit since the
   identifier is local.

### Verify with

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q <each-of-the-9-files>
# Expect: same count, all green
```

### Gotcha

- **Helper functions that take `KEY` as default argument** (e.g.,
  `tests/test_tdmpc2.py:697 _build_plan_params(cfg, key=jax.random.PRNGKey(0))`)
  do NOT count as module-level — the call only happens when the helper
  runs. Leave those alone.

---

## Task 3 — Delete `tests/test_train_tdmpc2_h{1..5}.py`

**Cost:** 5 min.

### Files

```
tests/test_train_tdmpc2_h1.py
tests/test_train_tdmpc2_h2.py
tests/test_train_tdmpc2_h3.py
tests/test_train_tdmpc2_h4.py
tests/test_train_tdmpc2_h5.py
```

All five `subprocess.run(cwd="/home/stevenman/.../.worktrees/tdmpc2-impl")`.
Verified `.worktrees/` does not exist. Tests are dead.

### Diff

```bash
git rm tests/test_train_tdmpc2_h1.py tests/test_train_tdmpc2_h2.py \
       tests/test_train_tdmpc2_h3.py tests/test_train_tdmpc2_h4.py \
       tests/test_train_tdmpc2_h5.py
```

### Verify with

```bash
uv run python -m pytest --collect-only -q -m slow | tail -3
# Expect: 5 fewer slow tests than before (was 15, now 10)

uv run python -m pytest -q
# Expect: same default-lane pass count as before (these were `slow`,
#         not in default lane)
```

### Done when

- [x] Files removed.
- [x] `pytest -m slow --collect-only` collects 10 not 15.
- [x] Default lane unchanged.

### Gotcha

- Don't replace them with new smoke tests in the same commit. If owner
  wants a TDMPC2 train smoke, that's a separate task. Keep this PR
  surgical.

### Commit

```
test: delete dead worktree-path TDMPC2 train smoke tests

test_train_tdmpc2_h{1..5}.py all subprocess.run cwd=.worktrees/tdmpc2-impl/
which no longer exists. They were behind @pytest.mark.slow so default
lane never saw them, but `pytest -m slow` was a 5-failure trap.

TDMPC2 component coverage stays in tests/test_tdmpc2.py (48 unit tests,
CPU-clean).
```

---

## Task 4 — Rewrite or delete `tests/test_tdmpc2_i3_eval_isolation.py`

**Cost:** 30 min (rewrite) or 5 min (delete).

The test logic — assert `run_eval` does not mutate `state.prev_mean` —
is genuinely valuable and protects a non-obvious invariant. The
current implementation is a 100-line embedded subprocess script that
imports from a deleted worktree.

### Option A — Delete (recommended if owner is in cleanup mode)

```bash
git rm tests/test_tdmpc2_i3_eval_isolation.py
```

The invariant is *probably* still upheld by the current TDMPC2 code,
but without the test we'll only catch a regression at training time.

### Option B — Rewrite (recommended if owner cares about TDMPC2)

Replace the subprocess script with an in-process call. The current
script imports:

```python
sys.path.insert(0, '/home/stevenman/.../.worktrees/tdmpc2-impl')
from train_tdmpc2 import (
    build_modules, init_train_state, build_train_config_from_tdmpc2,
    run_warmup, run_eval, _pipe_obs,
)
from jax_rl.algos.tdmpc2 import make_update_step, build_world_model_optimizer, build_policy_optimizer
```

Post-relocation the imports become:

```python
from scripts.train_tdmpc2 import (
    build_modules, init_train_state, build_train_config_from_tdmpc2,
    run_warmup, run_eval, _pipe_obs,
)
from jax_rl.algos.tdmpc2 import make_update_step, build_world_model_optimizer, build_policy_optimizer
```

Pull the entire embedded `script` string body up into the test
function directly. Remove the `subprocess.run` call. Mark
`@pytest.mark.gpu` (CheetahRun needs MJX). Keep `@pytest.mark.slow` —
this builds full TDMPC2 modules and runs a couple of warmup/collect
steps; it's not a unit test.

The `_KEY` mismatch and PRNG-skip-on-eval pattern are preserved
because the test logic is still the same.

### Verify with (Option B)

```bash
uv run python -m pytest -q -m "slow and gpu" tests/test_tdmpc2_i3_eval_isolation.py
# Expect: 1 passed (or 1 failed if the invariant is actually broken — that's
# the point of the test)
```

### Gotchas

- **`scripts.train_tdmpc2` may not import as a module** if `scripts/`
  isn't on `sys.path`. Repo root IS on `sys.path` per
  `pyproject.toml:pythonpath = ["."]`, so `scripts.train_tdmpc2`
  should resolve. Verify with `uv run python -c "from scripts import
  train_tdmpc2"` first.
- **The subprocess version was probably defensive about JAX state
  pollution** between tests. In-process means PRNG state, jitted
  closures, and any module-level globals from `train_tdmpc2.py` leak
  into other tests. If you see flakes after this rewrite, that's why
  — wrap the whole test body in a fresh function and avoid module-
  level imports of `train_tdmpc2`.

### Recommendation

**Default to Option A (delete).** The invariant *should* be guarded
by `tests/test_tdmpc2.py`'s component tests. If owner wants to keep
the explicit isolation guarantee, do Option B but treat it as a
separate session.

---

## Task 5 — Make `deploy/test_policy_runner.py::test_policy_runner_loads_and_infers` synthetic + add rejection test

**Cost:** 1 hr.

### Files

- `deploy/test_policy_runner.py` lines 11-42 (rewrite the first test;
  add a second test below it)
- Reference: `tests/test_artifact_contract.py:test_validate_shared_actor_files_passes_on_complete_dir`
  lines 89-97 (synthetic shared-actor ckpt precedent)

### Diff

Replace lines 11-42 with two tests:

```python
import json
import os
import tempfile

import numpy as np
import pytest


def _make_synthetic_shared_actor_ckpt(td: str, *, obs_dim: int = 48,
                                       action_dim: int = 12, algo: str = "fast_sac",
                                       hidden: int = 16):
    """Build a minimal valid shared-actor ckpt at `td`. Returns nothing.

    Synthesizes the exact files PolicyRunner.from_checkpoint() reads:
    meta.json, actor_params.npy, orbax/.
    """
    from jax_rl.training.artifact_contract import KIND_SHARED_ACTOR, stamp_meta

    meta = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "algo": algo,
        "train_config": {"n_frame_stack": 1},
        "obs_schema": {
            "state": ["gyro", "accelerometer", "gravity",
                      "joint_pos_offset", "joint_vel", "last_act", "command"],
        },
    }
    meta = stamp_meta(meta, KIND_SHARED_ACTOR)
    with open(os.path.join(td, "meta.json"), "w") as f:
        json.dump(meta, f)

    # Minimal pytree for an MLP actor — exact structure depends on the
    # algo's network. PolicyRunner uses a numpy MLP, so feed it a flat
    # weights/biases dict matching whatever PolicyRunner expects.
    # Inspect deploy/policy_runner.py:_load_actor_params to see the
    # exact pytree shape and replicate it here with random small floats.
    # See gotcha below.
    actor_params = {...}  # algo-specific; see gotcha
    np.save(os.path.join(td, "actor_params.npy"), actor_params, allow_pickle=True)
    os.makedirs(os.path.join(td, "orbax"), exist_ok=True)


def _make_synthetic_tdmpc2_ckpt(td: str):
    """Build a minimal TDMPC2 ckpt — should be rejected by PolicyRunner."""
    from jax_rl.training.artifact_contract import KIND_TDMPC2, stamp_meta

    meta = {
        "obs_dim": 24, "action_dim": 6, "algo": "tdmpc2",
        "train_config": {"n_frame_stack": 1},
    }
    meta = stamp_meta(meta, KIND_TDMPC2)
    with open(os.path.join(td, "meta.json"), "w") as f:
        json.dump(meta, f)
    np.savez(os.path.join(td, "actor_params.npz"))
    np.savez(os.path.join(td, "world_model_params.npz"))


def test_policy_runner_loads_synthetic_shared_actor_ckpt(tmp_path):
    """PolicyRunner loads a hermetic synthetic shared-actor ckpt and infers."""
    from deploy.policy_runner import PolicyRunner

    _make_synthetic_shared_actor_ckpt(str(tmp_path), obs_dim=48, action_dim=12)
    runner = PolicyRunner(str(tmp_path))

    assert runner.obs_dim == 48
    assert runner.action_dim == 12

    obs = np.zeros(48, dtype=np.float32)
    action = runner.get_action(obs)
    assert action.shape == (12,)
    assert np.all(np.abs(action) <= 1.0 + 1e-6)
    assert not np.any(np.isnan(action))


def test_policy_runner_rejects_tdmpc2_ckpt(tmp_path):
    """PolicyRunner refuses a TDMPC2 ckpt with redirect message.

    Closes consumer-side artifact-kind rejection coverage gap (codex audit
    2026-04-27 §"Artifact contracts implicit and split").
    """
    from deploy.policy_runner import PolicyRunner

    _make_synthetic_tdmpc2_ckpt(str(tmp_path))

    with pytest.raises((ValueError, FileNotFoundError)) as exc:
        PolicyRunner(str(tmp_path))
    msg = str(exc.value)
    assert "tdmpc2" in msg.lower() or "actor_params.npy" in msg
```

### Verify with

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q deploy/test_policy_runner.py
# Expect: 10 passed (was 9; +1 for the new rejection test; the loads_and_infers
#         is replaced not removed, so net +1)
```

### Gotchas

- **`actor_params` pytree shape is algo-specific.** I sketched
  `actor_params = {...}`. Before writing the test, read
  `deploy/policy_runner.py` carefully — find `_load_actor_params` or
  whatever method reads the npy. Match its expected pytree exactly
  (likely `{"params": {"Dense_0": {"kernel": ..., "bias": ...}, ...}}`
  for an MLP). If you guess wrong, the test fails with a confusing
  shape error. Worth doing right the first time.
- **`PolicyRunner` may construct an MLP and require specific layer
  count/sizes.** If `obs_dim=48, action_dim=12` is hardcoded in
  `PolicyRunner` setup, leave the synthetic at those dims (which is
  what I did above).
- **The rejection assertion is loose** — it accepts either ValueError
  (from `assert_artifact_kind`) or FileNotFoundError (from the legacy
  load path). Tighten it after reading `deploy/policy_runner.py:__init__`
  to know which path TDMPC2 hits.

### Commit

```
test(deploy): replace local-ckpt-dependent test with synthetic; add TDMPC2 rejection

test_policy_runner_loads_and_infers used to scan checkpoints/ and
pytest.skip if absent — CI never saw it. Replace with a hermetic
synthetic shared-actor ckpt fixture. Add a second test asserting
PolicyRunner rejects TDMPC2 artifacts via assert_artifact_kind.

Closes codex audit gap: consumer-side artifact-kind rejection coverage.
```

---

## Task 6 — Fix `test_run_offpolicy_loop_stub_env_cpu` + parameterize across off-policy algos

**Cost:** 1 hr.

### Files

- `tests/test_offpolicy_loop.py` (rewrite the first test)

### Diff 1: bug fix

Line 88 currently:

```python
bundle = EnvBundle(
    env=env, env_step=stub_step, env_state=initial_state,
    eval_env=env, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
    critic_obs_dim=None, has_privileged=False, dict_obs=False,
    key=jax.random.PRNGKey(0),
)
```

Add `num_envs=NUM_ENVS,`:

```python
bundle = EnvBundle(
    env=env, env_step=stub_step, env_state=initial_state,
    eval_env=env, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
    critic_obs_dim=None, has_privileged=False, dict_obs=False,
    key=jax.random.PRNGKey(0),
    num_envs=NUM_ENVS,
)
```

### Diff 2: extend across the 4 off-policy algos

The current test is SAC-specific. Don't `parametrize` — TD3 and FastTD3
have different ctor signatures (`actor_optimizer + critic_optimizer`,
no `alpha_optimizer`). Instead, refactor to a helper + 4 thin tests:

```python
def _stub_env_bundle(num_envs=2, obs_dim=4, action_dim=2):
    """Hermetic stub env bundle for loop tests. Returns EnvBundle."""
    @dataclass
    class StubEnvState:
        obs: jnp.ndarray
        reward: jnp.ndarray
        done: jnp.ndarray
        info: dict

    def stub_step(state, action):
        new_obs = state.obs + 0.01 * jnp.ones_like(state.obs)
        return StubEnvState(
            obs=new_obs,
            reward=jnp.ones((num_envs,)) * 0.5,
            done=jnp.zeros((num_envs,)),
            info={"truncation": jnp.zeros((num_envs,))},
        )

    initial_state = StubEnvState(
        obs=jnp.zeros((num_envs, obs_dim)),
        reward=jnp.zeros((num_envs,)),
        done=jnp.zeros((num_envs,)),
        info={"truncation": jnp.zeros((num_envs,))},
    )

    class StubEnv:
        action_size = action_dim
        def reset(self, keys): return initial_state
        def step(self, state, action): return stub_step(state, action)

    env = StubEnv()
    return EnvBundle(
        env=env, env_step=stub_step, env_state=initial_state,
        eval_env=env, obs_dim=obs_dim, action_dim=action_dim,
        critic_obs_dim=None, has_privileged=False, dict_obs=False,
        key=jax.random.PRNGKey(0),
        num_envs=num_envs,
    )


def _common_cfg(num_envs):
    return TrainConfig(
        env_name="StubEnv", num_envs=num_envs, total_timesteps=40,
        episode_length=100, eval_every_n_episodes=10**9,
        gamma=0.99, lr=3e-4, reward_scaling=1.0, n_frame_stack=1,
        handle_truncation=True,
    )


def _patch_eval(monkeypatch):
    import jax_rl.training.offpolicy_loop as ol_module
    def _noop_maybe_eval(*a, **kw):
        last_eval_eps = kw.get("last_eval_eps", a[7] if len(a) > 7 else 0)
        key = kw.get("key", a[8] if len(a) > 8 else None)
        return last_eval_eps, key
    monkeypatch.setattr(ol_module, "maybe_eval_and_checkpoint", _noop_maybe_eval)
    monkeypatch.setattr(ol_module, "final_eval_and_checkpoint", lambda *a, **kw: None)


def test_run_offpolicy_loop_stub_env_sac_cpu(tmp_path, monkeypatch):
    bundle = _stub_env_bundle()
    cfg = _common_cfg(num_envs=2)
    algo_cfg = SACConfig(hidden_dim=(32, 32), batch_size=8,
                         min_buffer_size=10, buffer_size=100,
                         grad_updates_per_step=1)
    optimizer = optax.adam(cfg.lr)
    alpha_opt = optax.adam(algo_cfg.alpha_lr)
    algo = SAC(config=algo_cfg, obs_dim=4, action_dim=2,
               optimizer=optimizer, alpha_optimizer=alpha_opt,
               gamma=cfg.gamma, critic_obs_dim=None)
    explore = lambda p, o, k: algo.select_action(p, o, k)

    _patch_eval(monkeypatch)
    monkeypatch.chdir(tmp_path)

    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="sac",
        env_bundle=bundle, explore_fn=explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
        seed=0, resume=None, use_wandb=False,
    )


def test_run_offpolicy_loop_stub_env_td3_cpu(tmp_path, monkeypatch):
    """Same shape, TD3 ctor (no alpha)."""
    from jax_rl.algos.td3 import TD3
    from jax_rl.configs.td3_config import TD3Config

    bundle = _stub_env_bundle()
    cfg = _common_cfg(num_envs=2)
    algo_cfg = TD3Config(hidden_dim=(32, 32), batch_size=8,
                        min_buffer_size=10, buffer_size=100,
                        grad_updates_per_step=1)
    actor_opt = optax.adam(cfg.lr)
    critic_opt = optax.adam(cfg.lr)
    algo = TD3(config=algo_cfg, obs_dim=4, action_dim=2,
               actor_optimizer=actor_opt, critic_optimizer=critic_opt,
               gamma=cfg.gamma, critic_obs_dim=None)
    explore = lambda p, o, k: algo.select_action(p, o, k)

    _patch_eval(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run_offpolicy_loop(
        cfg=cfg, algo_cfg=algo_cfg, algo=algo, algo_name="td3",
        env_bundle=bundle, explore_fn=explore,
        log_extra_fields=[],  # TD3 has no entropy/alpha
        log_extra_keys=[],
        seed=0, resume=None, use_wandb=False,
    )


def test_run_offpolicy_loop_stub_env_fast_sac_cpu(tmp_path, monkeypatch):
    """FastSAC: same ctor pattern as SAC."""
    # mirror SAC test but instantiate FastSAC, FastSACConfig
    ...

def test_run_offpolicy_loop_stub_env_fast_td3_cpu(tmp_path, monkeypatch):
    """FastTD3: same ctor pattern as TD3."""
    ...
```

### Verify with

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_offpolicy_loop.py
# Expect: 4 passed, 1 deselected (the slow CheetahRun test stays slow)
# If all 4 fail with EnvBundle error → you forgot num_envs=
# If 1 fails with attribute error → ctor signature wrong; check the algo's
# __init__ params
```

Remove the file's `pytestmark = pytest.mark.gpu` from Task 1 once these
tests pass on CPU. (They don't actually need GPU now.)

### Done when

- [x] All 4 off-policy stub-env tests pass on CPU.
- [x] Original `test_run_offpolicy_loop_sac_cheetah` (slow, GPU) still
  exists, still marked `slow`, still passes when run with `-m slow`.

### Gotchas

- **TD3 ctor takes `actor_optimizer + critic_optimizer`**, NOT a single
  `optimizer`. Don't typo this.
- **`log_extra_fields=[]` for TD3/FastTD3** — they don't have entropy
  or alpha. Pass empty lists, not `None`.
- **Some algos may require `critic_obs_dim` to be set explicitly to
  `obs_dim`** for the symmetric case. Verify by reading each algo's
  `__init__`.
- **The `_patch_eval` mock pulls `last_eval_eps + key` from positional
  arg 7/8.** This is a brittle interface — if `run_offpolicy_loop`
  ever changes its `maybe_eval_and_checkpoint` signature, the mock
  breaks silently. Long-term fix: have `run_offpolicy_loop` accept an
  optional `eval_fn` injection point. Out of scope here.

---

## Task 7 — DomainRand per-episode persistence regression test

**Cost:** 1 hr (revised up from 30 min after re-reading the wrapper).

### Files

- New file: `tests/test_domain_rand_persistence.py`
- Reference: `tests/test_domain_rand_compose.py` (mock-wrapper pattern)
- Source: `jax_rl/envs/wrappers/domain_rand.py`
  - line 100: `_KEY = '_dr'`
  - line 151: `state.info[f'{_KEY}_dr_fields']` is the persisted store
  - lines 192-219: where `step()` decides reset-vs-step DR fields

### What to test

The behavioral claim from `algo_port_protocol.md` §8: "non-done envs
see persistent DR fields across multiple steps."

**Hermetic test approach:**

The `DomainRandWrapper.step()` path requires a real `mjx_env.State`,
a real `vmap`-able `env.reset` and `env.step`, and at minimum a
single-spec `dr_specs`. The mock-wrapper pattern in
`test_domain_rand_compose.py` skips `__init__` entirely; for
persistence we must walk a full step path.

Two approaches:

**A. Mock the wrapper end-to-end with a fake `mjx_env.State` shape.**
~80 LOC. The test constructs a `types.SimpleNamespace`-based env that
returns a state with `.obs`, `.reward`, `.done`, `.info`, `.metrics`,
`.data` — fields the wrapper reads. Run reset → step → step with
done=False, assert `info["_dr_dr_fields"]` byte-identical across
steps for env 0. Run reset → step (done=True) → step, assert env 0's
DR fields *change* on the reset.

**B. Use a real MJX env (CheetahRun via `make_env_bundle`).** Simpler
to write but requires GPU → mark `gpu`. Defeats the "hermetic
regression test" goal.

**Recommendation: Option A.** Use the same `_mock_wrapper` helper as
`test_domain_rand_compose.py`, but bypass less of `__init__`:

```python
def _mock_wrapper_with_episode(model_field_name, model_field_array, specs,
                                episode_length=1000):
    """Mock wrapper that supports reset() and step()."""
    w = DomainRandWrapper.__new__(DomainRandWrapper)
    # Bypass __init__'s env probing
    w.env = ...  # types.SimpleNamespace with reset/step that return mjx_env.State
    w.episode_length = episode_length
    w.mode = "per_step"
    w._model_specs = specs
    w._runtime_specs = []
    w.dr_specs = specs
    return w
```

The fake `env.reset` and `env.step` need to return objects with
`.data, .obs, .reward, .done, .info, .metrics` — read
`mujoco_playground._src.mjx_env.State` to see the dataclass shape.

**Test bodies:**

```python
def test_dr_fields_persist_across_steps_for_non_done_env():
    """Non-done env's DR fields are byte-identical across N steps."""
    specs = [DRSpec(name="g0", type="model", field="geom_friction",
                    column=0, operation="set", min=0.7, max=0.7,
                    per_element=False)]
    w = _mock_wrapper_with_episode("geom_friction",
                                    jp.ones((10, 3)) * 0.5, specs)
    state = w.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    initial = state.info["_dr_dr_fields"]["geom_friction"]
    action = jp.zeros((4, 1))  # action_dim=1 stub

    state = w.step(state, action)
    after_step1 = state.info["_dr_dr_fields"]["geom_friction"]
    assert jp.array_equal(initial, after_step1)

    state = w.step(state, action)
    after_step2 = state.info["_dr_dr_fields"]["geom_friction"]
    assert jp.array_equal(initial, after_step2)


def test_dr_fields_resample_only_for_done_envs():
    """Env 0 reaches truncation; env 1 doesn't. DR field changes for env 0,
    stays for env 1."""
    specs = [DRSpec(name="g0", type="model", field="geom_friction",
                    column=0, operation="uniform", min=0.0, max=1.0,
                    per_element=False)]
    w = _mock_wrapper_with_episode("geom_friction",
                                    jp.ones((10, 3)) * 0.5, specs,
                                    episode_length=2)
    state = w.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    initial = state.info["_dr_dr_fields"]["geom_friction"]

    # Step twice — env 0 hits truncation, env 1 doesn't (depends on
    # truncation logic; tune episode_length so env 0 truncates)
    action = jp.zeros((2, 1))
    state = w.step(state, action)
    state = w.step(state, action)

    after = state.info["_dr_dr_fields"]["geom_friction"]
    # Env 0's DR field should differ post-truncate
    # Env 1's should match initial
    assert not jp.array_equal(initial[0], after[0])  # env 0 resampled
    assert jp.array_equal(initial[1], after[1])      # env 1 persistent
```

### Verify with

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_domain_rand_persistence.py
# Expect: 2 passed (or fail with the per-episode bug if regression hits)
```

### Done when

- [x] 2 hermetic CPU-only tests pass.
- [x] No marker needed (pure-numpy mock, no MJX).

### Gotchas

- **`mjx_env.State` is a flax dataclass** — your fake env.reset must
  return something with the right pytree structure. The simplest path:
  `from mujoco_playground._src.mjx_env import State` and instantiate
  it with stub fields. (`test_action_delay.py` does this.)
- **`jax.vmap` over the env.step path** — the wrapper expects `vmap`-
  compatible callables. If your fake `env.step` is plain Python, it'll
  trace as a static. Mark fake env's reset/step with `jax.jit` or
  return jax arrays directly.
- **`_where_done` merge logic is non-trivial.** Read lines 228-236 of
  `domain_rand.py` carefully before asserting on `info["_dr_dr_fields"]`
  — it goes through the same merge.
- **`episode_length=2` to force truncation in 2 steps.** The wrapper's
  truncation gate (line 225-226) compares `steps >= episode_length`.

This task is the most likely to take longer than the estimate. If
you're past 90 min, stop and ask owner.

---

## Task 8 — Resume warmup behavioral test

**Cost:** 1 hr. Hardest task in the playbook.

### Files

- New file: `tests/test_resume_warmup_behavior.py`
- Source: `jax_rl/training/offpolicy_loop.py` lines 47, 135-189
- Reference: `tests/test_offpolicy_loop.py` (stub-env pattern)

### What to test

From `.context/lessons/offpolicy.md` §"Resume Warmup":

> Action selection branch now gates `use_random` on
> `is_warmup AND (start_step == 0 OR resume_warmup == "random")`.

Behavioral claim: **on resume with `--resume-warmup policy` (default),
the warmup phase calls `explore_fn` with the loaded actor params, NOT
`jax.random.uniform`.**

### Approach

Test by side-channel: count calls to `jax.random.uniform` and
`explore_fn` during warmup. Use `monkeypatch.setattr` on `jax.random.uniform`
inside `offpolicy_loop` to a wrapper that increments a counter; same
for `explore_fn`.

```python
def test_resume_warmup_policy_does_not_call_random_uniform(tmp_path, monkeypatch):
    """When resume_warmup='policy' and start_step > 0, warmup uses explore_fn,
    NOT jax.random.uniform.

    Closes the resume-warmup behavioral coverage gap (codex audit
    2026-04-27 §"Resume warmup is opt-in random, default policy").
    """
    import jax_rl.training.offpolicy_loop as ol_module

    # Patch jax.random.uniform inside the loop module to count calls.
    real_uniform = ol_module.jax.random.uniform
    uniform_calls = []
    def _spy_uniform(key, shape, **kwargs):
        uniform_calls.append(shape)
        return real_uniform(key, shape, **kwargs)
    monkeypatch.setattr(ol_module.jax.random, "uniform", _spy_uniform)

    # Build stub env + tiny SAC (use _stub_env_bundle from Task 6 — extract
    # to tests/_loop_helpers.py during Task 6).
    bundle = _stub_env_bundle(num_envs=2)
    # ... cfg, algo, explore as in test_run_offpolicy_loop_stub_env_sac_cpu

    # Phase 1: cold start (start_step=0). Warmup should call uniform.
    monkeypatch.chdir(tmp_path)
    run_offpolicy_loop(..., resume=None, resume_warmup="policy")
    cold_uniform_calls = len(uniform_calls)
    assert cold_uniform_calls > 0, "cold start did not use random for warmup"

    # Phase 2: simulate resume. Persist a ckpt from phase 1 and reload.
    # (Requires save_checkpoint to have run during phase 1, which it does
    # by default — point resume= to that ckpt.)
    uniform_calls.clear()
    ckpt_dir = ... # the ckpt phase 1 wrote
    run_offpolicy_loop(..., resume=ckpt_dir, resume_warmup="policy")

    # On resume with policy mode, warmup should NOT call uniform for action.
    # uniform CAN be called for other things (action noise) but NOT for the
    # warmup-action shape (num_envs, action_dim) = (2, 2).
    warmup_uniform_calls = [s for s in uniform_calls if s == (2, 2)]
    assert len(warmup_uniform_calls) == 0, \
        f"resume_warmup=policy still used random.uniform for warmup actions: {warmup_uniform_calls}"
```

### Verify with

```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_resume_warmup_behavior.py
# Expect: 1 passed
```

### Gotchas

- **`jax.random.uniform` is called many times for many shapes.** SAC's
  reparameterization sampler also uses `jax.random.uniform` internally.
  The test must filter by shape `(num_envs, action_dim)` to isolate the
  warmup-action call. Verify by reading `offpolicy_loop.py:189` —
  `action = jax.random.uniform(ak, (num_envs, action_dim), ...)`.
- **Resume requires a real ckpt on disk.** Phase 1 of the test runs
  uninterrupted to a small `total_timesteps`, which writes a ckpt
  via `CheckpointManager`. Phase 2 reads that. If `total_timesteps`
  is too small, the ckpt isn't written. Set `total_timesteps=40` and
  ensure `save_every` triggers. Easiest: read
  `jax_rl/training/checkpointing.py:save_checkpoint` to confirm
  triggers.
- **`monkeypatch.setattr(ol_module.jax.random, "uniform", ...)` may
  not patch deeply enough.** If `offpolicy_loop` already imported
  `jax.random` in module scope, the patch hits the module-level
  binding. Verify with a debug print in the spy.
- **`start_step > 0` requires the resume path to set it.** Read
  `offpolicy_loop.py:135-143` — `start_step` comes from
  `metrics.csv`. Phase 1 must produce a `metrics.csv` for phase 2's
  resume to advance `start_step`.

This is genuinely hard test infrastructure. If the spy approach
fails, fall back to: patch `explore_fn` itself to count calls and
assert it's called during warmup (positive assertion instead of
negative). Easier to reason about.

---

## Task 9 — CPU-only GitHub Actions workflow

**Cost:** 30 min after Task 1.

### Files

- New file: `.github/workflows/tests.yml`

### Diff

(Don't write this until Task 1 is merged — markers must exist first.)

```yaml
name: Tests (CPU)

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  pytest-cpu:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install 3.13

      - name: Install dependencies
        run: uv sync --group dev --no-group docs

      - name: Run CPU tests
        env:
          JAX_PLATFORMS: cpu
        run: uv run python -m pytest -q
        # markers in pyproject.toml addopts already exclude gpu/warp/go2/network/slow
```

### Verify with

Push to a PR branch. CI should run, all green, in <5 min.

### Gotchas

- **`uv sync --group dev` may fail without `--no-group docs`** if docs
  deps include OS-level libs not on the runner. Test with the explicit
  exclusion.
- **`JAX_PLATFORMS=cpu` matters even though there's no GPU on the
  runner.** Without it, JAX tries to detect CUDA and prints a warning
  on every test. Set it to silence.
- **CPU-only JAX install:** `pyproject.toml` has
  `jax[cuda13]>=0.9.0`. On the GitHub runner, this should fall back
  to CPU automatically (cuda13 wheels need GPU at runtime, not install
  time). If `uv sync` complains, add a CI-specific override or split
  the cuda extra.
- **Don't hide the `slow` lane.** Add a separate `workflow_dispatch`-only
  job that runs `pytest -m slow` against the self-hosted GPU runner
  if/when the owner sets one up. Out of scope for this PR.

### Commit

```
ci: add CPU-only pytest workflow

Was: docs-only CI. Add a CPU pytest job using the marker-excluded
default lane (Task 1). Runs on push to main + every PR. Excludes
gpu/warp/go2/network/slow markers automatically via pyproject.toml
addopts.

Closes codex audit P1 (no pytest gate).
```

---

## Task 10 — Document named pytest invocations

**Cost:** 15 min after Task 1.

### Files

- `docs/contributing.md` (find the "Test" section; replace stale
  invocations)

### Verify with

```bash
grep -nA10 "pytest\|test suite" docs/contributing.md
```

Make sure the doc no longer says "Full test suite (~299 tests)" — that
number is stale. Codex flagged this. Current is 819 (or ~600 default
post-Task 1).

### What to add

Replace any existing test instructions with:

```markdown
## Tests

The default lane is CPU-only and hermetic — runs ~600 tests in 1-2 min:

\`\`\`bash
uv run python -m pytest
\`\`\`

To run subsets:

\`\`\`bash
uv run python -m pytest -m gpu              # MJX/CUDA tests (~200)
uv run python -m pytest -m "warp or go2"   # Go2 Warp surface (~70)
uv run python -m pytest -m deploy           # Deploy contract tests
uv run python -m pytest -m "not network"   # Everything except network-bound
uv run python -m pytest -m slow             # Long-running (CPU or GPU)
\`\`\`

Markers are defined in `pyproject.toml`. See `pytest --markers` for
the live list with descriptions.
\`\`\`
```

---

## Verification matrix (run after each task)

After every task lands, run this triage to make sure nothing regressed:

```bash
# 1. Default lane still green
JAX_PLATFORMS=cpu uv run python -m pytest -q

# 2. Pass count matches expected (write down the number per task)

# 3. Docs canaries still pass
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_docs_drift.py tests/test_docs_code_blocks.py

# 4. Markers list still correct (after Task 1)
uv run python -m pytest --markers | grep -E "gpu|warp|go2|deploy|network|slow"
```

If pass count drops without a corresponding deletion in the task,
**stop and investigate**. Don't accept "looks fine" — drift accumulates.

---

## Cross-references

- **Audit doc:** `.context/audits/2026-04-27_test_suite_audit.md`
- **Original codex audit:** `codex_audit.md` (top-level, untracked)
- **Algo-port protocol §9 (test hermeticity):** `.context/lessons/algo_port_protocol.md`
- **Resume warmup lesson:** `.context/lessons/offpolicy.md` §"Resume Warmup"
- **Env backend contract:** `.context/lessons/env_backends.md`
- **Structural plan with marker phase:** `.superpowers/plans/2026-04-25-structural-hardening.md` Phase 0
- **DomainRand persistence implementation:** `jax_rl/envs/wrappers/domain_rand.py` lines 100-260
- **EnvBundle dataclass + num_envs default:** `jax_rl/training/env_bundle.py:48`
- **Off-policy loop resume warmup gate:** `jax_rl/training/offpolicy_loop.py:185-189`
- **Synthetic-ckpt precedent:** `tests/test_artifact_contract.py:89-118`
- **Mock-wrapper pattern:** `tests/test_domain_rand_compose.py`
