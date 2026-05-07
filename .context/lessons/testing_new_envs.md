# Testing New Envs — Contract

> Guidance for any agent adding a new env, env wrapper, or env-adjacent
> code. Captures patterns that emerged from the 2026-04-27 test-suite
> cleanup. Pairs with `.context/lessons/algo_port_protocol.md` §9.
>
> **Default if you skip this:** your tests will OOM the GPU at collection
> time, fail on CI, depend on machine-local checkpoints, or silently
> pass while broken. All four happened in the prior repo state.

---

## §1. Marker discipline — gate at module scope

Pyproject registers six markers (`pyproject.toml`):
`gpu` / `warp` / `go2` / `deploy` / `network` / `slow`. Default
`addopts` excludes all of them: `addopts = "-m 'not (gpu or warp or go2
or network or slow)'"`. **Default lane is hermetic CPU.**

If your test cannot run on CPU without CUDA, you MUST mark it. Pick the
narrowest accurate set:

| Test needs | Marks |
|---|---|
| MJX physics / `jax.device_put` of large arrays | `[gpu]` |
| `warp-lang` callable + GPU | `[gpu, warp]` |
| Go2 Warp env (`WarpJoystick`, `Go2Bongo*`, terrain curriculum) | `[gpu, warp, go2]` |
| Deploy runtime (`deploy/policy_runner`, `Go2Interface`, sim2sim) | `[deploy]` (orthogonal to gpu) |
| `requests`/`urllib` to public internet | `[network]` |
| >30s wall-clock | `[slow]` |

Markers do NOT inherit. Tagging `[warp]` does not imply `[gpu]`. **Tag
the full set every time** so `-m gpu` and `-m warp` selectors both
catch it.

Stamp at module scope, not per-test, when the whole file is
intent-uniform:

```python
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.warp, pytest.mark.go2]
```

Per-test marks only when the file is mixed (e.g.,
`tests/test_deploy_e2e.py` has 9 hermetic + 2 GPU-Warp tests).

**Verify before committing:**
```bash
JAX_PLATFORMS=cpu uv run python -m pytest -q
```
should run your file's hermetic tests cleanly OR skip your GPU tests
silently. Default-lane pass count must not drop.

---

## §2. Hermetic-first — synthetic mock before real env

The test pyramid: most tests should be CPU-only, no MJX/Warp init, no
real env construction. Real-env smoke tests are the tip, not the base.

### When to mock the env

If your test asserts something about a **wrapper, observation pipeline,
reward spec, DR composition, control metadata, schema validation** —
mock the env. Examples in repo:

- `tests/test_domain_rand_compose.py` — mocks `mjx_model` via
  `types.SimpleNamespace`, bypasses `DomainRandWrapper.__init__` with
  `__new__`. 4 hermetic tests, no MJX.
- `tests/test_domain_rand_persistence.py` — same pattern, walks
  full reset → step → step path with a fake state dataclass.
- `tests/test_obs_spec.py` — pure pytree checks against
  `compute_obs(...)`. No env build.
- `tests/test_terrain_primitives.py` — flat array math; no env.

### When to use a real env

If your test asserts something about **physics integration, contact
mode, joint ordering on the real Unitree MJCF, observation values
matching MuJoCo state** — use the real env, mark `[gpu, warp, go2]`,
and accept the GPU dependency. Examples:

- `tests/test_go2_warp_env.py` — exercises actual
  `WarpJoystick.step()` rollouts.
- `tests/test_deploy_e2e.py:test_env_metadata_matches_deploy_constants`
  — verifies `get_control_metadata()` against `deploy/go2_constants.py`.

### When in doubt

Write the mock first. If you find yourself rebuilding 200 lines of
MJX state manually, that's the signal to mark `[gpu]` and use the real
env. Don't over-mock.

---

## §3. No module-level CUDA-allocating bombs

Per `.context/lessons/algo_port_protocol.md` §9: do NOT put
`jax.random.PRNGKey(...)`, `jnp.zeros(...)`, or env construction at
module scope. They run at **collection time** — before any test runs,
before any marker filter applies. On a tight GPU, that's a CUDA OOM
before the test gate can even skip your file.

**Bad:**
```python
KEY = jax.random.PRNGKey(0)        # ← collected eagerly
DUMMY_OBS = jnp.zeros((4, 48))     # ← allocates on import
ENV = WarpJoystick()               # ← builds Warp graph at collection
```

**Good:**
```python
@pytest.fixture
def key():
    return jax.random.PRNGKey(0)

def test_foo(key):
    obs = jnp.zeros((4, 48))
    ...
```

Or inline inside the test body. Either form fires only when the test
runs, not when pytest collects.

Imports of MJX/Warp packages at module scope are fine — they're
lightweight. The killer is *constructing* JAX arrays or envs.

---

## §4. Don't depend on local artifacts

Tests that scan `checkpoints/`, `wandb/`, `.temp/`, or hardcode
absolute paths are CI-invisible. They `pytest.skip` or fail on the
runner, so the only place they ever run is the dev box that produced
the artifact.

**Replace local-artifact dependencies with synthetic fixtures.**
Precedents in repo:

- `tests/test_artifact_contract.py:test_validate_shared_actor_files_passes_on_complete_dir`
  — synthesizes `meta.json` + `actor_params.npy` + `orbax/` in
  `tempfile.TemporaryDirectory()`.
- `deploy/test_policy_runner.py:_make_synthetic_shared_actor_ckpt`
  — synthesizes a full shared-actor ckpt with random weights for
  PolicyRunner inference.
- `tests/test_resume_warmup_behavior.py:_synthesize_resume_ckpt`
  — calls real `save_checkpoint(...)` with `algo.init()` output to
  produce a loadable orbax dir.

If your test needs a checkpoint, build one. If it needs a metrics.csv,
write one. `tmp_path` fixture gives you a unique scratch dir per test.

**Hardcoded paths are also banned.** No `/home/stevenman/.../...`
strings (verified absent post-cleanup; previous codex audit found 6
such tests, all deleted). Use `pathlib.Path(__file__).parent` for
fixtures inside the test tree.

---

## §5. Reuse existing helpers

Before writing scaffolding, check:

- `tests/_loop_helpers.py` — `_stub_env_bundle(num_envs, obs_dim,
  action_dim)`, `_common_cfg(num_envs)`, `_patch_eval(monkeypatch)`.
  Used by 4 off-policy loop smoke tests + the resume-warmup behavioral
  test. **Use this for any test that calls `run_offpolicy_loop`.**
- `tests/test_domain_rand_compose.py:_mock_wrapper` — mock-wrapper
  pattern for any `DomainRandWrapper` subclass test.
- `jax_rl.training.artifact_contract` — `KIND_SHARED_ACTOR`,
  `KIND_TDMPC2`, `stamp_meta`, `assert_artifact_kind`,
  `validate_shared_actor_files`. Use for any ckpt-shape test.

Do NOT copy these into your file. Import them. If the helper doesn't
fit, extract a new one alongside (`tests/_<topic>_helpers.py`,
underscore-prefixed so pytest doesn't collect it).

---

## §6. Test the non-obvious; skip the obvious

Don't re-test what the framework or env spec already enforces:

- **Don't:** assert `state.obs.shape == (num_envs, obs_dim)` on a fresh
  env reset. The bundle layer already enforces this.
- **Don't:** assert that `env.step(state, action)` returns a state with
  `.reward, .done, .info`. Brax/MJX contract guarantees it.
- **Don't:** assert default config values match constants. Refactor-
  fragile, no real coverage.

**Do** test:

- Wrappers' invariants (e.g., DR persistence across non-done steps,
  frame-stack tile-on-reset, action-delay queue advancement).
- Reward sign / scaling / clipping — these have bugs.
- Observation schema round-trips (env → ckpt → deploy obs builder).
- Control metadata stamping (`get_control_metadata()` outputs survive
  to deploy).
- Cross-cutting contracts (e.g., truncation flag plumbing,
  `info["truncation"]` populated for `cfg.handle_truncation=True`).
- Behavioral gates (e.g., resume-warmup gate at
  `offpolicy_loop.py:186` — covered by
  `tests/test_resume_warmup_behavior.py`).
- Regression for any bug you fix. Test BEFORE patch (TDD pattern from
  global CLAUDE.md).

---

## §7. New env? Specific tests to write

If you're adding a new locomotion / manipulation env:

1. **Obs schema test** — env's `_obs_groups` matches what
   `schema_from_obs_groups` produces, and matches what deploy's
   `ObsBuilder` will read. Hermetic. Pattern: `deploy/test_policy_runner.py::test_schema_extractor_resolves_include_group`.

2. **Control metadata test** (Go2-class only) — `get_control_metadata()`
   returns Kp/Kd/action_scale/dts/contact_mode/joint_order matching
   `deploy/go2_constants.py`. Mark `[gpu, warp, go2, deploy]`. Pattern:
   `tests/test_deploy_e2e.py::test_env_metadata_matches_deploy_constants`.

3. **Single-step smoke** — env reset + 1 step doesn't crash, obs has
   expected dim. Mark `[gpu, warp]` if MJX/Warp needed. Pattern:
   `tests/test_go2_warp_env.py`.

4. **DR spec test if env has DR** — composition of multiple specs on
   same field works. Hermetic mock, no real env. Pattern:
   `tests/test_domain_rand_compose.py`.

5. **Bundle test** — `make_env_bundle(cfg, seed=0)` returns populated
   `EnvBundle` with correct `obs_dim`/`action_dim`/`backend_kind`.
   Pattern: `tests/test_env_bundle.py`. Mark `[gpu]` since most envs
   are MJX.

6. **Off-policy loop smoke** — only if your env has an SAC/TD3 preset.
   Reuse `tests/test_offpolicy_loop.py` pattern + `_loop_helpers`.
   Stub-env version is hermetic; real-env version is `[slow, gpu]`.

If your env is NOT in the SAC/PPO preset list yet, you don't need a
loop smoke. Wait until preset lands.

---

## §8. Verification checklist before commit

```bash
# Default lane stays green + same-or-higher pass count
JAX_PLATFORMS=cpu uv run python -m pytest -q

# Drift canaries unchanged
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_docs_drift.py tests/test_docs_code_blocks.py

# Your file passes specifically
JAX_PLATFORMS=cpu uv run python -m pytest -q <your_test_file>

# Marker selectors agree with intent
uv run python -m pytest --collect-only -q -m gpu | grep <your_test>   # if [gpu]
```

If your file is GPU-marked, also verify it actually runs on a GPU
machine before claiming it works — `[gpu]` is excluded from default
lane, so a typo in the marker name silently passes the CI gate
without ever running.

---

## §9. Common mistakes (history)

- **Hardcoded worktree paths** — 6 tests deleted on 2026-04-30 because
  they referenced `.worktrees/tdmpc2-impl/` which got cleaned up. If
  your test uses `subprocess.run(cwd=...)` with an absolute path,
  rewrite to import in-process or use `cwd=REPO_ROOT` from
  `pathlib.Path(__file__).parent.parent`.
- **`pytest.importorskip("warp")` is not a CUDA gate.** `warp-lang` is
  in `pyproject.toml`, so the import always succeeds; the tests then
  fail on first JAX-on-CUDA call. Use `[gpu]` marker.
- **`@pytest.mark.slow` is not a substitute for `[gpu]`.** Slow CPU
  tests exist; GPU tests should be `[gpu]`. The 2026-04-27 cleanup
  reclassified ~10 conflated cases.
- **Eagerly probing `jax.devices()` at module scope.** Same problem as
  PRNGKey at module scope — initializes the backend, allocates GPU
  memory at collection time. Push to fixture or test body.

---

## §10. Pointers

- Marker config: `pyproject.toml` lines 71-77
- Hermetic patterns: `tests/test_artifact_contract.py`, `tests/test_domain_rand_compose.py`
- Synthetic ckpt: `deploy/test_policy_runner.py:_make_synthetic_shared_actor_ckpt`
- Loop helpers: `tests/_loop_helpers.py`
- Resume warmup spy: `tests/test_resume_warmup_behavior.py:_install_uniform_spy`
- Test-suite audit: `.context/audits/2026-04-27_test_suite_audit.md`
- Algo port test rules: `.context/lessons/algo_port_protocol.md` §9, §11
- CI workflow: `.github/workflows/tests.yml`
