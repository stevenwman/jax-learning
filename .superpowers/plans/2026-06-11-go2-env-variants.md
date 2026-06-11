# Go2 Warp Variants-as-Data + OSC Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One declaration per named Go2 Warp env (config + train overrides + class) in an env-side registry; loud failure on unknown Go2 names; OSC controller logic consolidated into its component.

**Architecture:** New import-light `go2_warp_variants.py` holds an `EnvVariant` table + a single `go2_config()` builder. `mjx_backend.py` registers Go2 envs by looping the table; `env_presets.py` resolves Go2 presets from it. PR 2 moves the OSC math from the host env into the `OSC` component. Safety net: transitional equality tests pin new configs/presets against legacy output BEFORE legacy deletes.

**Tech Stack:** Python, ml_collections ConfigDict, mujoco_playground registry, pytest. Repo rule: `uv run python` always. Worktree: ALL commands run from `/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/go2-osc-impedance` — **cwd does NOT persist between Bash calls; `cd` there at the start of every command.**

**Spec:** `.superpowers/specs/2026-06-11-go2-env-variants-design.md` (read it first).

**Commit rule:** NO `Co-Authored-By` lines, ever (user rule, overrides defaults).

---

## File map

| File | Role |
|---|---|
| Create `jax_rl/envs/locomotion/go2_warp_variants.py` | `EnvVariant`, `go2_config()` builder, `GO2_WARP_VARIANTS` table (29 entries). Module-level imports: stdlib + `ml_collections` ONLY. Curriculum/PosTrack config factories use function-local imports inside the variant's `config` callable. |
| Create `tests/test_go2_warp_variants.py` | Equality tests (transitional → snapshot), registration+cls test, preset no-fallback test |
| Create `tests/data/go2_warp_variants_snapshot.json` | Frozen config snapshot (Task 5) |
| Modify `jax_rl/training/env_backends/mjx_backend.py` | Go2 block → loop over table + `_resolve_cls` |
| Modify `jax_rl/configs/env_presets.py` | `_resolve_go2_variant` helper wired into 6 getters; legacy Go2 entries deleted |
| Modify `scripts/train_{sac,td3,fast_sac,fast_td3,flashsac}.py` | `--eval-every` help text fix |
| Modify `jax_rl/envs/locomotion/go2_warp_components.py` | PR 2: OSC owns `_run_osc`/`_feet_in_body`/`_compute_nominal_foot_body` + its state |
| Modify `jax_rl/envs/locomotion/go2_warp_joystick.py` | PR 2: delete "OSC mechanics" section; `default_config()` delegates to `go2_config()` |
| Rewrite `tests/test_go2_osc_env.py` | PR 2: construct via `WarpJoystick(config=...)`, access `env._controller._…` |
| Delete (Task 5/7) `go2_warp_osc_joystick.py`, `go2_warp_osc_var_impedance.py`, `go2_warp_osc_rough.py` | After dependents migrate. `go2_osc.py` (torque math) is NOT touched. |

Reference current code before editing: `mjx_backend.py:27-226` (legacy closures), `env_presets.py` (preset tables + getters at lines ~285/324/366/474/598/605), `go2_warp_components.py:260-402` (controllers), `go2_warp_joystick.py:347-406` (OSC mechanics).

---

## PR 1 — variants-as-data

### Task 1: `go2_warp_variants.py` + transitional config-equality test

**Files:** Create `jax_rl/envs/locomotion/go2_warp_variants.py`, `tests/test_go2_warp_variants.py`.

- [x] **Step 1.1: Write the failing equality test**

```python
"""tests/test_go2_warp_variants.py — variants-as-data registry tests."""
import numpy as np
import pytest


def _deep_eq(a, b, path=""):
    """ConfigDict-aware deep equality with float tolerance 0 (exact)."""
    da, db = a.to_dict(), b.to_dict()
    def rec(x, y, p):
        assert type(x) == type(y) or (isinstance(x, (int, float)) and isinstance(y, (int, float))), f"{p}: {x!r} vs {y!r}"
        if isinstance(x, dict):
            assert x.keys() == y.keys(), f"{p}: keys {sorted(x)} vs {sorted(y)}"
            for k in x: rec(x[k], y[k], f"{p}.{k}")
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), f"{p}: len"
            for i, (xi, yi) in enumerate(zip(x, y)): rec(xi, yi, f"{p}[{i}]")
        else:
            assert x == y, f"{p}: {x!r} != {y!r}"
    rec(da, db, path)


def test_variant_configs_match_legacy_registration():
    """TRANSITIONAL: every variant's config == the legacy-registered config.

    Relies on mjx_backend's legacy closures still being registered. After
    Task 5 this test is REPLACED by the snapshot test."""
    from mujoco_playground import registry as pg_registry
    import jax_rl.training.env_backends.mjx_backend  # noqa: F401  (registers legacy)
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    for name, v in GO2_WARP_VARIANTS.items():
        legacy = pg_registry.get_default_config(name)
        new = v.config()
        _deep_eq(legacy, new, path=name)


def test_variants_file_is_import_light():
    import sys, subprocess
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import jax_rl.envs.locomotion.go2_warp_variants; "
         "bad = [m for m in ('mujoco', 'jax', 'mujoco_playground') if m in sys.modules]; "
         "print(','.join(bad))"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"heavy imports leaked: {out.stdout}"
```

NOTE for executor: verify `pg_registry.get_default_config` exists (`uv run python -c "from mujoco_playground import registry; print(registry.get_default_config)"`). If the API differs, read `mujoco_playground/_src/registry.py` and use the equivalent (the config factory is stored at registration; `mujoco_playground._src.locomotion._cfgs[name]()` is the fallback).

- [x] **Step 1.2: Run it — must fail with ModuleNotFoundError (variants file absent)**

`cd /home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/go2-osc-impedance && uv run python -m pytest tests/test_go2_warp_variants.py -x -q`

- [x] **Step 1.3: Write `jax_rl/envs/locomotion/go2_warp_variants.py`**

Module-level imports: `dataclasses`, `typing`, `ml_collections.config_dict` ONLY.

`go2_config()` builds the complete ConfigDict from literals. Base values = the current body of `go2_warp_joystick.default_config()` (ctrl_dt 0.02, sim_dt 0.004, episode_length 1000, Kp 20.0, Kd 0.5, action_repeat 1, soft_joint_pos_limit_factor 0.95, the noise/reward/command blocks, push interval 350, impl "warp", contact_mode "training", naconmax 4*8192, naccdmax 4000, njmax 100 — transcribe verbatim from `go2_warp_joystick.py:27-87`).

```python
def go2_config(
    *,
    controller: str = "joint_pd",          # "joint_pd" | "osc" | "var_impedance"
    osc_kp=None, osc_kd=None,              # (3,) lists; required for cartesian controllers
    use_op_space_inertia: bool = True,     # False = Jt-impedance ablation
    target_mode: str = "abs_body",
    stiffness_granularity: str = "per_foot",   # var_impedance only
    damping_action: bool = False,              # var_impedance only
    var_s=(0.25, 2.0), var_zeta=(0.5, 2.0),
    motor: str = "ideal",                  # "ideal" | "torque_speed" | "physical"
    terrain: str | tuple = "flat",         # "flat" | (profile, amplitude)
    push=(0.75, 0.75),
    action_scale: float | None = None,     # default 0.5 joint_pd / 0.12 cartesian
) -> config_dict.ConfigDict:
```

Semantics (must reproduce legacy outputs EXACTLY):
- `controller="joint_pd"`: no `osc` block at all (controller_from_config picks JointPD by absence).
- `controller="osc"`: `cfg.action_scale=0.12` (unless overridden); `cfg.osc = config_dict.create(target_mode=..., use_op_space_inertia=..., gravity_ff="none", ridge=1e-4, kp=list(osc_kp), kd=list(osc_kd))`. Default gains when osc_kp is None: kp=[3000.0,3000.0,4000.0], kd=[110.0,110.0,130.0]. **Key order inside `osc` must match the legacy factory** (target_mode, use_op_space_inertia, gravity_ff, ridge, kp, kd) in case dict comparison is order-sensitive anywhere downstream.
- `controller="var_impedance"`: osc block as above PLUS `var_s_min=0.25, var_s_max=2.0, stiffness_granularity=..., damping_action=..., var_zeta_min=0.5, var_zeta_max=2.0` (this order — matches `go2_warp_osc_var_impedance.default_config`).
- `motor="torque_speed"`: `cfg.torque_speed_model=True`. `motor="physical"`: both `torque_speed_model=True` and `physical_armature=True`. `"ideal"`: leave the base False/False.
- `terrain=(profile, amp)`: `cfg.rough_profile=profile; cfg.rough_amplitude=amp; cfg.rough_seed=0`.
- `push=(vmin, vmax)`: sets `push_config.vel_min/vel_max`.

`EnvVariant` + table:

```python
@dataclasses.dataclass(frozen=True)
class EnvVariant:
    config: object                    # Callable[[], ConfigDict]
    cls: str = "WarpJoystick"
    train: dict = dataclasses.field(default_factory=dict)
    algo: dict = dataclasses.field(default_factory=dict)   # {algo_name: {field: val}}
    notes: str = ""
```

The 29 entries (exact values; gains traps flagged):

| name | config call | cls | train (Task 4 adds more) |
|---|---|---|---|
| Go2WarpJoystickFlat | `go2_config()` | | |
| Go2WarpJoystickFlatTorqueSpeed | `go2_config(motor="torque_speed")` | | |
| Go2WarpJoystickFlatNoAccel | `go2_config()` | WarpJoystickNoAccel | |
| Go2WarpJoystickUnitree | `go2_config(action_scale=0.25)` | WarpJoystickNoAccel | |
| Go2WarpJoystickFlatPhysical | `go2_config(motor="physical")` | | |
| Go2WarpJoystickFlatHardKick | `go2_config(push=(0.5, 2.5))` | | |
| Go2WarpOscJoystickFlat | `go2_config(controller="osc")` | | |
| Go2WarpOscJoystickFlatJt | `go2_config(controller="osc", use_op_space_inertia=False, osc_kp=[1500.0,1500.0,2500.0], osc_kd=[60.0,60.0,80.0])` | | |
| Go2WarpOscJoystickFlatKp025/Kp05/Kp2/Kp4 | `go2_config(controller="osc", osc_kp=[s*3000.0,s*3000.0,s*4000.0], osc_kd=[110.0*s**0.5,110.0*s**0.5,130.0*s**0.5])` for s in 0.25/0.5/2.0/4.0 — **MUST be the computed expressions, NOT rounded literals** (legacy computes them; Kp05 kd≈77.78,91.92 ≠ the SoftPhysical literals 78/92) | | |
| Go2WarpOscJoystickFlatKp05HardKick | Kp05 gains + `push=(0.5, 2.5)` | | |
| Go2WarpOscFlatSoftPhysical | `go2_config(controller="osc", osc_kp=[1500.0,1500.0,2000.0], osc_kd=[78.0,78.0,92.0], motor="physical")` — **literal 78/92, NOT computed** | | per_step+eval (Task 4) |
| Go2WarpOscVarImpedanceFlat | `go2_config(controller="var_impedance")` | | |
| Go2WarpOscVarImpedanceAxisFlat | `go2_config(controller="var_impedance", stiffness_granularity="per_axis")` | | |
| Go2WarpOscVarFlatPhysical | `go2_config(controller="var_impedance", motor="physical")` | | Task 4 |
| Go2WarpOscVarAxisFlatPhysical | + `stiffness_granularity="per_axis"` | | Task 4 |
| Go2WarpOscVarDampingFlatPhysical | + `damping_action=True` | | Task 4 |
| Go2WarpOscVarDampingAxisFlatPhysical | + `stiffness_granularity="per_axis", damping_action=True` | | Task 4 |
| Go2WarpOscVarImpedanceHardKickFlat | `go2_config(controller="var_impedance", push=(0.5, 2.5))` | | |
| Go2WarpOscVarImpedanceAxisHardKickFlat | + `stiffness_granularity="per_axis"` | | |
| Go2WarpJointRoughUni | `go2_config(motor="physical", terrain=("uniform", 0.07))` | | Task 4 |
| Go2WarpOscRoughUni | `go2_config(controller="osc", osc_kp=[1500.0,1500.0,2000.0], osc_kd=[78.0,78.0,92.0], motor="physical", terrain=("uniform",0.07))` | | Task 4 |
| Go2WarpOscVarRoughUni | `go2_config(controller="var_impedance", motor="physical", terrain=("uniform",0.07))` | | Task 4 |
| Go2WarpOscVarAxisRoughUni | + per_axis | | Task 4 |
| Go2WarpJoystickCurriculum | lazy: `def _cfg(): from jax_rl.envs.locomotion.go2_warp_curriculum import default_config; return default_config()` | WarpJoystickCurriculum | `{"reset_mode": "per_step"}` |
| Go2WarpJoystickCurriculumTorqueSpeed | same + `cfg.torque_speed_model = True` | WarpJoystickCurriculum | `{"reset_mode": "per_step"}` |
| Go2WarpFlatPosTrackProto | lazy: `from jax_rl.envs.locomotion.go2_warp_flat_postrack import default_config` | WarpFlatPosTrack | |

**Rough/HardKick ordering trap:** legacy rough factories set keys in order: osc gains first (if any), then `rough_profile, rough_amplitude, rough_seed, torque_speed_model, physical_armature`. Legacy hardkick wraps base then sets `push_config.vel_min/max` (mutating the existing sub-dict, key order unchanged). Legacy `_physical` sets `torque_speed_model` then `physical_armature` AFTER the var/osc factory ran. If the deep-eq test fails only on ordering, the test's dict comparison ignores order (it compares by keys) — ordering matters only if a downstream consumer iterates keys; it doesn't. Equality test as written compares values, not order — fine.

- [x] **Step 1.4: Iterate until the equality test passes for all 29 names**

`cd <worktree> && uv run python -m pytest tests/test_go2_warp_variants.py -x -q` → 2 passed. Mismatch output shows exact path (e.g. `Go2WarpOscJoystickFlatKp05.osc.kd[0]`) — fix the variant table, not the test.

- [x] **Step 1.5: Commit** — `git add jax_rl/envs/locomotion/go2_warp_variants.py tests/test_go2_warp_variants.py && git commit -m "feat(go2): variants-as-data table + builder, pinned equal to legacy configs"`

### Task 2: registration flips to the table

- [x] **Step 2.1: Add failing registration+cls test** to `tests/test_go2_warp_variants.py`:

```python
def test_registry_uses_variant_cls():
    from mujoco_playground import registry as pg_registry
    import jax_rl.training.env_backends.mjx_backend  # noqa: F401
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    # construction-only; pick the 6 non-default-cls + 2 default-cls names to keep runtime sane
    check = {n: v for n, v in GO2_WARP_VARIANTS.items()
             if v.cls != "WarpJoystick"} | {
        "Go2WarpJoystickFlat": GO2_WARP_VARIANTS["Go2WarpJoystickFlat"],
        "Go2WarpOscFlatSoftPhysical": GO2_WARP_VARIANTS["Go2WarpOscFlatSoftPhysical"]}
    for name, v in check.items():
        env = pg_registry.load(name)
        assert type(env).__name__ == v.cls, f"{name}: {type(env).__name__} != {v.cls}"
```

(Constructing envs compiles MuJoCo models — seconds each on CPU, acceptable. If `pg_registry.load` defaults to GPU/warp put_model and is slow, constructing 8 envs is still fine.)

- [x] **Step 2.2: Run — currently PASSES against legacy registration for cls'd names?** Legacy already registers the right classes, so this test passes pre-change. That's fine — it's a regression pin, not TDD red. Note it and move on.

- [x] **Step 2.3: Replace the Go2 Warp block in `mjx_backend._register_custom_envs`** (lines ~42-226 for the Go2 part; KEEP bongo, G1, splitbelt, factory blocks) with:

```python
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS

    def _resolve_cls(cls_name: str):
        from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick, WarpJoystickNoAccel
        from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
        from jax_rl.envs.locomotion.go2_warp_flat_postrack import WarpFlatPosTrack
        return {"WarpJoystick": WarpJoystick, "WarpJoystickNoAccel": WarpJoystickNoAccel,
                "WarpJoystickCurriculum": WarpJoystickCurriculum,
                "WarpFlatPosTrack": WarpFlatPosTrack}[cls_name]

    for _name, _v in GO2_WARP_VARIANTS.items():
        _reg(_name, _v.config, cls=_resolve_cls(_v.cls))
```

Preserve `_reg` and the explanatory comments about controller-from-config. Delete the now-unused legacy closures (`_warp_osc_default_config_jt`, `_make_osc_kp_config`, `_OSC_BASE_KP/KD`, `_physical`, `_osc_soft_physical`, `_hardkick`, `_bind_rough_cfg`, the torque-speed/unitree/noaccel/physical loops for Go2) — the imports from `go2_warp_osc_joystick/var_impedance/rough` in this file go away too. **Equality test now compares table-to-table (registration sources the same callables), so it degenerates — that's expected; Task 5 replaces it with the snapshot.**

- [x] **Step 2.4: Run** `uv run python -m pytest tests/test_go2_warp_variants.py tests/test_go2_warp_env.py tests/test_go2_warp_curriculum_env.py -x -q` → green.

- [x] **Step 2.5: Commit** — `refactor(go2): register Go2 Warp envs from the variants table`

### Task 3: preset resolution + kill silent fallback

- [x] **Step 3.1: Transitional preset-equality + failing no-fallback test**

```python
def test_go2_presets_resolve_and_unknown_raises():
    from jax_rl.configs import env_presets as ep
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    getters = [("fast_sac", ep.get_fast_sac_preset), ("flash_sac", ep.get_flash_sac_preset),
               ("fast_td3", ep.get_fast_td3_preset), ("sac", ep.get_sac_preset),
               ("td3", ep.get_td3_preset)]
    for name in GO2_WARP_VARIANTS:
        for algo_name, g in getters:
            cfg, algo_cfg = g(name)
            assert cfg.env_name == name
        ppo_cfg = ep.get_preset(name)        # PPO getter returns a BARE TrainConfig (no tuple)
        assert ppo_cfg.env_name == name
    with pytest.raises(ValueError):
        ep.get_preset("Go2WarpNopeDoesNotExist")
    for _, g in getters:
        with pytest.raises(ValueError):
            g("Go2WarpNopeDoesNotExist")
    # splitbelt names must NOT raise (excluded family)
    ep.get_fast_sac_preset("Go2WarpSplitbelt")
```

Plus a TRANSITIONAL check (delete in Step 3.5): before removing legacy dict entries, for every Go2 name present in each legacy `*_PRESETS` table, assert old-entry tuple == new resolution output (dataclass equality). Write it inline as a script-style assertion in the test, run once, then convert: after deletion the test keeps only explicit expected-value asserts for the migrated `train` deltas (curriculum/splitbelt-era values): e.g. `get_fast_sac_preset("Go2WarpJoystickCurriculum")[0].reset_mode == "per_step"`.

- [x] **Step 3.2: Implement `_resolve_go2_variant` in `env_presets.py`** (per spec — returns None for non-Go2 and `Go2WarpSplitbelt*`/excluded; raises for unknown `Go2Warp*`). Wire as the FIRST branch of all 6 getters. **PPO caveat:** `get_preset` (line ~605) returns a bare `TrainConfig`, not a tuple — in that getter use only the cfg half of the resolution (apply `v.train` overrides to the PPO base TrainConfig; `v.algo` has no "ppo" entries today). Do NOT change `get_preset`'s return signature (train_ppo.py callers depend on it). Transcribe existing Go2 entries from each preset table into variant `train`/`algo` dicts: read every `*_PRESETS["Go2Warp..."]` entry in `env_presets.py` (fast_sac: Flat/TorqueSpeed/NoAccel/Unitree/Curriculum/CurriculumTS; flash_sac: Flat/TorqueSpeed/Curriculum/CurriculumTS; fast_td3/sac/td3/ppo: grep) and diff each against its algo base — the diffs (e.g. `reset_mode="per_step"`, `episode_length`) go into `train` if TrainConfig-level and algo-agnostic, into `algo[algo_name]` if algo-specific. Splitbelt entries STAY in the tables (excluded family).
- [x] **Step 3.3: Run transitional equality once** → green, then delete the legacy Go2 (non-splitbelt) entries from all preset tables.
- [x] **Step 3.4: Run full preset tests** `uv run python -m pytest tests/test_go2_warp_variants.py tests/test_env_presets.py tests/test_algo_configs.py tests/test_tdmpc2_presets.py -x -q` → green (update `test_env_presets.py` if it asserts fallback behavior for Go2 names).
- [x] **Step 3.5: Commit** — `refactor(configs): Go2 presets resolve from variants table; unknown Go2Warp names raise`

### Task 4: defaults fix (intentional behavior change)

- [x] **Step 4.1: Failing test:** every variant whose name contains `Osc` or ends in `Physical` or `RoughUni` has `train["reset_mode"]=="per_step"` and `train["eval_every_n_episodes"]==500`; assert via `get_fast_sac_preset` output. EXCEPTIONS: names already migrated with explicit historical values keep them (curriculum keeps per_step + its existing eval default — do not add eval_every to curriculum).
- [x] **Step 4.2: Add `{"reset_mode": "per_step", "eval_every_n_episodes": 500}` to the `train` dict of:** all 8 Osc* flat/JT/Kp-sweep/hardkick variants, all var-impedance variants, SoftPhysical, the 4 *FlatPhysical, JoystickFlatPhysical, the 4 RoughUni. (`Go2WarpJoystickFlat` benchmark family, NoAccel, Unitree, TorqueSpeed, HardKick-joint, PosTrackProto: UNCHANGED.)
- [x] **Step 4.3: Fix `--eval-every` help text** in `scripts/train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py`: replace the stale "every 512 episodes" with "every 5000 episodes; Go2 OSC/physical presets set 500". `train_flashsac.py` (line ~467) has NO stale string — its help is just "Evaluate every N episodes"; append the same default note there.
- [x] **Step 4.4: Run + commit** — `feat(go2): OSC/physical variants default to per_step DR + eval every 500 episodes`

### Task 5: snapshot pin + legacy factory deletion

- [x] **Step 5.1: Generate snapshot** `tests/data/go2_warp_variants_snapshot.json`: `{name: variant.config().to_dict()}` for all 29, via a small inline script (`json.dump(..., indent=1, sort_keys=True, default=list)`). Replace `test_variant_configs_match_legacy_registration` with `test_variant_configs_match_snapshot` (same `_deep_eq` against the JSON; cast lists/tuples consistently).
- [x] **Step 5.2: Migrate remaining importers of the legacy factory modules.** `grep -rn "go2_warp_osc_joystick\|go2_warp_osc_var_impedance\|go2_warp_osc_rough" --include="*.py" jax_rl/ scripts/ tests/ deploy/ docs/ | grep -v __pycache__`. Known dependents: `go2_warp_osc_rough.py` (imports the other two — being deleted), `tests/test_go2_osc_env.py` (rewritten in Task 7 — for now switch its imports to `WarpJoystick + go2_config`), back-compat re-exports of decode helpers (`_N_STIFFNESS, log_action_scale, var_action_size, impedance_gains`) — their canonical home is already `go2_warp_components`; update any importer to that path.
- [x] **Step 5.3: Delete** `go2_warp_osc_joystick.py`, `go2_warp_osc_var_impedance.py`, `go2_warp_osc_rough.py`. `go2_warp_joystick.default_config()` body is replaced by `from .go2_warp_variants import go2_config; return go2_config()` (no cycle: variants imports nothing from env modules at module level).
- [x] **Step 5.4: Full suite** `uv run python -m pytest tests/ -x -q` (GPU tests included; kill zombie GPU procs first: `nvidia-smi | grep python`). Fix fallout.
- [x] **Step 5.5: Commit** — `refactor(go2): delete legacy OSC config-factory modules; snapshot-pin variant configs`

### Task 6: docs regen

- [x] **Step 6.1:** `uv run python docs/scripts/gen_env_presets.py && uv run python docs/scripts/gen_cli_reference.py` (per CLAUDE.md cross-reference table). Commit regenerated docs — `docs: regen env presets + CLI reference for variants table`.

## PR 2 — finish OSC extraction (behavior-identical)

### Task 7: move OSC mechanics into the component

- [x] **Step 7.1:** In `go2_warp_components.py`, move from `go2_warp_joystick.py`: `_compute_nominal_foot_body`, `_feet_in_body`, `_run_osc` (verbatim bodies incl. the STABILITY-CRITICAL comment) as `OSC` methods taking `(self, env, ...)` where they need `env.mjx_model`/`env._torso_body_id`/`env._act_to_joint`/`env._apply_torque_speed_limit`. Controller state becomes `self._kp, self._kd, self._use_lambda, self._ridge, self._target_mode, self._foot_site_ids, self._leg_dof_ids, self._torque_limit, self._nominal_foot_body` — `setup(env)` reads from env, writes ONLY to self. `VarImpedance` inherits; its `apply` passes decoded gains into `self._run_osc(env, data, deltas, kp, kd)`.
- [x] **Step 7.2:** Delete the "OSC mechanics" section from `go2_warp_joystick.py` (lines ~347-406) and the function-local `controller_from_config` docstring references to host-owned mechanics. Host keeps `_apply_control` delegation unchanged.
- [x] **Step 7.3:** Rewrite `tests/test_go2_osc_env.py`: build envs as `WarpJoystick(task="flat_terrain", config=go2_config(controller="osc"))` / `config=go2_config(controller="var_impedance", ...)`; replace `env._osc_foot_site_ids` → `env._controller._foot_site_ids`, `env._feet_in_body(...)` → `env._controller._feet_in_body(env, ...)`; keep every behavioral assertion (FK parity, action sizes, hold probe) intact. `tests/test_go2_osc.py` untouched, must stay green.
- [x] **Step 7.4:** Run GPU OSC tests + variant tests: `uv run python -m pytest tests/test_go2_osc_env.py tests/test_go2_osc.py tests/test_go2_warp_variants.py -x -q`.
- [x] **Step 7.5: Commit** — `refactor(go2): OSC component owns its mechanics + state; host loses planted attrs`

### Task 8: full verification

- [x] **Step 8.1:** Full suite: `uv run python -m pytest tests/ -q` → all green (capture count).
- [x] **Step 8.2:** Smoke run (background, ~200k steps): `cd <worktree> && uv run python scripts/train_fast_sac.py --env Go2WarpOscFlatSoftPhysical --total-timesteps 200000 --num-envs 256 --seed 0 --wandb --wandb-project go2-osc-impedance` → verify: banner shows per_step DR wrapper active, eval fires (eval_every 500 episodes ≈ every ~500k steps won't fire in 200k — pass `--eval-every 50` explicitly to force evals in-smoke), no NaN, sps sane (compare ~earlier OSC runs), checkpoint + meta.json written, meta lists env_name + control block.
- [x] **Step 8.3:** Doc sweep per CLAUDE.md: journal `.context/journals/2026-06-11.md` (refactor + defaults change cut date), `.context/TODO.md`, `.context/AGENT_HANDOFF.md` (env table + "adding a Go2 variant" workflow), `.context/lessons/go2.md` if anything reusable. Commit docs.

**Verification skills:** @superpowers:test-driven-development for every task; @superpowers:verification-before-completion before claiming done.
