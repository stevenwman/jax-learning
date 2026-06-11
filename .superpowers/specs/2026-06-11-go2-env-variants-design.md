# Go2 Warp Env Variants-as-Data + OSC Extraction Completion

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Scope decisions (Steven):** Go2 Warp family only; fix OSC/physical training defaults in the same refactor.

## Problem

Audit (2026-06-10) found four structural problems on the env side:

1. **Patch-chain configs.** A named env like `Go2WarpOscFlatSoftPhysical` is defined as a closure in `jax_rl/training/env_backends/mjx_backend.py` that patches the output of `go2_warp_osc_joystick.default_config()`, which itself patches `go2_warp_joystick.default_config()`. No single place shows the final config; answering "what is this env" requires replaying mutations across 3 files.
2. **Science in infra.** `_register_custom_envs` in `mjx_backend.py` is ~420 lines of experiment definitions (gains, kick DR, motor models) inside a backend plumbing file.
3. **Two registries, silent fallback.** The same env-name string keys both the Playground registration (env physics) and `env_presets.py` (training HPs). `get_fast_sac_preset` silently returns base defaults for unknown names — the OSC envs therefore train with `reset_mode="legacy"` (no domain randomization) and `eval_every_n_episodes=5000` (≈ zero mid-run evals on a 5M-step run) without any indication.
4. **Half-finished controller extraction.** `OSC.setup()` (in `go2_warp_components.py`) plants `env._osc_kp`, `env._osc_foot_site_ids`, etc. onto the host env; the OSC math (`_run_osc`, `_feet_in_body`, `_compute_nominal_foot_body`) stayed in `go2_warp_joystick.py` and reads those planted attributes back. The white-box consumers are `tests/test_go2_osc_env.py` (constructs `WarpOscJoystick`/`WarpOscVarImpedance` directly, reads `env._osc_foot_site_ids`, calls `env._feet_in_body`) — PR 2 must migrate that test to the new surfaces. `WarpOscJoystick` is an otherwise-empty subclass whose only dependents are that test and `WarpOscVarImpedance` (which inherits from it).

## Goals

- One declaration per named Go2 Warp env: full env config + training overrides + reset mode, in one record, env-side.
- `mjx_backend.py` reduced to plumbing for the Go2 family.
- Unknown `Go2Warp*` name in preset lookup raises instead of silently falling back.
- OSC/physical variants train with DR (`per_step`) and a sane eval cadence by default.
- OSC controller logic lives in one file; no attribute planting; vestigial class deleted.

## Non-goals

- No manager-based / runtime-dispatch env framework (explicitly rejected; `step()` stays a single readable function).
- No migration of G1, splitbelt, bongo, factory, or DMC/gym envs (later, if the pattern proves out).
- No typed-dataclass replacement of `ml_collections.ConfigDict` (woven through Playground's `MjxEnv`/`registry.load` contract).
- No changes to algos, training loop, buffers, networks, deploy code.

## Design

### PR 1 — variants-as-data

#### New file: `jax_rl/envs/locomotion/go2_warp_variants.py`

Import-light (stdlib + `ml_collections` only — no mujoco/Playground/jax), so `env_presets.py` can import it cheaply.

```python
@dataclass(frozen=True)
class EnvVariant:
    config: Callable[[], config_dict.ConfigDict]  # full env config, one builder call
    cls: str = "WarpJoystick"      # env class NAME; resolved lazily in mjx_backend
    train: dict = field(default_factory=dict)         # TrainConfig overrides (all algos)
    algo: dict[str, dict] = field(default_factory=dict)  # per-algo HP overrides, keyed by algo_name
    notes: str = ""

GO2_WARP_VARIANTS: dict[str, EnvVariant] = { ... }
```

Names migrating (29 — the list below is exhaustive): the `WarpJoystick`-host family —
`Go2WarpJoystickFlat`, `...FlatTorqueSpeed`, `...FlatNoAccel` (cls `WarpJoystickNoAccel`), `...Unitree` (cls `WarpJoystickNoAccel`),
`...FlatPhysical`, `...FlatHardKick`,
`Go2WarpOscJoystickFlat`, `...FlatJt`, `...FlatKp{025,05,2,4}`, `...FlatKp05HardKick`,
`Go2WarpOscFlatSoftPhysical`,
`Go2WarpOscVarImpedance{Flat,AxisFlat}`, `Go2WarpOscVar{,Axis,Damping,DampingAxis}FlatPhysical`,
`Go2WarpOscVarImpedance{,Axis}HardKickFlat`,
`Go2Warp{Joint,Osc,OscVar,OscVarAxis}RoughUni`,
`Go2WarpJoystickCurriculum`, `...CurriculumTorqueSpeed` (cls `WarpJoystickCurriculum`),
`Go2WarpFlatPosTrackProto` (cls `WarpFlatPosTrack`).
Excluded: `Go2WarpSplitbelt*` (own host class, own file — out of scope), `Go2BongoHandstand*`, G1, factory, DMC/gym.

#### One config builder

`go2_config(...)` in the variants file (or a sibling `go2_warp_config_builder.py` if the variants file gets crowded), with explicit axes:

```python
def go2_config(
    *,
    controller="joint_pd",            # "joint_pd" | "osc" | "var_impedance"
    osc_kp=None, osc_kd=None,         # required when controller != "joint_pd"
    use_op_space_inertia=True,        # False = Jt impedance ablation
    target_mode="abs_body",
    stiffness_granularity=None,       # "per_foot" | "per_axis" (var_impedance)
    damping_action=False, var_s=(0.25, 2.0), var_zeta=(0.5, 2.0),
    motor="ideal",                    # "ideal" | "torque_speed" | "physical"
    terrain="flat",                   # "flat" | ("uniform", amplitude)
    push=(0.75, 0.75),
    action_scale=None,                # default: 0.5 joint_pd, 0.12 cartesian
) -> config_dict.ConfigDict
```

Builds the complete ConfigDict in one pass from the base values currently in `go2_warp_joystick.default_config()`. No factory patches another factory's output. Every variant's declaration shows its distinguishing values literally (e.g. `osc_kp=[1500, 1500, 2000]`).

The existing `default_config()` functions in `go2_warp_osc_joystick.py` / `go2_warp_osc_var_impedance.py` / `go2_warp_osc_rough.py` become one-line calls into `go2_config(...)` during the transition, then delete once nothing imports them.

#### Registration

`mjx_backend._register_custom_envs`: the Go2 Warp block collapses to

```python
from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
for _name, _v in GO2_WARP_VARIANTS.items():
    _reg(_name, _v.config, cls=_resolve_cls(_v.cls))
```

`_resolve_cls` maps class names to classes via local imports (keeps heavy imports out of the variants file). G1/splitbelt/bongo/factory registration blocks are untouched.

#### Preset resolution

In `env_presets.py`, a single helper used by every `get_<algo>_preset`:

```python
def _resolve_go2_variant(env_name, base_cfg, base_algo, algo_name):
    v = GO2_WARP_VARIANTS.get(env_name)
    if v is None:
        if env_name.startswith("Go2Warp") and env_name not in _EXCLUDED:  # splitbelt
            raise ValueError(f"unknown Go2 Warp env {env_name!r}; known: {sorted(GO2_WARP_VARIANTS)}")
        return None   # not Go2 — caller falls through to existing behavior
    cfg = dataclasses.replace(base_cfg, env_name=env_name, **v.train)
    algo = dataclasses.replace(base_algo, **v.algo.get(algo_name, {})) if v.algo.get(algo_name) else base_algo
    return cfg, algo
```

- Wired into `get_fast_sac_preset`, `get_flash_sac_preset`, `get_fast_td3_preset`, `get_sac_preset`, `get_td3_preset` (and the PPO preset getter if one exists for Go2).
- Existing hand-written Go2 entries in `FAST_SAC_PRESETS` / `FLASH_SAC_PRESETS` / etc. migrate into the variants' `train` dicts and the old entries delete. Per-algo-only deltas (e.g. FlashSAC `grad_updates_per_step`) go in `algo`.
- DMC/gym names keep today's silent fallback — legitimate there.

#### Defaults fix (behavior change, scoped)

- All OSC, var-impedance, and `*Physical` variants declare `train={"reset_mode": "per_step", "eval_every_n_episodes": 500}`.
- Variants that already had explicit preset entries (`Go2WarpJoystickFlat` benchmark family, curriculum) keep their exact current values — benchmark history stays comparable.
- Fix the stale `--eval-every` help text in the off-policy train scripts ("default: every 512 episodes" → actual default 5000).

### PR 2 — finish the OSC controller extraction (behavior-identical)

In `go2_warp_components.py`:
- Move `_run_osc`, `_feet_in_body`, `_compute_nominal_foot_body` from `go2_warp_joystick.py` into the `OSC` component (as methods; `VarImpedance` inherits).
- Controller state moves onto the controller: `self._kp`, `self._kd`, `self._use_lambda`, `self._ridge`, `self._target_mode`, `self._foot_site_ids`, `self._leg_dof_ids`, `self._torque_limit`, `self._nominal_foot_body`. `OSC.setup(env)` reads what it needs from the env (sites, stall torque, model) but writes nothing onto it.
- The stability-critical comment on the 250 Hz substep recomputation moves with `_run_osc` verbatim.
- `WarpJoystick._apply_control` delegation is unchanged; the host loses the "OSC mechanics" section entirely.
- Delete `WarpOscJoystick` AND `WarpOscVarImpedance` (which inherits from it) — after PR 1, both are config presets over `WarpJoystick`; their files survive only while `default_config*` factories are still imported during the PR-1 transition, then delete.
- **Update `tests/test_go2_osc_env.py`:** construct envs as `WarpJoystick(config=go2_config(controller="osc", ...))` (or via registry name), and read controller state from `env._controller` (e.g. `env._controller._foot_site_ids`, `env._controller._feet_in_body(...)`) instead of planted `env._osc_*` attrs. Test intent (FK parity, hold probe, action-size checks) is preserved — only the access path changes.
- `tests/test_go2_osc.py` (synthetic-leg math on `go2_osc.compute_leg_impedance_torque`) is untouched by both PRs and must stay green as-is.

Ordering: PR 1 first (it rewrites the config factories PR 2's file deletion depends on). PR 2 is independent in logic but sequenced after to avoid rebase churn.

## Back-compat

- Every existing registered name keeps working verbatim — checkpoint `meta.json` `env_name` resolution, `record_video.py`, and `sim2sim_direct.py` are unaffected.
- `projects/mud_eval/` imports only `jax_rl.algos.fast_sac` + checkpoint contract — unaffected. Its `mud_osc.py` doc references `go2_osc.py` by name; `go2_osc.py` (the torque math) is NOT moved by this design, only its callers.
- `deploy/` reads `meta["control"]` — unchanged (`get_control_metadata` untouched).

## Verification

1. **Config-equality transitional test:** before deleting the legacy factories, a test builds every migrated name's ConfigDict via the old patch-chain AND the new builder and asserts deep equality (pure python, no GPU). Legacy factories delete in the same PR after the test passes; the test then pins the new builder against a frozen snapshot of the dicts (committed as JSON) so future edits to the builder are intentional.
2. **No-fallback test:** every name in `GO2_WARP_VARIANTS` resolves a preset for each off-policy algo; a made-up `Go2WarpNope` raises.
3. **Registration test:** every variant name loads via `pg_registry` (construction only, no stepping; CPU).
4. **Existing suite green:** `uv run python -m pytest tests/` — `test_env_presets.py`, `test_go2_warp_env.py`, `test_go2_warp_curriculum_env.py` likely need updates for the new resolution path; `test_go2_osc_env.py` is rewritten by PR 2 (see above).
5. **Smoke run:** 200k steps of `train_fast_sac.py --env Go2WarpOscFlatSoftPhysical` (now with DR + eval cadence) before any real training jobs. Per project convention, no cross-run bit-identity gating (GPU nondeterminism); within-run sanity only.

## Risks

- **Builder kwarg creep:** if `go2_config` grows past ~15 params it's recreating the patch-chain as kwargs. Mitigation: params map 1:1 to the physical axes listed above; anything more exotic stays a literal ConfigDict field set at the declaration site.
- **Hidden name dependents:** scripts/docs may reference deleted helper factories (`_make_osc_kp_config` etc.). Mitigation: repo-wide grep before deletion; `docs/scripts/gen_env_presets.py` re-run (per CLAUDE.md cross-reference table).
- **Behavior change surprises:** OSC variants now train with DR — intentional, but invalidates loose comparisons against earlier no-DR OSC runs. Mitigation: journal entry stating the cut date; old runs remain labeled by their meta.json.
