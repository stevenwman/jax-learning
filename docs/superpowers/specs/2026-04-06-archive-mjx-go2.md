# Archive MJX Go2 Environment — Design Spec

**Date:** 2026-04-06
**Status:** Approved

## Motivation

The MJX (pure JAX) Go2 env used Menagerie's simplified MJCF which couldn't match the real robot (wrong damping, wrong actuator type, sphere-only collisions). The Warp env using unitree's MJCF solved all sim2sim issues (FastSAC 276.5, walks 20s+ on CPU). There's no reason to maintain two parallel Go2 environments — Warp is the sole path forward.

## Scope

Remove MJX Go2 env from active codebase. Archive (not delete) the files. Update all docs and site content so nothing references `Go2JoystickFlat` as if it still works.

**Out of scope (separate task):** Kp/Kd DR stays inline in `go2_warp_joystick.py` for now. Moving it to the DR wrapper requires a DR wrapper redesign to support runtime params (not just model-level DR). That's a separate spec — see TODO.

## Code Changes

### Archive files

Move to `jax_rl/envs/locomotion/archive/`:
- `go2_base.py` — MJX base class (Menagerie XML loader + overrides)
- `go2_joystick.py` — MJX joystick env
- `go2_cpu.py` — CPU parity env for MJX pipeline

Move to repo root `archive/`:
- `record_video_cpu.py` — CPU recording script (imports `go2_cpu`, MJX-pipeline only)

Move to `tests/archive/`:
- `test_go2_env.py` — tests for MJX Joystick

### Modify

- **`env_setup.py`** — remove `Go2JoystickFlat` registration (import + register block)
- **`env_presets.py`** — remove `Go2JoystickFlat` PPO preset entry
- **`record_video.py`** — remove `Go2JoystickFlat` from `ENV_DEFAULTS` dict
- **`go2_constants.py`** — add `# ARCHIVED` comment above MJX-only constants (`SCENE_FLAT_XML`, `ROOT_BODY`)
- **`go2_warp_joystick.py`** — no changes (Kp/Kd DR stays inline, deferred to DR wrapper v2)

### Leave as-is (debug tools, not production)

`tools/pd_debug.py`, `tools/kinematic_sweep.py`, `tools/ragdoll_test.py`, `tools/brax_baselines/train_brax_ppo_go2.py` — these import MJX Joystick. They'll break but they're one-off debug scripts, not part of the training pipeline.

## Doc Updates

### Critical — functional references

| File | Change |
|------|--------|
| `README.md:57-58` | Remove MJX training example (`--env Go2JoystickFlat`) |
| `README.md:76` | Remove `Go2JoystickFlat` bullet from env list |
| `README.md:100` | Change CLI example to `Go2WarpJoystickFlat` |
| `deploy/README.md:121` | Change training command to Warp |
| `deploy/README.md:153` | Update PD gains to Kp=20/Kd=0.5 (Warp values) |
| `deploy/README.md:174` | Verify joint order for Warp (unitree actuator order differs) |
| `docs/reference/env-presets.md` | Re-run `gen_env_presets.py` (auto-generated from code) |
| `docs/reference/cli-flags.md:44` | Change example env to `Go2WarpJoystickFlat` |
| `docs/api/envs.md:9` | Remove `go2_joystick.Joystick` autodoc directive |
| `docs/tutorials/sim2real.md:20-25` | Rewrite comparison table — MJX is now archived, Warp is the only path |
| `train_ppo.py:362` | Update `--env` help text example to `Go2WarpJoystickFlat` |
| `train_ppo_fast.py:444` | Same |
| `train_offpolicy.py:4,382` | Same (docstring + help text) |
| `deploy/obs_builder.py:35` | Update docstring reference from `go2_joystick.py` to `go2_warp_joystick.py` |
| `deploy/sim2sim_direct.py:124` | Update comment from `go2_base.py` to `go2_warp_base.py` |

### Project context docs

| File | Change |
|------|--------|
| `.context/AGENT_HANDOFF.md` | Update env listing (single env), remove "MJX frozen" policy, fix example commands |
| `.context/NEW_AGENT_PROMPT.md` | Single env description, note MJX archived |

### Keep as-is (historical value)

- All journal entries — document the debugging journey
- All lesson files — document why MJX was insufficient
- `docs/tutorials/train-locomotion.md` — explains why Warp was chosen
- `docs/reference/lessons-learned.md` — valuable debugging lessons

## Deferred: DR Wrapper v2 (Runtime Params)

Kp/Kd DR currently lives inline in `go2_warp_joystick.py` because these are runtime Python params (external PD), not MuJoCo model fields. The DR wrapper only handles model-level randomization (`mjx.Model` fields like friction, mass). To unify all DR under the wrapper, a redesign is needed where the wrapper can also inject per-env runtime params (Kp/Kd scales, custom force params, action delay ranges, etc.) into `state.info`. This is tracked as a separate TODO item.

## Test Plan

1. Run `uv run pytest tests/ -v` — all tests pass (no MJX Go2 tests to fail)
2. Verify `Go2WarpJoystickFlat` still loads and steps: `uv run python -c "from jax_rl.training.env_setup import make_envs; ..."`
3. Verify `Go2BongoHandstand` unaffected
4. `mkdocs build` succeeds with no broken references
5. Grep for `Go2JoystickFlat`, `go2_base`, `go2_cpu`, `go2_joystick` (not warp) across all non-archived `.py` files — zero functional hits
