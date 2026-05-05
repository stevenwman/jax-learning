# Splitbelt Continuation Seed Prompt (2026-05-05)

> Paste this into the next session if context was compacted. Doc paths are absolute from repo root. The whole point: skip you reading 50 commits to figure out where you are.

You're picking up an in-progress JAX RL project. The active branch is `new_slate_linen`. Today's date: 2026-05-05+. Read in this order before doing anything:

1. `.context/AGENT_HANDOFF.md` — project overview + current Go2 env table (search "Go2WarpSplitbelt" — three rows)
2. `.context/journals/2026-05-03-splitbelt-env-build.md` — full splitbelt build journal (3 days, all bug stories)
3. `.context/lessons/splitbelt.md` — 6 load-bearing lessons from this work
4. `.context/TODO.md` — top section is splitbelt completion + open follow-ups

## Current state of splitbelt work

Three env variants registered + tested:

| Env | Obs | Schedule | Eval @ 1M FastSAC | Ckpt |
|---|---|---|---|---|
| `Go2WarpSplitbelt` | blind (joystick-aligned) | tied(0.5) | 72.3 ± 13.4 (v2 ckpt invalidated by obs rename — DON'T use for inference) | n/a |
| `Go2WarpSplitbeltDR` | blind | random_per_episode (v∈[0.3,1.5], ratio∈[0.5,2.0]) | not trained | n/a |
| `Go2WarpSplitbeltPoseDR` | pose_track (world body pos+upvec+fwdvec) | random_per_episode | **280.6 ± 207.8** | `checkpoints/20260505_184757_fast_sac_go2warpsplitbeltposedr_seed0` |

PoseDR is idealized (NOT real-robot deployable — actor sees ground-truth world-frame pose). Use it as the strongest baseline for stabilization studies.

## Hard rules / invariants

- `uv run python` always; no `python`/`python3`.
- No `Co-Authored-By` in commits.
- RTX 5080 16GB + splitbelt 14-actuator model: use `XLA_CLIENT_MEM_FRACTION=0.55` + `--num-envs 512`. 1024 envs OOMs Warp graph during eval env construction.
- Belt schedule semantics is biomech "drag speed": positive value = belt drags foot backward. Joint qvel and ctrl are NEGATED in env code. See spec §6.3.
- Splitbelt blind obs schema = joystick NoAccel (45d, same field names + order). Future Go2-family envs should preserve this for cross-deploy.
- `Go2WarpEnv.__init__:64-67` clobbers `actuator_forcerange`. Splitbelt's `_post_init` re-mutates + re-puts model. Smoke assertion right after — failure means MJX semantics changed.
- Don't trust `<contact data="found">` against MuJoCo plane geoms (margin-fires). Use position-based termination if needed. Splitbelt uses `splitbelt_geom.foot_belt_id` for off-belt detection.
- `lax.scan` over `action_repeat` requires reset & step `state.metrics` dicts to have IDENTICAL keys. Splitbelt initializes `splitbelt/term_cause` in reset so step's pytree matches.

## Open work in priority order

1. **OOD eval on PoseDR ckpt at unseen belt speeds** — most valuable next move. PoseDR was trained with belt speed sampled from [0.3, 1.5]. Eval at fixed v ∈ {0.5 (in-dist), 1.5 (boundary), 2.0, 2.5 (OOD)}. Need to override env's `schedule_kind` and `schedule_params` at eval time (or build a new fixed-speed eval env). Compare drift trajectories + episode lengths. This was user's stated motivation for the pose_track variant.
2. **A1 protocol probe with PoseDR ckpt** — record `tied_split_tied(v_warm=0.5, vL_split=0.5, vR_split=1.0, t1=200, t2=600)` rollout. Run offline analyzer (`jax_rl/envs/locomotion/splitbelt_analysis.py`). Measures `recovery_time` + `after_effect`. First real adaptation-benchmark data without per-protocol training.
3. **Train splitbelt with `error` obs_mode** (cmd_track_error + drift_xy in actor). Closes asymmetric-AC actor-blindness gap. Compare to PoseDR (pos directly) and to blind v2 (no translation signal).
4. **Per-protocol algo presets (A1/A2/A3/A4)** — fresh brainstorm/spec/plan cycle. Reuse schedule samplers; add per-protocol algo configs.
5. **History-mode `FrameStackWrapper` wiring in `mjx_backend`** — still deferred. Required for A3 protocol.

## Tools / patterns to reuse

- Train: `XLA_CLIENT_MEM_FRACTION=0.55 uv run python scripts/train_fast_sac.py --env <name> --num-envs 512 --total-timesteps 1000000 --reset-mode per_step --seed N`
- Record: `MUJOCO_GL=egl uv run python scripts/record_video.py --checkpoint <ckpt>/best --max-steps 1250 --no-early-term`
- Sidecar gait analyzer: `jax_rl.envs.locomotion.splitbelt_analysis` (numpy lib; `detect_step_events`, `step_lengths`, `step_length_asymmetry`)
- Tests: `JAX_PLATFORMS=cpu uv run python -m pytest -q tests/test_splitbelt_*.py` (hermetic). GPU: `uv run python -m pytest -q -m "gpu and warp and go2" tests/test_splitbelt_*.py`.
- Default lane baseline: 754 passed before splitbelt work; should stay green.

## Pattern: the multi-agent audit cycle

Spec → plan → 4 rounds of parallel multi-agent audits (spec-consistency / plan-vs-spec / codebase-grounding / testing-contract / env-backend / algo-agnosticism). Each round caught real issues. Asymptote at round 4 (1 high + 1 blocker after R3 fixes). User's framing: "stop iterating once we're catching only correctness issues my own fixes introduce."

## Don't do these (lessons learned the hard way)

- Don't add speculative gates / abstractions during plan-fix cycles — they introduce new bugs faster than they catch them.
- Don't retrain a ckpt unless a use case demands it. v2 ckpt got invalidated by obs rename; retraining "for hygiene" was wrong.
- Don't trust the `floor_found` sensor. Don't use `gravity_body[2]` for upright detection — use `get_upvector()`.
- Don't kill running trainings without confirming the user actually wants the change driving the kill.

## Working-with-this-user notes

- Direct, terse, "yuh" / "rip" / "ngl" — engineering signals not noise.
- Pushback when warranted; don't yes-machine.
- Surface assumptions before non-trivial work; let user correct.
- Caveman mode active by hook (terse fragments, drop articles, code blocks normal).
- Train in background with `ScheduleWakeup ~270s` for first eval check.
