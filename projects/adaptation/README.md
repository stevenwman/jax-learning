# Adaptation Project

Research project: studying within-episode + across-episode adaptation in
locomotion policies on perturbed treadmills (split-belt, etc.). Uses Go2
quadruped on the splitbelt env family (`jax_rl/envs/locomotion/go2_warp_splitbelt.py`).

## What lives here

- `sweeps/` — project-specific orchestrators that load a trained ckpt and
  drive a focused analysis. They reuse `scripts/record_video.py` and the
  shared `jax_rl.utils.eval` infra; they're not generic enough for top-level.
- `videos/` — generated rollout videos and `_traj.npz` sidecars. Gitignored.
- `artifacts/` — saved sweep CSVs / npz / plots. Gitignored.

Run everything from repo root:
```bash
cd /home/stevenman/Desktop/Work/Research/jax-learning
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python projects/adaptation/sweeps/eval_physics.py \
    --checkpoint checkpoints/<ckpt>/best --pairs 0.5,0.5 0.5,1.5 ...
```

## What does NOT live here (intentionally)

- **Env code** stays in `jax_rl/envs/locomotion/` — `go2_warp_splitbelt.py`,
  `splitbelt_schedules.py`, `splitbelt_geom.py`, `splitbelt_analysis.py`.
  The env is shared infra registered via `mjx_backend.py`; any ckpt trained
  on it depends on the lib path.
- **Tests** stay in `tests/test_splitbelt_*.py`.
- **Generic scripts** stay in top-level `scripts/`: `record_video.py`,
  `train_*.py`. These are used by every project.
- **Lessons** stay in `.context/lessons/splitbelt.md` (cross-cutting).
- **Specs / plans** stay in `.superpowers/{specs,plans}/`.
- **Checkpoints** stay in `checkpoints/` (env name is in the path so it
  remains tractable across projects).

## Sweep scripts

| Script | Purpose |
|---|---|
| `sweeps/eval_ood.py` | Reward-mean OOD sweep across tied speeds OR (vL,vR) pairs |
| `sweeps/eval_physics.py` | Cold-hard physics-metric sweep — termination cause, lag, sway, survival. Use this, not eval_ood, for failure-mode analysis. |
| `sweeps/eval_a1.py` | Within-episode adaptation probe under `tied_split_tied` schedule. Reports per-phase touchdown rate + drift. |
| `sweeps/record_at_speed.py` | Record a single rollout video at a fixed `tied(v)` belt speed |
| `sweeps/record_sweep.py` | Record multiple videos at various (vL,vR) pairs into one folder |

## Current state (2026-05-07)

PoseDR v2 — `checkpoints/20260506_195126_fast_sac_go2warpsplitbeltposedr_seed0/best`:
- Trained 1M FastSAC on `random_per_episode` belt schedule
- Eval 378.9 ± 113.4 (16 ep)
- Physics sweep: **0% term at tied 0.5; 75-100% term at every other tested condition.**
  Tilt is the dominant failure mode (91/144 ep across 9 conditions).

## Open follow-ups

- Train PoseDR v3 with broader DR range (`ratio ∈ [0.3, 3.0]`) + heavier
  orientation reward (test the tilt hypothesis from `lessons/splitbelt.md`).
- Train splitbelt with `error` obs_mode (`cmd_track_error + drift_xy` in
  actor) — deployable variant of pose_track.
- Train PoseDR with `tied_split_tied` schedule for actual A1 within-episode
  adaptation study (current PoseDR sees mid-episode belt change as OOD).
- Per-protocol algo presets (A1/A2/A3/A4) — fresh brainstorm/spec/plan cycle.

See `.context/TODO.md` (top section) for the live priority list.

## References

- Spec: `.superpowers/specs/2026-05-02-splitbelt-treadmill-env-design.md`
- Plan: `.superpowers/plans/2026-05-02-splitbelt-treadmill-env.md`
- Lessons: `.context/lessons/splitbelt.md`
- Build journal: `.context/journals/2026-05-03-splitbelt-env-build.md`
- Seed prompt for fresh agent: `.context/SEED_PROMPT_2026_05_05.md`
