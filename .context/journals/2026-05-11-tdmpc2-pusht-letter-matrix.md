# 2026-05-11 — TDMPC2 on PushT letter matrix

## What

Started the TDMPC2 counterpart to the SAC V2 letter matrix
([`.context/studies/2026-04-22_pusht_letter_matrix.md`](../studies/2026-04-22_pusht_letter_matrix.md)).
First TDMPC2 run on a gym-backend env (PushT) after the env-backend
refactor (2026-04-26).

5 specialists + 1 DR, sequential pipeline:

| shape | budget | best step | mppi | prior | gap |
|---|---|---|---|---|---|
| tee  | 500k → killed @ 250k | 150k | **815** | 645 | +170 |
| l    | 200k | 150k | **738** | 686 | +52 |
| k    | 200k | 50k  | **727** | 297 | +430 |
| s    | 200k | 100k | **451** | 127 | +324 |
| dr   | 500k (IN FLIGHT) | 150k so far | 574 | 180 | +394 |

(eval = 8-episode mean, eval-every=25k after v2 pipeline restart)

## Why

- See whether MPPI planning + world model gets better cross-shape OOD
  transfer than the SAC V2 specialists (~0-20% off-diagonal).
- Validate the gym backend path for TDMPC2 (post env-backend refactor).
- Establish a TDMPC2 row in the cross-algo PushT letter matrix.

## Highlights

### Tee 250k collapse — caught by dual-mode eval

Step 50k: mppi=386, prior=336, gap=+50 \
Step 150k: mppi=815 [best] \
Step 200k: mppi=775 \
Step 250k: **mppi=307, prior=732, gap=-425** ← world model went bad

Single-shape pattern: peak ~step 150k, then world model degrades while
policy prior keeps improving. The `mppi_return / prior_return /
mppi_prior_gap` triple in `run_eval` (runtime.py) was decisive — without
the prior baseline we'd think the agent regressed; with it, we see
**the model regressed, not the policy**.

Acted on this: killed pipeline, pivoted to 200k specialists (vs original
500k) + 500k DR. Saved ~3h on the pipeline.

### Plumbing wins from env-backend refactor

- `TDMPC2_PRESETS["PushT"]` works via gym backend, MPPI planning still
  JIT'd on world model.
- `--env-kwargs '{"block_shape":"tee",...}'` plumbed through CLI →
  `dataclasses.replace(train_cfg, env_kwargs=...)`.
- `run_eval` dispatches MPPI/prior rollout via `backend_kind` ∈
  {"mjx","gym"} — gym path is a Python serial loop (since SyncVectorEnv
  can't be JIT-batched), MJX path is the original `lax.scan`.

### S-shape physics tunneling

Recorded `videos/tdmpc2_s_best_step100k.mp4` at default seed: ep1 had
**cov=0.0%** — visual inspection showed agent half-clipped through S
boundary. S has 18 convex wedge polys vs T/L/K with 2-3, so pymunk's
default solver (`iterations=10`) struggles. Recorded again at seed=3000
and got 92.8% / 85.2% / 95.2%(succ) / 93.6% — confirming the 0% was a
sim artifact, not a policy failure.

Knobs filed under `projects/pusht/TODO.md` Backlog. Try `space.iterations=50`
+ wall radius 2→5 first (cheapest).

## Decisions

- **Eval every 25k** (not 50k) for finer-grain visibility of the
  collapse pattern. Negligible cost on PushT.
- **Best ckpt = max mppi over training run** (existing behavior). Saved
  us when tee, l, k all degraded post-peak.
- **DR at 500k, not 2M** — TDMPC2 sample-efficiency >> SAC. SAC needed 5M
  for DR=73.9%; TDMPC2 DR @ 150k = mppi=574. Will revisit if DR plateaus
  far below specialists.
- **S knobs deferred to after matrix completes** — physics change would
  break SAC V2 comparability; defer until we have the TDMPC2 row first.

## Next

- Wait for DR @ 500k (~19:30 ETA).
- Run 5×4 matrix via `projects/pusht/scripts/tdmpc2_letter_matrix.py`.
- Append matrix table + comparison to SAC V2 to this journal.
- Update `.context/studies/2026-04-22_pusht_letter_matrix.md` with TDMPC2 row.
- Then: S physics tuning (see `projects/pusht/TODO.md`).

## Files touched / created

- New: `projects/pusht/` (README, TODO, scripts, videos, artifacts)
- New: `projects/pusht/scripts/tdmpc2_letter_matrix.py` (5×4 MPPI eval)
- New: `projects/pusht/scripts/record_tdmpc2_pusht.py` (gym-render mp4)
- New: `projects/pusht/scripts/run_letter_matrix_pipeline.sh` (sequential train)
- New: `projects/pusht/videos/tdmpc2_{tee,l,k,s}_best_step*.mp4`
