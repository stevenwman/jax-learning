# PushT Project

Research project: 2D planar pushing on the PushT env family (gym-pusht +
custom letter shapes T/L/K/S). Focus: cross-shape OOD generalization,
domain-randomization (DR), and contrasting model-free (SAC, FastSAC) vs
model-based (TDMPC2) on the same keypoint observation.

## Status (2026-05-11)

- **SAC letter matrix V2 (N=10 dense KP obs)** — DR row-mean 73.9% sto /
  57.8% det. Specialists 87-94% on-shape, ~0-20% off-shape. See
  [`.context/studies/2026-04-22_pusht_letter_matrix.md`](../../.context/studies/2026-04-22_pusht_letter_matrix.md).
- **TDMPC2 letter matrix** — IN FLIGHT 2026-05-11. Sequential training
  T/L/K/S × 500k + DR × 2M (gym backend, MPPI planning). Tee step-150k
  ckpt already hitting 95.9% cov in 12 steps (`videos/tdmpc2_tee_step150k.mp4`).

## What lives here

| Subdir | Contents |
|---|---|
| `scripts/` | PushT-specific orchestrators: `tdmpc2_letter_matrix.py` (5×4 MPPI eval), `record_tdmpc2_pusht.py` (gym-render rollout → mp4) |
| `videos/` | Rendered MPPI/SAC rollouts. Gitignored. |
| `artifacts/` | Saved matrix CSVs / .md / plots. Gitignored. |
| `specs/` | PushT-specific design specs (none yet). |
| `plans/` | PushT-specific implementation plans (none yet). |

Run from repo root:

```bash
# Train one shape via TDMPC2 (gym backend)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python scripts/train_tdmpc2.py \
  --env PushT --total-timesteps 500000 --num-envs 8 --seed 0 \
  --eval-every 50000 --ckpt-dir .temp/tdmpc2_pusht_tee_500000 \
  --env-kwargs '{"obs_type":"keypoints","block_shape":"tee","reward_mode":"contact_gated","coverage_shape":"log_barrier","coverage_eps":0.01}' \
  --wandb --wandb-project pusht-tdmpc2-letter

# Record an MPPI rollout
uv run python projects/pusht/scripts/record_tdmpc2_pusht.py \
  --ckpt-dir .temp/tdmpc2_pusht_tee_500000/best --shape tee \
  --num-episodes 2 --out projects/pusht/videos/tdmpc2_tee_best.mp4

# Run the 5×4 letter matrix (after all 5 ckpts trained)
uv run python projects/pusht/scripts/tdmpc2_letter_matrix.py \
  --ckpts .temp/tdmpc2_pusht_tee_500000 .temp/tdmpc2_pusht_l_500000 \
          .temp/tdmpc2_pusht_k_500000 .temp/tdmpc2_pusht_s_500000 \
          .temp/tdmpc2_pusht_dr_2000000 \
  --n-episodes 5
```

## What does NOT live here (intentionally)

- **Env code** stays in `jax_rl/envs/manipulation/pusht/` — shapes.py,
  pusht_env.py. Shared infra.
- **Generic scripts** — `scripts/train_tdmpc2.py`, `scripts/train_sac.py`,
  `scripts/record_video.py`. Used by every project.
- **Lessons** — `.context/lessons/pusht_design.md`,
  `.context/lessons/pusht_physics.md`, `.context/lessons/pusht_rewards.md`.
  Cross-cutting.
- **Studies** — `.context/studies/2026-04-21_pusht_transfer_matrix.md`,
  `.context/studies/2026-04-22_pusht_letter_matrix.md`. Repo-wide history.
- **Specs / plans** for repo-wide work stay in `.superpowers/{specs,plans}/`.
- **Checkpoints** stay in `.temp/` (training-output convention).

## Key numbers

| Algo | Obs | DR row-mean | Notes |
|---|---|---|---|
| SAC (V1, padded) | 27d (N=11 zero-pad) | 32.9% sto / 24.8% det | Pad-leak hypothesis |
| SAC (V2, dense) | 25d (N=10 dense) | **73.9% sto / 57.8% det** | Pad-leak fix |
| TDMPC2 (in flight) | 25d (N=10 dense) | TBD | MPPI on world model |
