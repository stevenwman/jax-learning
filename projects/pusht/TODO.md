# PushT Project — TODO

Repo-wide TODO at `.context/TODO.md` (cross-project items). This file is
PushT-specific.

## Open

- [ ] **TDMPC2 letter matrix run** — IN FLIGHT 2026-05-11. Sequential
  training T/L/K/S × 500k + DR × 2M (pipeline at `/tmp/tdmpc2_pusht_pipeline.sh`,
  PID 3966504, log `/tmp/tdmpc2_pipeline.log`, wandb
  `pusht-tdmpc2-letter`). ETA ~12-20h.
- [ ] **Run 5×4 TDMPC2 matrix** once all 5 ckpts trained
  (`projects/pusht/scripts/tdmpc2_letter_matrix.py`).
- [ ] **Update study doc** with TDMPC2 row in
  `.context/studies/2026-04-22_pusht_letter_matrix.md` once matrix done.
- [ ] **Commit `projects/pusht/scripts/` to repo** (currently scripts
  exist but matrix.py is also in `/tmp/`). Move pipeline.sh into the
  project too.

## Backlog

- [ ] **Fix S-shape physics tunneling** — observed in
  `videos/tdmpc2_s_best_step100k.mp4`: contact model lets agent half-clip
  through S boundary. Root cause: S has 18 convex wedge polys vs T/L/K's
  2-3 — many small contacts overwhelm pymunk solver defaults.
  Knobs in [`jax_rl/envs/manipulation/pusht/pusht.py:665`](../../jax_rl/envs/manipulation/pusht/pusht.py#L665):
    1. `space.iterations = 50` (default 10) — solver stiffness
    2. Wall `radius=2 → 5` (lines 672-675) — thicker walls
    3. `collision_slop = 0.01` (default 0.1) — tighter penetration
    4. `dt = 0.01 → 0.005` + 2× substeps — best accuracy, 2× slowdown
    5. Poly `radius=2` in `shapes.py` builders — rounded-corner CCD
  Try #1+#2 first (cheapest). Caveat: breaks SAC V2 comparability —
  either retrain SAC matrix on v3 physics, or label TDMPC2 row "v3 phys".
- [ ] **HP tuning for TDMPC2 PushT** — current preset uses
  `make_tdmpc2_config(action_dim=2, episode_length=300)` defaults
  (horizon=3, latent_dim=512). Sweep horizon ∈ {3,5,7}.
- [ ] **FastSAC PushT letter matrix** — counterpart to SAC V2 + TDMPC2
  for a 3-way comparison.
- [ ] **Cross-algo matrix doc** — combine SAC/FastSAC/TDMPC2 numbers in
  one figure.

## Done

- [x] **2026-04-21** Initial transfer matrix (state obs).
- [x] **2026-04-22** Letter matrix V1 (N=11 zero-padded) — pad-leak found.
- [x] **2026-04-24** Letter matrix V2 (N=10 dense) — pad-leak fix.
- [x] **2026-04-26** Env-backend refactor (PushT now on gym backend).
- [x] **2026-05-11** TDMPC2 plumbing — gym backend dispatch + PushT
  preset + `--env-kwargs` CLI.
- [x] **2026-05-11** PushT TDMPC2 video recorder
  (`projects/pusht/scripts/record_tdmpc2_pusht.py`).
