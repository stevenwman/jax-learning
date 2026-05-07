# Skill Discovery Project

Research project: unsupervised skill discovery for locomotion. Train a policy
without task rewards that learns a *family* of distinct, conditionable behaviors
(`z`-conditioned). Goal: reusable behaviors for downstream task composition,
fine-tuning, or selection — eventually for Go2 hardware deployment.

## Status (2026-05-07)

- **SD-A + SD-B** complete — DIAYN aux module, manager, train script, parity tests, 1M validation on CheetahRun.
- **SD-B Wave D Gate 2** closed via Ant MJX port (3 seeds × 1M, all PASS numerical visual gate).
- **METRA baseline contrast** complete — 3 seeds × 1M METRA on AntMJXClassic. Hypothesis "METRA Lipschitz constraint produces wider xy-spread than DIAYN" REJECTED at default reference HPs (DualLam → 0 in all 3 seeds). Documented as informative null with follow-up: try `dual_dist="l2"`.
- Pending: SD-C (Go2 deploy), SD-D (D3 factor decomposition), SD-E (DUSDi factored MI).

## Headline figures

- [`figures/ant_classic_diayn_3seeds.png`](figures/ant_classic_diayn_3seeds.png) — DIAYN 3-seed canonical xy-trajectory diversity (closes SD-B Gate 2)
- [`figures/ant_classic_diayn_vs_metra_3seeds.png`](figures/ant_classic_diayn_vs_metra_3seeds.png) — 6-panel DIAYN-vs-METRA comparison (null delta)
- [`figures/ant_classic_vs_v5_seed0.png`](figures/ant_classic_vs_v5_seed0.png) — Classic (27d) vs v5 (105d) obs comparison; richer obs lets discriminator hide skill diffs in cfrc minutiae

## What lives here

| Subdir | Contents |
|---|---|
| `scripts/` | Project-specific orchestrators: `plot_skill_xy.py` (DIAYN-canonical xy figure + numerical gate), `stitch_skill_xy_panels.py` (multi-panel matplotlib stitcher) |
| `lessons/` | `diayn_cheetah.md`, `diayn_ant.md`, `metra_ant.md` — narrative-locked findings |
| `references/` | Source code extracts (METRA, DIAYN, D3, DADS), validation methodology |
| `plans/` | Implementation plans: SD-A (config/prior/factors/manager), SD-B (off-policy DIAYN loop + Wave A-D), Ant MJX port, METRA on Ant |
| `specs/` | `2026-04-28-skill-discovery.md` — canonical SD-A through SD-E design spec |
| `figures/` | All headline xy-trajectory PNGs (DIAYN + METRA, per-seed + stitched panels) |
| `artifacts/` | Gitignored — future ad-hoc analysis outputs (CSVs, npz) |

Run everything from repo root:
```bash
cd /home/stevenman/Desktop/Work/Research/jax-learning
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python projects/skill-discovery/scripts/plot_skill_xy.py \
    --checkpoint checkpoints/<ckpt> \
    --rollouts-per-skill 3 --rollout-length 500 \
    --output projects/skill-discovery/figures/<output>.png
```

## What does NOT live here (intentionally)

- **Library code** — `jax_rl/skill_discovery/` (config, prior, factors, diayn, metra, manager, checkpointing) stays in the main library; reusable across projects.
- **Env code** — `jax_rl/envs/locomotion/ant.py`, `xmls/ant.xml` — env classes are infra, used by anything that wants Ant.
- **Generic train scripts** — `scripts/train_skill_discovery.py`, `scripts/record_video.py`, `scripts/train_sac.py` stay top-level. Project-specific orchestrators live here.
- **Tests** — `tests/test_metra.py`, `tests/test_ant_*.py`, `tests/test_skill_discovery_*.py` stay in `tests/` (library-scoped).
- **Journals** — chronological session entries stay in `.context/journals/`. They reference this project's docs by absolute path (`projects/skill-discovery/...`).
- **Checkpoints** — `checkpoints/<timestamp>_*` flat at repo root (env name encoded in dirname keeps cross-project tractability).
- **Logs** — `.temp/logs/` is the workspace-level log scratch.

## Cross-references

- Workspace lesson index: [`.context/LESSONS.md`](../../.context/LESSONS.md) (points back to lessons here)
- Workspace TODO: [`.context/TODO.md`](../../.context/TODO.md) §"Skill Discovery (V2)" tracks high-level phase status
- Companion projects: [`projects/adaptation/`](../adaptation/) (splitbelt locomotion adaptation)
