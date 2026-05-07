# Skill Discovery — TODO

Repo-wide TODO is at `.context/TODO.md` (only items spanning multiple
projects). This file is skill-discovery-specific.

> **2026-05-02 retitle:** old "Phase 6A–6E" replaced by SD-A through SD-E in
> `specs/2026-04-28-skill-discovery.md`. The old bullets were optimistic
> about ObsSpec/RewardSpec being sufficient — the V2 audit found off-policy
> buffer, frame-stack, normalization, checkpoint, and deploy contracts also
> need work.
> Informed by D3 paper (arXiv:2508.19953) and leggedrobotics/d3-skill-discovery.
> See `references/d3_skill_discovery.md`.

## Status (2026-05-07)

| Phase | Status | Headline |
|---|---|---|
| SD-A | ✅ DONE | config + priors + factor registry + DIAYN aux + manager (5 commits, 31/31 tests) |
| SD-B | ✅ DONE | vanilla SAC + DIAYN training loop, 1M × 3 seeds CheetahRun, Wave D Gate 1+2 closed |
| Ant MJX port | ✅ DONE | AntMJXClassic (27d) + AntMJX (105d), 3/3 Classic seeds PASS visual gate |
| Ant METRA contrast | ✅ DONE | Null delta vs DIAYN; DualLam → 0 in all 3 seeds (default `dual_dist="one"` too generous) |
| SD-C | open | Go2 deployable obs DIAYN |
| SD-D | open | Deploy/export contract for fixed skills (sim only) |
| SD-E | open | D3-style factorization + hardware |

## Done — DIAYN baseline + Ant + METRA

### SD-A: Contract and scaffolding ✅
**Plan:** `plans/2026-05-02-skill-discovery-sd-a.md`
- [x] Pure config + priors + factor registry + DIAYN aux + SkillManager. Unit tests only. No env, no train script. Landed 2026-05-02 (commits 84883a7, 0317db7, 593ddf4, bd0bcf8, 725eced). 31/31 tests pass; zero regressions.

### SD-B: vanilla SAC + DIAYN training loop on CheetahRun ✅
**Plan:** `plans/2026-05-02-skill-discovery-sd-b.md` (vanilla SAC override; FastSAC re-enters at SD-C/E)

**Wave A + B + C (code) — DONE 2026-05-03:**
- [x] Extend `ObsPipeline.make_buffer` with generic `extra_obs_dims` — `fb0bcbc`
- [x] Skill checkpoint wrapper (`jax_rl/skill_discovery/checkpointing.py`) — `b2a433b`
- [x] `jax_rl/training/skill_offpolicy_loop.py` w/ sample-time intrinsic reward + aux update — `7d6154d`
- [x] `scripts/train_skill_discovery.py` CLI — `9c19ade`
- [x] `scripts/record_video.py --skill-index` — `47d41cc`
- 13/13 SD-B unit tests pass; 0 regressions on existing 54 buffer/pipeline tests

**Wave D (acceptance runs) — COMPLETE:**
All commands prefix with `XLA_CLIENT_MEM_FRACTION=0.55` for CheetahRun (default 0.7 OOMs at eval-scan graph creation on 16 GB GPU with Warp env). Ant variants use `XLA_PYTHON_CLIENT_PREALLOCATE=false` instead.
- [x] Phase 5.0 resume guard — PASS (5K ckpt + resume; current_z fresh, no NaN; both runs below min_buffer so disc_loss-equivalence moot)
- [x] Phase 5.1 10K smoke (seed 0) — PASS (120 grad updates, skill diversity emerging, no NaN)
- [x] Phase 5.2 100K validation (seed 0) — PASS (DiscA 0.199→0.480, gate 0.225 cleared by 70K, ckpt `20260503_102424_*`, log `.temp/logs/sd_b_phase_5_2_seed0_100k.log`, 165s wall-clock)
- [x] Phase 5.3 1M × 3 seeds CheetahRun — Gate 1 PASS all 3 seeds (per-skill spread > 0.5× max-skill mean); Gate 2 (visual diversity) **CLOSED via AntMJX port (2026-05-05)**. CheetahRun ckpts: `20260503_104453_*`, `20260503_111229_*`, `20260503_113956_*`. Total wall-clock 1h22m serial.

### Ant MJX port (closes Wave D Gate 2) ✅
**Plan:** `plans/2026-05-03-ant-mjx-port.md`
- [x] AntMJXClassic (27d) + AntMJX (105d) variants ported to MJX/Warp. 3/3 Classic seeds PASS numerical visual gate (max-pairwise > 3m OR circular heading-std > 30°). Seed 1: z6=+199.6 highest return (stand-upright skill near origin); z7=-227 drives max-pairwise via 1.4m -x trail. Return-rank ≠ xy-spread on Ant — survive_reward dominates if policy stays healthy.
  - Headlines: `figures/ant_classic_diayn_3seeds.png` + `figures/ant_classic_vs_v5_seed0.png`
  - Lesson: `lessons/diayn_ant.md`

### Ant METRA baseline contrast (2026-05-07) ✅
**Plan:** `plans/2026-05-05-ant-metra.md`
- [x] 3 seeds × 1M METRA on AntMJXClassic with default reference HPs. Original hypothesis "METRA's Lipschitz constraint produces wider xy-spread than DIAYN" REJECTED. Avg max-pairwise 0.763m (METRA) vs 0.767m (DIAYN); avg heading-std 71° vs 86°. All 3 METRA seeds had `DualLam → ~0.05` (Lipschitz constraint never engaged) so we effectively ran "DIAYN with different reward + continuous z". `dual_dist="one"` constant-1 too generous for Ant's small step-to-step state changes; future ablation should try `dual_dist="l2"`.
  - Headlines: `figures/ant_classic_metra_3seeds.png` + `figures/ant_classic_diayn_vs_metra_3seeds.png` (6-panel)
  - Lesson: `lessons/metra_ant.md`

## Open — next branches

### ~~Cheap follow-up: METRA `dual_dist="l2"` ablation~~ — DONE 2026-05-07, ALSO NULL
- [x] Tried `dual_dist="l2"` seed 0 × 1M. Same degenerate equilibrium as `one`: DualLam → 0.044 (vs `one`'s 0.052), PhiAlign → 0.080 (basically zero). Visual gate PASS via heading-std (68°), max-pairwise 0.443m WORSE than `one` seed 0's 1.108m. Confirms METRA-on-Ant ceiling is NOT a constraint-shape issue — phi stays small in both regimes so cst_penalty saturates at +slack either way. See `lessons/metra_ant.md` §5c. Default reverted to `"one"`. Next-likeliest fix: 256×256 phi (D3 fork sizing, ~2h).

### SD-C: Go2 DIAYN with deployable obs
- [ ] Target `Go2WarpJoystickUnitree` (45d hardware-conservative obs, action_scale=0.25)
- [ ] First DIAYN factor: command-conditioned behavior class or base-velocity response (named extractor, not magic indices)
- [ ] Per-skill eval rollouts, fall rate, behavior summary

### SD-D: Deploy/export contract for fixed skills (sim only)
- [ ] Skill-aware deploy obs composer; explicit dim check: `(raw_dim * n_frame_stack) + skill_dim == runner.obs_dim`
- [ ] CLI: `--skill-index`, `--skill-vector`, `--skill-mode fixed`
- [ ] ONNX sidecar or `deploy_contract.json` next to `actor.onnx`
- [ ] `deploy_go2.py` real-mode rejects skill checkpoints unless `meta["skill_discovery"]["hardware_ready"] = true`
- [ ] Fix `sim2sim_direct.py` action target construction to use checkpoint metadata, not deploy constants

### SD-E: D3-style factorization + hardware
- [x] METRA aux module + Lagrangian dual update — landed 2026-05-07 (`jax_rl/skill_discovery/metra.py`, commit `b9ed1fe`)
- [x] `unit_sphere` prior — landed 2026-05-07 (commit `2752bee`)
- [ ] Dirichlet prior (deferred from SD-A)
- [ ] Named factor extractors for base xy, heading, base height, roll/pitch
- [ ] Style factor + safety penalties (D3 Tables 9/10) — load-bearing for hardware
- [ ] Symmetry augmentation (4-fold for quadruped)
- [ ] Within-episode skill resampling (`resample="fixed_steps"`)
- [ ] Hardware readiness gate + real-robot deploy

### METRA on Humanoid (apples-to-apples with paper)
- [ ] METRA reference uses Humanoid (376d obs); our Ant null-delta may be Ant-specific. Run METRA on `HumanoidRun` (already in mujoco_playground) at 3 seeds × 1M to test whether default HPs engage there.

### DUSDi (factored discrete skills with anti-MI penalty)
- [ ] Spec/plan cycle pending. Bigger novel method than METRA. Most info gain if successful.

## Cross-references

- Workspace lesson index: [`.context/LESSONS.md`](../../.context/LESSONS.md)
- Workspace TODO: [`.context/TODO.md`](../../.context/TODO.md) (cross-project only)
- Project README: [`README.md`](README.md)
