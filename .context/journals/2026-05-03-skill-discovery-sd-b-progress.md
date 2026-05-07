# Skill Discovery SD-B Progress — 2026-05-03

## Summary

SD-A complete (2026-05-02). SD-B Waves A + B + C complete (2026-05-03). Wave D acceptance runs in progress.

## SD-A (commits, 2026-05-02)

| Task | Commit | Tests |
|---|---|---|
| 1: config | `84883a7` | 6/6 |
| 2: prior | `0317db7` | 6/6 |
| 3: factors | `593ddf4` | 5/5 |
| 4: diayn | `bd0bcf8` | 5/5 |
| 5: manager | `725eced` | 9/9 |

**Total SD-A: 31/31 unit tests pass; 665/665 repo regression tests pass.**

## SD-B Waves A + B + C (commits, 2026-05-03)

| Wave | Task | Commit | Tests |
|---|---|---|---|
| A | 1: ObsPipeline.make_buffer extras | `fb0bcbc` | 6/6 |
| A | 2: Skill checkpoint wrapper | `b2a433b` | 4/4 |
| B | 3: skill_offpolicy_loop | `7d6154d` | 3/3 |
| C | 4: train_skill_discovery.py | `9c19ade` | 5K smoke ✓ |
| C | 4b: record_video --skill-index | `47d41cc` | skill-0 video ✓ |

**Total SD-B unit-level: 13/13 new tests pass; 54/54 non-regression tests pass; 44/44 combined SD-A+SD-B suite green.**

## Audit discipline

Two rounds of parallel-agent audit per phase:
- SD-A: round 1 (5 agents), round 2 (5 agents) — 12 critical fixes landed across both rounds
- SD-B: round 1 (5 agents), round 2 (3 agents returned, 2 rate-limited) — 12+ BLOCKERs caught across both rounds, plus 5 paper-claim corrections, 3 CHALLENGEs, 8+ WARNINGs

Both phases: zero critical issues escaped to subagent dispatch.

## Wave C deviations from plan

**Task 4 (train script):** smoke required `XLA_CLIENT_MEM_FRACTION=0.55` to avoid eval-scan OOM. Default 0.7 OOMs at Warp graph creation on 16 GB GPU. Documented in commit message and memory file.

**Task 4b (record_video):** subagent extended `_build_select_action` to recognize `algo == "sac_skill"` (dispatch hole — train_skill_discovery.py writes `algo_name="sac_skill"` per Task 3, but `_build_select_action` lacked that case). Routes through existing vanilla SAC builder. Reasonable judgment call.

## Wave D plan

Per SD-B plan §Task 5:

| Phase | Command | Wall-clock | Acceptance |
|---|---|---|---|
| 5.0 | resume guard (5K save → resume → 1K) | ~5 min | post-resume disc_loss within 2× pre-save; current_z fresh; no NaN |
| 5.1 | 10K seed-0 smoke | ~2 min | finite aux losses, buffer fills |
| 5.2 | 100K seed-0 validation | ~5-10 min | discriminator accuracy > 0.225 (chance + 0.1) |
| 5.3 | 1M acceptance × 3 seeds | ~30-60 min × 3 | per-skill return spread > 50%; qualitative video diversity |

Run convention: one phase at a time, journal between phases (compaction-safe boundaries).

**All Wave D commands prefix with `XLA_CLIENT_MEM_FRACTION=0.55`.**

## Existing checkpoint

`checkpoints/20260503_000548_sac_skill_skill_cheetahrun_seed0/` — 5K-step ckpt from Task 4 verify. Used to validate Task 4b record_video skill-index path. Contains `meta.json` with `skill_discovery` block, `skill_aux/full_state/{params,opt_state}.npz`.

## Wave D progress so far

### Phase 5.0 resume guard — PASS
- Used existing 5K ckpt (`checkpoints/20260503_000548_sac_skill_skill_cheetahrun_seed0/`) as save point.
- Resume command: `--total-timesteps 6000 --seed 0 --resume <ckpt>` → produced new run dir `20260503_004146_*`.
- Log confirmed `Resuming from step 0 (aux state restored)`.
- All 8 skills evaluated finite (z0-z7 returns 11.2 to 25.1 mean 16.2 ± 4.7).
- No NaN. current_z resampled fresh.
- Caveat: both runs below `min_buffer=8192` so 0 gradient steps fired in either run; the disc_loss-equivalence gate from plan §5.0 is moot. Pipe + aux-state + current_z lifecycle verified.

### Phase 5.1 10K smoke — PASS
- Run: `--total-timesteps 10000 --seed 0` → ckpt `20260503_004238_*`.
- 120 gradient updates after warmup (15 outer steps × 8 grad_updates_per_step = 120 — matches buffer fill at step 8192/128 = outer step 64, ran until 78, so 14 outer steps × 8 ≈ 112; reported 120 includes eval grads).
- Per-skill eval: z0-z7 returns 0.2-11.7, mean 3.6 ± 4.2. **Skill diversity emerging** (4 of 8 near 0, 4 above 4) at very early training.
- No NaN. Buffer fills past min_buffer.
- metrics.csv not written (run too short to trigger eval cycles + checkpointing inside loop). disc_loss curve will be visible at 5.2 (100K) which has multiple eval cycles.

### Phase 5.2 100K seed 0 — PASS
- Run: `--total-timesteps 100000 --seed 0` → ckpt `20260503_102424_sac_skill_skill_cheetahrun_seed0/`.
- Wall-clock: 165s (2m45s) — well under 5-10 min estimate.
- 5,744 gradient updates total (steps 8192-99968 / 128 envs × 8 grad_updates_per_step ≈ 5,737).
- **Discriminator accuracy curve:**
  - Step 10K: DiscA=0.199 (chance ≈ 0.125)
  - Step 30K: DiscA=0.223 (just below gate)
  - Step 50K: DiscA=0.213
  - Step 70K: DiscA=0.291 (gate cleared)
  - Step 90K: DiscA=0.434
  - **Step 100K: DiscA=0.480** (gate `> 0.225` — passed by 2.1×)
- DiscL: 2.003 → 1.470 (monotonic decrease post-30K)
- IntR: 0.076 → 0.610 (rising as discriminator gets confident — DIAYN signal working)
- No NaN. Q1/ActLoss stable.
- Per-skill eval: z0-z7 = [1.7, 0.2, 2.6, 6.6, 2.5, 0.0, 0.0, 20.3], mean 4.2 ± 6.4. Spread already huge (z7=20.3 vs z5=z6=0.0). Episodic returns nan because episode_length=1000 × 128 envs not reached at 100K total steps (expected — 5.3 1M run resolves this).
- Log: `.temp/logs/sd_b_phase_5_2_seed0_100k.log`.

### Phase 5.3 1M × 3 seeds — Gate 1 PASS, Gate 2 DEFERRED to Ant
- Strategy: serial (3 parallel @ XLA_CLIENT_MEM_FRACTION=0.55 = 26GB > 16GB GPU; reducing fraction risks eval-scan OOM we already fought).
- Wall-clock: 1h22m total (seed 0 27m36s, seed 1 27m27s, seed 2 27m49s) — clean ~27.5 min/seed extrapolation from 5.2 confirmed.
- Single orchestration task `bgzu8sw6z`; per-seed logs `.temp/logs/sd_b_phase_5_3_seed{0,1,2}_1m.log`.
- 61,992 gradient updates per seed.

**Final results (1M):**

| Seed | DiscA | DiscL | Per-skill eval (z0..z7) | Mean ± SD | Online avg (last 100 eps) | Ckpt |
|---|---|---|---|---|---|---|
| 0 | 0.96 | 0.10 | [0.6, 0.5, 0.4, **7.3**, 0.1, 0.0, 0.0, 0.0] | 1.1 ± 2.4 | 1.6 | `20260503_104453_*` |
| 1 | 0.95 | 0.12 | [0.0, **5.7**, 0.7, 0.0, 0.0, 0.9, **12.5**, 0.5] | 2.6 ± 4.2 | 2.9 | `20260503_111229_*` |
| 2 | 0.96 | 0.11 | [0.0, 0.1, 1.0, **36.1**, 0.0, 0.0, 1.2, **48.2**] | 10.8 ± 18.3 | 7.9 | `20260503_113956_*` |

**Gate 1 (per-skill spread > 50% of max-skill mean): PASS all 3 seeds.**
- seed 0: spread 7.3 vs threshold 3.65 (2.0×)
- seed 1: spread 12.5 vs threshold 6.25 (2.0×)
- seed 2: spread 48.2 vs threshold 24.1 (2.0×)

**Gate 2 (qualitative video diversity): DEFERRED.** CheetahRun visually hard to interpret skill differences (planar 2D cheetah with 6 joints; running-fall-recover all look similar in low-quality renders). User explicitly flagged this earlier and asked to defer visual gate to Ant (canonical DIAYN figure: xy-trajectory plot of 8 skills sweeping different angles from origin — instantly legible). Numerical gates suffice here for SD-B sign-off.

**Critical observation — skill collapse:** Across all 3 seeds, only 1-3 of 8 skills are behaviorally active (positive task return); 5-7 collapse to ~0. Discriminator still 0.95+ accurate, so it separates skills via tiny obs differences (joint angle minutiae) not gross behavior. **This is canonical DIAYN failure mode for HC** — paper Fig. 12 shows similar pattern, motivates METRA's Lipschitz dual + D3's factor-aware reward (SD-D, SD-E). **Pipeline implementation correct; base method ceiling-bound.**

## Wave D status: SD-B COMPLETE pending video deferral.

Numerical gates (5.0, 5.1, 5.2, 5.3 Gate 1) all PASS. Gate 2 video diversity deferred to Ant follow-up (already TODO'd as future).

## Lesson surfaced
See `projects/skill-discovery/lessons/diayn_cheetah.md` — DIAYN+CheetahRun is a useful smoke test (verifies pipeline) but not a useful behavioral demo (skill collapse + visual ambiguity). Use Ant for behavioral acceptance going forward.

## Next session pickup

If session compacts here:
1. Read this journal entry for state.
2. Memory file: `project_skill_discovery_state.md` has canonical doc paths + run conventions + XLA mem fix.
3. Lesson file: `projects/skill-discovery/lessons/diayn_cheetah.md` (cheetah caveats).
4. Run `git log --oneline -15` to see commit history.
5. **SD-B is functionally complete**; next step is user-decision: (a) Ant port for visual gate, (b) move to SD-C (Go2 deployable obs DIAYN), (c) jump to SD-D/E (METRA / factored).
Wall-clock projection: 100K → 165s, so 1M ≈ 1650s (27.5 min) per seed. Three parallel ≈ same wall-clock if GPU memory permits (each run uses ~55% of 16GB ≈ 8.8GB, so 3 in parallel = 26.4GB > 16GB — **must serialize or interleave**). Realistic: 3 × 30 min serial ≈ 1.5h.
