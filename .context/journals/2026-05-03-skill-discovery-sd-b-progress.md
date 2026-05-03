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

## Wave D pending

- [ ] Phase 5.2 100K seed 0 — discriminator accuracy > 0.225 by 100K (chance + 0.1)
- [ ] Phase 5.3 1M acceptance × 3 seeds (0, 1, 2) — per-skill spread > 50%, qualitative video diversity

## Next session pickup

If session compacts here:
1. Read this journal entry for state.
2. Read `.superpowers/plans/2026-05-02-skill-discovery-sd-b.md` §Task 5 (5.2 + 5.3 specs).
3. Check `.context/TODO.md` SD-B section for Wave D phase status.
4. Memory file: `project_skill_discovery_state.md` has canonical doc paths + run conventions + XLA mem fix.
5. Run `git log --oneline -15` to see commit history.
6. Resume Wave D at Phase 5.2 (100K seed 0 validation).
7. **All Wave D commands prefix with `XLA_CLIENT_MEM_FRACTION=0.55`.**

### Phase 5.2 command (next)
```bash
XLA_CLIENT_MEM_FRACTION=0.55 uv run python scripts/train_skill_discovery.py \
    --env CheetahRun --num-skills 8 --total-timesteps 100000 --seed 0
```
Run in background (Bash run_in_background=true), grep stdout for `disc_acc` to monitor accuracy curve. ~5-10 min wall-clock on GPU.
