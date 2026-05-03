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

## Next session pickup

If session compacts here:
1. Read this journal entry for state.
2. Read `.superpowers/plans/2026-05-02-skill-discovery-sd-b.md` §Task 5.
3. Check `.context/TODO.md` SD-B section for Wave D phase status.
4. Memory file: `project_skill_discovery_state.md` has canonical doc paths + run conventions.
5. Run `git log --oneline -15` to see commit history.
6. Resume Wave D at the next-pending phase.
