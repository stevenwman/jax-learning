# DIAYN + CheetahRun: useful smoke test, weak behavioral demo

> Lesson from SD-B Wave D acceptance runs (2026-05-03). DIAYN trained on
> CheetahRun (MJX, 8 skills, 1M steps, 3 seeds) reliably hits the
> numerical gate (discriminator accuracy 0.95+ at 1M, per-skill return
> spread > 50% of max-skill mean), but visually inspecting the resulting
> skills is a poor signal of method success or failure. Use Ant for the
> behavioral / visual gate from now on.

---

## §1. What DIAYN+CheetahRun gives you

**Reliable signals (use these):**
- Pipeline correctness — does training loop produce finite losses, fill
  buffer, run discriminator + SAC updates, save aux state, resume from
  ckpt? Yes if DiscA climbs above chance (1/num_skills + 0.1) within
  ~100K steps.
- Discriminator implementation correctness — does logsoftmax+one-hot
  cross-entropy converge to ≥0.9 by 1M? Yes for 8 skills on this env.
- Reward plumbing correctness — does intrinsic reward magnitude track
  discriminator confidence? IntR rises from 0.07 to ~2.0 in line with
  DiscL falling.

**Weak / misleading signals (don't read too much into):**
- Per-skill task-return diversity — most skills (5-7 of 8) collapse to
  ~0 task return at 1M, even though discriminator is 95%+ accurate.
  Discriminator separates skills by tiny obs differences (joint angle
  minutiae) not gross body-frame trajectories.
- Online avg return — single-digit values for most skills tell you the
  collapsed skills are stuck/jittery; doesn't differentiate "DIAYN
  working as intended" from "DIAYN broken."
- Visual rollouts — the planar 2D cheetah running, falling, jittering,
  and recovering all look superficially similar in standard renders.
  Hard to call diversity from videos alone.

## §2. Why this happens (canonical DIAYN limit)

DIAYN's mutual-information objective `I(s; z)` is upper-bounded by
`log(num_skills)` and is reachable through *any* discriminable s-vs-z
mapping. Nothing in the objective requires those mappings to correspond
to behaviorally-distinct trajectories — minute postural differences
suffice. This is exactly what motivates METRA (Park 2024 — Lipschitz
dual constraint on representation distance), D3 (Cathomen 2025 — factor
extractors + half-normal L2 distance), and DUSDi (factored discrete
skills with anti-MI penalty).

Original DIAYN paper acknowledges this: Fig. 12 in Eysenbach et al.
2018 shows HalfCheetah skills clustering near origin with most being
minor variants of stationary postures.

## §3. What to use instead for behavioral / visual gates

**Ant (DIAYN App. D.3, Fig. 4):** xy-trajectory plot of N skills sweeping
distinct angles from origin. Instantly legible diversity check —
human eye can call "this is N distinct directional skills" or "this
collapsed to 2-3 directions." Two routes:
- (a) CPU Gym Ant-v5 via existing `gym_backend.py:200` — one-shot
  figure for paper / write-up.
- (b) MJX Ant port — required only if Ant DIAYN runs become routine.

Don't pull Brax for this; one env doesn't justify a third backend lib.
Single Ant figure → route (a).

## §4. SD-B sign-off implication

**Numerical gates (5.0, 5.1, 5.2, 5.3 Gate 1) are sufficient for SD-B
acceptance.** Visual gate (5.3 Gate 2) deferred to Ant follow-up. Don't
re-litigate; the cheetah skill-collapse pattern is expected DIAYN
behavior, not an implementation bug.

## §5. Practical numbers (CheetahRun, 8 skills, vanilla SAC, 1M, 3 seeds)

- Wall-clock per seed (RTX 5080, MJX, Warp env, batch 512, 8 grad/step,
  128 envs, XLA_CLIENT_MEM_FRACTION=0.55): ~27.5 min.
- DiscA at 1M: 0.95-0.97 across seeds.
- Active skills (return > 1.0): 1-3 of 8 per seed.
- Max-skill mean return: 7-50, varies wildly seed-to-seed (seed 2 had
  z7=48.2; seed 0 best was z3=7.3).
- 3 parallel runs at fraction=0.55 OOMs (3×8.8GB > 16GB). Either run
  serial or drop fraction to 0.25 (untested, eval-scan OOM risk).

## §6. When to revisit this lesson

- If a future SD-D / SD-E run shows >5 of 8 skills behaviorally active
  on Ant or HC: that's a real win for the new method (METRA / D3 /
  DUSDi vs DIAYN baseline). Document the contrast against this entry.
- If a future DIAYN regression on the same env config shows DiscA
  climbing but skill spread *less* than seen here (max-skill mean < 5
  across all seeds): something broke. Bisect against ckpts referenced
  in the 2026-05-03 progress journal.
