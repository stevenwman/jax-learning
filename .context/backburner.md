# Backburner

Investigations noted but deferred — root cause unknown, needs isolation/sweep
to answer. Keep short; promote to a lesson or TODO when someone takes it on.

---

## L-specialist late-training collapse (2026-04-21) — RESOLVED via N=10 dense obs

**V1 observation (N=11 zero-pad):**
- Peak sto cov 0.86 @ step ~900k, then crashed to 0.1-0.5 for final 800k. Q1L spiked to 1e+2 near 2M.

**Resolution (V2, N=10 dense obs, 2026-04-24):**
Same L training config on N=10 dense obs: trained stable to 2M, final Q1L ≈ 2 (no spike), final sto cov 0.846 @ 2M. Best eval return 706.69.

Hypothesis 1 (zero-pad destabilizes) was correct. The shape-ID leak forces policy to branch on pad-pattern; under SAC's entropy bonus this bifurcation becomes unstable. Dense obs eliminates the branching.

**Artifacts (for record):**
- V1 ckpt (collapsed): `checkpoints/20260421_221953_pusht_sac_contact_gated_seed0/`
- V2 ckpt (stable): `checkpoints/20260424_131444_pusht_sac_contact_gated_seed0/`

**Artifacts:**
- ckpt: `checkpoints/20260421_221953_pusht_sac_contact_gated_seed0/`
- log: `logs/letters27_l_*.log`
- videos: `.temp/pusht_l27_ep{0-4}.mp4`

---

## DR-keypoint late-training collapse at 5M (2026-04-24) — RESOLVED via N=10 dense obs

**Observed (V1, N=11 zero-pad):**
- `obs_type=keypoints` (27d padded), `target_entropy_scale=1`, 5M steps.
- Peak sto cov 0.33 @ ~2M, then critic blowup: Q1L=2.5e+03, alpha 0.098 runaway.

**Resolution (V2, N=10 dense obs, 2026-04-24):**
Switching from zero-padded keypoints to N=10 arc-length-dense keypoints (no padding) fixed both the cross-shape transfer AND the late-training collapse. V2 DR 5M ran cleanly for full 5M, peak sto cov 0.864 @ step 4.7M, no critic blowup. DR matrix row-mean jumped 32.9% → 73.9% sto. The instability was downstream of the pad-leak (multimodal target = one-policy-per-pad-pattern); remove the leak, critic stabilizes.

See `.context/studies/2026-04-22_pusht_letter_matrix.md` V2 section.

**Artifacts:**
- V1 ckpt (collapsed): `checkpoints/20260423_223824_pusht_sac_contact_gated_seed0/`
- V2 ckpt (stable): `checkpoints/20260424_143821_pusht_sac_contact_gated_seed0/`

---

## Action chunking off-policy (deferred 2026-04-20)

See `.context/TODO.md` push-T section for full writeup. Bundled with planned
flow-matching off-policy work — action chunking without flow actor breaks
action correlation. Revisit when flow-matching SAC infra lands.

---

## DR + FS=3 mixed signal on cross-shape (2026-04-21)

See `.context/studies/2026-04-21_pusht_transfer_matrix.md` §7. FS=3 helps
on triangle (+21pp) but hurts on tee (−31pp) vs FS=1 DR baseline. No net
win at 2M steps. Unclear whether more training, explicit velocity features,
or longer FS window (5-10) would flip this.

**What to try:**
- 5M step DR_fs3 vs 5M DR_fs1 (compare converged, not 2M)
- Explicit velocity obs instead of FS (block_vel + agent_vel = +4d vs +10d FS)
- Per-shape network branch instead of shared (breaks shape-agnostic property though)

**Priority:** low. Main matrix conclusions don't depend on FS.
