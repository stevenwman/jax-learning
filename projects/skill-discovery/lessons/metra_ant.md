# DIAYN vs METRA on Ant — null-delta result

> Lesson from Ant METRA baseline contrast (2026-05-06 → 2026-05-07). 3 seeds × 1M
> METRA on `AntMJXClassic` (the env we just used to close SD-B Wave D Gate 2)
> with default METRA hyperparameters from the reference repo. **Hypothesis:
> METRA's Lipschitz dual constraint produces wider xy-spread than DIAYN's MI
> objective. REJECTED at 3 seeds.** METRA matches DIAYN within seed-to-seed
> noise; the Lipschitz constraint never engaged (`DualLam → ~0.05` in all 3
> seeds). Companion to `lessons/diayn_cheetah.md` and
> `lessons/diayn_ant.md`.

---

## §1. Headline numbers (3 seeds × 1M, AntMJXClassic, 8 skills)

| Method | Seed | DiscA / DualLam | max-pairwise (m) | heading-std (°) | gate |
|---|---|---|---|---|---|
| DIAYN  | 0 | DiscA 0.998 | 0.431 | 79.4 | PASS |
| DIAYN  | 1 | DiscA 1.000 | **1.197** | 81.8 | PASS |
| DIAYN  | 2 | DiscA 1.000 | 0.673 | 96.1 | PASS |
| **DIAYN avg** | | | **0.767** | **85.8** | |
| METRA  | 0 | DualLam 0.052 | **1.108** | 74.0 | PASS |
| METRA  | 1 | DualLam 0.050 | 0.558 | 58.9 | PASS |
| METRA  | 2 | DualLam 0.062 | 0.623 | 81.4 | PASS |
| **METRA avg** | | | **0.763** | **71.4** | |

Side-by-side panels:
- `figures/ant_classic_metra_3seeds.png` — 3-panel METRA only
- `figures/ant_classic_diayn_vs_metra_3seeds.png` — 6-panel DIAYN top row, METRA bottom row
- `figures/ant_classic_diayn_vs_metra_seed0.png` — 2-panel side-by-side (seed 0 only, headline)

## §2. What went wrong with METRA — `DualLam → 0` degenerate equilibrium

All 3 METRA seeds showed the same training trajectory:
- `DualLam` initialized at 30.0
- Decayed monotonically to ~0.05 by step 1M (in all seeds)
- `PhiAlign` final magnitude < 0.02 in all seeds (acceptance gate was > 0.1 — FAIL)

**Mechanism:** with `dual_dist="one"` (constant 1) and `dual_slack=1e-3`:
```
cst_penalty = jp.minimum(1 - mean(‖Δφ‖²), 1e-3)
loss_dual = log_dual_lam * stop_gradient(cst_penalty.mean())
```
At initialization, `phi` is near-identity-ish so `mean(‖Δφ‖²)` is small (~0.001). Therefore `cst_penalty ≈ +0.999 → clamped to +1e-3` (constraint hugely satisfied). Adam descends `log_dual_lam` because gradient is positive, so `λ = exp(log_dual_lam)` decays.

Once `λ ≈ 0`, the phi loss reduces to `-mean(alignment) = -mean((phi_y - phi_x) · z)`. Phi has incentive to grow ‖Δφ‖ in the z direction, but **without the Lipschitz upper bound** there's nothing forcing phi to actually change between consecutive states — the network can learn `phi(s) ≈ const + ε(s)·z` where ε is small. Result: `phi_alignment` stays near zero, intrinsic reward stays near zero, SAC has no signal to differentiate skills via phi.

**But the policy still produces visible skill diversity.** Why?
- Early in training (before phi degenerated), there WAS a phi-driven reward gradient that biased SAC's actor toward skill-conditional policies.
- That early bias persists in the actor weights even after phi collapses.
- At inference, evaluating along basis vectors z = e_k samples 8 distinct conditioning patterns; the actor responds with 8 distinct flailing patterns.

This is genuinely a "bad" METRA — phi has not learned a state representation that captures meaningful state distances. But the figure looks reasonable because we're inspecting actor behavior, not phi geometry.

## §3. Why this matters

The original hypothesis was that METRA's Lipschitz constraint would produce **wider physical state coverage** than DIAYN — the canonical METRA pitch is "skills travel further because phi-distances scale with environment-distances." On AntMJXClassic, that didn't happen. METRA matches DIAYN on max-pairwise (0.763 vs 0.767m) and is slightly worse on angular spread (71° vs 86°). **The Lipschitz constraint never engaged**, so we ran what was effectively a "DIAYN with a different reward formula and continuous z prior" rather than a true METRA experiment.

Two ways to read this:
- **(a) METRA needs better hyperparameters on this env.** The default `dual_dist="one"` with `dual_slack=1e-3` is too generous; phi never gets pressured to grow. Try `dual_dist="l2"` (constraint is ‖s'-s‖² instead of constant 1) which auto-scales with actual state changes. Or boost phi initialization to put us immediately in the constraint-violated regime.
- **(b) Phi network architecture is too shallow / too narrow for Ant 27d.** METRA reference uses 1024×1024 phi for Humanoid (376d obs); we use 1024×1024 for Ant (27d obs) — 38× obs-to-hidden expansion vs reference's 2.7×. Phi may have far more capacity than needed, leading to easy collapse to constant-output minima. D3 fork uses 256×256 phi, possibly better proportioned for our scale.

Both are tunable parameters of an ablation, NOT fundamental method limits. So the lesson is **not "METRA doesn't work"**, it's **"METRA-with-default-hyperparameters does not exceed DIAYN on AntMJXClassic"**. A future ablation should try `dual_dist="l2"` first (single-line change, smallest effort).

## §4. Surprising secondary finding — visible diversity despite phi collapse

The METRA seed 0 figure (`ant_classic_metra_seed0.png`) shows skills fanning out radially in 8 distinct directions, **comparable in legibility to DIAYN's best seed**. This is despite training metrics showing METRA's phi network degenerated.

**Implication:** the visual diversity gate (`max_pairwise > 3m OR heading_std > 30°`) measures actor behavior, not phi quality. An actor that responds differently to 8 different basis-vector inputs will produce 8 different trajectories — regardless of whether the upstream representation learning was successful.

A stricter test of "did METRA actually learn a useful state representation" would be to evaluate phi: does it cluster physically-similar states close together in phi-space? Does it spread physically-different states apart? We didn't do this evaluation.

For SD purposes, the visual gate suffices to call "diversity achieved", but it does NOT confirm METRA is doing its specific job (Lipschitz state-distance preservation). A future lesson should add a `phi_state_distance_correlation` test for that.

## §5. Engineering gotchas

- **Phi 1024×1024 with skill_dim=8 → 1.09M params** — fine on 16GB GPU. Wall-clock ~106 min/seed (1.9× DIAYN's 56 min, due to phi double-forward + larger net + buffer next_obs I/O). All within initial 70-90 min projection band.
- **`dual_dist="one"` constant 1 is universal for METRA reference + D3 fork.** Both report this works for their envs. Doesn't generalize to ours; needs ablation.
- **Sign convention for `dual_lam_loss`** matters but plan-locked correctly: `loss = log_dual_lam * stop_grad(cst_penalty.mean())`, Adam descent. Verified against METRA reference `metra.py:292-300` AND D3 fork `d3-skill-discovery/.../metra.py:445`. Plan's risk register documented this, plan audit caught a doc-only confusion that would have misled an implementer to flip the sign — DON'T flip it.
- **`unit_sphere` prior at training, basis vectors at inference** — METRA was trained with z ~ uniform on S^{N-1}. We rendered figures with z = standard basis vectors (each is on the unit sphere — valid input). METRA reference's plotting code does the same per `metra.py:561-569`. Distribution mismatch is real but mild; consistent with their published figures.
- **`<framelinacc>` sensor in ant.xml** still required for `cfrc_ext` — same as DIAYN port. Inherited cleanly; no METRA-specific issue.

## §5c. `dual_dist="l2"` ablation (2026-05-07) — also null

Per §3 follow-up: tried `dual_dist="l2"` on seed 0 (single seed, 1M). Hypothesis: `cst_dist = mean(‖s'-s‖²)` auto-scales with actual state changes, putting cst_penalty in a regime where Adam doesn't immediately saturate at +slack. Should keep λ active and force phi to grow.

Reality: same degenerate equilibrium.

| Run | DualLam final | PhiAlign final | max-pairwise | heading-std |
|---|---|---|---|---|
| `one` seed 0 (baseline) | 0.052 | 0.007 | 1.108 m | 74.0° |
| `l2` seed 0 (ablation) | 0.044 | 0.080 | **0.443 m** | **68.0°** |

`l2` is *worse* than `one` on both visual-gate metrics for seed 0. Per-skill mean −1613 ± 212 vs `one`'s −464 ± 386 (worse task return too).

**Takeaway: the failure isn't constraint shape — `‖Δφ‖²` stays close to zero in both regimes, so cst_penalty saturates at +slack either way and λ decays.** The phi network at 1024×1024 with random init produces near-constant outputs early; without external pressure, it never breaks out of this fixed point.

This rules out the "default `one` was wrong on Ant" hypothesis — both constraint types fail the same way. **Future METRA ablations on Ant must address phi architecture (try 256×256 per D3 fork), phi initialization (warm-start with pretrained features?), or move to a larger env where step-to-step state changes are big enough to make cst_penalty meaningfully negative early in training (Humanoid 376d).**

Ckpt: `checkpoints/20260507_120054_sac_skill_skill_antmjxclassic_seed0`
Log: `.temp/logs/ant_classic_metra_l2_1m_seed0.log`
Figure: `figures/ant_classic_metra_l2_seed0.png`

Default `_METRA_DUAL_DIST` reverted to `"one"` (METRA reference value).

## §6. When to revisit

- ~~If a future ablation tries `dual_dist="l2"` on AntMJXClassic...~~ — DONE, also null. See §5c.
- If a future ablation uses 256×256 phi (D3 fork choice) with `dual_dist="one"` and shows engagement, the lesson is "phi was over-parameterized for our env scale". This is now the next-most-likely fix; budget ~2h.
- If 256×256 phi also degenerates, the right contrast environment is probably Humanoid (METRA's reference benchmark, 376d obs) — Ant may be too low-dim for METRA's representation to find meaningful state distances.
- For SD-D / SD-E (D3 factor decomposition or full DUSDi), use the AntMJXClassic baseline numbers from §1 as the DIAYN-vs-* contrast, but FLAG that METRA-baseline is an inconclusive 4-seed (3 `one` + 1 `l2`) experiment.

## §7. Filesystem references

| Item | Path |
|---|---|
| Plan | `plans/2026-05-05-ant-metra.md` |
| Lessons (cross-env) | `lessons/diayn_cheetah.md`, `lessons/diayn_ant.md`, this file |
| Source extracts (verbatim METRA hparams) | `references/skill_discovery_source_extracts.md` §METRA (lines 77-136) |
| METRA aux module | `jax_rl/skill_discovery/metra.py` |
| Manager extension | `jax_rl/skill_discovery/manager.py` (METRA branch) |
| Train CLI | `scripts/train_skill_discovery.py --algo metra` |
| Figure script | `scripts/plot_skill_xy.py` (algo-agnostic) |
| Stitcher | `scripts/stitch_skill_xy_panels.py` |
| Ckpts | `checkpoints/20260506_210720_*` (seed 0), `_225834_*` (seed 1), `20260507_004802_*` (seed 2) |
| Logs | `.temp/logs/ant_classic_metra_1m_seed{0,1,2}.log` |
| Headline figure | `figures/ant_classic_diayn_vs_metra_3seeds.png` (6-panel) |
