# Skill Discovery — Validation Methodology + Method Tradeoffs

**Date:** 2026-05-02
**Purpose:** Resolve the validation gap. Skill discovery has no single benchmark. This doc catalogues how each paper proves its method works, what tradeoffs exist between methods, and how alternatives map to D3.
**Sources:** Deep-audit reports of DIAYN, METRA, D3, DADS, DUSDi, SkiLD (parallel agent research, 2026-05-02).
**Related:**
- Spec: `.superpowers/specs/2026-04-28-skill-discovery.md`
- **Source extracts: `.context/references/skill_discovery_source_extracts.md`** ← ground-truth values from each paper's official repo (file:line cites). Read alongside this doc.
- D3 reference: `.context/references/d3_skill_discovery.md`
- SD-A plan: `.superpowers/plans/2026-05-02-skill-discovery-sd-a.md`

---

## Why this doc exists

Standard RL benchmarks (return on a reward function) don't apply. There is no "correct" skill set — only "diverse," "useful," "stable," "deployable" sets. Validation in this field is multi-metric, partly qualitative, and varies by paper. To know our DIAYN/METRA/D3 implementation is correct, we need a paper-grounded eval contract per phase, not vibes.

---

## Part 1 — Validation methods catalogued across 6 papers

| Metric | DIAYN | METRA | D3 | DADS | DUSDi | SkiLD |
|---|---|---|---|---|---|---|
| **State coverage** (bin count, std-of-means) | Qualitative trajectory plots only | **Headline.** Bin count over training, 8 seeds, 95% CI ribbons. Fig 5. | **Headline.** Std of mean states across 10K+ skill samples (Table 2). | Trajectory std-dev under fixed z (lower = more predictable). Fig 6. | Not primary metric. | % rollouts inducing each hard local-dependency graph. Fig 5. |
| **Discriminator accuracy** | Implicit (training loss = pseudo-reward) | N/A (no discriminator) | Used per-DIAYN factor | N/A (uses dynamics model) | Sub-metric in ablation | Used per-skill |
| **Skill-following fidelity** | N/A | N/A | **Cosine similarity** between commanded z and realized direction (METRA factors) / discriminator posterior (DIAYN factors) | Implicit (model predictability) | N/A | Graph-match indicator |
| **Disentanglement** | N/A | N/A | Implicit via factorization | N/A | **DCI score** (Disentanglement, Completeness, Informativeness) — **headline**, Table 1. ~50× over DIAYN-MC. | N/A |
| **Downstream task return** | HRL meta-controller, 3-5 seeds | **HRL goal reaching, 4-8 seeds 95% CI** | **Hierarchical PPO on rough-terrain waypoint** (Table 3); zero-shot, frozen skills | **Zero-shot MPPI in skill space** (no RL!); Fig 7-8 | 13 downstream tasks via PPO meta-controller; Fig 4 | 10 sparse-reward long-horizon tasks; finetune curves |
| **Safety / illegal contacts** | N/A | N/A | **Headline for hardware**: % illegal contacts per body part (Table 1). Style on/off ablation: base 4.04% → 0.03%. | N/A | N/A | N/A |
| **Qualitative** | Videos, t-SNE, reward histograms per skill | xy trajectory plots colored by z (Fig 3) | Roll/pitch coverage maps per skill (Fig 5); real-robot demos (Fig 3) | Per-skill rendered videos | Project-page skill grids | Graph-induced behavior grids |
| **Sample budget per run** | ~1M steps | 300K (pixel buffer) – 1M (state) | 1 day RTX 3090 (~2K envs × tens of M steps) | ~2M on-policy; off-DADS 300K real-D'Kitty | 4M unsupervised pretrain + downstream | 2-10M pretrain + 1-5M finetune |
| **Seed count + CI** | **3-5 seeds, std-dev shading, no CIs** | **8 seeds, 95% CI** ← gold standard | **5 seeds, mean ± std** | **Unstated** ← red flag | **3 seeds** | **5 seeds** |
| **Hardware** | None | None | **ANYmal-D zero-shot** | **D'Kitty real-robot (off-DADS)** — 20 robot-hours, hand-supervised resets | None | None |

### Patterns across papers

1. **State coverage is universal proxy for "did skills diverge."** Quantified as bin count (METRA), std-of-means (D3), or dependency-graph variety (SkiLD). Qualitative only in DIAYN.
2. **Downstream task is the universal end-test.** Frozen skills + meta-controller. Method varies: PPO over z (D3, DUSDi), MPPI in skill space (DADS), zero-shot goal-z (METRA).
3. **Hardware safety has a distinct metric** (D3): illegal-contact percentages. None of the sim-only papers have this.
4. **Seed counts vary 3–8.** METRA's 8 seeds + 95% CI is the methodological gold standard; D3's 5 seeds is acceptable for a hardware paper. DIAYN/DUSDi at 3 seeds is the floor.
5. **Discriminator accuracy is rarely a headline metric** — it's a sanity-check sub-metric. The headline is what skills *do* in state space, not how confident the discriminator is.

### Failure modes documented

- **DIAYN skill collapse** (URLB benchmark, CIC paper): "no competence-based approach achieves SOTA on URLB" — DIAYN learns small static skill sets, sensitive to alpha (entropy), num_skills, and early-termination dynamics. On DM Control Hopper without early reset, "both algorithms collapse." On Walker-Run 2M+100k: DIAYN 242±11 vs APT 344±28.
- **METRA λ explosion** if constraint persistently violated; λ slack design (`min(ε, 1 − ‖Δφ‖²)`) intentionally caps gradient.
- **DADS improper bound** (Appendix C): the variational ratio is not strictly a lower bound. Empirical only.
- **DADS hardware reality**: 20 robot-hours, motor replacements, hand-supervised resets, episode length dropped 200 → 60.
- **D3 acknowledged limits**: loco-manip failed; obstacle avoidance never emerges; symmetry suppresses single-foot lifts; factor list + algo choice manual.
- **SkiLD limitation**: requires multi-object / contact-rich envs to have signal — pCMI between rigid-body factors of one robot is near-saturated.

---

## Part 2 — Method tradeoff matrix

| Method | Skill type | Core objective | Strength | Cost / weakness | Hardware proven |
|---|---|---|---|---|---|
| **DIAYN** | discrete one-hot | `I(s; z)` via discriminator `q(z\|s)` | foundational, simple, fast | mode collapse, metric-agnostic, fails in pixel/high-dim, sensitive to entropy + num_skills | No |
| **METRA** | continuous on unit sphere | `(φ(s')−φ(s))ᵀz` with Lipschitz constraint via Lagrangian dual | scales to pixel, big spatial coverage, principled metric awareness | dual-λ tuning, sample-inefficient (UTD 1/16), no hardware demo | No |
| **DADS** | continuous in `[-1,1]^D` | `I(s'; z \| s)` via forward dynamics model `q_φ(s'\|s,z)`; planning via MPPI in skill space | predictable transitions, **zero-shot composition without RL**, real-robot deployed | improper bound (Appendix C), x-y prior often required, 20 robot-hours per real run, sample-inefficient | **Yes** (D'Kitty) |
| **D3** | factorized DIAYN+METRA + style + λ-weights | per-factor objectives sum + style + safety + symmetry-augmented PPO | hardware-deployed quadruped, factorization explicit, operator slider deploy, ablations cover style/symmetry/λ | manual factor + algo choice, no loco-manipulation, no obstacle avoidance, no online arbitration | **Yes** (ANYmal-D) |
| **DUSDi** | factored discrete | `Σ_i [I(s^i; z^i) − λ I(s^¬i; z^i)]` — positive within-factor + negative cross-factor MI | provable disentanglement (DCI metric), value-Q decomposition reduces variance | requires factor schema, sim only, demonstrated on toys + iGibson (no quadruped) | No |
| **SkiLD** | causal interaction graph + diversity index | reward inducing dependency graph + DIAYN-inner diversity term | learns interaction structure, useful for contact-rich + multi-object | requires multi-factor envs with non-trivial pCMI, dynamics model + ε threshold, hierarchical PPO | No |

### Compatibility / orthogonality

- **D3 ⊥ DUSDi**: D3 has prescribed factors but no explicit cross-factor independence loss. DUSDi's `−λ I(s^¬i; z^i)` term is **drop-in compatible** as a regularizer over D3's factors. Cost: extra discriminator per factor.
- **D3 ⊥ DADS**: DADS replaces *what* objective each factor uses (predictability instead of discriminability/Wasserstein). On the position factor, DADS could swap for METRA — same goal (continuous, model-aware skills), different bound. Bonus: DADS gives MPPI composition for free, which would replace D3's hierarchical PPO with planning. **Major tradeoff**: DADS's improper bound vs METRA's principled Lipschitz constraint.
- **D3 ⊥ SkiLD**: SkiLD's pCMI shines when factors couple intermittently (object pickup, terrain contact). Pure free-running locomotion has dense factor coupling — pCMI is near-saturated, no signal. Useful **only if** we extend to loco-manipulation (which D3 explicitly failed at).
- **METRA ⊥ DIAYN**: directly competing on state coverage. METRA wins on pixels and scale; DIAYN wins on simplicity + interpretability (one-hot skills). D3 uses both, choosing per factor.
- **DADS ⊥ DIAYN**: DADS is "DIAYN with a forward model instead of a posterior model." Different bound direction, different reward signal. DADS wins on transition predictability + planning; DIAYN wins on training simplicity.

---

## Part 3 — Mapping to our SD-A → SD-E plan

### What's already in our plan and validated by the literature
- DIAYN (foundation) + METRA (scaling) + D3 (hardware) is a defensible spine.
- One-hot prior for SD-A through SD-C (matches DIAYN paper, matches D3 for DIAYN factors).
- Sample-time intrinsic reward for off-policy (paper precedent: nobody does collection-time for SAC; off-DADS uses importance-sampling clipping for the same problem).

### What we should add given the research

1. **Concrete validation gates per phase.** See Part 4 below.
2. **Seed/CI convention.** Adopt **5 seeds with mean ± std** (matches D3, the most relevant prior). 8 seeds (METRA gold standard) only if compute permits and a result is contentious.
3. **Per-skill rollout protocol.** Standardize: for each fixed z, run M episodes (M=10 sim, M=3 hardware), report mean ± std of (return, fall rate, behavior summary). D3 doesn't fix M precisely — we should.
4. **Open-impl-questions doc.** DIAYN paper doesn't pin discriminator MLP size; METRA paper doesn't pin λ_init/slack/lr in the readable HTML. We must read source repos before SD-B coding (already in spec but emphasize).

### What we should evaluate adding outside D3 scope

- **DUSDi negative-MI penalty** is cheap to add, gives a measurable disentanglement number (DCI). Worth a SD-E ablation: D3 vs D3+DUSDi on Go2 factors. Hypothesis: D3 factors (xy, heading, height, roll/pitch) on a rigid body are not naturally independent (heading correlates with xy direction), so an explicit independence penalty might help skill interpretability.
- **DADS-style dynamics model** is more invasive. Worth flagging as SD-F/G (post-D3) only if D3 hierarchical PPO fails to compose skills well downstream. Don't bake into the spec yet.
- **SkiLD** is irrelevant unless we add manipulation. Note for future, don't plan for now.

### What we should explicitly not pursue *for the current locomotion target*

- **Pixel-based skills** (METRA's headline) — Go2 deploy uses proprioception, no cameras. Skip METRA's pixel ablations entirely.
- **DADS MPPI hardware composition** — needs full forward dynamics model + 20 robot-hours per task. Disproportionate cost.

### What stays on the radar for manipulation extension

- **SkiLD causal graph discovery.** Pure free-running locomotion has dense factor coupling — pCMI between rigid-body factors of one robot is near-saturated, no signal. **But** if we extend to loco-manipulation (which D3 explicitly failed at: "pushing degenerated to unsafe collisions") or stationary manipulation (Go2 arm? table-top robot? bimanual?), SkiLD's interaction-graph discovery becomes load-bearing. The pCMI signal lights up exactly when factors couple intermittently — robot↔object, object↔object — which is the manipulation regime.
  - **Decision:** keep SkiLD-style interaction-graph hooks orthogonal to the D3 locomotion stack. Don't bake into SD-B/C/D, but design the factor extractor registry (SD-A) so a future `InteractionGraphFactor` can plug in without refactor. Concretely: factor extractors return `(batch_array, metadata_dict)` rather than just an array, so dependency-graph factors can pass `g_target` alongside their state-factor reads. **Cost in SD-A:** ~5 lines of API.
  - **Re-evaluate at:** start of any manipulation phase (post-SD-E, or earlier branch if hardware target shifts).

---

## Part 4 — Validation contract per SD phase

### SD-A: scaffolding

**Validation goal:** numerical correctness of equations, no behavioral claims.

- [ ] Discriminator accuracy on synthetic separable data (skill i ↔ obs = i) reaches >50% (4-skill chance is 25%) within 50 grad steps. Already in plan Task 4.
- [ ] `compute_intrinsic_reward` deterministic for fixed aux state + batch.
- [ ] `compute_intrinsic_reward` changes when aux state updates (already in plan Task 5).
- [ ] No real env, no real training run.

**No paper to reproduce — this is unit-test layer only.**

### SD-B: DIAYN training loop on simple env

**Validation goal:** match DIAYN-paper qualitative behavior on a published env.

- [ ] **Smoke (10K steps CPU):** finite aux losses, no NaN, buffer fills.
- [ ] **Discriminator accuracy curve:** plot accuracy over training. Should rise above chance (`1/num_skills`) within first 100K steps.
- [ ] **Per-skill task-return histogram:** at 1M steps, run M=10 episodes per fixed skill, plot histogram of *env return* under each skill. Should show diversity (not all skills get same return). Matches DIAYN App. D.4 protocol.
- [ ] **Replicate DIAYN HalfCheetah qualitative:** at 1M steps, render 1 video per skill (use `record_video.py`). Skills should show diverse gaits (run forward, run backward, jump, flip, etc.). Subjective check.
- [ ] **Seed variance:** **3 seeds minimum** for SD-B smoke (matches DIAYN paper); upgrade to 5 seeds if results are contentious.
- [ ] **Buffer regression:** existing `test_jax_replay_buffer.py` asymmetric-critic tests still pass.

**Acceptance threshold:** mean per-skill return spread (max − min across skills) > 50% of any single skill's return at 1M steps. Loose threshold; tightened in SD-C.

### SD-C: Go2 DIAYN with deployable obs

**Validation goal:** match D3 paper diversity protocol on Go2 with a single DIAYN factor.

- [ ] **Per-skill state coverage** (D3 Table 2 protocol): run **1000+ skill samples** (i.e., 1000 random z's, one rollout each); report std-of-mean-states for the factor's tracked dimension. Higher = more diverse skills.
- [ ] **Per-skill rollout aggregates** (D3-style): for each of the K=4 (DIAYN one-hot) skills, run M=10 episodes; report mean ± std of:
  - episode return
  - episode length (proxy for fall rate; D3 uses illegal-contact %)
  - command-tracking error (since SD-C uses command-conditioned skill factor)
  - per-leg torque RMS
- [ ] **Skill-following fidelity** (D3-style cosine similarity): if our DIAYN factor is "command-conditioned behavior class," the skill should correlate with realized base velocity direction. Cosine similarity > 0.5 mean across skills.
- [ ] **Headline plot:** xy trajectory per skill (matches METRA Fig 3 protocol). Color by skill index.
- [ ] **Seed variance: 3-5 seeds**, mean ± std.
- [ ] **No hardware.** Sim only.

**Acceptance threshold:** state-coverage std rises above the no-skill (vanilla SAC) baseline by at least 2×. **Fall rate ≤ 20%** per skill (looser than D3's hardware bar; SD-C is sim-only).

### SD-D: deploy contract for fixed skills (sim only)

**Validation goal:** sim deployment with explicit contract; **no hardware**.

- [ ] `deploy_go2.py --sim --skill-index 0` runs a fixed skill end-to-end without contract errors.
- [ ] Dim check: `(raw_dim * n_frame_stack) + skill_dim == runner.obs_dim`.
- [ ] Real-mode rejects skill checkpoints (`hardware_ready=false`).
- [ ] ONNX sidecar / `deploy_contract.json` round-trips.
- [ ] `sim2sim_direct.py` matches sim training rollout to within tolerance (existing sim2sim gate, extended for skills).

**No paper to reproduce — this is contract-correctness layer.**

### SD-E: D3 full factorization + hardware

**Validation goal:** replicate the load-bearing D3 ablation tables.

- [ ] **D3 Table 1 replication** (style on/off, sim): with-style vs without-style, report illegal-contact % per body part (base, shank, thigh) and per-skill task return. Style should reduce contacts ≥10×.
- [ ] **D3 Table 2 replication** (algo choice ablation, sim): DIAYN-only, METRA-only, D3 mixed, on Go2 factors. Report state coverage per factor. Mixed should beat single-method on at least 2 factors.
- [ ] **D3 Table 3 replication** (downstream nav, sim): hierarchical PPO over frozen skills on rough-terrain waypoint task. Report mean reward, heading error, position error, termination ratios.
- [ ] **D3 Fig 5 replication**: roll/pitch coverage map with/without symmetry.
- [ ] **DUSDi DCI score** (extra): compute DCI on D3 factors. Report as a number; this is informational, not a gate.
- [ ] **Hardware readiness gate**: `meta["skill_discovery"]["hardware_ready"] = true` only after passing all sim ablations + a sim2real walkability test (no falls in 60s sim2sim with operator-slider z).
- [ ] **Hardware deploy** (real Go2): per fixed skill, M=3 trials, report success/fall, qualitative video. Match D3's operator-slider protocol.
- [ ] **Seed variance: 5 seeds** for sim ablations, 3 trials per skill on hardware (compute / wall-clock permitting).

**Acceptance threshold:** **at minimum** match D3's qualitative claim on 2 of 3 ablations (style, factorization, downstream). Hardware: zero base contacts in 60s of operator-driven skill execution.

---

## Part 5 — Open implementation questions — RESOLVED

All blocking questions resolved via source-code audits 2026-05-02. Detailed values + file:line cites in `.context/references/skill_discovery_source_extracts.md`. Brief table:

| Question | Resolution |
|---|---|
| DIAYN discriminator MLP sizes | DIAYN ref: `[300, 300]` plain ReLU. D3 ref: `SimBa[256, 256]` ELU. Spec ships `[256, 256]` plain in `AuxNetConfig`; SD-C/E may upgrade to SimBa. |
| DIAYN: discriminator on `s` or `s'`? | **Current state s** (confirmed `diayn.py:175-180`). Common confusion: DADS uses `s'`. |
| DIAYN obs: full state or factor slice? | DIAYN ref uses **full state**. Our SD-C uses factor extractor output (D3 pattern). |
| METRA: target φ network used? | **None** — only Q-targets (`iod/metra.py`). |
| METRA: dual λ init / lr / slack ε | `dual_lam_init=30` (log-parameterized); `dual_lam_lr=5e-4` (D3); `dual_slack=1e-3` (NOT 1e-5). |
| METRA: dual_dist? | `'one'` default — constraint is `‖Δφ‖²≤1` (constant). L2 is non-default ablation. |
| METRA: z continuous prior? | `N(0, I) → project to unit sphere`. |
| D3 learning rate? | `1e-3` adaptive schedule (NOT 1e-4 as paper text). |
| D3 λ sampling? | Half-normal `|N(0,1)|^skew` then L2-normalized. NOT Dirichlet, NOT sum-1. |
| D3 per-factor value? | 6 fully separate critic MLPs; UCB wired but disabled (β=0). |
| D3 already includes DUSDi penalty? | **Yes** — `skill_disentanglement=True`, `lambda_skill_disentanglement=0.1`. |
| D3 skill resampling cadence? | 375 steps (7.5s @ 50Hz) all factors. |
| D3 ships eval scripts? | **No** — we build our own for Tables 1/2/3. |
| Per-skill eval episode count M | M=10 sim, M=3 hardware (our convention; no paper fixes M). |
| Per-skill state-coverage sample count | 1K first run, 10K post sim2real (our convention). |
| DADS L (alt-z samples)? | **L=100** in shipped configs (NOT 500 as paper text). |
| DADS numerical guard? | Hard `np.clip(±50)`, no logsumexp — confirms "improper bound" caveat. |
| DUSDi anti-discriminator? | `anti=False` default in shipped config; D3 turns it on. |
| SkiLD lower policy? | Rainbow DQN by default for discrete envs; only upper graph-PPO is on-policy. |

---

## Part 6 — Recommendations

1. **Stay with D3 as north star.** It's the only paper proven on quadruped hardware with a clean ablation structure. Our SD-A → SD-E plan tracks it.
2. **Adopt 5-seed mean ± std as default eval convention.** Matches D3. Upgrade case-by-case.
3. **Add DUSDi DCI as informational metric in SD-E.** Cheap, gives a number for "are our factors actually independent."
4. **Defer DADS** for hardware-target locomotion; revisit only if D3 hierarchical PPO underperforms (DADS MPPI alternative). **Keep SkiLD design hooks open** — manipulation extension is on the user's radar, and SkiLD's interaction-graph approach is the strongest candidate for contact-rich / multi-object skill discovery. The factor extractor registry (SD-A) should accept metadata-bearing factors so a future SkiLD-style `InteractionGraphFactor` plugs in without refactor.
5. **Read source repos before each phase implementation start.** DIAYN architecture and METRA Lagrangian details are not fully specified in paper text. The audits of both papers explicitly flagged Appendix F / source defaults as the ground truth.
6. **Patch SD-A plan acceptance** with the SD-B "discriminator on `s` not `s'`" fix and the M=10 per-skill rollout convention.
