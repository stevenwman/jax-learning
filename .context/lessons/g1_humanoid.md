# Unitree G1 Humanoid (Warp env) — Lessons

> Initial G1 build + first joystick training: 2026-05-07.
> Env: `jax_rl/envs/locomotion/g1_warp_joystick.py`,
> XML: `xmls/g1_warp_scene_flat.xml` (vendored playground feet-only).

---

## Reward `jp.clip(reward, 0, ...)` silently kills negative-weighted terms (2026-05-07)

**What happened:** G1 v2 had `termination=-100` weight to push policy
hard against falling. After 500k steps, returns flat at 0.0 ± 0.0,
no learning. Diagnosis: Go2's `step()` clipped reward as
`jp.clip(sum(rewards) * dt, 0.0, 10000.0)`. The lower bound 0.0
nullified the -100 termination cost: at fall step, raw reward
≈ -100 × 0.02 = -2, but clip(−2, 0, ...) = 0. Policy saw "fall
== standing-still" → no gradient signal.

**Fix:** Remove the lower clip:
```python
reward = sum(rewards.values()) * self.dt   # no jp.clip lower bound
```

Once removed (v3), gradient flowed: returns went −3.1 (negative as
expected from -100 fall penalty) and entropy collapsed 18 → 0.9
(policy committed). Q correlation 0.915 (healthy critic).

**Lesson:** A reward clip with a non-negative floor erases any
negative-weighted term whose magnitude exceeds positive contributions
in that step. Humanoids especially: termination penalties, joint-limit
costs, posture costs are often the dominant signal. Don't clip rewards
at 0 unless you genuinely want to discard negative gradient. Go2's
clip happened to work because its rewards stayed positive in practice;
G1's recipe inherits the bug.

**Generic rule:** clip(r, lo, hi) where lo > -∞ should be flagged in
any RL env code review. If you must clip, use `clip(r, -1000, 10000)`
or similar — let negative gradient through.

---

## FastSAC paper defaults work fine on humanoid — earlier "broken architecture" claim was wrong (2026-05-08)

**Updated finding (supersedes the section below):** When user pushed back
that "vanilla SAC works on Humanoid, why wouldn't FastSAC?", I actually
read the holosoma source (https://github.com/amazon-far/holosoma) and
ran one more test: FastSAC + `--obs-norm` + `--reward-scaling 0.2`.

| Config | Eval @ 5M |
|---|---|
| FastSAC default | -0.7 |
| ... 6 isolation variants ... | all -1 to +0.6 |
| **FastSAC + obs-norm + reward-scale 0.2 (paper)** | **28.2 ± 0.5** |
| FlashSAC default | 26.8 ± 0.8 |

**The single critical missing knob was `obs_normalization=True`.** Paper
sets this to True; we had it False by default. The `reward-scaling 0.2`
also helped but obs-norm is the dominant factor. Both algos use C51,
both have similar architecture once normalization is matched.

What I got wrong in the earlier diagnostic:
1. Claimed "FlashSAC uses regression Q" — wrong, it's also C51
2. Then claimed "FlashSAC's residual blocks + BatchNorm are the magic" —
   wrong, FastSAC's MLP works fine with proper obs/reward normalization
3. Spent 7 isolation runs on scalar HP knobs (tau, gamma, target entropy,
   grad updates per step, etc.) without ever flipping `obs_normalization`
   to True
4. The `.context/archive/FAST_ALGOS_LIT_MISMATCH.md` doc had the obs
   normalization flag listed as "FIXED" but it was only fixed at the
   CLI/config level — not enabled in the preset I was using

**Lesson:**
- When a paper specifies normalization (obs, reward, advantage), enable
  it. Don't compare against a "no-normalization" baseline and conclude
  the algo is broken.
- Always read the paper's source code config when a published algo "fails"
  on a task it was specifically designed for. Holosoma is the FastSAC
  paper's source — should have looked there first.
- `obs_normalization` defaulting to False is dangerous; should be True
  for any humanoid locomotion preset.

---

## (DEPRECATED) FastSAC stalls where FlashSAC progresses on humanoid — likely C51 support range, not distributional-vs-regression (2026-05-07)

**This section is deprecated as of 2026-05-08 — see above.** The
isolation runs in this section identified Q-bias drift, alpha collapse,
and Q-corr inversion as fingerprints of a "broken critic" — but the
actual root cause (no obs normalization) wasn't tested. Keeping the
content for the diagnostic-pattern observations only.

**What happened:** Same env config (G1 v6, full unitree reward match),
two algorithms, drastically different outcomes after 2M steps:

| Metric | FastSAC | FlashSAC |
|---|---|---|
| Eval return | -0.7 ± 0.1 | **+6.4 ± 3.5** |
| Online return | -1.0 | **+3.7** |
| Q correlation | **-0.3 (broken)** | 0.75 (healthy) |
| Avg episode steps | 20 | **119** |

FastSAC's Q anti-correlated with MC returns: critic was actively
learning the wrong value landscape. FlashSAC's Q tracked properly.

**CORRECTION (2026-05-07):** Initial diagnosis claimed "FlashSAC
uses regression Q, FastSAC uses C51." Wrong — both use C51 categorical
critics with the same `make_support()` / `logits_to_q()` infrastructure.
The actual difference is the support range:

- FastSAC default: `v_min=-20, v_max=+20` → 101 atoms × 0.4 reward units
- FlashSAC default: `v_min=-5, v_max=+5` → 101 atoms × 0.1 reward units

On G1 where returns range [-3, +27], FastSAC's [-20, 20] support has
~75 atoms in the active region (rest wasted at the negative edge),
and atom resolution is 4× coarser. FlashSAC's [-5, 5] gives much
finer per-bin resolution where the actual return density lives.

Other differences (also possibly contributing):
- FlashSAC has residual blocks in actor/critic (vs FastSAC's MLP)
- FlashSAC dynamically normalizes rewards (`RewScale ~0.36` in logs)
- FastSAC does not

**Isolation runs (2026-05-07, 5M each on G1 v7 env w/ gait reward):**

| Config | Eval | Q corr | Q bias |
|---|---|---|---|
| FastSAC default (v_min=-20, v_max=+20) | -0.7 | -0.3 | — |
| + tight C51 support (-5/+5) [v8] | -0.3 | 0.32 | — |
| + reward scaling 0.36 [v9] | 0.6 | 0.7 | — |
| + tau 0.01, gamma 0.99, delay 2 [v10] | -0.3 | -0.13 | 1.7-2.2 |
| + target_entropy_scale 0.5 [v11, killed @52%] | -0.5 | -0.09 | 2.69 |
| + grad_updates_per_step 1 [v12] | -0.8 | -0.34 | 1.4 → 3.7 |
| + tes 0.5 + gups 1 combo [v13, killed @42%] | -1.7 (worse!) | 0.59 | — |
| **FlashSAC default** [v7] | **26.8** | 0.17 | 0.07 |

**Pattern across all FastSAC variants**: Q-bias drifts upward (critic
increasingly over-estimates) and Q-corr eventually goes negative.
Classic SAC critic divergence on high-dim continuous control. None of
the standard scalar knobs (target entropy, grad updates per step,
support range, reward scaling, tau, gamma, policy delay) prevent it.

Conclusion: scalar hyperparam differences are NOT the cause. v10 has
all of FlashSAC's hyperparams except architecture. The remaining
FlashSAC advantages are:
1. **Inverted residual blocks + BatchNorm + weight normalization** in
   actor/critic networks (vs FastSAC's plain MLP)
2. **Adaptive `RewScale`** that tracks return std dynamically (vs
   FastSAC's fixed `--reward-scaling`)
3. **G_max=5 clipping** and **sigma_target=0.15** entropy convention

The architecture is doing most of the work. Reimplementing residual
blocks + per-batch normalization in FastSAC would essentially replicate
FlashSAC. Practical takeaway: **use FlashSAC for humanoid**.

**Lesson:**
1. C51 distributional critics need careful `v_min, v_max` tuning
   matched to actual return distribution. The default
   `[-20, 20]` is too wide for tasks with rewards in [-2, +2] range
   (atom resolution wasted) AND too narrow for tasks where returns
   genuinely span 100s. Either case: effective resolution suffers.
2. **Algo failure ≠ env failure.** Tried 6 env iterations chasing the
   FastSAC issue (reward weights, spawn pose, action_scale, sim_dt,
   adding alive bonus). All marginal. Switching algo (or just the
   support range) gives 8× improvement on eval at the same step count.
   Diagnose Q correlation early.
3. When debugging "policy stuck in local optimum" with off-policy
   methods, check Q correlation BEFORE blaming env reward shape.
   Negative Q corr means critic is broken — fixing env won't help.
4. **Don't over-claim algo differences.** I initially said FlashSAC
   was non-distributional. Wrong. Always verify the algo internals
   before writing the lesson. Architecturally similar algos can
   diverge purely on hyperparameter defaults.

---

## Humanoid SAC reward landscape: alive bonus floor + small term penalty (2026-05-07)

**What happened:** Iterated through G1 reward configs:
- v3: termination=-100, no alive → robot converges to "fall fast"
  (hits the fixed -2 termination cost amortized over short episode,
  rest of return is small positive accumulation; "fall fast" minimizes
  cumulative cost).
- v4 (longer 2M): same, no help.
- v5: added alive=0.15 (matches unitree). Marginal improvement: still
  ~-3 returns at 720k.
- v6: full unitree match: termination=0, alive=0.15, base_height=-10,
  joint_deviation_legs=-1.0, action_rate=-0.05, dropped clipped reward.
  → FlashSAC reaches +6.4 (FastSAC stalls at -0.7 due to C51 issue).

**Diagnosis of v3-v5 failure mode:**
- High termination penalty (-100) creates a local optimum where
  "fall in N steps" pays roughly the same as "fall in 2N steps" (the
  -2 cliff dominates the per-step accumulation).
- Without alive bonus, every step alive pays approximately 0
  (tracking_lin_vel ≈ 1 × weight 1 = 1, but ang_vel costs and joint
  deviation costs ≈ -1, net 0).
- So the gradient between "fall fast" and "fall slow" is nearly flat;
  policy commits to whatever local optimum it stumbles into first
  (fall fast, since random sigma-1 actions immediately destabilize).

**Unitree's recipe** (from
`unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`):
- alive = +0.15 (per-step)
- termination cost = 0 (no penalty)
- base_height = -10 (target 0.78m), strong "stay tall"
- flat_orientation_l2 = -5
- joint_deviation_legs = -1.0 (.*hip_roll, .*hip_yaw)
- action_rate = -0.05
- dof_pos_limits = -5
- gait = +0.5 (CPG-like phase) — we don't have this yet
- Curriculum cmd ranges: start at ±0.1, expand to ±1 only after
  terrain progress

**Lesson:**
1. **Per-step survival bonus > one-time termination cliff.** The
   gradient from "live longer" must be present at every step, not
   concentrated at the death step. Alive bonus + alive bonus +
   alive bonus accumulating across an episode gives a smooth signal.
2. **Match a known-working recipe before tuning.** I tried 5
   configurations of varying creativity before finding that
   unitree's published recipe just works (with FlashSAC). Cheaper
   to copy first, modify only after validation.
3. **Narrow cmd ranges early.** Asking a humanoid to walk at vx=1.0
   from step 1 with random init is a curriculum mistake. Start with
   cmd ≈ 0 (stand still), add walking later.
