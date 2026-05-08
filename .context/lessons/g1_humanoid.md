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

## Splitbelt-G1: works but plateaus at eval ~25 — actor needs belt obs to break the gap (2026-05-08)

Ported `G1WarpSplitbeltEnv` from `g1_warp_joystick`. Subclass adds belt
actuators (slide+vel), schedule sampling at reset, off-belt termination,
splitbelt info dict. Reuses `splitbelt_geom` + `splitbelt_schedules` from
the existing splitbelt infrastructure.

| Run | Variant | Eval @ 5M |
|---|---|---|
| v18 | flat HoloSoft (FastSAC paper) | **292** |
| v19 | splitbelt-tied 0.5 | 26.4 ± 19.5 |
| v20 | splitbelt-DR (random_per_episode v∈[0.3,1.0]) | 22.8 ± 6.3 |
| v21 | splitbelt-DR + off_belt_term=False | 21.6 ± 8.7 |
| v22 | splitbelt-DR + treadmill_drift=-5 | 18.6 ± 6.2 |

**Splitbelt costs ~10× the eval ceiling vs flat (292 → ~25)**, even
holding algo + reward set + spawn pose constant. Diagnostics tried:

- **off_belt_termination disabled (v21)**: no improvement. Hypothesis was
  that swing-foot lateral drift falsely fires off-belt at y boundary
  ±0.500m. Disabling didn't recover the gap, so this isn't dominant.
- **treadmill_drift cost (v22, weight -5)**: penalize body xy from
  origin → "stay anchored" gradient. Didn't help — policy saw it as
  one more cost without the observation to act on.
- All variants: ~20-30 mean, max episode 50-65, episode length ~50 steps
  avg. Max 100 steps in best cases. Robot CAN walk on belts in some
  trajectories, just not consistently.

**Diagnosis**: actor obs is proprio + cmd=0 (no belt info). On flat
ground proprio is enough — feet feel the ground via joint torques. On
moving belts, proprio gives ambiguous signal: "joint torque rising" can
mean "I'm pushing into ground" or "ground is pulling me along". Without
belt_vel in obs, policy has to infer indirectly from gyro/joint
deflection — way harder.

**Splitbelt-G1 specific quirks** (port pitfalls):
1. **Joint order in scene XML**: include order matters. With
   `treadmill_splitbelt.xml` first, belt slide joints occupy `qpos[0,1]`
   and the freejoint shifts to `qpos[2:9]` (not `qpos[0:7]`). Keyframe
   qpos layout breaks → robot spawns upside-down at x=0.76, z=0. **Fix**:
   include `g1_feetonly_splitbelt.xml` FIRST so freejoint occupies
   `qpos[0:7]`.
2. **Joint slicing**: parent G1WarpJoystick's reward / obs lambdas
   originally used `data.qpos[7:]` (all 31 joints in splitbelt model
   = 29 body + 2 belts). Need to slice `[7:7+NUM_ACTUATORS]` (=29) to
   match `_default_pose` shape. Same for `data.qvel[6:]` and `jnt_range`.
3. **Vendored `g1_feetonly_splitbelt.xml`**: stripped the inline
   `<contact>` block (refs to "floor" geom which splitbelt scene doesn't
   define) and `<sensor>` block (scene defines all sensors). Just the
   model.
4. **Subclass `_post_init` bypass**: parent G1WarpJoystick._post_init
   loads keyframe "knees_bent" with `qpos[7:]` → would 31-element on
   splitbelt model. Patched parent to skip if `_init_q`/`_default_pose`
   already set, so subclass sets them first.

**Future paths to close the gap to flat-G1 (eval 292)**:
- **(highest leverage)** Add belt_vel + drift_xy to actor obs (like Go2's
  "informed" / "error" obs modes). Even just a 2-d belt_vel addition
  should help substantially — actor knows which way feet are being
  dragged.
- **PoseDR-style obs**: actor sees world body xyz + upvec + fwdvec
  (idealized; not deployable). Ground-truth localization shortcuts
  inferring drag.
- **Curriculum**: start with v=0.1 belts, ramp up. Robot learns to walk
  on slow belts first.
- **Longer training**: 5M is short for this task. Try 20M if compute
  permits.

Videos in `projects/adaptation/videos/g1_v19_splitbelt_tied/`,
`g1_v20_splitbelt_dr/`, `g1_v21_splitbelt_dr_no_offbelt/`, and
`g1_v22_splitbelt_drift/`.

---

## Holosoma reward weights @ 0.5× penalties = working G1 walker, eval 292 (2026-05-08)

After paper-match unlocked FastSAC (eval 28 on light-penalty Flat env),
ported holosoma's full G1 reward set (`config_values/loco/g1/reward.py:
g1_29dof_loco_fast_sac`) including missing terms (`close_feet_xy`,
`feet_ori`, per-joint pose weights, alive=10).

**With full penalty weights**: G1WarpJoystickHolo, FlashSAC 5M → eval
**10.9 ± 1.7**. WORSE than light-penalty Flat. Penalties dominate before
the policy learns.

**With penalties × 0.5** (matching holosoma's `penalty_curriculum`
initial state where `min_scale=0.5`): G1WarpJoystickHoloSoft, FlashSAC
5M → eval **273.9 ± 2.0**, FastSAC paper-match 5M → eval **292.1 ± 0.6**.
Full 1000-step episode survival, gait emerges (videos at
`projects/adaptation/videos/g1_v17_flashsac_holosoft/` and
`g1_v18_fastsac_holosoft_papermatch/`).

Why holosoma's weights work at 0.5×: their training never actually
reaches 1.0× because their adaptive `PenaltyCurriculum` ramps up only
when avg_epl > 750 — robot rarely hits that. So holosoma's *effective*
weights are ~0.5× throughout training. Static `HoloSoft` preset
replicates this without curriculum infrastructure.

| Run | Algo | Env | Eval @ 5M |
|---|---|---|---|
| v7 | FlashSAC | Flat (light) | 26.8 |
| v15 | FastSAC paper-match | Flat (light) | 28.2 |
| v16 | FlashSAC | Holo (full pen.) | 10.9 |
| v17 | FlashSAC | HoloSoft (pen ×0.5) | 273.9 |
| **v18** | **FastSAC paper-match** | **HoloSoft (pen ×0.5)** | **292.1** |

**Lesson:**
1. **Reward magnitude matters more than weight ratios.** Holosoma's full
   weights (`alive=10, action_rate=-2, orientation=-10`) are correctly
   *balanced* but absolutely too punishing for early exploration. Halving
   the penalty terms preserves balance, halves total signal magnitude
   per-step, lets policy survive long enough to find walking.
2. **Adaptive curriculum is roughly equivalent to picking the right
   static `min_scale`** for many tasks. If the robot can't reach
   `level_up_threshold`, the curriculum's `min_scale` is what it spends
   most time at. Static preset matching that initial state is a good
   shortcut before investing in adaptive logic.
3. Per-joint pose weights matter: `[0.01, 1, 5, 0.01, 5, 5]` × 2 (legs)
   leaves hip-pitch + knee free for stride; `[50] × 17` (waist + arms)
   locks upper body rigid. Uniform pose weight (e.g. our earlier `[1] × 29`)
   forces all 29 DOFs equally toward default — kills the natural gait.
4. **Ad-hoc reward iteration without checking the paper's reference is a
   trap.** I burned 13 wrong-headed runs (v1-v14) tuning weights from
   scratch. Reading `holosoma/config_values/loco/g1/` in 30 seconds
   would have given the answer.

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
