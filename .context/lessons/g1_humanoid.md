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

## FastSAC's C51 distributional critic fails on humanoid; FlashSAC works (2026-05-07)

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

**Suspected root cause:** C51 distributional critic with
`v_min=-20, v_max=+20` support and 101 atoms. On G1 with sparse
positive reward (alive 0.15) and clustered negative-cost terms
(joint deviations -1.0 each, base height -10), the per-step reward
distribution is heavily peaked near small negatives. The C51 categorical
distribution may struggle to allocate atoms across this skewed,
narrow-support range, especially when the policy explores rapidly
(early-fall episodes have very different return distributions than
later partial-stand episodes).

FlashSAC uses plain regression Q (single scalar), no support
discretization. Just MSE between Q(s,a) and TD target. Insensitive
to return distribution shape.

**Fix:** Use FlashSAC for humanoid joystick task. FastSAC remains
appropriate for tasks where rewards are bounded near zero and
relatively symmetric (Go2 quadruped at eval ~280 worked fine).

**Lesson:**
1. C51 distributional critics need careful `v_min, v_max` tuning
   matched to actual return distribution. The default
   `[-20, 20]` is too wide for tasks with rewards in [-2, +2] range
   (most atoms unused) AND too narrow for tasks where returns can
   genuinely span 100s. Either case: the critic's effective resolution
   is wasted.
2. **Algo failure ≠ env failure.** Tried 6 env iterations chasing the
   FastSAC issue (reward weights, spawn pose, action_scale, sim_dt,
   adding alive bonus). All marginal. Switching algo to FlashSAC: 8x
   improvement on eval at the same step count, same env config.
   Diagnose Q correlation early.
3. When debugging "policy stuck in local optimum" with off-policy
   methods, check Q correlation BEFORE blaming env reward shape.
   Negative Q corr means critic is broken — fixing env won't help.

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
