# Push-T Reward & Critic Lessons

Reward shaping, log-barrier curvature, success-bonus tuning, and the FastSAC C51 mismatch on the push-T benchmark. Obs/action choices in `pusht_design.md`; eval/metric calibration in `pusht_eval.md`.

---

## Zero-Action Attractor in Velocity/Teleport Modes (2026-04-18)

**What happened:** On push-T with shaped reward, position-PD mode converged to eval +143 (47% success). Velocity-delta and teleport modes plateaued at -85 to -90 despite same obs, same reward, same training budget. Shaping tweaks (r_block_vel 2→10, added r_pusher_vel_toward_block) marginally helped vel/tele but never closed the gap.

**Diagnosis (with per-component reward + velocity logging):** pusher_vel_mag 0.80 m/s in pos mode vs 0.06 m/s in vel/tele. Even with 4× cap bump (vel 0.01→0.04 m/step, tele 0.25→1.5 m/s), pusher used only 7-13% of the cap. Policies learned small actions and stayed there.

**Root cause:** in vel/teleport modes, `action=0 → ctrl=current → pusher holds still`. Self-consistent attractor. Policy has no local gradient to explore "what if I moved aggressively" — any aggressive action reverts to hover next step because the new ctrl again equals current pusher position. In pos-PD mode, `action=0 → ctrl=[0,0] ≠ current` so PD always yanks pusher toward origin at kp×err force. Accidentally forces motion even at zero action. Exploration-friendly by construction.

**Implications:**
- Position-PD is the surprisingly-good default for pushing tasks trained from scratch with RL.
- Velocity/teleport match imitation-literature conventions (gym-pusht, DP) but those use expert demos as exploration crutch — not RL-from-scratch.
- Don't assume equivalent action spaces have equivalent trainability. The attractor structure matters.

**Fix options (none fully closed gap in our tests):**
1. Bypass PD entirely for vel mode: directly set `qvel[pusher]=action*scale`. Removes attractor but breaks contact force coupling.
2. Large action bonus term `+α||action||` to counter the hover equilibrium. Fragile.
3. Accept: pos-PD for RL, reserve vel/tele for BC/imitation comparisons.

**Lesson:** before blaming HPs, check if your action parameterization has a fixed-point at `action=0`. If yes, and the fixed-point is inside the workspace, expect cautious policies. Add diagnostic `pusher_vel_mag / block_vel_mag / pusher_to_block` to metrics when A/B-ing action modes — the numbers tell you whether policy is action-limited or structural.

---

## Reward Shaping Strength Is a Dial, Not a Monotonic Knob (2026-04-18)

**What happened:** Bumping `r_block_vel` scale from 2.0 → 10.0 tanked pos mode from eval +143 → -1.7. Bigger shaping reward made training worse.

**Mechanism:** at 10.0, the reward became: _achievable only if you shove the block around_. Angle-matching (`r_angle`) stopped mattering in relative terms — drowned by block_vel gradients. Policy learned to hustle the block back-and-forth for r_block_vel rather than patiently rotating to the target angle. Returns got mostly-positive-but-not-solving: task-reward noise but benchmark-low success rate (47% → 3.5%).

**Reverting `r_block_vel` to 2.0** restored eval +143 at cost of losing the new `r_pusher_vel` term (net −25 from the perturbation, better than -1.7 but still below +143).

**Lesson:** when shaping, the ratio between shaping terms and ground-truth (pos+angle error) must stay bounded. A shaping term that dominates becomes the policy's real objective — and if it's not aligned with the actual task, you optimize the wrong thing. Tuning heuristic: keep shaping reward magnitude ≤ 0.5× max task reward. If a bigger shaping term helps early exploration, anneal it down over training (curriculum on reward weights).

---

## Always Log Per-Component Reward + Velocity Magnitudes When Shaping (2026-04-18)

**What happened:** Spent several iterations blindly tweaking reward term weights without visibility into which component was dominant. Added env-side metrics (`r_pos`, `r_angle`, `r_approach`, `r_block_vel`, `r_pusher_vel`, `pusher_vel_mag`, `block_vel_mag`, `pusher_to_block`) then ran a quick diag rollout on each checkpoint. Immediately saw pusher_vel 24× smaller in vel/tele vs pos. One diag run answered several open hypotheses.

**Pattern:**
1. Emit every reward component as a scalar metric from env (in `state.metrics`).
2. Emit proxy metrics for "is the policy even _using_ the action space?" (`pusher_vel_mag`, distance to relevant bodies).
3. Rollout 200 steps on the trained checkpoint, print mean of each metric.
4. Diff metrics across configs instead of only comparing eval reward.

**Lesson:** per-component reward logging is essentially free (µs per step) and turns "why is this worse" from a multi-hour A/B into a 10-line diff. Bake it into env design from day 1 for any env with more than 2 reward terms.

---

## Tunable `success_threshold` for Sparse-Reward Tractability (2026-04-19)

**What happened:** Default `success_threshold=0.95` (DP convention) is unreachable by humans (max LeRobot demo = 0.9489) and by pure RL (peak ~0.89). Sparse reward mode `1 if coverage > 0.95 else 0` therefore gives literally zero signal during from-scratch training.

**Fix:** added `success_threshold` kwarg to `PushTEnv`. Defaults to 0.95 (parity), can be lowered for tractable sparse RL:
```python
env = PushTEnv(reward_mode="sparse", success_threshold=0.85)
```

**Reporting convention reminder:** push-T literature reports **max coverage per episode**, not binary success rate. DP scores ~0.91-0.95, BC LSTM ~0.55-0.74, ours ~0.84-0.89. Don't compare against 0.95 termination flag.

---

## FastSAC C51 Critic Is Wrong Choice for Bounded-Reward Manipulation (2026-04-19)

**What happened:** Early training attempts used `FastSAC` (C51 distributional critic, atoms over `[v_min, v_max]` range). Training diverged because `v_max` defaults to 20 in the paper preset, but cumulative contact_gated reward over 300 steps can hit ~150. Critic atoms don't cover actual Q range → critic is "blind" beyond v_max → policy can't improve past that ceiling.

**Fix:** switch to vanilla SAC (scalar Q, unbounded). Same task works immediately (once TimeLimit is also fixed).

**When to use each (updated from earlier locomotion-centric lessons):**
- **FastSAC / C51**: good for **unbounded** locomotion rewards, paper-tuned for humanoid / Go2 scale. Needs `v_min`/`v_max` sized to actual discounted Q.
- **Vanilla SAC**: better for **bounded, short-horizon, shaped-reward** tasks like push-T. No atom-range landmine.

**Lesson:** distributional critics are optimization tools, not reward-range magic. Always sanity-check that `[v_min, v_max]` covers `reward_min * horizon` to `reward_max * horizon` under your gamma and episode length. If it doesn't, the distributional critic is actively worse than a scalar one.

---

## Log-Barrier Coverage Reward Beats Linear by +8pp (2026-04-20)

**What happened:** Baseline `contact_gated` used `r_coverage = coverage_clip` — linear in `[0, 1]`. Policy plateaued around 85% cov because marginal reward was flat (1pp gain = 0.01 reward regardless of coverage level). Geometry of coverage metric is nonlinear: 30→60% is gross-motor, 90→95% is pixel-precision. Linear underpays precision work.

**Fix:** `coverage_shape="log_barrier"` → `r_coverage = -log(1 - clip(cov/threshold, 0, 1) + ε)` with `ε=0.01`.

```
cov    linear   log_bar
0.5    0.53     0.73
0.7    0.74     1.30
0.9    0.95     2.77
0.95+  1.00     4.60 (ceiling at ε=0.01)
```

Marginal reward near goal ∝ `1/(1-cov+ε)` — at cov=0.9, gradient is 9× steeper than at cov=0. Policy actually pursues the last few percent.

**Evidence:**
- v9 linear (full stack): 0.852 sto
- baseline_logbar (same stack, log_bar only): **0.933 sto** (+8pp, det std 5× tighter)

**Cost:** Q magnitudes grow ~3-5× (ep_r_avg from 130 → 500). No instability observed at `reward_scale=0.1, grad_clip_norm=1.0`. Monitor Q1 trajectory for early-training blowup signs.

**When NOT to use:** if coverage metric is already well-distributed (e.g. dense geometric progress), log_barrier adds instability without gain. Best applied to metrics where the last few % are disproportionately hard (IoU, pose overlap, SSIM).

**Lesson:** match reward curvature to metric curvature. Linear reward on a geometrically-nonlinear metric caps learning where marginal task difficulty exceeds marginal reward. Log-barrier is the standard interior-point shape for this.

---

## Bigger Success Bonus Doesn't Raise Ceiling on Unreachable Thresholds (2026-04-20)

**What happened:** tested `success_bonus: 50 → 200` on full-stack log_bar config. Expected: bigger terminal pull → policy reaches higher coverage. Actual: det 0.867 → 0.914 (+5pp tighter), sto 0.933 → 0.933 (unchanged).

**Why:** with `success_threshold=0.95` and peak coverage observed during training ~0.94, the policy never crossed threshold → never sampled the bonus. Making an unobserved terminal larger doesn't change learning.

Det tightened because Q near goal has less variance (policy converges to the same near-threshold trajectory), but the stochastic ceiling is set by the unreachable threshold, not the bonus magnitude.

**Lesson:** before tuning success bonus, verify policy actually hits successful terminations during training. If terminal is never sampled, its magnitude is irrelevant to learning — only to offline analysis. Alternative: lower threshold until terminations happen during training (tradeoff: caps learning at the threshold).
