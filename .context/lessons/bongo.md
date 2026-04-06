# Bongo Board Handstand Lessons

---

## Always Verify Policy Behavior Visually (2026-04-01)

**What happened:** FastSAC on Go2BongoHandstand reached eval 397 — impressive on paper. Recorded a video and the robot was balancing on the flat ground next to the board, not doing a handstand on the board at all.

**Root cause:** No termination condition for feet touching the floor. The robot learned it could slide off the board, land on the ground, and balance there easily. The `com_above_support` reward still pulled it toward the board center, but there was nothing forcing it to stay ON the board.

**Fix:** Added foot-floor contact termination — any of the 4 foot geoms contacting the floor = instant episode end. Uses existing `FL_floor_found` etc. contact sensors from the scene XML.

**Lesson:** High eval scores mean nothing without visual verification. Reward hacking is silent in logs. ALWAYS record a video of your best policy before celebrating.

---

## CMA-ES Hard Rejects Poison the Population (2026-04-01)

**What happened:** CMA-ES for handstand keyframe optimization. Invalid poses (head underground, feet above head) returned cost=1e6. At some sigma values, the entire population was rejected — all 128 candidates returned 1e6. CMA-ES couldn't learn anything because every candidate looked equally bad.

**Root cause:** Hard rejection gives no gradient information. If 90% of the population is invalid, CMA-ES can't distinguish "almost valid" from "completely wrong" — both are 1e6.

**Fix:** Switched to soft penalties. Instead of `return 1e6` for head near ground, use `penalty += (threshold - geom_z) * weight`. CMA-ES can now see that geom_z=0.01 is worse than geom_z=0.04, and move toward valid regions.

**Exception:** Tripod detection (head resting ON ground as a stable 3rd contact point) uses hard reject because it's an exploit, not a soft constraint. The pose is stable and would get high survival score — soft penalty can't compete with 8000 steps of survival reward.

**Lesson:** For continuous optimization, prefer smooth penalties over hard boundaries. Reserve hard rejects only for exploit detection where the invalid pose would otherwise score well.

---

## Constrained DOFs Should Be Computed, Not Optimized (2026-04-01)

**What happened:** Initially had `base_z` as a CMA-ES parameter (10 dims total). The optimizer found poses where the robot was underground, floating, or with feet at inconsistent heights. Many bugs traced back to `base_z` being independent of the actual leg geometry.

**Root cause:** Given a pitch angle and joint configuration, there is exactly ONE base_z that puts the front feet on the ground. Making it a free parameter violates this constraint — the optimizer explores impossible poses.

**Fix:** `_solve_base_z()` — forward kinematics once, measure feet height, adjust base_z. Reduced to 7 params. Eliminated all spawn-height bugs in one shot.

**Lesson:** If a quantity is determined by other parameters, compute it — don't optimize it. Free dimensions that violate physical constraints waste optimizer budget and create bugs.

---

## MuJoCo Euler Uses Degrees by Default (2026-04-01)

**What happened:** Bongo board roller had `euler="1.5708 0 0"` to rotate the cylinder 90° from Z-axis to Y-axis. The roller appeared upright (unrotated) in renders.

**Root cause:** MuJoCo's default angle unit is degrees, not radians. `euler="1.5708 0 0"` = 1.57 degrees, not π/2 radians. The Go2 XML has `<compiler angle="radian"/>` but the bongo board XML (included separately) doesn't inherit it consistently.

**Fix:** Use `quat="0.7071 0.7071 0 0"` instead of euler. Quaternions are unit-independent and work correctly regardless of the `<compiler angle>` setting in any including file.

**Lesson:** When writing MJCF files that will be `<include>`d by other files, use `quat` for rotations — it's immune to the angle convention. Never assume radians unless you control the `<compiler>` block.

---

## Use Contact Sensors, Not Position Heuristics, for Termination (2026-04-02)

**What happened:** Initial termination used position checks: `base_z < 0.15`, `geom_xpos[i][2] < 0.03` for head-near-ground, `board_tilt² > 0.25`. These are all approximations — a geom's xpos is its center, not its surface. A 5cm-radius sphere at xpos z=0.04 is already touching the ground, but the check says it's fine.

**Root cause:** Position is a proxy for contact. MuJoCo already solves exact collision detection — use it.

**Fix:** Named the 3 torso collision geoms in vendored go2.xml (`torso_box`, `torso_cyl`, `torso_nose`). Added `<contact>` sensors in the scene XML for all combinations of torso×{floor, board}, feet×floor, and board×floor. Termination is now purely contact-based — no thresholds to tune.

**Lesson:** For termination conditions involving "did X touch Y", always use MuJoCo contact sensors (`<contact>` in `<sensor>`), not position heuristics. Contact sensors are exact, threshold-free, and JIT-compatible via `data.sensordata`. Position checks are approximations that need manual tuning and can miss edge cases.

---

## Reward Hacking Closes Every Loophole You Leave Open (2026-04-02)

**What happened:** Three separate exploits found across 4 training runs:
1. **Ground balancing** (run 2): Robot slid off board, balanced on flat ground. Eval 397. Fixed with foot-floor contact termination.
2. **Board slam** (run 3): Robot could slam the board flat, creating a stable platform. Fixed with board-floor contact termination.
3. **Head tripod** (CMA-ES): Head resting on ground + 2 feet = 3-point stable contact. Fixed with torso contact termination.

**Root cause:** Each fix closed one exploit, and the policy immediately found the next one. The optimizer (both CMA-ES and SAC) is adversarial — it will find ANY stable configuration that maximizes reward, regardless of whether it looks like a handstand.

**Lesson:** When designing termination for trick/skill tasks, enumerate ALL ways the robot could cheat:
- What body parts could touch the ground that shouldn't?
- Could the robot leave the apparatus entirely?
- Could the apparatus itself become a stable platform?

Add termination for each. A survival bonus without comprehensive termination just rewards finding exploits faster.

---

## Checkpoint Resume Doesn't Save the Replay Buffer (2026-04-02)

**What happened:** Resumed training from 20M → 50M steps. Returns dropped from ~70 to ~7 for the first few thousand steps, then recovered to ~40 within a few thousand more. Alpha (entropy coefficient) spiked from 0.002 to 0.008.

**Root cause:** Orbax checkpoint saves actor/critic params + optimizer state + step count. The replay buffer (millions of transitions, ~2GB) is NOT saved. On resume, the buffer is empty — the critic's Q estimates are based on the old buffer distribution, but it's now getting fresh on-policy data from a different state distribution.

**Expected behavior:** The dip is normal and recovers quickly as the buffer refills (~5-10k steps). The alpha spike is SAC's auto-tuning reacting to the distribution shift. Not a bug.

**Lesson:** When resuming off-policy training, expect a transient performance drop. The policy weights carry the learned behavior, but the critic needs fresh data to recalibrate. Don't panic at the initial dip — watch for recovery over the next few thousand steps.

---

## Frame Stacking is Critical for Balance Tasks (2026-04-03)

**What happened:** PPO on bongo handstand plateaued at eval ~24 (48% of max) across multiple reward configs and 50-100M steps. Adding `--frame-stack 3` (3× obs stacking) jumped to eval **46.9** (94% of max) with same rewards.

**Root cause:** Single-frame obs gives positions and velocities, but balance requires knowing accelerations — is the board tilt increasing or recovering? Frame stacking gives the policy implicit access to second derivatives of all state variables. The existing `last_act` in obs provided some temporal context but not enough.

**Contrast with locomotion:** Frame stacking didn't help Go2 joystick locomotion (eval 276.5 without vs 271.3 with). Locomotion is mostly a steady-state task where velocity is sufficient. Balance on an unstable platform is fundamentally about reacting to acceleration.

**Lesson:** For dynamic balance/stabilization tasks on unstable platforms, frame stacking (or equivalent temporal context) is not optional — the policy literally cannot solve the task without it. Test early.

---

## Regularization Penalties Can Suppress Necessary Corrective Actions (2026-04-03)

**What happened:** Added joint velocity cost (-1.0) and bumped torque cost (-0.5 → -1.0) to reduce jerky behavior. PPO peaked at 15.9 then regressed to ~11 over 100M steps. Previous run without these penalties reached 23.7.

**Root cause:** Balancing on a bongo board requires aggressive corrective movements. The velocity/torque penalties taught the policy "don't move much" which directly conflicts with "stay balanced." The policy found a local optimum of minimal movement that delayed falling by ~2s but couldn't sustain balance.

**Nuance:** The penalties were kept for the frame-stacking run (PPO4) which hit 46.9 — suggesting the penalties aren't fatal when the policy has enough temporal context to plan corrections efficiently. Frame stacking let the policy make smaller, better-timed corrections instead of large reactive ones.

**Lesson:** Don't add regularization penalties to hard tasks until the policy can solve the base task. They can prevent learning entirely if the optimal strategy requires exactly what you're penalizing.

---

## PPO Entropy Collapse Signals Premature Commitment (2026-04-03)

**What happened:** PPO entropy dropped from 7.8 → -0.89 over 780 iterations (25M steps). All 512 envs produced identical returns (train return std ≈ 0). Policy locked into one strategy and polished it rather than exploring alternatives.

**Root cause:** Default PPO entropy coefficient (0.01) provides only a weak exploration incentive. For hard tasks where the initial random policy dies quickly, the first strategy that survives a few steps gets reinforced heavily. With near-deterministic actions, every env plays out identically — no diversity to discover better strategies.

**Implication:** Low return variance in PPO training is not a sign of stability — it's a sign of dead exploration. Healthy training should show spread as different envs stumble into different strategies.

**Lesson:** Monitor entropy and return variance together. If both collapse early and returns are below the theoretical max, consider higher entropy coefficient, entropy annealing schedule, or population-based approaches.
