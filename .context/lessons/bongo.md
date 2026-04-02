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
