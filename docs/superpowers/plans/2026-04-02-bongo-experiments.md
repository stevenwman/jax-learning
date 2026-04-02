# Bongo Handstand — Experiment List

**Context:** FastSAC plateauing at eval 50-70 (out of ~600 theoretical max).
Robot survives ~2s then falls. Need to diagnose whether it's reward shaping,
action authority, exploration, or the task itself.

## Experiments (priority order)

### 1. Remove pushes
Velocity kicks every 200 steps kill the robot before it learns basic balance.
Disable during initial training. Add back after policy converges.

Change: `push_interval=99999` (effectively off)

### 2. Full config export at training start
Currently guessing what episode_length, action_scale, etc. actually ran.
Print the full env config + algo config + train config at training start.

### 3. Cost-based reward (replace exp-based)
Current exp rewards saturate — almost no gradient between "kinda balanced" and
"perfectly balanced." Switch to quadratic costs with a survival ceiling.

**New reward structure:**
```
per_step = survival_bonus - w1*(gravity_error²) - w2*(board_tilt²)
         - w3*(com_offset²) - w4*(height_error²) - w5*(roller_pos²)
         - w6*(torques) - w7*(action_rate²)
```

**Key design:** normalize each cost by its max expected value so they're
all [0, 1] before weighting. Then weights = relative importance.

| Term | Raw range | Normalization | Weight |
|------|-----------|---------------|--------|
| survival | constant 1.0 | — | 10.0 |
| gravity_error² | 0 to 4 | /4 | 8.0 |
| board_tilt² | 0 to 0.5 | /0.5 | 6.0 |
| com_offset² | 0 to 0.1 | /0.1 | 5.0 |
| height_error² | 0 to 0.1 | /0.1 | 3.0 |
| roller_pos² | 0 to 0.05 | /0.05 | 2.0 |
| torques | varies | /max_torque_norm | 0.5 |
| action_rate² | 0 to ~1 | /1 | 0.5 |

Per step (perfect): 10 * 0.02 = 0.2
Per step (bad but alive): (10 - 8 - 6 - 5) * 0.02 = -0.18 → clipped to 0
Episode max (250 steps): 50

### 4. Full action scale
`action_scale=1.0` — let the policy use the full joint range. Currently 0.3
limits deviations to ±0.3 rad. The robot might need larger corrections on
the rocking board.

### 5. More init randomization
Increase reset perturbation ranges to force robustness from the start:
- Joint angles: ±0.05 → ±0.1 rad
- Base xy: ±2cm → ±5cm
- Yaw: ±5deg → ±10deg
- Board tilt: ±2deg → ±5deg

## Run plan

**Run A (quick wins):** Remove pushes + full action scale. Everything else same.
See if the ceiling lifts.

**Run B (reward redesign):** Cost-based reward from #3. Compare to Run A.

**Run C (full):** Best of A/B + init randomization + pushes back on.
