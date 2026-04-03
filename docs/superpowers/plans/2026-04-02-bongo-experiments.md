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

### 6. Target entropy -6
Current `target_entropy=0` makes policy nearly deterministic at convergence.
Original SAC uses `-dim(A)=-12`. Try `-6` as middle ground — more exploration
without going full noisy. Could break out of local optimum.

Change: `--target-entropy -6` CLI flag (if exists) or algo config override.

## Run plan

**Run A (quick wins):** Remove pushes + full action scale + negative rewards. ← RUNNING
- action_scale=1.0, push_interval=99999, clip(-10000, 10000)

**Run B (exploration):** Run A + target_entropy=-6.

**Run C (reward redesign):** Cost-based reward from #3. Compare to Run A/B.

**Run D (robustness):** Best of above + more init randomization + pushes back on.

## Results

| Run | Config | Best eval | Steps | Notes |
|-----|--------|-----------|-------|-------|
| v1 (no floor term) | old rewards, scale=0.5, pushes, ep=1000 | 397 | 20M | CHEATING — ground balance |
| v2 (floor term) | old rewards, scale=0.5, pushes, ep=1000 | 71 | 20M | Honest, plateaued |
| v3 (resumed) | same as v2 | 89 | 50M | Marginal improvement |
| v4 (tuned) | survival=5, scale=0.3, pushes, ep=1000 | 65 | 20M | Plateaued ~50-70 |
| A | scale=1.0, no pushes, neg rewards, ep=250 | 119 | 30M | Plateau 80-100 |
| B2 | A + entropy=-6 + arm termination | 101 | 36M | Harder w/ arm term, OOM crash |
| C | B2 + cost-based rewards | 28 | 25M | Normalized quadratic costs, plateaued 15-28 |
| D (abandoned) | C + wider init rand | 2.6 | 100k | Init rand too aggressive, instant deaths |
| PPO-C | PPO + config C, ep=250 | running | 50M | On-policy, asymmetric critic |
