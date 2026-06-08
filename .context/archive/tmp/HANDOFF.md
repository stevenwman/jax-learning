# Handoff — Factory PegInsert, Phase 3 reward redesign in progress

Date: 2026-06-01
Branch: `factory-peg-insert` (worktree at
`/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/factory-peg-insert/`)
Last commit: `1254666 fix(factory): engaged/success gates require tilt AND xy AND z (Phase 3 v14)`

## Goal

Port IsaacLab Factory PegInsert to MuJoCo Warp, train SAC end-to-end
until peg actually inserts. The task: 8mm capsule peg welded to a Franka
Panda gripper, descend into an 8.1mm bore with 0.05mm radial clearance
and 25mm depth. Phase gate is "≥0.5 reward/step at 5M timesteps with the
**deterministic-deploy** policy doing **honest physical insertion**."

Phases 0/1/2 are closed. Phase 3 (SAC training) is the open work.

## Current state

### What works
- Env, OSC controller, IK reset, weld coherence, render pipeline — all
  correct. 41 hermetic tests pass; gpu/warp tests pass.
- Training runs end-to-end (`uv run python scripts/train_sac.py --env
  FactoryPegInsert --total-timesteps 5000000 --wandb`). 5M takes ~10 min
  on the RTX 5080.
- Single-env recording (`scripts/record_video.py`) takes ~12 s with the
  scan fast path; per-env free-camera defaults bake the Phase 2 view
  for FactoryPegInsert.
- Geometry verified — bore opening at world z=0.075, bore tile centroid
  at (0.6, 0), tile inner radius 4.05mm.

### What doesn't work yet
- Phase 3's deterministic-deploy policy does **not** physically insert
  the peg. The latest checkpoint (v14, seed 14) keeps the peg vertical
  (tilt 1.77° avg) and descends, but parks at xy_dist ≈ 2.4 cm above
  the bore — bolt sidewall blocks further xy correction once z is low.

### v14 deterministic deploy numbers (the "honest" baseline)
```
peg xy_dist : mean=0.024  max=0.031   (need <0.005 for engaged)
peg z       : 0.116 → 0.054           (above bore floor 0.048)
tilt        : mean=1.77°  max=2.15°   (mostly within 2° gate)
total reward: 437.6 / 900 steps        (gate 450)
Engaged     : 0 / 900   Success: 0 / 900
```

## Files we're actively editing

| Path | Status |
|---|---|
| `jax_rl/envs/manipulation/factory/reward.py` | Under redesign. v14 has the multi-term v11 design + a tilt gate. **Next step rewrites this file** (see plan below). |
| `jax_rl/envs/manipulation/factory/factory_peg_insert.py` | Config knobs being added to keep up with reward changes. Just bumped `engage_xy_threshold` 0.03 → 0.005. Watch the `default_config()` block. |
| `tests/manipulation/factory/test_factory_reward.py` | Track every reward signature change; `test_reward_per_step_max_bound` is the canary. |
| `scripts/record_video.py` | Stable since `19e7847` (scan fast path, free-cam defaults, `--stochastic` flag). |

Don't touch these without good reason:
- `controller/osc.py` (Phase 2 OSC — stable, tested)
- `controller/action_chain.py` (Phase 1 EMA/denorm/clip)
- `assets/peg_insert/scene.xml` (lighting + weld + solver settings tuned)

## Reward design — the saga we lived through

13 5M training runs to get to v14. Notable wreckage:

| run | reward change | deterministic deploy | what we learned |
|---|---|---|---|
| v1-v5 | various tweaks | Return 45-191, collapse | **Root cause: `actuator_mode="position_pd"` default disabled the OSC.** Policy actions had zero effect. Fixed in v6 via default flip to `"motor"`. |
| v6 | first OSC engagement | Return 1810 **EXPLOITED** | `is_engaged`/`is_success` checked only z. Policy parked peg 7cm off-axis, dropped through air past the bolt for free +2/step. Added xy gate to bonuses. |
| v7-v9 | xy gate tuning (2.5mm → 1cm → 3cm) | 226-360 | None small enough to be honest. Loosening the gate alone doesn't help. |
| v10 | add `r_xy_proximity` term | 360 | Helped the baseline but no insertion. |
| v11 | add `r_aligned_descent = z_progress * xy_gate` | Return **1236**, engaged **779/900** | Looked like a win — celebrated and merged. **Was actually a dishonest exploit**: peg hovered at xy=2.8cm, tilt unknown, z=0.035 — beside the bolt. The 3cm engaged threshold was too loose; bonus fired without physical engagement. |
| v12 | full rewrite to 3 clean terms (xy attractor + sigmoid-gated descent + success) | training mean 815, **deploy 130** | Razor-sharp sigmoid xy gate left a flat-zero plateau outside 5mm. Policy drifted to corner with no gradient back. |
| v13 | widen sigmoid → soft squashing gate | training mean 3340, **deploy 102** | Training-mean inflated by exploration noise; deterministic policy collapsed to a different attractor than the training distribution. Reverted to v11's reward in `19e7847`. |
| v14 | v11 reward + tilt gate (2° cone) + xy 5mm | Return 394, **honest** | First honest result. Policy aligns (tilt mean 1.77°), descends, but can't get xy below 5mm so neither bonus fires. |

Tilt gate math (in `reward.py:_peg_aligned`):
```python
# peg z-axis in world (closed-form from quat (w,x,y,z))
z_axis_world = [2(xz+wy), 2(yz-wx), 1−2(x²+y²)]
# gripper-down convention: -world_z·peg_z_axis > cos(2°)
aligned = -z_axis_world[2] > 0.9994
```

## Why v14 plateaus

Action cube is `pos_action_bounds = (0.02, 0.02, 0.10)`: policy can only
command target_pos ±2cm from the bore opening top in xy. Hand-tilt under
OSC dynamics swings the welded peg an extra ~5mm in xy. So peg can
physically reach xy_dist ≈ 2.5cm but the 5mm engaged gate is a sub-cube
SAC has to learn to hold inside via fine torque control.

The reward shaping at xy=2.4cm gives ~0.5/step (xy_attractor + tilt
attractor + partial aligned_descent + keypoint shape terms). At xy=5mm
the reward would be ~1.5/step (engaged fires). At xy<2.5mm + seated z,
peak ~5+/step (engaged + success + max shape).

SAC needs a clean gradient pulling it from the 2.4cm attractor to the
5mm zone. Current reward has competing terms (3 keypoint bells +
xy_proximity + aligned_descent + 2 binary bonuses) which collectively
give a flat-ish landscape near 2.4cm.

## Next step — proposed phased reward rewrite (NOT YET APPLIED)

Idea is to encode the task structure into the reward:

1. **Above bore opening** (`peg_z > 0.096` = peg tip above z=0.075):
   peg has full xy freedom. Reward ONLY xy + tilt alignment. No descent
   incentive — we don't want SAC to dive while still off-axis.
2. **Below entry** (`peg_z ≤ 0.096`): peg geometrically constrained.
   Reward descent **only when xy + tilt aligned**; penalize low-z without
   alignment (peg jammed on bolt side = bad equilibrium).

Pseudocode for the new `compute_reward` (replaces the v11 multi-term
sum):

```python
ENTRY_Z   = 0.096            # peg body z at which tip enters bore
ALIGN_XY  = 0.005            # bore inner clearance
TILT_OK   = 0.9994           # cos(2°)

# Soft phase factor: ~1 above entry, ~0 below
phase_above = jax.nn.sigmoid((peg_z - ENTRY_Z) * 200.0)

# Always-on alignment attractors
r_xy   = squashing_fn(xy_dist, 100.0, 0.0)         # max 0.5 at xy=0
r_tilt = squashing_fn(tilt_rad, 20.0,  0.0)        # max 0.5 at tilt=0

# Phase A — above the bore entry: don't reward descent, only alignment
r_A = phase_above * 2.0 * (r_xy + r_tilt)          # max 2.0

# Phase B — below entry: descent ONLY when aligned, penalty otherwise
xy_ok    = jax.nn.sigmoid((ALIGN_XY - xy_dist) * 500.0)
tilt_ok  = jax.nn.sigmoid((peg_z_axis_world_dot_neg_world_z - TILT_OK) * 500.0)
aligned  = xy_ok * tilt_ok
r_B_desc = (1.0 - phase_above) * aligned * z_progress       # max 1.0
r_B_pen  = (1.0 - phase_above) * (1.0 - aligned) * (-0.5)   # negative if dead-end

# Terminal bonus
r_succ   = 5.0 * is_success(... xy<2.5mm AND tilt<2° AND z_seated)

return r_A + r_B_desc + r_B_pen + r_succ
```

Per-step bounds:
- Init (peg_z=0.116, xy=0, tilt=0): r_A ≈ 2.0 max, others 0 → r ≈ 2.0
- Above entry + half-aligned: r_A ≈ 1.0, others 0 → r ≈ 1.0
- Below entry + aligned + seated: r_B_desc ≈ 1.0, r_succ = 5.0 → r ≈ 6.0
- Below entry + misaligned: r_B_pen ≈ -0.5 → r ≈ -0.5

User approved the geometry and the phased structure but we haven't
written the code yet. **Start by reading `reward.py` end-to-end**, then
replace `compute_reward`'s shape stack with the phased version above.
Keep `is_engaged`/`is_success` as-is (with the v14 tilt gate). Update
`test_reward_per_step_max_bound` to the new max (~6.0). Then retrain
seed 15.

## Other open items

- **`v14 ckpt deterministic-stochastic discrepancy`** — v12/v13 showed
  training-mean returns (3340) that the deterministic deploy couldn't
  reproduce (102). v14 doesn't have this issue but watch for it after
  the next reward rewrite. The `--stochastic` flag on record_video is
  for diagnosing this.
- **DR wrappers** not applied at reset. Adding per-episode hand /
  bolt-pose noise will probably stabilize the converged policy but is
  Phase 4 scope.
- **Action cube xy = 2cm + hand-tilt swing = ~2.5cm peg reach.** If the
  new reward can't drag SAC inside that 5mm engaged zone, consider
  tightening pos_action_bounds further (e.g. 1cm) so the policy's
  reachable workspace IS the alignable zone.
- **Engagement bonus may never fire** with vanilla SAC at 5M even with
  the phased reward. Consider curriculum (start engaged_xy_threshold
  loose, tighten over training).

## How to verify

After the reward rewrite, expected flow:

```bash
# 1. Hermetic tests still pass
JAX_PLATFORMS=cpu uv run pytest tests/manipulation/factory/ \
    -k "not gpu and not warp" --no-header -q

# 2. Reward diag — see what the trained policy collects per phase
uv run python scripts/factory/diag_reward.py

# 3. Train 5M (~10 min)
nohup uv run python scripts/train_sac.py --env FactoryPegInsert \
    --total-timesteps 5000000 --seed 15 --wandb \
    > .tmp/logs/factory_full_5m_v15_phased.log 2>&1 & disown

# 4. Record + analyze
CKPT=$(ls -td checkpoints/*sac_factorypeginsert_seed15 | head -1)
MUJOCO_GL=egl uv run python scripts/record_video.py \
    --env FactoryPegInsert --checkpoint "$CKPT" --max-steps 900 \
    --out .tmp/recordings/factory_v15.mp4

# 5. Check whether peg actually inserts (xy_dist, tilt, peg_z, engaged/success counts)
uv run python -c "
import numpy as np
d = np.load('.tmp/recordings/factory_v15_traj.npz')
q = d['qpos']
xy = np.linalg.norm(q[:, 9:11] - [0.6, 0], axis=1)
z = q[:, 11]
tilt_align = 1 - 2 * (q[:, 13]**2 + q[:, 14]**2)
tilt = np.degrees(np.arccos(np.clip(-tilt_align, -1, 1)))
print(f'xy mean={xy.mean():.4f}  min={xy.min():.5f}')
print(f'z range [{z.min():.4f}, {z.max():.4f}]')
print(f'tilt mean={tilt.mean():.2f}°')
print(f'Engaged steps: {((xy<0.005) & (tilt<2) & (z<0.075-0.0225)).sum()}/900')
print(f'Success steps: {((xy<0.0025) & (tilt<2) & (z<0.075-0.001)).sum()}/900')
"
```

Phase 3 closes honestly when:
- Engaged steps > ~50/900 (peg actually enters bore)
- Success steps > 0/900 (peg seats fully)
- Deterministic deploy reproduces the training-mean return
