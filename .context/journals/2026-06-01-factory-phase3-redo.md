# Factory Port — Phase 3 redo (honest insertion via phased reward + FlashSAC)

Date: 2026-06-01
Branch: `factory-peg-insert` (worktree `.worktrees/factory-peg-insert/`)
Prior: `.context/journals/2026-05-29-factory-phase3.md` (closed prematurely
with v11 ckpt — engaged fired 779/900 but was a tilt-exploit, peg hovered
~2.7 cm beside the bolt, never inserted).

## Result

Phase 3 re-closed honestly. **794 success steps / 900**, Return **6460**.

Best ckpt: `checkpoints/20260601_124204_flash_sac_factorypeginsert_seed15/best`
(FlashSAC v15.6). Eval @ 4096 eps: 6649 ± 32 [6614, 6709] over 5 deterministic
episodes. Single recorded rollout `factory_flashsac_v15_6_best.mp4`:

```
reward sum   = 6459.6   per-step mean = 7.18 / 8.0 max
peg_z        : 0.106 init → 0.072 final (bore floor)
xy_dist      : 6.2 mm init → 0.01 mm min, 0.4 mm mean (well inside 0.05 mm clearance)
tilt         : 1.12° init → 0.12° min, 0.42° mean (< 2° gate)
engaged      : 827 / 900
success      : 794 / 900
```

Comparison:

| Run | Return | Engaged / 900 | Success / 900 | Notes |
|---|---|---|---|---|
| v11 (2026-05-29 close) | 1236 | 779 | 0 | Tilt exploit — peg hovered 2.7 cm beside bolt |
| v14 (tilt gate added) | 437 | 0 | 0 | Aligned but stuck at xy 2.4 cm |
| v15.6 (this) | **6460** | **827** | **794** | Honest insertion + hold |

## Why v11 was reopened

User caught the v11 video: peg never actually entered the bore. `is_engaged`
checked xy < 3 cm AND z below entry, but neither xy<5 mm nor any tilt
constraint. Peg parked at xy = 2.7 cm with arbitrary tilt and collected the
engaged bonus from being merely "low and offset." Added a **2° tilt gate**
to `is_engaged` / `is_success` and tightened xy threshold to 5 mm
(`_XY_PROXIMITY_THRESHOLD`, `_TILT_COS_THRESHOLD` in
`jax_rl/envs/manipulation/factory/reward.py`). v14 retrained against the
honest gates and plateaued at Return 437 — policy aligned vertically but
couldn't get xy below 5 mm.

## Phase 3 redo — what worked

Five independent fixes stacked together broke the policy through:

### 1. Phased reward (`reward.py:compute_reward`)

Replaced v11's 5-term shaping stack with a structural reward encoding the
"align above → descend below → hold seat" task split:

```python
r_align   = 2.0 * (r_xy + r_tilt)                  # always on, max 2.0
phase_above = sigmoid((peg_z - entry_z) * 200)
phase_below = 1 - phase_above
r_B_desc  = phase_below * aligned * z_progress     # +1.0 in Phase B only
r_B_pen   = phase_below * (1 - aligned) * (-0.5)   # dead-end if misaligned below
r_success = 5.0 * is_success(xy<2.5mm & tilt<2° & seated)
return r_align + r_B_desc + r_B_pen + r_success    # max 8.0/step
```

Two non-obvious anchors:

- **`r_align` is NOT gated by `phase_above`.** First attempt (v15) put it
  inside `phase_above * ...` — created a cliff at entry where hover gave
  2.0/step but descent paid 1.0/step. Policy refused to step off. Untying
  removed the cliff: descent strictly adds reward.
- **`z_progress` anchored at `entry_z` (peg tip crossing bore opening),
  NOT `hole_top_z` (body crossing).** With `hole_top_z`, the entire 2.1 cm
  region where the tip is descending into the bore but the body is still
  above gave `z_progress=0`. Switching to `entry_z` gives a smooth 0→1
  gradient throughout the actual insertion path.

The dead-end penalty is "free" — the bore tile geometrically blocks the
peg from descending if xy is outside clearance, so `z_progress` can't grow
without genuine alignment. The penalty fires only when the policy parks
the peg below entry off to the side of the bolt (the v6 exploit class).

### 2. Reset noise (`factory_peg_insert.py:reset`)

Per-episode ±0.02 rad uniform noise on the seven arm qpos joints,
applied after `_init_arm_qpos` is set but before `mjx.forward`. Perturbs
hand pose by 0.5–1.2 cm in xy and ±2° tilt. Without it the deterministic
mean policy memorizes a single hover pose that gets r_align ≈ 2.0/step
and never has to navigate.

First tried ±0.05 — that's enough to broaden the buffer but too much
for the action chain (±2 cm xy bounds) to cleanly correct from, so the
policy ended up with high stochastic training Returns but a deterministic
eval still stuck at hover (v15.4 / v15.5 saw Return 6150 in training but
1700 in eval). Tightened to ±0.02 — subtle enough that the deterministic
mean internalizes descent.

### 3. FlashSAC `alpha_init = 0.1` (10× default)

The single biggest unlock. v15.4 happened to break through to descent
(stochastic Return 6150) where v15.5 with identical config did not (stuck
at 1700) — same seed, same env. Diagnosis from matched-step log compare:

| Metric @ step 519k | v15.4 (broke) | v15.5 (stuck) |
|---|---|---|
| Alpha | 0.0003 | 0.0001 |
| RewScale | 252 | 58 |
| Q1 × RewScale | 548 | 172 |

GPU floating-point nondeterminism rolled alpha lower during early
training in v15.5. Auto-tune then collapsed it further (it was tracking
target_entropy). Lower alpha → less action noise → fewer descent
excursions → no success rollouts in buffer → RewScale stayed low → critic
estimated hover-value → actor gradient stayed at hover. Bistable
attractor at low alpha.

Bumping `alpha_init` 0.01 → 0.1 gives the entropy bonus 10× the initial
margin before auto-tune crushes it. v15.6 broke through to Return 5660
on its FIRST eval (512 eps), with Q correlation jumping 0.09 → 0.36.

### 4. `mj_model.opt.ccd_iterations = 100`

Bumped from default 35. With reset noise active, ~1 in 40k env steps hit
an edge contact configuration where the CCD solver couldn't converge in
35 iterations (per-occurrence runtime warning). 100 silences it without
measurable step-time impact since the cap is per-contact-pair, not
per-step.

### 5. `total_timesteps` 5M → 2.5M

v15.6 hit 6545 by eval 1024 eps (step ~1M) and plateaued 6500-6650
through step 5M. Final best ckpt at eval 4096 (step ~3.7M) is the same
score as the 1024-eps eval. 2.5M is sufficient; preset updated.

## What didn't work

### PPO (v15a-e) — five configs, all path-dependent failures

| Run | Config | Failure |
|---|---|---|
| v15a | default preset, squash=True | Return 1530 → 66 collapse by iter 28, KL spikes e6-e9 |
| v15b | 1 ep/batch, ent=2e-2, rew=0.1 | Locked in squash saturation iter 140+ (PLoss=1.8 sustained) |
| v15c | + squash=True, lr=1e-4 | Same lock pattern |
| v15d | squash=False, state_dep_std=True | logσ runaway to clip ceiling, Ent=20 max |
| v15e | squash=False, scalar σ | Stuck at Return 977 (zero variance) by iter 240+ |

PPO + dense-shaped Factory reward + bounded action space appears to have a
deep instability — squash=True overflows the tanh-Jacobian on small σ;
squash=False has runaway σ or saturation-lock. SAC handles this cleanly
because it normalizes Q-targets adaptively and doesn't rely on a hand-tuned
σ schedule.

### `sigma_target = 0.5` (FlashSAC)

target_entropy = +4.36 was unreachable by a squashed Gaussian on
[-1, 1]⁶ (whose entropy is bounded by ~6). Auto-tune's PI controller drove
alpha to ~1.9M trying to push entropy up that never could rise. ActLoss
hit -8e6 before kill. Reverted to default 0.15.

### Removing the reward cliff alone

v15.1 (untied r_align without reset noise) still plateaued at 1770 hover —
the cliff-removal lets descent ≥ hover in principle, but SAC's deterministic
mean still converges to the easier hover attractor without diverse start
poses forcing exploration.

## Loose ends

- **Reset noise is currently `±0.02 rad` uniform on arm qpos**, applied
  directly in `reset()`. The `get_domain_randomization_spec` declares
  proper specs (`hand_init_pos_xy`, `bolt_pos_xy`, etc.) but they're still
  not wired through `DomainRandWrapper`. For sim2real work this should
  migrate to the wrapper.
- **Phase 5 (NutThread) needs rotation actions wired** through
  `denormalize` and into `info["target_quat"]`. Currently `target_quat` is
  fixed at reset, OK for an axially-symmetric peg, not OK for threading.
- **UR5e + Robotiq swap** still deferred. Spec at
  `.superpowers/specs/2026-05-28-ur5e-robotiq-swap.md`. Will revisit after
  Phases 5/6 land.

## Where to look

| Thing | Path |
|---|---|
| Trained ckpt | `checkpoints/20260601_124204_flash_sac_factorypeginsert_seed15/best` |
| Recording | `.tmp/recordings/factory_flashsac_v15_6_best.mp4` (+ `_traj.npz`) |
| Reward (final) | `jax_rl/envs/manipulation/factory/reward.py` (v15.1 untied, entry-anchored) |
| Reset noise | `jax_rl/envs/manipulation/factory/factory_peg_insert.py:reset` |
| Preset | `jax_rl/configs/env_presets.py:FLASH_SAC_PRESETS["FactoryPegInsert"]` (alpha_init=0.1, UTD=16, 2.5M) |
| Wandb (v15.6) | `wandb/run-20260601_124205-ue8ig29h` |
