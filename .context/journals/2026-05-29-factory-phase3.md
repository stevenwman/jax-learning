# Factory Port — Phase 3 (SAC training to assembly behavior)

Date: 2026-05-29
Branch: `factory-peg-insert` (worktree at `.worktrees/factory-peg-insert/`)
Plan: `.superpowers/plans/2026-05-27-factory-peg-insert-mvp.md`
Spec: `.superpowers/specs/2026-05-27-factory-mjx-warp-port.md`
Prior journals:
  - `2026-05-27-factory-phase0.md` (asset extract + Warp drop spike)
  - `2026-05-28-factory-phase2.md` (OSC + Visual Gate 2)

## Result

Phase 3 gate (≥0.5 reward/step at 5M timesteps) **PASSED**.

Best checkpoint: `checkpoints/20260529_174438_sac_factorypeginsert_seed11`
(v11 in the iteration ledger below). 5M SAC steps in ~10 min on the RTX
5080. Deterministic deploy:

  - Return / episode:  **1236**  (gate threshold 450)
  - Per-step reward:   1.37 (theoretical max ≈ 4.25)
  - Engaged steps:     **779 / 900** (peg is over the bolt and below
    bore_top for 87 % of the episode)
  - Success steps:     0 / 900 (peg never quite hits the 2.5 mm
    physical-insertion gate, but it does honest tracking inside the
    1 cm engaged gate)

MP4: `.tmp/recordings/factory_phase3_solved.mp4`.

## What it took (v1 → v13 forensics)

Phase 3 ran thirteen 5M trains. Each iteration is in git as a separate
commit; this is the postmortem.

### v1 (commit `0d3d062`) — collapsed at 45/episode

Bad keypoint coefs `(100, 2)/(500, 2)/(1500, 0)` from the plan spec
made the baseline reward near-zero at init. SAC sat. Fixed in v3.

### v2 — same collapse, target_entropy_scale=2.0

Misdiagnosed the v1 collapse as alpha decay. Alpha stayed higher but
the policy still collapsed at 45/episode. Killed early.

### v3 (commit `01f7004`) — correct Isaac coefs `(5, 4)/(50, 2)/(100, 0)`

Baseline reward at init went from 0.001 → 0.164. Still collapsed, at
the new baseline equilibrium of 191/episode.

### v4 — target_entropy_scale=0.0, same collapse

Alpha controlled to keep entropy ≥ 0, still didn't escape the baseline
shaping plateau.

### v5 — ema_factor 1.0 (direct action), same 191 collapse

Suspected EMA dampening. Wrong diagnosis.

### Root cause discovered (commit `2594fc7`) — actuator_mode default

**The big one.** `default_config().actuator_mode` was still
`"position_pd"` (the Phase 1 placeholder controller). In that branch
of `env.step`, `ctrl[:7]` is hard-coded to `DEFAULT_ARM_QPOS` and the
OSC torque is discarded. All five SAC runs (v1..v5) trained against
an env where the policy's actions had ZERO effect on dynamics.
Return 45/191 was just the baseline shaping for the IK-resolved init
pose with the arm frozen.

Diagnostic: replayed v5's saved actions through `env.step` with
`actuator_mode="motor"` — peg dropped from z=0.116 to z=-0.052 (peg
below the hole). Same actions, opposite outcome.

Fix: one-line default flip in `default_config`. Tests updated
(`test_default_actuator_mode_is_motor`).

### v6 (commit `2594fc7`) — Return 1810 but EXPLOITED

First run with actual OSC. SAC learned to drive the peg to the corner
of the action cube (0.65, -0.05, ~0) — 7 cm xy offset from the bolt —
and bumped its z low. Engaged + success bonuses fired purely on z
because `is_engaged`/`is_success` had no xy gate. Free 2.0/step ×
700 steps = 1810/episode for a peg that never went near the hole.

### v7 (commit `6e89f24`) — xy gate added (2.5 mm) → 258 collapse

Added the `is_centered` xy proximity check that IsaacLab uses. v6's
exploit closed but SAC now has no reachable bonus zone (5 cm action
cube can't be navigated to 2.5 mm). Policy stayed at baseline 258.

### v8 (commit `4840890`) — tight bounds + loose engaged gate → 226

Shrank `pos_action_bounds.xy` to 2 cm; loosened engaged xy threshold
to 1 cm. Peg still parked 3 cm offset because hand tilt swings the
welded peg an extra cm in xy. Bonus didn't fire.

### v9 (commit `81cd123`) — engaged xy 3 cm → 256

Loosened engaged gate further. Engaged still never fired (peg landed
just outside the 3 cm radius).

### v10 (commit `0d3d062`) — explicit r_xy_proximity term → 360

Added a fourth shaping term. Reward landscape improved at the
baseline (0.27/step vs 0.20/step before), but the gradient toward
the bonus zone still wasn't strong enough.

### v11 (commit `d3ee0cd`) — gated z-descent reward → **1236, gate PASSED**

This was the breakthrough. Added a fifth term:
```
r_aligned_descent = z_progress * (2 * squashing(xy_dist, 80, 0))
```
Multiplicative xy-gate × z-progress. Descent now earns no reward when
peg is off-axis. Forces the correct ordering: align xy first, then
descend. Engaged fired 779/900 steps, peg honestly tracked the bonus
zone for 87 % of the episode.

### v12 / v13 — redesigned reward, regressed

Tried a "clean 3-term" rewrite (xy attractor + gated descent + success
bonus; dropped the keypoint bells, engaged bonus, and xy proximity
term). Training mean ballooned (v13 hit 3340 train mean) but
deterministic deploy collapsed back to 102/episode. The running
training mean reflected exploration noise, not the converged policy.

Diagnosis: the v11 design's engaged bonus acts as a wide basin-of-
attraction anchor. Without it, SAC's deterministic policy has nothing
to hold it inside the gradient-rich region; nearby modes pull it
away.

Reverted to v11's reward in commit `19e7847`. v11 stands as the
Phase 3 close.

## Other fixes that landed during Phase 3

- `scripts/record_video.py`:
  - Scan-based fast path for MJX backends. 900-step single-env
    recording dropped from "several minutes" to ~12 s by replacing the
    Python loop with a `jax.lax.scan` that emits qpos/qvel/action/
    reward arrays and does ONE device→host transfer. Commit `b1b877a`.
  - Per-env free-camera defaults via `ENV_FREE_CAM_DEFAULTS`. Factory
    PegInsert ships with the Phase 2 view (lookat 0.55/0/0.15,
    distance 0.95, az 115, el -18). No more `--cam-distance` per
    invocation. Commit `0d3d062`.
  - `--stochastic` flag (sample from policy distribution instead of
    the mean) for diagnosing multi-modal policies. Commit `19e7847`.

- Scene lighting dimmed (scene.xml). Earlier `headlight ambient=0.6`
  + `scene_key diffuse=1.2` produced washed-out frames. Halved values
  give a clean exposure with the Phase 2 cam. Commit `1950f9c`.

- Reward bug: `is_engaged`/`is_success` checked only z. Added the xy
  gate (commit `6e89f24`). Without this v6's policy exploited the
  z-only bonus by parking the peg 7 cm offset and dropping it past
  the bore.

- `info["clip_anchor"]` separated from `info["fixed_pos"]`. The action
  chain now clips against the bore opening top (matching IsaacLab's
  `fixed_pos_obs_frame`); the reward keeps using the hole body origin
  as target. Earlier conflation silently broke engaged/success math
  for two iterations. Commit `01f7004`.

## Open items / known limitations

- **Deterministic policy doesn't hit the 2.5 mm success gate.**
  Engaged fires reliably (779/900), success doesn't (0/900). Final
  insertion precision needs either longer training (15M+), behavior
  cloning from scripted insertion demos, or a curriculum that tightens
  the xy threshold progressively. Out of Phase 3 scope.
- **DR wrappers not applied at reset.** The DR spec exists in
  `get_domain_randomization_spec` but `reset()` uses deterministic
  nominal values. Phase 3 trained on a single configuration; turning
  DR on may shift the converged policy.
- **target_quat is fixed for the whole episode.** Phase 5 (NutThread)
  needs rotation actions wired through `denormalize` and into the OSC
  target_quat.
- **v12/v13 lesson: deterministic policy can diverge from training
  mean.** Multi-modal SAC + small alpha. Worth documenting in
  `.context/LESSONS.md`.

## Where to look

| Thing | Path |
|---|---|
| Trained checkpoint | `checkpoints/20260529_174438_sac_factorypeginsert_seed11` |
| Final policy MP4 | `.tmp/recordings/factory_phase3_solved.mp4` |
| Training logs | `.tmp/logs/factory_full_5m_v*.log` |
| Reward implementation | `jax_rl/envs/manipulation/factory/reward.py` (v11 design) |
| Env config | `jax_rl/envs/manipulation/factory/factory_peg_insert.py:default_config` |
| Wandb run (v11) | `wandb/run-*sac_factorypeginsert_seed11*` |
