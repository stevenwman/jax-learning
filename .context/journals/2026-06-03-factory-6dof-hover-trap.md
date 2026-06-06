# Factory Port — 6-DOF action wiring + hover trap diagnosis

Date: 2026-06-03
Branch: `factory-peg-insert` (worktree `.worktrees/factory-peg-insert/`)
Prior: `.context/journals/2026-06-01-factory-phase3-redo.md` (v15.6 closed Phase 3 at Return 6545).

## Result

**Hover trap broken via altitude-weighted r_align.**

| Run | Setup | Return | Per-step |
|---|---|---|---|
| v15.6 baseline | 3-DOF, no DR, ep=900 | 6545 | 7.27/step |
| v1–v10 | 6-DOF + various probes, ep=450 | 700–890 | 1.5–2.0/step (pure hover) |
| **v11** | **6-DOF + altitude r_align, no DR, ep=450** | **3120** [3004, 3205] | **6.93/step** |

v11 ckpt: `checkpoints/20260603_182811_flash_sac_factorypeginsert_seed0/best`. Trajectory min peg_z=0.0715 (full insertion, plate top=0.075). Recording: `.tmp/recordings/peginsert_v11_altitude_6dof.mp4`.

## What was happening

Two compounding regressions since the v15.6 baseline:
1. **6-DOF wiring** (this session) — added rotation actions through `denormalize` and into `info["target_quat"]` via accumulating `quat_mul`, deferred from Phase 5 NutThread (per spec).
2. **Hole-pose DR** (added in commit `7cf631b`, 2026-06-01) — ±2cm xy, ±1cm z mocap noise per episode.

All initial smoke trains (v1–v10) hit a tight plateau at Return ≈ 880 [≈ 2.0/step]. Every probe failed:
- σ_target sweep 0.15 → 0.19 → 0.30: same plateau
- DR off, 6-DOF on: same plateau
- rot_threshold 0.097 → 0.01: same plateau
- Reward gating attempts (approach-gate, 5× r_B_desc, widened aligned gate): same plateau
- 2M steps (4× the v15.6-equivalent budget): same plateau

## Root cause

`r_align = 2.0 * (r_xy + r_tilt)` was **z-invariant**. Hovering at any altitude with good xy + tilt paid 2.0/step. Critic Q-surface was flat in z — no gradient pulling policy down.

Two failure modes the v15.6 reward depended on but only 3-DOF could exploit:

1. **3-DOF accidentally worked** because pos-only exploration occasionally hit (aligned ∧ low_z) — the rare conjunction that fires `r_B_desc`. Once observed, critic learned descent paid → policy descended.

2. **6-DOF broke this conjunction.** Rotation exploration (any std > 0) flickers `aligned_tilt` off most of the time. The (aligned ∧ low_z) sample pair almost never appears in the replay buffer → critic stays Q-flat in z → policy mean drifts upward (path of least resistance).

A secondary issue confirmed via v9–v10 actor probes: **entropy sink into rot dims.** SAC's `target_entropy` is a scalar sum constraint. The actor distributes per-dim variance to satisfy it as cheaply as possible. With `rot_threshold=0.01` (small physical impact), rot dim variance was "free" entropy that didn't affect reward. Pos exploration was starved (0.089 abs mean vs 0.158 for rot in v9).

Adding `r_rot_cost = -0.5 * Σ(ema_rot²)` (v10) fixed entropy distribution (pos abs mean jumped to 0.518, 6× more), but **didn't fix the Q-flat-in-z problem.** Policy still hovered because critic still had no z gradient.

## The fix (3 lines in `reward.py`)

```python
# altitude_bonus: 0 at z = entry_z + 10cm, 1 at z = entry_z.
altitude_bonus = jp.clip((entry_z + 0.10 - peg_z) / 0.10, 0.0, 1.0)
r_align = 2.0 * (r_xy + r_tilt) * altitude_bonus
```

Now r_align strictly increases as peg descends toward entry. Q surface has a clean downward gradient. Critic learns `Q(low_z, aligned) > Q(high_z, aligned)` immediately, no aligned-∧-low_z conjunction needed. Policy mean drifts downward.

v11 reaches 6.93/step within 5% of v15.6's 7.27/step ceiling, **at half the episode length**, with full 6-DOF + altitude-weighted reward.

## Probe sequence (post-mortem)

| Probe | What I changed | Result | What it taught |
|---|---|---|---|
| v1–v3 | σ_target sweep 0.15→0.30 | 831/842/804 | Entropy was satisfied; bottleneck not exploration capacity |
| v4 | approach-gated r_align | 768 | Killed alignment signal at start — wrong gating |
| v5 | r_B_desc × 5, drop r_B_pen | 702 | r_B_desc never fires; aligned gate too tight |
| v6 | wide aligned gate (30mm, 10°) | 807 | Policy never descended → wide gate didn't help |
| v7 | 3-DOF + DR (rot=0) | 809 [529, 1829] | **Bimodal** — 3-DOF sometimes solves; 6-DOF never |
| v8 | 6-DOF + no DR | 877 | DR isn't the bottleneck |
| v9 | rot_threshold 0.097→0.01 | 888 | Action probe: rot std 0.158 > pos 0.089 (**entropy sink confirmed**) |
| v10 | + r_rot_cost = -0.5·Σema² | 885 | Entropy redistributed (pos 0.518), but Q still flat in z |
| **v11** | **altitude_bonus on r_align** | **3120** | **Hover trap broken** |

## Meta-lessons (added to `lessons/offpolicy.md`)

1. **Heterogeneous action dims break SAC's entropy assumption.** Scalar `target_entropy` + per-dim cost asymmetry = entropy sink. Watch when action dims have different physical effect magnitudes.

2. **Reward diagnostic: does each axis of the action space have a Q gradient?** If you can hover and collect ~max reward, critic can't learn that axis even with infinite exploration. Look for "flat valleys" in the reward landscape, especially with conjunction-gated terms.

3. **Probe before you tune.** v9 actor stats (action mean + std per dim) took 30 seconds to compute and immediately confirmed the entropy sink hypothesis. Should have been the first diagnostic, not the 6th.

4. **3 lines of reward shape > 8 algo/hyperparam probes.** Once root cause is identified, the fix is usually small. Time wasted on entropy/algo tuning was time not spent diagnosing Q-flatness.

5. **Conjunction-gated rewards are fragile.** `r_B_desc = phase_below * aligned * z_progress` — three multiplicands all need to be ~1 simultaneously. Each factor independently unlikely → product vanishes during exploration. Replace conjunctions with continuous shaping when possible.

## Loose ends

- **DR still off** in this branch state (PROBE block at `factory_peg_insert.py:hole_pos_xy_noise=0.0`). Need to verify v11's altitude reward holds up with DR on. Should be fine — DR adds task variance, not reward-shape problems.
- **Rot_threshold reverted to original 0.097**, σ_target=0.30. These are reasonable defaults; could tune further but not critical.
- **GearMesh needs the same altitude_bonus port** if Phase 1 is to ship. GearMesh's reward also calls into `reward.py:compute_reward`, so the fix propagates automatically.

## Where to look

| Thing | Path |
|---|---|
| v11 ckpt (best) | `checkpoints/20260603_182811_flash_sac_factorypeginsert_seed0/best` |
| v11 recording | `.tmp/recordings/peginsert_v11_altitude_6dof.mp4` |
| Reward (with altitude_bonus) | `jax_rl/envs/manipulation/factory/reward.py:146-159` |
| 6-DOF wiring | `jax_rl/envs/manipulation/factory/factory_peg_insert.py:716-733` |
| Action chain (rot helpers) | `jax_rl/envs/manipulation/factory/controller/action_chain.py` |
| WandB v11 | run name `20260603_182811_flash_sac_factorypeginsert_seed0` |
