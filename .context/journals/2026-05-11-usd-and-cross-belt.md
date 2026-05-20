# 2026-05-11 — USD export pipeline + splitbelt cross-belt termination

## What

Two tracks landed in one session:

**A. USD trajectory replay pipeline** — turn a saved `_traj.npz` + env XML
into a USD file viewable in Blender / Omniverse / usdview. First step
toward "extract trajectory → render pretty in external tool" workflow
(analog of Isaac → Blender pipelines).

- New script: [`scripts/export_usd.py`](../../scripts/export_usd.py).
  Loads env's `mj_model`, replays qpos sequence through
  `mujoco.usd.exporter.USDExporter`, writes `<out>/frames/frame_N.usd`
  (single file with all time samples) + `<out>/assets/texture_*.png`.
- record_video.py: HD/quality flags added (`--resolution W H`,
  `--video-quality 1-10`). Renderer was hardcoded 640×480; now CLI-
  configurable. `record_sweep.py` mirrors both flags.
- Tested end-to-end: PoseDR splitbelt trajectory → 9.8 MB `.usd` (178
  prims, 1250 time samples @ 60fps). Pixar's USD format; Blender 3.5+
  imports natively. Phase 1 viewer: `usdview` (Pixar CLI, needs PySide6
  pip install) or Blender's USD import.

  Caveat: WSL2 + Blender = Wayland passthrough fails (OpenGL 4.3 not
  exposed via WSLg). Windows-native Blender is the practical viewer.

**B. Cross-belt termination** — `Go2WarpSplitbeltEnv._get_termination`
previously had 3 causes (fall_torso, off_belt, tilt). User flagged a
gap: nothing terminated when a grounded foot landed on the OPPOSITE
belt from its spawn assignment. That's the real biomech-experiment
fail (subject puts wrong foot on wrong belt) and was hiding inside
the `tilt` bucket. Added 4th cause:

```python
# FL,RL (robot-left, world +y) → right belt (id=1); FR,RR → left belt (id=0)
natural_belt = jp.array([1, 0, 1, 0], dtype=jp.int32)
foot_grounded = foot_pos_world[..., 2] < 0.02
cross_belt = jp.any(
    (foot_belt_id != natural_belt) & (foot_belt_id != -1) & foot_grounded
)
term_cause = jp.where(... cross_belt, jp.int32(4), ...)
```

Belt-name vs foot-name convention mismatch caught at smoke-test: robot
FL foot spawns at world +y → right belt id=1, not the "left" id you'd
expect from the FL prefix. Fixed mapping `[1,0,1,0]` not `[0,1,0,1]`.

eval_physics.py extended with 4th column `cross`.

## Re-eval results

PoseDR v2 ckpt (`20260506_195126_fast_sac_go2warpsplitbeltposedr_seed0`,
1M FastSAC) re-evaluated under new termination. Same actor weights;
only env termination logic changed. 16 ep × 9 conditions @ ep_len 1250.

Training support: `vL ∈ [0.3, 1.5]`, `ratio ∈ [0.5, 2.0]`.

| # | (vL, vR) | ratio | OOD | term% | fall | off | tilt | cross | surv | finalDx |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5, 0.5 | 1.0× | in | 0% | 0 | 0 | 0 | 0 | 1250 | -0.02 |
| 2 | 1.0, 1.0 | 1.0× | in | 0% | 0 | 0 | 0 | 0 | 1250 | -0.07 |
| 3 | 0.5, 1.0 | 2.0× | in (edge) | **88%** | 0 | 0 | 4 | **10** | 549 | -0.17 |
| 4 | 0.5, 1.5 | 3.0× | ratioOOD | **94%** | 0 | 0 | 2 | **13** | 289 | -0.27 |
| 5 | 0.3, 0.9 | 3.0× | ratioOOD | **100%** | 0 | 0 | 2 | **14** | 424 | -0.17 |
| 6 | 0.3, 1.5 | 5.0× | ratioOOD | **100%** | 0 | 0 | 9 | **7** | 209 | -0.85 |
| 7 | 0.5, 0.2 | 0.4× | absOOD + ratioOOD | 81% | 0 | 0 | **13** | 0 | 944 | -0.38 |
| 8 | 1.0, 0.3 | 0.3× | ratioOOD | 75% | 0 | 1 | **11** | 0 | 965 | -1.82 |
| 9 | 1.5, 1.5 | 1.0× | in (edge) | **100%** | 0 | 8 | 8 | 0 | 677 | -7.03 |

Old eval (3-cause) read row #3 as "75% term, mostly tilt"; new eval
reveals it's actually 88% term, 10/16 cross-belt + only 4 tilt. The
extra terminations come from episodes that previously "survived" via
crossing feet onto a single belt.

**Asymmetry by failure mode**:
- R-faster (rows 3–6, vR > vL): cross-belt dominates. Robot dragged
  rightward, FL/RL feet end up on left belt.
- L-faster (rows 7–8, vL > vR): zero cross-belt, all tilt. Policy can't
  pull right-side feet to the faster left belt — keyframe/gait bias.
- Tied at edge of support (row 9): no cross possible; fails by
  off-belt drift (8/16, ~7m backward drag) + tilt (8/16).

Same magnitude differential produces opposite failure mode based on
direction. Spawn keyframe + uniform DR ≠ symmetric robustness exposure.

## Notable observations

1. **Termination scheme is load-bearing for diagnostics.** Aggregating
   distinct failure modes into one bucket hides directional asymmetry.
   Add fine-grained causes (cross-belt, off-belt, knee-contact) and
   surface per-OOD breakdown.
2. **Re-evaluating with stricter termination = cheap diagnostic.** No
   retrain. Reveals what failures the policy was "cheating around"
   during training (e.g. cross-foot survival episodes).
3. **Real biomech splitbelt experiments reject cross-belt episodes** —
   our env should match. Old setup overcounted success.

## Files changed

- `scripts/export_usd.py` (new)
- `scripts/record_video.py` — `--resolution`, `--video-quality` flags
- `projects/adaptation/sweeps/record_sweep.py` — same flags piped
- `jax_rl/envs/locomotion/go2_warp_splitbelt.py` — cross_belt term
- `jax_rl/envs/locomotion/go2_rendering.py` — bigger cmd arrows
  scaled to robot size, length-normalized so small cmd still visible
- `projects/adaptation/sweeps/eval_physics.py` — 4th cause column
- `projects/adaptation/lessons/splitbelt.md` — new lesson at
  "Cross-belt detection reveals tilt failures were mostly belt-crossover"

## Open / next

1. **Retrain PoseDR with cross-belt termination active** — current
   policy was trained without it, so it's not optimizing to avoid
   crossover. Fresh 1M FastSAC should produce a meaningfully different
   gait (less dependence on cross-foot stance for stability).
2. **Symmetrize spawn / DR** to address direction asymmetry. Either
   mirror-augment trajectories or randomize spawn keyframe orientation.
3. **Phase 2 USD pipeline**: Blender material setup, HDRI env light,
   Cycles render — turn the .usd into a publication-grade clip.
