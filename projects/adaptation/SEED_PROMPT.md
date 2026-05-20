# G1 Humanoid + Splitbelt — Continuation Seed (2026-05-08)

> Paste this into next session if context compacted. All paths absolute from
> repo root. Active branch: `new_slate_linen`. Today's date: 2026-05-08+.

## Read first (in order)

1. `.context/lessons/g1_humanoid.md` — full v1→v22 progression, every wrong
   turn + the actual fixes. Critical: don't repeat the "FastSAC architecture
   broken" diagnostic — paper-match (`--obs-norm --reward-scaling 0.2`) was
   the actual missing knob.
2. `projects/adaptation/TODO.md` — splitbelt project todo (Go2 + G1 unified).
3. `projects/adaptation/lessons/splitbelt.md` — Go2 splitbelt lessons (foot
   tunneling, belt sign convention, etc. — many port directly to G1).
4. `jax_rl/envs/locomotion/g1_warp_joystick.py` — G1 flat env.
5. `jax_rl/envs/locomotion/g1_warp_splitbelt.py` — G1 splitbelt env.

## Current state

### Working ckpts (use these as starting points)

| Env | Ckpt | Eval | Notes |
|---|---|---|---|
| **G1WarpJoystickHoloClearance** | `checkpoints/20260508_105608_fast_sac_g1warpjoystickholoclearance_seed0/best` | **266 ± 5 @ 1M** | **REAL WALKER**. foot_z p90 0.075-0.088 (target 0.09), forward drift +1.41 m / 33s, qvel_x +0.07 m/s. Use as flat-G1 baseline. |
| G1WarpJoystickHoloSoft | `checkpoints/20260508_014027_fast_sac_g1warpjoystickholosoft_seed0/best` | 292.1 ± 0.6 @ 5M | shuffles, foot_z p90 0.043. Survives but doesn't really walk. |
| G1WarpJoystickHoloLift | `checkpoints/20260508_093213_fast_sac_g1warpjoystickhololift_seed0/best` | 345 @ 5M | feet_phase weight 5→10, sigma .008→.005. Still shuffles (foot_z p90 0.046). Higher reward but no real lift improvement. |
| G1WarpJoystickHoloWide | `checkpoints/20260508_090201_fast_sac_g1warpjoystickholowide_seed0/best` | 285 @ 5M | cmd_a=[0.5,0.3,0.5]. Drifts further but still shuffles. |
| G1WarpSplitbeltInformed | `checkpoints/20260508_095811_fast_sac_g1warpsplitbeltinformed_seed0/best` | ~20 @ 5M | belt_vel R^2 in actor obs. Falls in 67 steps. |
| G1WarpSplitbeltInformedTied | `checkpoints/20260508_102715_fast_sac_g1warpsplitbeltinformedtied_seed0/best` | ~10 @ 5M | tied(0.5) + belt_vel obs. Falls in 30 steps. |
| G1WarpSplitbeltClearanceTied | `checkpoints/20260508_112432_fast_sac_g1warpsplitbeltclearancetied_seed0/best` | peak 32 @ 4.55M, ep_len 19-45 across 8 seeds | HoloClearance reward + tied(0.5) + informed obs. Better than InformedTied but still falls. |
| G1WarpSplitbeltTied / G1WarpSplitbelt | various v19-v22 ckpts | ~25 ± 10 | splitbelt baseline plateau. |

Videos: `projects/adaptation/videos/{g1_holoclearance,g1_holowide,g1_hololift,g1_splitbelt_informed}/`.
Diagnostics CSV per ckpt in `projects/adaptation/diagnostics/foot_probe_*.csv`.

### Foot-lift breakthrough via linear clearance reward (2026-05-08)

3 reward configs (HoloSoft / HoloWide / HoloLift) all plateaued at
foot_z p90 ≈ 0.045 m vs 0.09 m target. Diagnostic showed feet_phase
reward at 0.79 (NOT saturated) — gradient alive but policy stuck in
"low foot, low cost" Pareto basin.

**HoloClearance fixed it**: linear contact-gated bonus
`feet_clearance_swing = sum_i min(foot_z_i, 0.10) * (1 - contact_i)`,
weight 30 (`default_config_holosoma_clearance` in g1_warp_joystick.py).
Result: foot_z p90 0.075-0.088, real forward walking, eval 266 at 1M
steps already (faster convergence than HoloSoft 5M).

**Why it worked**: linear reward has constant gradient below target,
unlike `exp(-err²/sigma)` which has near-zero gradient when far from
target. Contact-gating means stance foot doesn't pay → reward only
fires when policy actually lifts. Adds bonus on TOP of feet_phase
(complementary), doesn't replace it.

See `.context/lessons/g1_humanoid.md` "Foot-lift shuffle is a
reward-Pareto issue" lesson for full progression + linear-vs-exp logic.

### Cross-belt termination added; PoseDR re-eval reveals direction asymmetry (2026-05-11)

`Go2WarpSplitbeltEnv` now has 4th term cause: foot grounded on opposite
belt from spawn assignment (`natural_belt=[1,0,1,0]` for [FL,FR,RL,RR]
since FL spawns at world +y = right belt). Re-eval of PoseDR v2 ckpt
(same actor, new env term) under 9 (vL, vR) conditions × 16 ep:

- R-faster (ratio>1) **OOD**: cross-belt dominates (10–14/16 ep). Robot
  dragged rightward, feet end up on wrong belt.
- L-faster (ratio<1) OOD: zero cross-belt, all tilt. Policy can't pull
  right-side feet onto faster left belt — spawn/keyframe bias.
- Tied at edge of support (1.5,1.5): no cross possible; off-belt drift
  + tilt mix.

Old eval read row #3 (ratio 2×) as 75% term mostly tilt; new eval = 88%
term, 10/16 cross-belt. Termination scheme was hiding asymmetry.

Action items: retrain PoseDR with cross-belt active; symmetrize spawn /
DR to fix L/R asymmetry. See journal `2026-05-11-usd-and-cross-belt.md`
+ lesson `lessons/splitbelt.md` "Cross-belt detection reveals tilt
failures were mostly belt-crossover".

### USD trajectory replay pipeline (Phase 1, 2026-05-11)

`scripts/export_usd.py` replays a saved `_traj.npz` qpos sequence
through `mujoco.usd.exporter.USDExporter` → single `.usd` with time
samples + textures. Tested on PoseDR splitbelt: 9.8 MB, 1250 frames @
60fps, 178 prims. Viewable in Blender (3.5+ native USD import) /
usdview / Omniverse. Phase 2 (Cycles material setup, HDRI light) TBD.

`record_video.py` + `record_sweep.py` got `--resolution W H` and
`--video-quality 1-10` flags (renderer was hardcoded 640×480).

### Splitbelt-G1: bottleneck is dynamics, not actor blindness (2026-05-08)

Tested seed prompt #2a (add belt_vel to actor obs). Three variants:
- `G1WarpSplitbeltInformed` (random_per_episode v∈[0.3,1.0]): eval ~20,
  falls in 67 steps. Belt vR can hit 1.4 m/s — too fast for early policy.
- `G1WarpSplitbeltInformedTied` (v=0.5 fixed both belts): eval ~10,
  falls in 30 steps. Even slow tied belts crash the policy quickly.
- `G1WarpSplitbeltClearanceTied` (HoloClearance reward + tied(0.5) +
  informed obs): peak eval 32 / 5 ep at 4.55M, but per-seed rollouts
  show ep_len 19-45 (mean ~33) — same crash, just more variable.

Conclusion: actor blindness was NOT the bottleneck. The G1 control
policy itself can't survive belt drag from-scratch even with the
foot-lifting reward set proven on flat. Likely paths forward:

a. **Transfer-init from HoloClearance ckpt** — load the 266-eval flat
   walker as starting actor params, then train splitbelt. The flat
   walker already lifts feet and walks forward; should survive belt
   drag much longer. Need ckpt-loading at train start.

b. **Belt-speed curriculum**: start v_range=(0.0, 0.05) for first 1M
   steps, ramp to (0.3, 1.0) by 5M.

c. **Both** — transfer + curriculum.

### Reward set: HoloSoft (proven for survival, not foot-lift)

`default_config_holosoma_soft()` in `g1_warp_joystick.py`. Holosoma's G1
fast_sac reward set with **penalty terms × 0.5** (matches their
`PenaltyCurriculum.min_scale=0.5`). Key weights: alive=10,
tracking_lin_vel=2, tracking_ang_vel=1.5, feet_phase=5 (sigma 0.008),
per-joint pose weights (legs hip_pitch+knee=0.01 free, waist+arms=50
locked), close_feet_xy=-5, feet_ori=-2.5, action_rate=-1, orientation=-5.

### Algo: FastSAC paper-match (NOT default)

```bash
XLA_CLIENT_MEM_FRACTION=0.55 uv run python scripts/train_fast_sac.py \
  --env G1WarpJoystickHoloSoft --num-envs 256 \
  --total-timesteps 5000000 --reset-mode per_step --seed 0 \
  --buffer-size 1000000 --reward-scaling 0.2 --obs-norm \
  --wandb --wandb-project g1-warp-joystick
```

The two key flags `--obs-norm` and `--reward-scaling 0.2` are paper
defaults. WITHOUT them FastSAC stalls at eval -1. WITH them, eval 292.

## Open follow-ups (priority order)

### 1. Train HoloClearance further / multi-seed

HoloClearance peaked at eval 266 by 1M steps. Train ran to 5M but only
1 eval period was logged (eval_every_n_episodes=5000 = ~1M-step eval
cadence). Worth either:
- Re-evaluate the 5M `best` ckpt with more episodes (current: 1 eval
  block of 5 ep)
- Train multi-seed (1, 2, 3) for variance check
- Cmd-wide variant (HoloClearance + cmd_a=[0.5,0.3,0.5]) — now that
  foot lift works, see if wide-cmd tracks better

### 2. Splitbelt-G1: stop plateau at 25 → use HoloClearance as init

Splitbelt baselines + Informed + InformedTied all crash in 30-67
steps. The bottleneck is fundamental control, not actor obs. New plan:

a. **Transfer-init from HoloClearance** — load the 266-eval flat
   walker as starting actor params, then train splitbelt. The flat
   walker already lifts feet and walks forward; should survive belt
   drag much longer. Need ckpt-loading at train start (currently new
   actor inits randomly).

b. **Belt-speed curriculum**: start v_range=(0.0, 0.1) for first 1M
   steps, ramp to (0.3, 1.0) by 5M. Linear or avg-epl-based ramp.
   Standalone fix even without transfer-init.

c. **HoloClearance with splitbelt scene** — register a new env that
   uses the splitbelt XML but cmd_zero (stand on belts) and the
   clearance reward. Pure-from-scratch but with the proven reward.

### 3. Per-protocol splitbelt presets (A1/A2/A3/A4)

Once flat-splitbelt eval is reasonable, port the 4-protocol family from
Go2 splitbelt (within-episode adaptation, context-conditioned, meta-RL,
continual). Schedule samplers already in `splitbelt_schedules.py`.

## Hard rules (don't violate)

- `uv run python` always; no bare `python` / `python3`.
- No `Co-Authored-By` in commit messages.
- All training runs with `--wandb` by default.
- record_video.py during live training: pass `--out projects/adaptation/videos/<dir>/<name>.mp4` to avoid the active-checkpoint `best/` overwrite.
- RTX 5080 + Warp + 256 envs: `XLA_CLIENT_MEM_FRACTION=0.55`. Buffer 4M
  OOMs; use `--buffer-size 1000000`.
- Don't claim "policy works" from eval reward alone — record video, watch.
  Eval 292 with shuffling feet = "survives" not "walks".
- Per-joint pose weights are critical: `[0.01, 1, 5, 0.01, 5, 5]×2` legs +
  `[50]×17` waist+arms. Uniform pose weights kill gait.
- Splitbelt scene XML: include G1 model FIRST, treadmill apparatus SECOND.
  Reverse breaks freejoint qpos layout (caused upside-down spawn in v19 dev).

## Don't do these (mistakes I made)

- Don't tune scalar SAC hyperparams (tau, gamma, target_entropy) without
  first checking obs/reward normalization. v8-v13 burned 7 isolation runs
  on this; v15 with `--obs-norm` instantly worked.
- Don't use Go2's `jp.clip(reward, 0, 10000)` pattern. Lower-clip kills
  termination penalty gradient. We removed for G1.
- Don't apply holosoma's full penalty weights blindly (v16 → eval 11). Use
  HoloSoft (×0.5) preset.
- Don't claim FastSAC's architecture is the issue. Both FastSAC and
  FlashSAC use C51; obs normalization was the actual fix.

## User collaboration notes

- User direct, terse. "yuh" / "rip" / "ngl" engineering signals not noise.
- Pushes back when warranted. Listen — earlier "vanilla SAC works on
  Humanoid, why wouldn't FastSAC?" forced me to read paper source and find
  the real fix.
- Caveman mode active by hook (terse fragments, code blocks normal).
- Likes concrete actionable proposals over open-ended discussion.
- Background train + ScheduleWakeup pattern fine.

## Quick orientation commands

```bash
# Verify env constructs + steps
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -c "
import jax, jax.numpy as jp
import jax_rl.training.env_setup
from mujoco_playground import registry
env = registry.load('G1WarpJoystickHoloSoft')  # or G1WarpSplitbelt
state = env.reset(jax.random.PRNGKey(0))
state2 = env.step(state, jp.zeros(env.action_size))
print(f'reward={float(state2.reward):.4f}')
"

# Record video on a frozen ckpt
MUJOCO_GL=egl uv run python scripts/record_video.py \
  --checkpoint <ckpt>/best --max-steps 1000 \
  --out projects/adaptation/videos/<dir>/<name>.mp4

# Hermetic tests (CPU)
JAX_PLATFORMS=cpu uv run python -m pytest -q tests/
```

## Last commit when this prompt was written

`687d1a3 docs(g1): splitbelt-G1 baseline + lessons (eval ~25 plateau)`

## Prior seed prompt (preserved for ref)

[Splitbelt Continuation Seed Prompt 2026-05-05 — Go2 splitbelt PoseDR.
Content was: 3 env variants, FastSAC PoseDR eval 280→378 with tunneling
fix, Go2-specific. G1 work overlaid this; original problem space largely
moved on.]
