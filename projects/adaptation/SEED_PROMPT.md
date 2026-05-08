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

| Env | Ckpt | Eval @ 5M | Notes |
|---|---|---|---|
| G1WarpJoystickHoloSoft | `checkpoints/20260508_014027_fast_sac_g1warpjoystickholosoft_seed0/best` | **292.1 ± 0.6** | flat walking, FastSAC paper-match. Full 1000-step survival. **SHUFFLING foot-lift, not full stride.** |
| G1WarpJoystickHoloSoft | `checkpoints/20260508_011324_flash_sac_g1warpjoystickholosoft_seed0/best` | 273.9 ± 2.0 | same env, FlashSAC. Comparable shuffling. |
| G1WarpSplitbeltTied / G1WarpSplitbelt | various v19-v22 ckpts | ~25 ± 10 | splitbelt baseline plateau. Mean 50-step survival. Robot walks on belts in some episodes (max 65). |

Videos for all of the above in `projects/adaptation/videos/g1_v15_*` through
`g1_v22_*`. v18 is the canonical "flat walking" demo — but user noted it
shuffles, doesn't truly lift feet.

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

### 1. Encourage real foot lifting on flat (user observation: "both shuffling")

Current eval 292 episodes show robot surviving full episodes but
shuffling — not picking feet up cleanly. Hypotheses:

a. **`feet_phase` reward saturates near foot-z=0**: with sigma=0.008 and
   swing_height=0.09, exp(-error/sigma) is near 1 if foot stays low and
   target rz also near 0 during stance phase. Policy learns "keep foot
   near floor" satisfies the reward without lifting during swing.
   Diagnostic: print `_reward_feet_phase` value on a shuffled rollout —
   should already be ~1.0 even without lifting.
   Fix candidates:
   - Bump swing_height 0.09 → 0.15 (less reward for low-foot)
   - Tighten sigma 0.008 → 0.002 (sharper penalty for not tracking)
   - Add explicit `feet_clearance` reward (reward foot z while not in contact)

b. **cmd ranges too narrow** (cmd_a = [0.1, 0.1, 0.1]). Tracking_lin_vel
   reward = exp(0)=1 at vel=0, exp(-(0.1)²/0.25)=0.96 at full cmd.
   Saturates fast → no incentive to move. **Test**: bump cmd_a to
   [0.5, 0.3, 0.5] (closer to holosoma's `limit_ranges` of [-0.5, 1.0],
   [-0.3, 0.3], [-0.2, 0.2]) and retrain.

c. **Push events** (holosoma has interval_range_s=(5,5),
   magnitude_range=(0.1, 2.0) for vel kick). Forces recovery learning;
   without them, policy never learns external perturbations so it stays
   stiff. Implement: env.step adds random qvel kick every ~5s.

### 2. Splitbelt-G1: close 25 → ~200 gap

Bottleneck (per lesson doc): actor obs lacks belt info. Highest leverage:

a. **Add belt_vel to actor obs** (port Go2 splitbelt's `informed` mode).
   Modify `g1_warp_splitbelt.py` _post_init's `_obs_groups`:
   ```python
   ObsTerm("belt_vel", lambda info, **kw: info["splitbelt"]["belt_vel"], 0.0)
   ```
   Then retrain G1WarpSplitbelt. Expect substantial jump.

b. **Curriculum on belt speed**: start v_range=(0.1, 0.2), ramp to
   (0.3, 1.0). Holosoma-style avg-epl-based or simple step-count linear.

c. **Multi-seed**: current is only seed=0. v19-v22 high variance suggests
   3-5 seeds for stable mean.

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
