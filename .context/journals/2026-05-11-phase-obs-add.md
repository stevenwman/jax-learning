# 2026-05-11 — G1 phase obs + stand_prob (transitional flags)

## What

Added two opt-in flags to `G1WarpJoystick`:
- `cfg.gait_phase_obs` (default `False`)
- `cfg.command_config.stand_prob` (default `0.0`)

Both default off so legacy ckpts (HoloSoft 292, HoloClearance 266,
HoloClearanceWide v1 239) keep loading against unchanged obs schema /
cmd distribution. New `HoloClearanceWide` preset (v2 onwards) flips
`gait_phase_obs=True`, `stand_prob=0.2`.

## Why

Holosoma G1 source review (amazon-far/holosoma) showed their official
actor preset is `g1_29dof_loco_single_wolinvel`:

- Actor sees `base_ang_vel`, `projected_gravity`, `command_lin_vel`,
  `command_ang_vel`, `dof_pos`, `dof_vel`, `actions`, `sin_phase`,
  `cos_phase` — but **NOT** `base_lin_vel`.
- Critic gets `base_lin_vel` in privileged group.

Ours already matched the wolinvel pattern (linvel was privileged-only)
BUT lacked the phase clock. Quadruped tolerates phase-less obs because
stance dominates; humanoid dynamic balance needs the gait clock to map
cmd→stride. Lock-cmd probes on HoloClearanceWide v1 confirmed this
phenotype: actor tracked cmd_x=0.10 at 91% but cmd=0.5 at 18% and
fully ignored yaw — no temporal grounding to plan against.

Also added `stand_prob` since holosoma's `g1_29dof_command` uses 0.2 —
explicit "hold still" training. Otherwise policy averages over
nonzero-cmd distribution and never internalizes cmd=0 as a real mode.

## How (back-compat strategy)

Flags default off. Conditional append at obs-build site:

```python
if self._config.gait_phase_obs:
    self._obs_groups["state"].append(ObsTerm("sin_phase", ...))
    self._obs_groups["state"].append(ObsTerm("cos_phase", ...))
```

`sample_command` adds `jp.where(bernoulli(stand_prob), 0, cmd)` which
is a no-op when `stand_prob=0`.

Smoke test confirms back-compat:
- HoloSoft: actor=96 priv=208 (unchanged ✓)
- HoloClearance: actor=96 priv=208 (unchanged ✓)
- HoloClearanceWide v2: actor=100 priv=212 phase=True stand_prob=0.2
- SplitbeltClearanceTied: actor=98 priv=214 (unchanged ✓)

## Removal plan

Once obs schema stabilizes (likely when we have a fast walker we're
happy with and want to commit to ONE schema for all G1 work):

1. Retrain any ckpt we want to keep against the new schema.
2. Remove `gait_phase_obs` flag from default_config; always-on the
   sin/cos append.
3. Remove `stand_prob` config field; bake the holosoma value into all
   presets (or pick a value).
4. Delete every `TRANSITIONAL FLAG (2026-05-11)` comment.
5. Update lesson doc + delete this journal note's "removal plan"
   section.

Search-target string: `TRANSITIONAL FLAG (2026-05-11)`.

## Open

- Did not fire HoloClearanceWide v2 yet — pending user signal. v1
  ckpt at `checkpoints/20260511_141911_fast_sac_g1warpjoystickholoclearancewide_seed0/best`
  remains the comparison baseline.
