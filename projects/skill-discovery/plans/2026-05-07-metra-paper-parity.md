# METRA Paper-Parity Run on AntMJXClassic

> **Single-experiment plan**, not a multi-stage build. Tests whether our METRA implementation
> matches METRA paper (Park 2024) results on the same env (Ant) with the same HPs.

## Why

The 2026-05-07 ablation (`lessons/metra_ant.md`) showed METRA on AntMJXClassic
degenerated regardless of `dual_dist` (`one` and `l2` both null). We initially
labeled this as "Ant-may-be-too-low-dim" but METRA paper explicitly uses Ant
as a state-based benchmark (Fig 6/7) and shows skills traveling many meters
in distinct directions. **Our null is a paper-parity gap, not an env-scale
limit.**

Most-likely culprit: SAC actor/critic hidden_dim **(256, 256)** in our config
vs **(1024, 1024)** in METRA paper. Plus lr 1e-3 (ours) vs 1e-4 (paper).
Both could matter — bigger SAC = more capacity to express skill-conditional
policies; smaller lr = more stable updates over the long horizon.

## What

Single seed × 1M METRA on `AntMJXClassic` with paper HPs:

```
SAC actor hidden:     (1024, 1024)   # paper
SAC critic hidden:    (1024, 1024)   # paper
TrainConfig.lr:       1e-4           # paper
SACConfig.alpha_lr:   1e-4           # paper
SACConfig.alpha_init: 0.01           # paper
SACConfig.batch_size: 256            # paper

# Already match paper:
phi_hidden:           (1024, 1024)   # already
phi_lr:               1e-4           # already
dual_lr:              1e-4           # already
dual_lam_init:        30.0           # already
dual_slack:           1e-3           # already
dual_dist:            "one"          # paper default, reverted after l2 ablation
target_entropy_scale: 0.5            # = -|A|/2 = -4 for Ant ✓
tau:                  5e-3           # already
gamma:                0.99           # already
unit_sphere prior:    yes            # already
```

Buffer 1M (paper). Num envs 64 (ours; paper uses single-env collect but parallel
batched updates should be equivalent off-policy, modulo replay-ratio tuning).
Episode length 1000.

## Command

```bash
PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/train_skill_discovery.py \
    --env AntMJXClassic --algo metra --num-skills 8 \
    --total-timesteps 1000000 --seed 0 \
    --buffer-size 1048576 --num-envs 64 \
    --actor-hidden-dim 1024,1024 \
    --critic-hidden-dim 1024,1024 \
    --lr 1e-4 \
    --alpha-lr 1e-4 \
    --alpha-init 0.01 \
    --batch-size 256 \
    > .temp/logs/ant_classic_metra_paper_parity_seed0.log 2>&1
```

Wall-clock projection: ~120-150 min (4× SAC capacity → larger forward + backward; some compensation from smaller batch_size 256 vs 512).

## Memory check

Before launching, confirm fits in 16GB:
- Phi: 1024×1024 = ~1.1M params
- SAC actor: 35→1024→1024→16 = ~1.1M params
- SAC critic ×2 (twin Q): 35→1024→1024→1 = ~1.1M each
- Target Q ×2: same
- Adam state ×3 = 18M params × 4 bytes = ~72 MB
- Buffer 1M × ~80 bytes/element = ~80 MB
- Warp graph ~2-3 GB
- Total ~5-7 GB. Fits if no other GPU users.

GPU coordination: another agent running G1 FastSAC (11 GiB). Wait until that completes (or check it has finished) before launching this.

## Acceptance gates

| Gate | Pass | Fail (hypothesis confirmed) |
|---|---|---|
| **Lipschitz engagement** | DualLam stays ≥ 1 by 100K (constraint actually pressures phi) | DualLam → 0 same as default config |
| **Phi alignment** | PhiAlign mean > 0.1 by 500K (real directional learning) | PhiAlign ≈ 0 same as before |
| **xy spread** (visual gate) | max-pairwise > 3.0 m on figure | max-pairwise < 1.5 m |
| **Heading-std** | > 30° (always passed before) | unchanged |

If gates pass: **METRA implementation is correct, prior null was HP issue (small SAC).** Next:
- Add this preset to `env_presets.py` as `_METRA_PAPER_SAC_CFG` for reuse
- Run seeds 1, 2 (~5h total serial)
- Update `lessons/metra_ant.md` §3 / §6: rephrase from "Ant may be too low-dim" to "Ant works with paper HPs; default 256×256 SAC was undersized"
- Stitch new 6-panel DIAYN-vs-METRA(paper-parity) figure
- Promote METRA as legitimate baseline contrast for SD-D

If gates fail: **METRA implementation may have deeper issue.** Next:
- Compare phi outputs at init between paper repo (single-step trace) and ours
- Check phi normalization / output activation (paper uses linear output, no unit-norm)
- Consider single-env collect with replay_ratio=1 to mimic paper's update pattern exactly
- Or accept METRA-on-our-stack ceiling and move to SD-C

## Risk register

| Risk | Mitigation |
|---|---|
| 1024×1024 SAC + 1024×1024 phi blows GPU memory | Try with --batch-size 256 first (paper); if OOM, reduce buffer to 524K |
| lr=1e-4 too slow → no progress in 1M | Paper uses 2M-4M steps for some envs. If null at 1M, extend to 2M |
| Replay ratio mismatch (parallel 64 envs vs paper single env) | grad_updates_per_step=8 in our config (=128 effective updates/sec at 64 envs); paper does ~1 update per env step (=1 update/sec at 1 env). Ours is 128× higher. May overfit on small buffer. Try `--grad-updates-per-step 1` if first run shows instability. |
| Other GPU consumer | Wait for G1 FastSAC to finish (currently 11 GiB) before launching |

## Status

- [x] CLI flags added to `scripts/train_skill_discovery.py` (commit pending)
- [x] CheetahRun default-flag smoke verified — existing behavior preserved
- [ ] GPU coordination check
- [ ] Launch + monitor
- [ ] Render figure + apply gates
- [ ] Update lesson with conclusion
