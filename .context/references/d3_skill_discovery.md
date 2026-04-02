# D3: Divide, Discover, Deploy — Skill Discovery Reference

**Paper:** arXiv:2508.19953
**Repo:** https://github.com/leggedrobotics/d3-skill-discovery
**Robot:** ANYmal-D quadruped (sim-to-real validated)
**Framework:** IsaacLab + PPO, 2048 parallel envs

## What it does

D3 factorizes the robot's state into components (position, heading, height, orientation) and assigns the best skill discovery algorithm per factor:
- **METRA** for unbounded dimensions (position — needs directional skills)
- **DIAYN** for bounded/discrete dimensions (heading — needs categorical skills)

Plus: safety constraints (style factor), symmetry exploitation (morphology mirroring), and factor weighting (λ) to balance conflicting objectives.

## Key contributions over DIAYN/METRA alone

1. **Per-factor algorithm selection** — don't use one method for everything
2. **Style factor** — extrinsic reward for stable postures (torques, contacts, height). Essential for hardware.
3. **Symmetry augmentation** — mirrors (s, z) using quadruped morphology. Latin square permutations for DIAYN, coordinate flips for METRA.
4. **Factor weighting λ** — sample from truncated Gaussian, normalize Σλ=1. Manages conflicting skills.
5. **Skill resampling** — resample z multiple times within episode (not just once)

## Method lineage

```
DIAYN (MI discriminator, 2018)
├─ DADS (dynamics-aware, 2020)
├─ DUSDi (factorized DIAYN, 2023)
│  └─ D3 (multi-algorithm per factor + safety + symmetry, 2025)
└─ CIC, LSD, APS (exploration variants)

METRA (Wasserstein dependency, 2019)
└─ D3 (uses METRA for directional factors)
```

## D3's state factorization for quadruped

| Factor | Dim | Algorithm | Why |
|--------|-----|-----------|-----|
| Base position | 2 | METRA | Unbounded, directional (walk forward vs sideways) |
| Heading | 2 | DIAYN | Bounded angle, categorical (face N/S/E/W) |
| Base height | 1 | METRA | Continuous (crouch vs stand tall) |
| Roll/pitch | 2 | DIAYN | Bounded, categorical (lean left vs right) |

## Training setup

- PPO with clip=0.2, lr=1e-4
- Actor: 3-layer MLP [512, 256, 128] ReLU
- DIAYN discriminator: [256, 256], Dirichlet α curriculum 0.05→1.0
- METRA: Lagrange multiplier 30.0, norm-matching σ=10.0
- Style rewards: -1e-3 (torques), -10 (height), -10 (orientation), -30 (contacts)
- 2048 envs on RTX 3090

## What we need for Go2 implementation

**Already have:** RewardSpec, ObsSpec, asymmetric critic, action delay, EncoderConfig.context_dim

**Need to build:**
- Skill priors (Dirichlet + Hypersphere)
- Discriminator network q_φ(z|s)
- Intrinsic reward module (DIAYN or METRA per factor)
- Go2 symmetry mirror functions (4-fold: FL↔FR, RL↔RR)
- Style factor + regularization penalties
- Factor weighting λ mechanism

## Key insight for our roadmap

DIAYN → METRA is still the right progression, but they're not sequential replacements — they coexist in the final system, assigned per state factor. Safety and symmetry are not Phase 7 extras; they're required for hardware deployment.
