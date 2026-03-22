# Benchmark Results — JAX RL Framework

**Last updated:** 2026-03-22

## CheetahRun (obs=17, action=6)

| Algo | Eval | Steps | Envs | sps | Checkpoint |
|---|---|---|---|---|---|
| PPO | 826 | 20M | 2048 | 65k | *no checkpoint (journal only)* |
| PPO | 666 | 20M | 2048 | 52k | `20260313_173851_cheetahrun_seed0` |
| Vanilla TD3 | 749 | 5M | 128 | 10k | `20260318_115651_td3_cheetahrun_seed0` |
| Vanilla SAC | 771 | 5M | 128 | 10k | `20260319_213016_sac_cheetahrun_seed0` |
| **FastTD3** | **880** | 86M | 1024 | 18k | `20260319_000622_fast_td3_cheetahrun_seed0` |
| FastSAC | 582 | 100M | 1024 | 15k | `20260319_194043_fast_sac_cheetahrun_seed0` |

**Takeaway:** Deterministic TD3 + C51 wins on low-dim (6-dim) actions. Vanilla SAC at 128 envs (771) beats FastSAC at 1024 envs (582) — the "Fast" recipe doesn't help SAC on this task.

## WalkerWalk (obs=24, action=6)

| Algo | Eval | Steps | Envs | sps | Checkpoint |
|---|---|---|---|---|---|
| PPO | 833 | 60M | 2048 | 65k | *no checkpoint (journal only)* |
| Vanilla TD3 | 955 | 5M | 128 | 10k | `20260318_124138_td3_walkerwalk_seed0` |
| Vanilla SAC | 975 | 5M | 128 | 7k | `20260318_182809_sac_walkerwalk_seed0` |

**Takeaway:** SAC and TD3 both near-optimal. PPO gets there eventually but 12x slower.

## HumanoidRun (obs=67, action=21)

| Algo | Eval | Steps | Envs | sps | Checkpoint |
|---|---|---|---|---|---|
| PPO | ~10 | 60M | 2048 | 65k | *no checkpoint (journal only)* |
| Vanilla TD3 | 4.3 | 5M | 128 | 10k | `20260318_125519_td3_humanoidrun_seed0` |
| Vanilla SAC | 207 | 5M | 128 | 7k | `20260318_111816_sac_humanoidrun_seed0` |
| Vanilla SAC (extended) | 426 | 20M | 128 | 4.4k | `20260320_162959_sac_humanoidrun_seed0` |
| FastTD3 | 665 | 100M | 1024 | 8k | `20260320_205102_fast_td3_humanoidrun_seed0` |
| **FastSAC** | **892** | 100M | 1024 | 12k | `20260321_010852_fast_sac_humanoidrun_seed0` |
| FastDSAC (128 envs) | 490 peak | 5M | 128 | 305 | `20260321_194023_fast_dsac_humanoidrun_seed0` |
| FastDSAC (1024 envs) | 316 peak | 53M | 1024 | 2.3k | *Inf'd at 53M, rerunning with guard* |

**Takeaway:** SAC's entropy-based exploration is decisive for 21-dim actions. TD3 can't explore. FastSAC (892) is the best result. FastDSAC works at 128 envs but struggles at 1024.

## CartpoleBalance (obs=5, action=1)

| Algo | Eval | Steps | Envs | Checkpoint |
|---|---|---|---|---|
| PPO | >=995 | 1M | 64 | *no checkpoint* |

## Key Findings

1. **Low-dim (6-dim actions):** TD3 > SAC. Deterministic policy + simple noise suffices.
2. **High-dim (21-dim actions):** SAC >> TD3. Entropy exploration is critical.
3. **C51 distributional critic helps at scale** for both TD3 (749→880) and SAC (426→892).
4. **Vanilla algos at 128 envs are competitive** with Fast variants at 1024 envs on low-dim tasks.
5. **FastDSAC (Huber + DEM) is promising but fragile** at 1024 envs. Works at 128 envs (paper scale).
