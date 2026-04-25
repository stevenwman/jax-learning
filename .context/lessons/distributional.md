# Distributional RL Lessons (C51, FastTD3, FastSAC, FastDSAC)

---

## FastTD3: Scale Matters — Don't Bring a Race Car to a Parking Lot

**Problem:** FastTD3 underperformed vanilla TD3 at 128 envs / 5M steps: 285 vs 749.

**Root cause:** FastTD3 is designed for 1024 envs / 750M steps (~9M gradient steps). Our 5M run had 155K gradient steps — 57x fewer.

**Paper-scale run confirmed:** 1024 envs, batch=8192, 86M steps → **880 eval** (vs vanilla 749). FastTD3 wins on final score; vanilla is more sample-efficient.

**Lesson:** "Fast" algorithms are fast at scale, not at small scale. Match the paper's operating regime (1024+ envs, 50M+ steps) or use the simpler algorithm.

---

## C51 Support Range (V_min/V_max) Is a Critical Hyperparameter

**Problem:** Default V_min=-10, V_max=10 on CheetahRun: Q values capped at ~9, returns at 154.

**Root cause:** C51 represents Q as categorical distribution over [V_min, V_max]. True Q beyond V_max → all mass piles at boundary.

**Fix:** Set V_min/V_max to cover actual Q range:
- Q ≈ avg_reward_per_step / (1 - gamma)
- CheetahRun: reward ~0.8, gamma=0.99 → Q ≈ 80 → V_max=150
- HumanoidRun: reward ~0.2, gamma=0.99 → Q ≈ 20 → V_max=50

**Lesson:** Distributional Q is hard-bounded by [V_min, V_max]. Get it wrong and the critic is blind beyond the boundary.

---

## C51 Cross-Entropy NaN — `-inf * 0 = NaN`

**Problem:** Even with env NaN guards, FastTD3 still NaN'd. `log_softmax` returns `-inf` for near-zero probability atoms. `projected * log_probs` → `0 * -inf = NaN`.

**Fix:** `jnp.maximum(log_softmax(...), -30.0)` — clamp log probs to -30 (`exp(-30) ≈ 1e-13`).

**Lesson:** Any cross-entropy loss using `log_softmax` needs a floor clamp. The `-inf * 0 = NaN` trap is silent.

**Codified (2026-04-25, B5.7 commit):** the clamp + sum pattern was duplicated 8 times across `fast_td3.py`, `fast_sac.py`, `flash_sac.py`. Now centralized in `jax_rl/utils/distributional.py` as `safe_log_softmax(logits, min_log=-30.0)` and `cross_entropy_categorical(target_probs, logits)`. Use the helpers; don't re-inline the clamp.

---

## SAC Variants Can't Match FastTD3 on Low-Dim Tasks — But That's OK

Exhaustive testing on CheetahRun (6-dim actions):
- FastSAC (C51, α=1.0): 375
- FastSAC OG (C51, α=0.001, max_σ=1.0): 447 peak
- FastDSAC + β: **567** — best SAC variant
- FastTD3: **880** — untouchable on this task

**Why:** 6 action dims — deterministic TD3 + Gaussian noise explores efficiently. SAC's entropy machinery is overhead.

**Two different "FastSAC" papers exist:**
- **Seo et al. 2025** (Berkeley) — C51, real robots
- **FastDSAC 2026** (different group) — Gaussian critic + DEM, HumanoidBench

Both benchmark on 21+ dim tasks. Neither claims SAC beats FastTD3 on low-dim. Test on HumanoidRun where structured exploration matters.

---

## Always Run the Simple Baseline Before the Fancy Version

Spent a full day on FastSAC/FastDSAC. Best: 582 on CheetahRun in 108 min. Then vanilla SAC: **771 in 8 minutes.**

**Lesson:** Run the simple baseline first. A baseline that takes minutes can save hours of wasted tuning.

---

## Verify Configs Against Source Code, Not Paper Text

Implemented FastTD3/FastSAC from paper text. Found **7+ critical mismatches** vs holosoma source code:
- tau=0.125 (25x different from standard 0.005)
- Tapered 3-layer networks (512→256→128 actor, 768→384→192 critic)
- SiLU activation (not mentioned in paper)
- 101 C51 atoms (not specified)
- Policy delay=4 for FastSAC (only in source code)
- No gradient clipping (max_grad_norm=0)
- No LR schedule (constant LR)

**The tau=0.125 NaN proves it:** FastTD3 hit 395 eval then NaN'd at 31M steps with tau=0.005. With 8 gradient steps per env step, tau=0.005 can't track the fast-changing online network.

**Lesson:** Papers omit critical implementation details. Always check the source code repo.

---

## Read the Whole Recipe, Not Just the Key Ingredients

Implemented FastSAC with α_init=0.001 and max_σ=1.0 only. Missed:
- **gamma=0.97** (we used 0.99)
- **AdamW with β2=0.95** (we used Adam β2=0.999)
- **weight_decay=0.001** (we had none)

**Lesson:** Extract ALL hyperparameters into a table before implementing. Missing "minor" params like gamma and optimizer settings can completely change behavior.

---

## FastDSAC: Paper Says "Gaussian NLL" but Code Uses Huber Loss

Implemented Gaussian NLL because the paper says "Gaussian distributional critic." NaN'd. Downloaded source code — actual loss is **Huber-based** (delta=50) with clamped ratio weighting. No `1/variance` anywhere.

**The paper's actual loss (from source code):**
```
ratio = clamp(mean_std² / (q_std_detach² + bias), 0.1, 10)
loss = mean(ratio * huber(q_mean, target_q, delta=50) + ...)
```

**Why stable:** Huber caps error at linear growth. Ratio clamped [0.1, 10]. No 1/variance division.

**Scaling from 128→1024 envs required:**
1. Buffer scaling: 51K→400K (proportional to env count)
2. Inf guard: `isinf()` alongside `isnan()` — MJX produces Inf from velocity overflow

**Key results:**
- 128 envs: 490 peak eval (no NaN)
- 1024 envs (400K buffer): 316 peak, survived 54.3M steps with Inf guard

**Lessons:**
1. "Gaussian distributional critic" does NOT mean Gaussian NLL loss — read the source.
2. Scaling envs requires proportional buffer scaling, not just UTD matching.
3. Guard against BOTH NaN AND Inf from physics engines.
4. Stress test edge cases directly (inject Inf/NaN) — 30 seconds vs 53M steps.
5. Match paper setup EXACTLY first (128 envs), then scale one variable at a time.

---

## FlashSAC Port: Four Porting Gotchas

**Context:** Porting FlashSAC from PyTorch to JAX/Flax. Five spec review passes caught 22 issues — three would have silently broken training.

### 1. Weight norm axis: Flax ≠ PyTorch kernel layout

PyTorch `nn.Linear` weight: `(output_dim, input_dim)` → `F.normalize(w, dim=-1)` normalizes each row.
Flax `nn.Dense` kernel: `(input_dim, output_dim)` → must normalize along `axis=0` (not `axis=-1`).

Getting this wrong produces unit-norm INPUT features instead of unit-norm OUTPUT neurons. Training runs without error but learns garbage. Always verify kernel layout conventions when porting.

### 2. Target BN stats are NOT copied from online

FlashSAC's target critics run with `train=True` to maintain their OWN BatchNorm running statistics. The Polyak EMA only updates learned parameters (kernels, BN scale/bias), NOT running buffers (mean/var). Copying online batch_stats to target would overwrite the target's independent statistics — subtle behavioral difference.

In PyTorch, `ema_update_parameters()` naturally skips buffers. In JAX/Flax, where params and batch_stats are separate pytrees, you must be deliberate about which pytrees get Polyak'd.

### 3. Truncation handling — Brax convention is correct

**Fixed 2026-04-12.** FastSAC/FastTD3/FlashSAC originally used `effective_done = max(done, truncation)` to zero bootstrap at truncation, with no TD mask on the loss. This was wrong given the wrapper semantics:

- `batch["done"]` from `EpisodeWrapper` = `terminated OR truncated`
- `batch["truncation"]` = `truncated AND NOT terminated`

Since `done` already includes timeouts, the `max(done, truncation)` line was a no-op. The real bug was the missing loss mask: on a pure-timeout row, the target is `r + 0` (reward only, no bootstrap because done=1), and the cross-entropy loss trains on it, teaching the network that `Q = r` at timeout steps. On long-horizon locomotion (Go2, Humanoid) this causes systematic Q underestimation proportional to (timeout rate × true tail value).

**Correct convention (Brax, matches SAC/TD3):**
- Target: `r + gamma * (1 - done) * V_next` — zero bootstrap on both term and timeout (next_obs is corrupted by AutoReset either way).
- Loss mask: `mask = 1 - truncation` — drop pure-timeout rows so the `r`-only target doesn't train the network. Term rows still contribute (their `r`-only target is genuinely correct).

This mirrors the PPO GAE fix in `.context/lessons/ppo.md` (zero deltas at timeouts).

**Related:** the `handle_truncation` constructor arg on all 5 off-policy algos is stored on `self` but never read. The real switch is `cfg.handle_truncation` in the training loop — when False, it stores zeros for truncation, making the mask a no-op.

### 4. BatchNorm running stats must follow the training state

In JAX/Flax, BatchNorm running stats (`mean`, `var`) are separate from learned params. When `select_action` is used for eval, it needs the *current* running stats — not the ones from `init()`. If batch_stats are captured once in a closure or stored as a default, eval will use all-zeros running stats → garbage actions.

Symptom: online return 686 but eval return 26. Fix: explicitly update the batch_stats reference from `training_state.actor_batch_stats` before every eval call. This is a Flax-specific gotcha — PyTorch BatchNorm stores running stats as buffers on the module, so they update in-place automatically.
