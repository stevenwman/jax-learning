# Off-Policy Lessons (SAC / TD3)

---

## Online Obs Normalization Is Incompatible with Off-Policy Replay

**Problem:** SAC on WalkerWalk diverged immediately — Q values to -15,000, alpha to 19+.

**Root cause:** At env reset, running variance ≈ 0 for some obs dims. Normalizing divides by `sqrt(0) + 1e-8` → ±2 billion stored in replay buffer. Old transitions become stale as statistics drift.

**The diagnostic trail:**
1. Q values diverging monotonically → TD targets corrupted
2. TD target = -47 million on first gradient step → next_obs corrupted
3. `next_obs` range in buffer: ±2 billion → normalization exploding
4. `norm_state` variance ≈ 0 after identical reset obs → division by near-zero

**Why PPO is immune:** On-policy — obs normalized and consumed in the same iteration.

**Two valid approaches for off-policy:**
1. Raw obs in buffer + normalize at sample time with `--obs-norm` (preferred — SAC Go2: eval 139 vs 97 without)
2. No normalization + Q LayerNorm only (simpler, but lower performance)

**Critical rule: NEVER normalize before buffer storage.**

**Update (2026-03-26):** Our `--obs-norm` flag implements approach #1 correctly. The original lesson "must NOT use online obs normalization" was misleading — the rule is: must NOT normalize BEFORE storing in the buffer.

---

## SAC Validation Results

| Environment | SAC Result | PPO Result | SAC Steps | PPO Steps | Notes |
|-------------|-----------|------------|-----------|-----------|-------|
| WalkerWalk | 975 avg, 995 max | 833 avg, 992 max | 5M | 60M | 12x more sample-efficient |
| HumanoidRun | 426 (vanilla SAC, 20M) | ~10 at 60M | 20M | 60M | SAC 42x PPO's best |
| HumanoidRun | **892** (FastSAC, 100M) | ~10 at 60M | 100M | 60M | SOTA for our framework |

---

## TD3 NaN on HumanoidRun — Gradient Clipping Is Not Optional

**Problem:** TD3 NaN'd at ~700k steps. Q1 spiked from 1.17 → 9.54 → NaN in a single log interval.

**Root cause:** No gradient clipping. 67-dim obs + 21-dim actions → Q-network gradients spike. One large gradient step destabilizes the network → NaN cascades.

**Why SAC was immune:** Entropy regularization (`alpha * log_prob`) keeps the actor's output distribution smooth.

**Fix:** `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))` + Q LayerNorm.

**Lesson:** Gradient clipping is essential for TD3 on high-dimensional tasks. SAC's entropy provides implicit stability that TD3 lacks.

---

## Replay Ratio Must Scale with num_envs

**Problem:** TD3 with `grad_updates_per_step=1` and 128 parallel envs — Q stuck near 0.

**Root cause:** Collecting 128 samples per step but only training on 256 (one batch). Model can't keep up.

**Fix:** Bumped `grad_updates_per_step` from 1 to 4 for 128-env preset.

**Lesson:** A 1:1 ratio for 128 envs means 1 gradient step per 128 samples. Use 4-8 gradient steps per env step for 128+ envs.

---

## TD3's Exploration Limits on High-Dimensional Tasks

TD3 reached 749 on CheetahRun (6-dim) but only 4.3 on HumanoidRun (21-dim).

**Why:** TD3 explores via additive Gaussian noise. In 21 dimensions, random perturbations almost never produce coordinated movements. SAC's entropy maximization is structured exploration.

**When to use which:**
- **TD3**: ≤10 action dims, faster wall-clock. Good for manipulation, simple locomotion.
- **SAC**: >10 action dims, tasks requiring coordinated multi-joint movement.
- **FastTD3**: C51 + large batches compensate for exploration weakness. 665 on HumanoidRun despite being deterministic.

---

## Target Entropy = 0 Is Not Optional for SAC at Scale

**Problem:** FastDSAC with target_entropy=-3.0: alpha collapsed to 0.012, eval peaked at 401. With target_entropy=0.0: alpha stayed at 0.65, eval 509.

**Root cause:** With negative target, the policy easily achieves that entropy level → alpha decays to zero → exploration pressure lost.

**When to use negative target entropy:** Standard SAC at small scale (128 envs, 5M steps) — the classic `-dim(A)` heuristic works.

**When to use target_entropy=0:** Large-scale training (1024 envs, 100M+ steps), high-dim action spaces, any SAC variant with DEM.

---

## AdamW Requires `params` in optimizer.update()

`optax.adamw` needs `optimizer.update(grads, opt_state, params=params)`. `optax.adam` doesn't. Weight decay is applied to the params themselves, not just the gradients.

---

## Optimizer Decoupling — Keep Algorithms Pure

PPO shouldn't own optimizer construction. Optimizers are external concerns.

**Before:** PPO internally created optimizers from config fields.
**After:** PPO takes `actor_optimizer` and `critic_optimizer` as constructor args. Train script builds them.

**Lesson:** Algorithms define *what* they optimize, not *how*. Makes swapping optimizers trivial.

---

## Staged Rewards Need Longer Training Budgets

**Problem:** SAC on PandaPickCube at 2M steps learned approach (reward ~604) but never lifted the cube. Box z stayed at 0.03 (table surface).

**Root cause:** PandaPickCube uses a gated reward — `box_target` reward (lift to target) only activates after `reached_box` flag (gripper within 1.2cm of box). At 2M steps the policy learned to reach the box (gripper_box reward) but hadn't explored the grasp-lift sequence enough to discover the gated reward.

**Fix:** 10M steps. Cube lifted to z=0.25, reward ~1386.

**Lesson:** When rewards are staged/gated (reward B only available after achieving condition A), training budget must be long enough to discover the full sequence. The first plateau is not convergence — it's the policy stalling at the first reward stage.
