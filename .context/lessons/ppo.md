# PPO Lessons

---

## Match Reference Implementation EXACTLY Before Investigating (2026-03-24)

**What happened:** Our PPO had 2x sample efficiency gap vs Brax PPO on the same env. We were about to launch multiple 100M-step training runs to "investigate" when we should have just diffed the code.

**Root cause (found in 5 minutes of code reading):**
1. **Value loss scaling:** Brax uses `0.5 * 0.5 = 0.25x`, we used `1.0x` → our critic gradients were 4x larger
2. **Advantage normalization scope:** Brax normalizes over the full batch, we normalized per-minibatch → higher variance gradients

**Lesson:** When a reference implementation outperforms yours, **diff the code first**. Don't throw compute at it. Every line that differs is a potential cause. "Match everything" means match the MATH, not just the config values.

---

## Entropy Coefficient Tuning — When Exploration Hurts

**What happened:** CheetahRun with `entropy_coef=0.001` — entropy grew unboundedly (9→16+), returns plateaued at ~248.

**Root cause:** With unconstrained `log_std` (clipped to [-5, 2]) and a positive entropy bonus, the optimizer pushes `log_std` toward the upper bound to maximize entropy. The policy becomes too noisy to learn anything useful.

**The fix:** Set `entropy_coef=0.0` for CheetahRun. Returns jumped to 503 at 10M steps.

**Diagnostic signals:**
- Entropy growing monotonically = entropy bonus is dominating task reward
- High KL + high clip fraction = policy changing too aggressively
- PLoss near 0 at end = policy gradient exhausted (needs LR annealing)

**Lesson:** Entropy bonus is environment-dependent. Simple tasks (CartpoleBalance) benefit from exploration (`entropy_coef=0.01`). Complex continuous control (CheetahRun) often needs `entropy_coef=0.0` to avoid the entropy-maximization trap.

---

## LR Annealing for Continuous Control

**Problem:** With constant LR (3e-4), policy loss converges to ~0 and clip fraction stays high (0.27-0.35) late in training. The policy keeps making large updates even when it should be fine-tuning.

**Fix:** Linear LR schedule annealing to 0 over total gradient steps:
```python
total_gradient_steps = num_iterations * num_updates_per_batch * num_epochs * num_minibatches
lr_schedule = optax.linear_schedule(lr, 0.0, total_gradient_steps)
optimizer = optax.adam(lr_schedule)  # optionally chain with clip_by_global_norm
```

**Lesson:** LR annealing lets the policy explore broadly early and fine-tune late. Standard practice for PPO on continuous control — Brax does this too.

---

## State-Independent vs State-Dependent log_std

**Problem:** HumanoidRun (21-dim actions) NaN'd repeatedly with state-dependent std (`nn.Dense` mapping features → log_std).

**Root cause:** Dense layer produces unpredictable initial log_std values. Outlier observations spike log_std → entropy explodes → NaN. With 21 action dims, the entropy is already ~30 nats from dimensionality alone (`0.5 * 21 * (1 + log(2pi)) ≈ 29.8`).

**Fix:** State-independent log_std (`nn.Param`), matching Brax PPO:
```python
log_std = self.param('log_std', nn.initializers.constant(jnp.log(init_noise_std)), (action_dim,))
log_std = jnp.broadcast_to(log_std, mean.shape)
log_std = jnp.clip(log_std, log_std_min, log_std_max)
```

**Lesson:** For PPO, state-independent std is more stable. Save state-dependent std for SAC where maximum entropy is the objective. Always clip log_std regardless of mode.

---

## Minibatch Count, Not Size, Is the Right Parameterization

**Problem:** Hardcoded `minibatch_size = min(2048, samples_per_iter)`. When we increased `num_steps` from 30 to 480, `samples_per_iter` went from 61k to 983k but minibatch_size stayed at 2048 — `num_minibatches` went from 30 to 480. With 16 epochs, that's 7,680 gradient steps per iteration — Brax uses 512.

**The coupling:** `grad_steps = num_minibatches x num_epochs`. Fixing `minibatch_size` means `num_minibatches` scales with data volume. Fixing `num_minibatches` keeps gradient steps constant regardless of data volume.

**Fix:** Parameterize by `num_minibatches` (default 32, matching Brax). Derive `minibatch_size = samples_per_iter // num_minibatches`.

**Lesson:** When comparing against a reference implementation, match the *structure* (what's fixed vs derived), not just the numbers.

---

## Batch-Level vs Per-Minibatch Advantage Normalization

Brax normalizes advantages over the entire batch (~983k samples) before splitting into minibatches. We were normalizing per-minibatch (~2k samples), giving noisy mean/std estimates.

With 30k+ samples per minibatch (after the fix), the difference is small. But for smaller setups or early in training, batch-level normalization is strictly better.

---

## Reshape Crash on Non-Divisible Batches

`obs[perm].reshape(num_minibatches, minibatch_size, -1)` silently assumes `N == num_minibatches * minibatch_size`. If not, it crashes.

**Fix:** Truncate the permutation: `perm = perm[:num_minibatches * minibatch_size]`.

**Lesson:** Always handle the case where data doesn't divide evenly.

---

## HumanoidRun Diagnostic Signals

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| Clip fraction | 0.05-0.20 | 0.50-0.80 (policy overshooting) |
| Entropy | Slowly decreasing | Growing monotonically (entropy bonus dominating) |
| VLoss | Stable, tracks returns | Collapsing to 0.00 (overfitting to batch) |
| KL | < 0.1 | Spikes > 1.0 (imminent NaN) |
| log_std | Slowly decreasing from ~0 | Growing or stuck at clip bounds |

---

## Interleaved Collect→Update (num_updates_per_batch)

**Problem:** Long rollout followed by many epochs → policy diverges from collection policy. Short rollouts + many epochs → overtrain on tiny batches → collapse.

**Brax's approach:** Instead of `collect(long) → update(many epochs)`:
```
for _ in range(num_updates_per_batch):  # 16 cycles
    data = collect(20 steps)             # fresh data each time
    sgd_update(data, num_epochs=1)       # just 1 epoch
```

**Key numbers (HumanoidRun):**

| Config | Grad steps/iter | Data freshness | Result |
|--------|----------------|----------------|--------|
| num_steps=480, epochs=4, updates=1 | 128 | 480 steps stale | Returns 7.6 (slow) |
| num_steps=20, epochs=4, updates=1 | 128 | Fresh but overtrained | Collapsed (entropy → -1.25) |
| num_steps=20, epochs=1, updates=16 | 512 | Fresh each cycle | Brax reference config |

**Lesson:** The right granularity is many short collect→update cycles, not fewer long ones.

---

## VLoss Oscillation from Synchronized Episode Resets

**Problem:** VLoss alternated between ~0.35 and ~61.5 every other iteration.

**Root cause:** The training wrappers reset all envs at the same timestep (episode_length=1000). The value function sees completely different return distributions on boundary vs mid-episode iterations.

**Why entropy goes negative:** With tanh-squash + single-sample entropy estimate: if a large gradient update briefly collapses std, the sample falls near the mode where log_prob is very high. Negative entropy is the artifact, not real divergence.

**Mitigation:** Brax's `num_updates_per_batch=16, num_steps=20` interleaves collections so episode phases are desynchronized.

---

## Truncation Handling in Auto-Reset Environments

**Problem:** HumanoidRun returns stuck at ~7.6 at 60M steps. Turned out to be a GAE correctness bug, not HPs.

**Root cause:** `AutoResetWrapper` auto-resets on timeout. After reset, `env_state.obs` is the **reset observation**, not the terminal observation. Our GAE used `V(reset_obs)` as bootstrap for truncated transitions → systematic negative bias.

**Final fix (matching Brax's `compute_gae`):**
```python
truncation_mask = 1.0 - truncations
termination = dones * (1.0 - truncations)
deltas = rewards + gamma * (1 - termination) * next_values - values
deltas = deltas * truncation_mask  # zero at truncation
gae = deltas[t] + gamma * gae_lambda * (1 - termination[t]) * truncation_mask[t] * gae
```

**Lesson:** When auto-reset corrupts the post-terminal observation, zero out the entire TD error at that step rather than trying to correct the bootstrap value.

---

## Tanh Squashing Requires Matched Entropy Computation

**Problem:** Enabled `squash=True` to match Brax. Immediate blowup — KL 296k, entropy 33→71, clip fraction 1.0.

**Root cause:** Raw Gaussian entropy was ~30 nats at `log_std=0`. Tanh-squashed entropy is ~14 nats. The entropy bonus on the inflated value pushed `log_std` to the clip maximum.

**Fix — single-sample entropy estimate (matching Brax):**
```python
def entropy_gaussian(log_std, mean=None, key=None, squash=False):
    if squash:
        std = jnp.exp(log_std)
        dist = distrax.Normal(loc=mean, scale=std)
        raw_actions = dist.sample(seed=key)
        log_prob = dist.log_prob(raw_actions)
        log_prob -= jnp.log(1 - jnp.tanh(raw_actions) ** 2 + 1e-6)
        return -log_prob.sum(axis=-1)
    else:
        return (0.5 * (jnp.log(2 * jnp.pi) + 1 + 2 * log_std)).sum(axis=-1)
```

**Lesson:** Entropy and log_prob must always be computed in the same space. If actions are tanh-squashed, entropy must include the Jacobian correction.

---

## Batch Volume Controls How Many SGD Epochs Are Safe

16 epochs over 61k samples = massive overtraining. Brax uses 16 epochs over 983k samples.

**Rule of thumb:** More SGD epochs require proportionally more data. If you increase epochs, increase batch size.

---

## Structural Parity, Not Just HPs

| What we thought was equivalent | What was actually different |
|-------------------------------|---------------------------|
| `num_updates_per_batch=16, num_epochs=1` | Brax: 1 collection → 16 epochs. Ours: 16 fresh collections → 1 epoch each |
| `num_steps=30` with 2048 envs (61k samples) | Brax collects 983k samples |
| `entropy_coef=1e-2` with raw Gaussian entropy | Brax uses tanh-corrected entropy (~14 nats vs ~30 nats) |

**Lesson:** Audit the *structure* (data flow, loss computation, distribution math), not just the *numbers*.

---

## Know When PPO Is the Wrong Tool

**Problem:** HumanoidRun returns stuck at ~7-10 after 60M steps. Brax PPO baseline got the same score.

**Literature confirms:** PPO on humanoid_run is a known failure case — the signal was there in the original dm_control paper from 2018.

**When to suspect algorithm limits vs implementation bugs:**
- Healthy metrics but flat returns → likely algorithm limits
- Reference implementation gets the same score → definitely algorithm limits
- Unhealthy metrics (KL spikes, entropy explosion, NaN) → implementation bug

---

## PPO Validation Summary

| Environment | Our Result | Target | Status |
|-------------|-----------|--------|--------|
| CartpoleBalance | >=995 @ 1M | >=950 @ 10M | PASS |
| CheetahRun | 826 @ 20M | >=700 @ 60M | PASS |
| HumanoidRun | ~7-10 @ 60M | matches Brax (~8-10) | PASS (algorithm limit) |

---

## CheetahRun Hyperparameter Journey

| Config | Return | Notes |
|--------|--------|-------|
| entropy_coef=0.001, 10M steps | ~248 | Entropy grew to 16+, policy too noisy |
| entropy_coef=0.0, 10M steps | ~503 | Entropy controlled, policy learning |
| entropy_coef=0.0, 20M steps | ~666 | More steps helped |
| + LR annealing, 20M steps | ~615 | Annealed too aggressively |
| Preset HPs (2048 envs, lr=1e-3, 16 epochs, reward_scaling=10) | **826** | Passed target at 20M steps |

---

## Remaining Performance Gap (RESOLVED)

*Gap closed. Our fast PPO beats Brax PPO on Go1 (27.3 vs 18 at 28.5M steps) and achieves eval 233 on Go2. The key fixes were value loss 0.25x scaling + full-batch advantage normalization, not the structural differences below.*

Historical notes:
1. Per-minibatch GAE recomputation — didn't matter once loss scaling was fixed.
2. State-dependent std — didn't matter for final performance.
