# Lessons Learned - JAX RL Framework

Active project lessons — debugging victories, algorithm pitfalls, and design decisions.
JAX/Flax fundamentals moved to LEARNER_LESSONS.md. Superseded content in outdated_lessons.md.

---

## Session 4: CheetahRun, Performance, & Tooling (Mar 2026)

### `jax.lax.scan` vs Python Loops — 542x Speedup

Benchmarked three PPO implementations on the same update workload:

| Variant | Per-update time | Speedup |
|---------|----------------|---------|
| `ppo.py` (Python loops, no JIT) | 2.712s | 1x |
| `ppo_jit.py` (JIT closures, Python epoch loops) | 0.089s | 30x |
| `ppo_scan.py` (JIT + `jax.lax.scan` for epochs) | 0.005s | 542x |

**Why scan matters:** For small MLPs (64x64), 95% of each call was Python-XLA dispatch overhead. `jax.lax.scan` compiles the entire epoch loop into a single XLA program — no Python roundtrips. The JIT-only version still had Python loops for epochs/minibatches, causing repeated dispatch.

**Lesson:** If your model is small and your loop body is fast, Python loop overhead dominates. `scan` eliminates it entirely.

---

### Entropy Coefficient Tuning — When Exploration Hurts

**What happened:** CheetahRun with `entropy_coef=0.001` — entropy grew unboundedly (9→16+), returns plateaued at ~248.

**Root cause:** With unconstrained `log_std` (clipped to [-5, 2]) and a positive entropy bonus, the optimizer pushes `log_std` toward the upper bound to maximize entropy. The policy becomes too noisy to learn anything useful.

**The fix:** Set `entropy_coef=0.0` for CheetahRun. Returns jumped to 503 at 10M steps.

**Diagnostic signals:**
- Entropy growing monotonically = entropy bonus is dominating task reward
- High KL + high clip fraction = policy changing too aggressively
- PLoss near 0 at end = policy gradient exhausted (needs LR annealing)

**Lesson:** Entropy bonus is environment-dependent. Simple tasks (CartpoleBalance) benefit from exploration (`entropy_coef=0.01`). Complex continuous control (CheetahRun) often needs `entropy_coef=0.0` to avoid the entropy-maximization trap.

---

### LR Annealing for Continuous Control

**Problem:** With constant LR (3e-4), policy loss converges to ~0 and clip fraction stays high (0.27-0.35) late in training. The policy keeps making large updates even when it should be fine-tuning.

**Fix:** Linear LR schedule annealing to 0 over total gradient steps:
```python
total_gradient_steps = num_iterations * num_updates_per_batch * num_epochs * num_minibatches
lr_schedule = optax.linear_schedule(lr, 0.0, total_gradient_steps)
optimizer = optax.adam(lr_schedule)  # optionally chain with clip_by_global_norm
```

**Lesson:** LR annealing lets the policy explore broadly early and fine-tune late. Standard practice for PPO on continuous control — Brax does this too.

---

### Optimizer Decoupling — Keep Algorithms Pure

PPO shouldn't own optimizer construction. Optimizers (LR schedule, grad clipping, future exotic optimizers like Muon) are external concerns.

**Before:** PPO internally created optimizers from config fields (`actor_lr`, `anneal_lr`, etc.)
**After:** PPO takes `actor_optimizer` and `critic_optimizer` as constructor args. Train script builds them.

**Lesson:** Same principle as dependency injection — algorithms define *what* they optimize, not *how*. Makes swapping optimizers trivial (Muon, Shampoo, etc.) without touching algorithm code.

---

### Orbax Checkpointing

- `ocp.StandardCheckpointer()` saves/restores arbitrary pytrees (works with Linen or NNX)
- **Must call `checkpointer.wait_until_finished()`** after save — otherwise process exit kills the async write thread ("cannot schedule new futures after shutdown")
- Save `meta.json` alongside checkpoint for config reconstruction (env name, hidden dims, obs/action dims)
- Checkpoint + metrics CSV + meta.json in timestamped directories: `checkpoints/{timestamp}_{env}_seed{seed}/`

---

### Video Recording — Two-Phase Approach

MuJoCo Playground's `env.render()` is CPU-side and can't be JIT'd. Solution:

1. **Phase 1 (GPU):** `jax.lax.scan` the rollout — fast, collects all states
2. **Phase 2 (CPU):** Render frames from saved states — slow but unavoidable

**Lesson:** Separate compute from rendering. The rollout itself is fast (~0.3s for 1000 steps including JIT). Rendering is the bottleneck (~50ms/frame).

---

### CheetahRun Hyperparameter Journey

| Config | Return | Notes |
|--------|--------|-------|
| entropy_coef=0.001, 10M steps | ~248 | Entropy grew to 16+, policy too noisy |
| entropy_coef=0.0, 10M steps | ~503 | Entropy controlled, policy learning |
| entropy_coef=0.0, 20M steps | ~666 | More steps helped, approaching target |
| + LR annealing, 20M steps | ~615 | Annealed too aggressively, policy froze late |
| Preset HPs (2048 envs, lr=1e-3, 16 epochs, reward_scaling=10) | **826** | Passed target at 20M steps |

Target was >=700 at 60M steps (MuJoCo Playground paper). Hit **826 at 20M** with proper HPs. Key changes: 2048 envs, shorter rollouts (30 steps), higher LR (1e-3), more epochs (16), reward scaling (10x).

---

## Session 5: HumanoidRun Debugging & Pipeline Robustness (Mar 2026)

### State-Independent vs State-Dependent log_std

**Problem:** HumanoidRun (21-dim actions) NaN'd repeatedly with state-dependent std (`nn.Dense` mapping features → log_std).

**Root cause:** Dense layer produces unpredictable initial log_std values. Outlier observations spike log_std → entropy explodes → NaN. With 21 action dims, the entropy is already ~30 nats from dimensionality alone (`0.5 * 21 * (1 + log(2pi)) ≈ 29.8`).

**Fix:** State-independent log_std (`nn.Param`), matching Brax PPO:
```python
# State-independent: single learned vector, same for all observations
log_std = self.param('log_std', nn.initializers.constant(jnp.log(init_noise_std)), (action_dim,))
log_std = jnp.broadcast_to(log_std, mean.shape)

# MUST clip in both modes — without this, entropy bonus pushes log_std → infinity
log_std = jnp.clip(log_std, log_std_min, log_std_max)
```

**Lesson:** For PPO, state-independent std is more stable. Save state-dependent std for SAC where maximum entropy is the objective. Always clip log_std regardless of mode.

---

### Minibatch Count, Not Size, Is the Right Parameterization

**Problem:** Hardcoded `minibatch_size = min(2048, samples_per_iter)`. When we increased `num_steps` from 30 to 480 (16x more data), `samples_per_iter` went from 61k to 983k. But minibatch_size stayed at 2048, so `num_minibatches` went from 30 to 480. With 16 epochs, that's 7,680 gradient steps per iteration — Brax uses 512.

**The coupling:** `grad_steps = num_minibatches x num_epochs`. Fixing `minibatch_size` means `num_minibatches` scales with data volume. Fixing `num_minibatches` keeps gradient steps constant regardless of data volume.

**Fix:** Parameterize by `num_minibatches` (default 32, matching Brax). Derive `minibatch_size = samples_per_iter // num_minibatches`.

**Lesson:** When comparing against a reference implementation, match the *structure* (what's fixed vs derived), not just the numbers. Brax fixes `num_minibatches=32` and `batch_size=1024`; we were fixing `minibatch_size=2048` which is a completely different parameterization.

---

### Batch-Level vs Per-Minibatch Advantage Normalization

Brax normalizes advantages over the entire batch (~983k samples) before splitting into minibatches. We were normalizing per-minibatch (~2k samples), giving noisy mean/std estimates.

With 30k+ samples per minibatch (after the fix), the difference is small. But for smaller setups or early in training, batch-level normalization is strictly better.

---

### Reshape Crash on Non-Divisible Batches

`obs[perm].reshape(num_minibatches, minibatch_size, -1)` silently assumes `N == num_minibatches * minibatch_size`. If not (e.g., 5000 samples with minibatch_size=1024 → 4 batches of 1024 = 4096 != 5000), it crashes.

**Fix:** Truncate the permutation: `perm = perm[:num_minibatches * minibatch_size]`.

**Lesson:** Always handle the case where data doesn't divide evenly. It works by coincidence for tuned presets but breaks for arbitrary configs.

---

### HumanoidRun Diagnostic Signals

Added `log_std_mean`, `log_std_min`, `log_std_max` to PPO metrics. Key observations:

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| Clip fraction | 0.05-0.20 | 0.50-0.80 (policy overshooting) |
| Entropy | Slowly decreasing | Growing monotonically (entropy bonus dominating) |
| VLoss | Stable, tracks returns | Collapsing to 0.00 (overfitting to batch) |
| KL | < 0.1 | Spikes > 1.0 (imminent NaN) |
| log_std | Slowly decreasing from ~0 | Growing or stuck at clip bounds |

---

### Interleaved Collect→Update (num_updates_per_batch)

**Problem:** With one long rollout (480 steps) followed by 4 epochs of updates, the policy used for collection diverges significantly from the policy being optimized. With short rollouts (20 steps) + 4 epochs, we massively overtrain on tiny batches — policy collapsed (entropy 25 → -1.25, clip fraction → 0.000).

**Brax's approach:** Instead of `collect(long) → update(many epochs)`, Brax does:
```
for _ in range(num_updates_per_batch):  # 16 cycles
    data = collect(20 steps)             # fresh data each time
    sgd_update(data, num_epochs=1)       # just 1 epoch
```

**Why it matters:**
- Policy stays close to the data-collecting policy (lower staleness)
- Episode resets naturally stagger across cycles (fixes VLoss oscillation from synchronized resets)
- Same total gradient steps (16 x 1 x 32 = 512) but each uses on-policy data
- `num_updates_per_batch=1, num_epochs=4` recovers our old behavior

**Key numbers (HumanoidRun):**

| Config | Grad steps/iter | Data freshness | Result |
|--------|----------------|----------------|--------|
| num_steps=480, epochs=4, updates=1 | 128 | 480 steps stale | Returns 7.6 (slow) |
| num_steps=20, epochs=4, updates=1 | 128 | Fresh but overtrained | Collapsed (entropy → -1.25) |
| num_steps=20, epochs=1, updates=16 | 512 | Fresh each cycle | Brax reference config |

**Lesson:** The right granularity is many short collect→update cycles, not fewer long ones. `num_updates_per_batch` generalizes our pipeline — setting it to 1 recovers the old behavior.

---

### VLoss Oscillation from Synchronized Episode Resets

**Problem:** VLoss alternated between ~0.35 and ~61.5 every other iteration. Even iterations (episode boundaries) had huge VLoss; odd iterations had near-zero VLoss.

**Root cause:** `wrap_for_brax_training` resets all envs at the same timestep (episode_length=1000). With 4096 envs all resetting simultaneously, the value function sees completely different return distributions on boundary iterations vs mid-episode iterations. It overfits to mid-episode predictions, then gets blindsided at boundaries.

**How Brax avoids this:** With `num_updates_per_batch=16` and `num_steps=20`, updates happen every 20 steps. After the first episode completes, the 16 sequential collection phases within an iteration mean envs are at different points in their episodes across update cycles.

**Lesson:** Synchronized resets + long rollouts = VLoss instability. Interleaved short rollouts naturally desynchronize episode phases.

---

### Truncation Handling in Auto-Reset Environments (Matching Brax)

**Problem:** HumanoidRun returns stuck at ~7.6 at 60M steps (reference: 50-200). Multiple HP tuning attempts couldn't close the gap. Turned out to be a GAE correctness bug, not a hyperparameter issue.

**Root cause:** `wrap_for_brax_training` auto-resets envs on episode timeout. After reset, `env_state.obs` is the **reset observation** (humanoid standing still), not the terminal observation (humanoid running). Our GAE used `values[t+1] = V(reset_obs)` as the bootstrap for truncated transitions. Since V(reset) << V(running), this created a systematic negative bias.

**Failed attempts:**
1. **Reward adjustment** (`gamma * V(s_t) * truncation`): massive synchronized reward spikes → NaN at iter 9
2. **Bootstrap correction + two-flag propagation** (`next_values = where(trunc, values, next_values)`): The V(s_t) bootstrap approximation grows biased as V improves, and two separate done flags for bootstrap vs propagation is error-prone

**Final fix (matching `brax.training.agents.ppo.losses.compute_gae`):**
```python
truncation_mask = 1.0 - truncations  # 0 at truncation, 1 otherwise
termination = dones * (1.0 - truncations)  # 1 only at true terminals

# TD errors: zero out entirely at truncation steps
deltas = rewards + gamma * (1 - termination) * next_values - values
deltas = deltas * truncation_mask  # zero at truncation

# Propagation: stop at both truncation and true terminal
gae = deltas[t] + gamma * gae_lambda * (1 - termination[t]) * truncation_mask[t] * gae
```

**What this achieves:**
- **Advantage at truncation = 0**: no gradient signal for policy at that step
- **Value target = V(s_t)**: return = A + V = 0 + V(s_t), so no critic gradient either
- **Propagation stops**: next episode's advantages don't leak in

You lose one transition of training signal per episode (0.1% with 1000-step episodes), but the value function stays unbiased. This is strictly better than trying to approximate V(terminal_obs) with V(s_t).

**Lesson:** When auto-reset environments corrupt the post-terminal observation, the cleanest fix is to zero out the entire TD error at that step rather than trying to correct the bootstrap value. Matches Brax's approach and avoids subtle bugs from splitting done flags.

---

## Session 6: Brax Parity Audit & Tanh Squashing (Mar 2026)

### Matching a Reference Implementation Requires Structural Parity, Not Just HPs

**Problem:** Copied Brax's hyperparameters (num_envs, num_steps, epochs, etc.) but training still failed or underperformed. Multiple runs NaN'd or plateaued at returns ~5 vs Brax's 50-200.

**Root cause:** Hyperparameters only work correctly within the structural context they were tuned for. Key structural differences that made the same HPs behave differently:

| What we thought was equivalent | What was actually different |
|-------------------------------|---------------------------|
| `num_updates_per_batch=16, num_epochs=1` | Brax: 1 collection → 16 epochs. Ours: 16 fresh collections → 1 epoch each |
| `num_steps=30` with 2048 envs (61k samples) | Brax collects `batch_size × num_minibatches × unroll = 1024 × 32 × 30 = 983k` samples |
| `entropy_coef=1e-2` with raw Gaussian entropy | Brax uses tanh-corrected entropy (~14 nats). Raw Gaussian entropy was ~30 nats — 2x the bonus |
| `squash=False` + `jnp.clip(action, -1, 1)` | Brax uses `tanh_normal` with proper Jacobian correction in log_prob and entropy |

**Lesson:** When matching a reference implementation, audit the *structure* (data flow, loss computation, distribution math), not just the *numbers* (learning rate, batch size, epochs). The same HP value can have completely different effects in a different structural context.

---

### Tanh Squashing Requires Matched Entropy Computation

**Problem:** Enabled `squash=True` to match Brax's `tanh_normal`. Immediate blowup — KL 296k, entropy exploding from 33 to 71, clip fraction 1.0.

**Root cause:** `entropy_gaussian(log_std)` computed raw Gaussian entropy: `0.5 * (log(2π) + 1 + 2*log_std)` summed over 21 dims = ~30 nats at `log_std=0`. But the actual entropy of the tanh-squashed distribution is ~14 nats (tanh compresses the distribution). The `entropy_coef=1e-2` bonus on the inflated value pushed `log_std` to the clip maximum.

**Fix — single-sample entropy estimate (matching Brax):**
```python
def entropy_gaussian(log_std, mean=None, key=None, squash=False):
    if squash:
        std = jnp.exp(log_std)
        dist = distrax.Normal(loc=mean, scale=std)
        raw_actions = dist.sample(seed=key)
        log_prob = dist.log_prob(raw_actions)
        # Tanh Jacobian correction
        log_prob -= jnp.log(1 - jnp.tanh(raw_actions) ** 2 + 1e-6)
        return -log_prob.sum(axis=-1)  # entropy = -E[log p(x)]
    else:
        # Closed-form Gaussian entropy
        return (0.5 * (jnp.log(2 * jnp.pi) + 1 + 2 * log_std)).sum(axis=-1)
```

**Lesson:** Entropy and log_prob must always be computed in the same space. If actions are tanh-squashed, entropy must include the Jacobian correction `log(1 - tanh(x)²)`. Using pre-transform entropy with post-transform actions is a guaranteed blowup.

---

### Batch Volume Controls How Many SGD Epochs Are Safe

**Problem:** With `num_steps=30, num_envs=2048` (61k samples) and `num_epochs=16`, entropy exploded and clip fraction hit 1.0.

**Root cause:** 16 epochs of SGD over 61k samples = massive overtraining. The policy changed so much that importance sampling ratios blew up. Brax uses 16 epochs too, but over 983k samples — 16x more data per epoch.

**Fix:** `num_steps=480` → `2048 × 480 = 983k` samples, matching Brax's collection volume. `minibatch_size = 983k / 32 = 30,720` also matches.

**Rule of thumb:** More SGD epochs require proportionally more data. If you increase epochs, increase batch size. If you can't increase batch size, reduce epochs.

---

### Remaining Performance Gap

Run 15 (all parity fixes) achieved stable training but only 5.3 avg return at 60M steps (target: 50-200). Remaining structural differences from Brax:

1. **Per-minibatch GAE recomputation** — Brax re-evaluates V(s) from current params inside each minibatch loss. Our GAE uses stored values from collection time, which go stale over 16 epochs.
2. **State-dependent std** — Brax's `tanh_normal` predicts both mean and scale from the network. Ours uses a state-independent learned `log_std` param.

---

## Session 7: PPO Validation & Algorithm Limits (Mar 2026)

### Know When PPO Is the Wrong Tool

**Problem:** HumanoidRun returns stuck at ~7-10 after 60M steps despite full Brax structural parity. Spent multiple sessions tuning HPs and fixing structural differences.

**Resolution:** Wrote `bench_brax_ppo.py` to run Brax's own PPO with the exact MuJoCo Playground config. Result: ~8-10 return — identical to ours. Checked the Playground paper (Figure 7): PPO is flat at ~10-20 through 100M steps. SAC reaches ~200 by 5M steps.

**Literature confirms this is expected:**
- dm_control paper (Tassa 2018): A3C scored 1.0/1000 on humanoid_run
- Brax paper: humanoid "does not find successful policies" with PPO even at 500M steps
- D4PG (off-policy, distributed) only reached ~464/1000 — the best reported score
- FastTD3 (2025) is current SOTA for humanoid locomotion

**Lesson:** Before spending days tuning an algorithm, check whether the algorithm is known to work on the task at all. A 5-minute literature search or a reference implementation sanity check can save days of HP tuning. PPO on humanoid_run is a known failure case — the signal was there in the original dm_control paper from 2018.

**When to suspect algorithm limits vs implementation bugs:**
- Your metrics look healthy (stable KL, reasonable clip fraction, no NaN) but returns are flat → likely algorithm limits
- Reference implementation gets the same score → definitely algorithm limits
- Metrics are unhealthy (KL spikes, entropy explosion, NaN) → implementation bug

---

### PPO Validation Summary

| Environment | Our Result | Target | Status |
|-------------|-----------|--------|--------|
| CartpoleBalance | >=995 @ 1M | >=950 @ 10M | PASS |
| CheetahRun | 826 @ 20M | >=700 @ 60M | PASS |
| HumanoidRun | ~7-10 @ 60M | matches Brax reference (~8-10) | PASS (algorithm limit, not bug) |

PPO implementation is validated. HumanoidRun requires off-policy methods (SAC, TD3, FastTD3).

---

## Session 8: Checkpoint Design & Inference Pipeline (Mar 2026)

### Orbax Restore Requires Exact Pytree Structure Match

**Problem:** `record_video.py` crashed restoring a checkpoint saved with `optax.chain(clip_by_global_norm, adam)` while the restore target was initialized with bare `optax.adam`. Error: `EmptyState vs dict`.

**Why it happens:** Orbax serializes the pytree structure of the optimizer state (not just the arrays). `optax.chain(A, B)` produces a tuple `(StateA, StateB)`. Bare `optax.adam` produces just `ScaleByAdamState`. These are structurally incompatible — orbax refuses to restore.

**Things that don't help:**
- `strict=False` on `StandardCheckpointer.restore()` — only for shape mismatches (array padding/truncation), not tree structure
- `partial_restore=True` — only available on the lower-level `PyTreeRestore` API, not `StandardCheckpointer`

**Targetless restore (emergency escape hatch):**
```python
# No target = restores as raw nested dict (UNSAFE warning, but works)
raw = ocp.StandardCheckpointer().restore(os.path.abspath(ckpt_dir))
actor_params = raw['training_state']['actor_params']
```
Use this when you need to extract specific leaves and don't have the exact pytree structure.

**Correct design:**
```python
# In train.py — save inference artifact separately from training artifact
np.save(os.path.join(ckpt_dir, "actor_params.npy"), {
    "actor_params": jax.device_get(training_state.actor_params),
    "norm_mean": jax.device_get(norm_state.mean),
    "norm_mean_of_squares": jax.device_get(norm_state.mean_of_squares),
    "norm_count": int(jax.device_get(norm_state.count)),
}, allow_pickle=True)

# In record_video.py — load without orbax
saved = np.load(params_path, allow_pickle=True).item()
training_state = training_state.replace(actor_params=saved["actor_params"])
```

**Lesson:** Separate inference artifacts from training artifacts. The orbax checkpoint is for resuming training (must match optimizer structure). A plain numpy file is for inference (no structure to match, immune to optimizer changes).

---

### Checkpoint Should Be Fully Self-Describing

**Problem:** `meta.json` stored only the network arch (hidden dims, activation, squash). `record_video.py` couldn't reconstruct the optimizer chain because `max_grad_norm` wasn't saved.

**Better approach:** Store the full training config as `dataclasses.asdict(cfg)` in meta.json. All hyperparameters, network config, and training setup in one place — sufficient to reproduce or resume any run without digging through code history.

```python
meta = {
    "obs_dim": obs_dim,
    "action_dim": action_dim,
    "train_config": dataclasses.asdict(cfg),  # full TrainConfig + nested PPOConfig
}
```

**Lesson:** A checkpoint that requires external knowledge to load is incomplete. The meta.json should answer: "given only this directory, can I fully reconstruct what was trained and how?"

---

### Synchronized Episode Resets Create Periodic VLoss Spikes

**Pattern observed in WalkerWalk training (episode_length=1000, num_steps=30, num_envs=2048):**
- Every ~33 iterations (1000/30 ≈ 33.3), all 2048 envs reset simultaneously
- VLoss spikes from ~1 to ~14,000, entropy collapses to -6.5
- Self-corrects within ~5 iterations

**Why entropy goes negative:** With tanh-squash + single-sample entropy estimate (`entropy = -log_prob`): if a large gradient update briefly collapses std, the sample falls near the mode where log_prob is very high (positive). Negative entropy is the artifact, not a sign of real divergence.

**Why it self-corrects:** The spike is a distribution shift (reset obs vs mid-episode obs), not a runaway update. Once the critic refits to the mixed distribution, everything stabilizes.

**Didn't prevent convergence** for WalkerWalk (833 avg return). But HumanoidRun NaN'd at these boundaries — higher-dim action space amplified the instability.

**Mitigation:** Brax's `num_updates_per_batch=16, num_steps=20` interleaves collections so episode phases are desynchronized across update cycles. After the first episode completes, subsequent update cycles have envs at different episode phases → no synchronized resets.

---

## Session 9: SAC Implementation & Off-Policy Pitfalls (Mar 2026)

### Online Obs Normalization Is Incompatible with Off-Policy Replay

**Problem:** SAC on WalkerWalk diverged immediately — Q values to -15,000, alpha to 19+, returns stuck at 32.

**Root cause:** At env reset, all 128 envs have near-identical observations (standing pose). The running variance estimate for some obs dimensions is ≈ 0. Normalizing `next_obs` (which differs slightly after one random step) divides by `sqrt(0) + 1e-8`, producing values of ±2 billion stored in the replay buffer.

**Why PPO is immune:** PPO is on-policy — obs are normalized and consumed in the same iteration. The norm stats match the data. SAC's replay buffer stores normalized obs that become stale as the running statistics evolve. Old transitions were normalized with statistics from 100k steps ago.

**The diagnostic trail:**
1. Q values diverging monotonically → something wrong with TD targets
2. TD target = -47 million on first gradient step → next_obs is corrupted
3. `next_obs` range in buffer: ±2 billion → normalization is exploding
4. `norm_state` variance ≈ 0 for some dims after identical reset obs → division by near-zero

**Fix:** Removed obs normalization entirely for SAC. The Q-network's LayerNorm provides implicit input normalization. Initialized norm_state as identity (mean=0, var=1) for checkpoint compatibility with record_video.py.

**Brax reference confirms:** Brax SAC does not normalize observations.

**Lesson:** On-policy normalization (update stats → normalize → use immediately) is safe. Off-policy normalization (update stats → normalize → store → sample much later) is fundamentally broken because the stored normalized values become stale as statistics drift.

**Update (2026-03-20):** The FastTD3 paper (holosoma source) actually uses obs normalization for off-policy — but they store **raw** obs in the buffer and normalize **at sample time** with current running statistics (`EmpiricalNormalization` with eps=1e-2). This is safe because buffer entries never go stale. They also use separate normalizers for actor and critic obs, and stop updating stats after a threshold. Our explosion happened because we normalized *before* storing — the exact antipattern they avoid.

Two valid approaches for off-policy:
1. No normalization + Q LayerNorm (our current approach, simpler)
2. Raw obs in buffer + normalize at sample time with large eps (paper's approach, may improve tasks with large obs scale differences)

---

### SAC Validation Results

| Environment | SAC Result | PPO Result | SAC Steps | PPO Steps | Speedup |
|-------------|-----------|------------|-----------|-----------|---------|
| WalkerWalk | 975 avg, 995 max | 833 avg, 992 max | 5M | 60M | 12x more sample-efficient, higher score |
| HumanoidRun | ~23+ at 1M (in progress) | ~10 at 60M | 5M target | 60M | Already 2x PPO's best at 20% of steps |

SAC matches the MuJoCo Playground paper's Figure 7 trajectory. WalkerWalk converged to >950 by 800k steps.

---

### TD3 NaN on HumanoidRun — Gradient Clipping Is Not Optional

**Problem:** TD3 on HumanoidRun NaN'd at ~700k steps. Q1 spiked from 1.17 → 9.54 → NaN in a single log interval. Episode count jumped from 641 to 10,497 (humanoid falling instantly from NaN actions).

**Root cause:** No gradient clipping. HumanoidRun has 67-dim obs and 21-dim actions — large enough that Q-network gradients can occasionally spike. Without clipping, a single large gradient step destabilizes the network, NaN propagates through the actor, and every env immediately terminates.

**Why SAC was immune:** SAC's entropy regularization acts as implicit gradient control — the `alpha * log_prob` term keeps the actor's output distribution smooth, preventing the sharp policy changes that trigger Q-value spikes. TD3 has no such mechanism.

**Fix:** Added `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))` for both actor and critic optimizers, plus Q LayerNorm for the HumanoidRun preset. Second run was completely stable through 5M steps.

**Lesson:** Gradient clipping is essential for TD3 on high-dimensional tasks. SAC's entropy regularization provides implicit stability that TD3 lacks. When porting between algorithms, don't assume stability properties transfer — each algorithm has its own failure modes.

---

### Replay Ratio Must Scale with num_envs

**Problem:** TD3 with `grad_updates_per_step=1` (vanilla) and 128 parallel envs showed Q values stuck near 0 and returns barely above random after 600k steps.

**Root cause:** Collecting 128 samples per step but only training on 256 (one batch). The model can't learn fast enough to keep up with data collection. Vanilla TD3's 1:1 ratio assumes single-env training.

**Fix:** Bumped `grad_updates_per_step` from 1 to 4 in the 128-env preset. Returns immediately started climbing.

**Lesson:** The replay ratio (gradient steps per env step) must scale with `num_envs`. A 1:1 ratio for 1 env = 1 gradient step per sample. For 128 envs, 1:1 means 1 gradient step per 128 samples — the model sees each sample roughly once before it's pushed out of the buffer. Use 4-8 gradient steps per env step for 128+ parallel envs.

---

### TD3's Exploration Limits on High-Dimensional Tasks

**Observation:** TD3 reached 749 on CheetahRun (6-dim action) and 955 on WalkerWalk (6-dim action), competitive with SAC. But on HumanoidRun (21-dim action), TD3 scored 4.3 vs SAC's 207.

**Why:** TD3 explores via additive Gaussian noise (`action + N(0, 0.1)`). In 21 dimensions, random perturbations almost never produce coordinated movements. SAC's entropy maximization actively searches for diverse high-reward behaviors — it's structured exploration, not random noise.

**When to use which:**
- **TD3**: Simpler tasks (≤10 action dims), faster wall-clock due to no entropy overhead. Good default for manipulation, simple locomotion.
- **SAC**: High-dimensional tasks (>10 action dims), tasks requiring coordinated multi-joint movement. Worth the extra complexity for humanoid-class problems.
- **FastTD3** (future): Uses distributional critic + large batches to compensate for TD3's exploration weakness. Achieves SOTA on humanoid tasks despite being deterministic.

---

### Python `if` vs `jax.lax.cond` Inside Traced Functions

**Problem:** `sac.select_action` used `if deterministic:` to branch between `tanh(mean)` and sampling. Worked fine during training (called from Python loop), crashed inside `jax.lax.scan` for video recording:
```
TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[]
```

**Why it worked during training:** `@jax.jit` traces `select_action` once per unique signature. When called from a Python loop with `deterministic=False` (a concrete Python bool), JIT traces the `False` branch and compiles it. The `if` is resolved at trace time, not runtime.

**Why it broke in scan:** `jax.lax.scan` traces its body function with abstract values. All arguments become tracers — including `deterministic`. A Python `if` on a tracer is illegal because JAX can't evaluate it at trace time.

**Fix:**
```python
# Before (Python control flow — fails inside scan)
if deterministic:
    return jnp.tanh(mean)
action, _ = sample_gaussian(mean, log_std, key, squash=True)
return action

# After (JAX control flow — works everywhere)
action, _ = sample_gaussian(mean, log_std, key, squash=True)
return jax.lax.cond(deterministic, lambda: jnp.tanh(mean), lambda: action)
```

**Rule of thumb:** If a function might ever be called inside `scan`, `vmap`, or another traced context, use `jax.lax.cond` for branching, not Python `if`. Functions that are ONLY called from Python loops can use `if` safely — but `lax.cond` is always safe.

---

### FastTD3: Scale Matters — Don't Bring a Race Car to a Parking Lot

**Problem:** FastTD3 (C51 distributional critic + Q averaging) underperformed vanilla TD3 at 128 envs / 5M steps: 285 eval vs vanilla TD3's 749.

**Root cause:** FastTD3 is designed for massive scale. The paper runs 1024 envs for ~750M total steps (~9M gradient steps). Our 5M-step runs had only 155K gradient steps — 57x fewer. The distributional critic adds computational overhead per gradient step without enough total steps to recoup the investment.

**The math:**
- Paper: 1024 envs × 12 updates/iter × ~750M steps → ~9M gradient steps × 32K batch = 300B total samples processed
- Our run: 128 envs × 4 updates/iter × 5M steps → 155K gradient steps × 512 batch = 80M total samples processed
- That's 3750x less total training compute

**V_min/V_max must cover actual Q range:** Paper uses [-10, 10] for their reward scale. CheetahRun with reward ~0.8/step has Q ≈ 80, so [-10, 150] needed. Q capped at V_max=10 → return capped at ~150 (the distributional equivalent of gradient clipping on the value function).

**Update — paper-scale run confirmed this:** 1024 envs, batch=8192, 12 grad updates/iter, 86M steps → **880 eval** (vs vanilla TD3's 749 at 5M). FastTD3 wins on final score and wall-clock (78 min for 86M steps at 18k sps), but vanilla TD3 is more sample-efficient (749 at 5M steps). Use FastTD3 when you have GPU compute to burn and want the best final score; use vanilla TD3 when samples are expensive.

**Lesson:** "Fast" algorithms are fast at scale, not at small scale. Running FastTD3 at 128 envs / 5M steps tests nothing — it's like benchmarking a multi-GPU distributed training framework on a single batch. Match the paper's intended operating regime (1024+ envs, 50M+ steps) or use the simpler algorithm.

---

### C51 Support Range (V_min/V_max) Is a Critical Hyperparameter

**Problem:** FastTD3 with default V_min=-10, V_max=10 on CheetahRun: Q values capped at ~9 (near V_max), returns capped at 154.

**Root cause:** C51 represents Q-values as a categorical distribution over fixed atoms in [V_min, V_max]. If the true Q exceeds V_max, all probability mass piles up at the boundary atom — the critic can't distinguish Q=10 from Q=100.

**Fix:** Set V_min/V_max to cover the actual Q-value range per environment:
- Q ≈ avg_reward_per_step / (1 - gamma)
- CheetahRun: reward ~0.8, gamma=0.99 → Q ≈ 80 → V_max=150
- HumanoidRun: reward ~0.2, gamma=0.99 → Q ≈ 20 → V_max=50

**Lesson:** Unlike scalar Q which is unbounded, distributional Q is hard-bounded by [V_min, V_max]. This is essentially a prior on the return range. Get it wrong and the critic is blind beyond the boundary.

---

### JIT the Hot Path — Non-JIT'd JAX Can Be Slower Than Numpy

**Problem:** Initial JAX replay buffer benchmark showed it was **5x slower** than numpy (1.85ms vs 0.34ms per sample at batch=512).

**Root cause:** `jax_array[random_indices]` without JIT dispatches a Python-level gather op every call. Numpy's C-compiled indexing + tiny CPU→GPU memcpy was faster than JAX's Python dispatch overhead.

**Fix:** Cache a JIT'd sample function per batch_size. After JIT warmup, sampling is constant ~0.13ms regardless of batch size (compiled GPU gather).

**Result:**
| Batch | Numpy | JAX (no JIT) | JAX (JIT'd) |
|-------|-------|-------------|-------------|
| 512 | 0.33ms | 1.85ms (5x slower!) | 0.12ms (2.7x faster) |
| 32,768 | 1.74ms | — | 0.13ms (13.3x faster) |

**Lesson:** JAX without JIT is slower than numpy for small ops due to Python dispatch overhead. Always JIT the hot path. "Put it in a jax.Array" is not enough — you need to JIT the operations on it too.

---

### jax.lax.scan Carry Cost — Large Buffer Arrays Kill Throughput

**Problem:** Scanning the inner gradient loop (sample + update × 8) with buffer arrays in the carry was 30% slower than a Python loop (6k vs 8.7k sps for SAC WalkerWalk).

**Root cause:** The scan carry included 6 buffer arrays (obs, next_obs, actions, rewards, dones, truncations) of shape (4M, dim). Even though they're not modified, JAX threads them through every scan iteration. At 4M entries, the carry overhead dominates the dispatch savings.

**Microbenchmark vs real training:**
- 100K buffer: scan was 1.17x faster (carry is cheap)
- 4M buffer: scan was 0.70x slower (carry overhead dominates)

**When scan helps:** Small carry state (just TrainingState, ~few MB). When scan hurts: large immutable data threaded through carry (buffer arrays, ~1GB).

**The right pattern:** Don't put the buffer in carry. Either:
1. Use `jax.make_jaxpr` tricks to avoid carry overhead (advanced)
2. Accept Python loop when carry would be large
3. Wait for JAX improvements to carry handling

**Lesson:** `jax.lax.scan` isn't free — carry size matters. Profile the full pipeline, not just the loop body.

---

### A Faster Component Doesn't Mean Faster Training

**Problem:** JAX buffer showed 4.8x faster sampling at batch=8192, but FastTD3 end-to-end throughput only improved ~1.5% (11.7k → 11.9k sps).

**Root cause:** Gradient steps (~6ms × 12 per env step) dominate wall-clock. Buffer sampling (0.72ms → 0.15ms × 12 = 6.8ms saved) is <10% of total step time. The 4.8x microbenchmark speedup translates to <2% end-to-end.

**When JAX buffer actually matters:**
- Scanning the inner gradient loop (eliminates Python dispatch between gradient steps)
- Very large batch sizes (32K+) where numpy transfer becomes significant
- Very high replay ratios (many samples per env step)

**Lesson:** Profile the full pipeline before optimizing a component. A 10x speedup on 5% of runtime = 0.5% end-to-end. The bottleneck here is Python dispatch between gradient steps, not buffer speed.

---

### SAC Variants Can't Match FastTD3 on Low-Dim Tasks — But That's OK

**Problem:** Exhaustive testing of SAC variants on CheetahRun (6-dim actions). None came close to FastTD3's 880.

**What we tried:**
- FastSAC (C51, α=1.0): 375 — alpha collapsed
- FastSAC OG recipe (C51, α=0.001, max_σ=1.0): 447 peak, degraded to 375 — too deterministic
- FastDSAC (Gaussian critic + DEM, no β): 509 — best without β
- FastDSAC + β (population diversity): **567** — best overall SAC variant
- FastTD3 (deterministic): **880** — untouchable on this task

**Why SAC can't win here:** CheetahRun has 6 action dims. Deterministic TD3 with simple Gaussian noise explores this space efficiently. SAC's entropy machinery (alpha tuning, stochastic sampling, log-prob computation) is overhead that doesn't buy anything in 6 dims. The exploration benefit of entropy maximization only matters when random perturbations can't cover the action space — i.e., high-dimensional tasks.

**Two different "FastSAC" papers exist:**
- **Seo et al. 2025** (arXiv:2512.01996) — the original, from Berkeley. Uses C51, α_init=0.001, max_σ=1.0. Reports results on **real robots**, not dm_control. Claims FastSAC works for whole-body tracking.
- **FastDSAC 2026** (arXiv:2603.12612) — different group, builds on Seo et al. Claims FastSAC is unstable, proposes Gaussian critic + DEM. Reports results on **HumanoidBench**.

Both papers benchmark on **high-dimensional tasks** (21+ action dims). Neither claims FastSAC beats FastTD3 on low-dim tasks like CheetahRun. Our result is consistent with the literature — the comparison that matters is HumanoidRun.

**Lesson:** Match the benchmark to the algorithm's intended regime. Testing SAC variants on CheetahRun is like testing 4WD on a flat highway — it works, but it's not where the advantage shows. Test on HumanoidRun (21 dims) where structured exploration matters.

---

### Always Run the Simple Baseline Before the Fancy Version

**Problem:** Spent a full day implementing and tuning FastSAC (C51 distributional) and FastDSAC (Gaussian distributional + DEM). Best result: 582 on CheetahRun after 100M steps and ~108 minutes. Then ran vanilla SAC as a baseline: **771 in 5M steps and 8 minutes.**

**The numbers:**
- Vanilla SAC (scalar Q, 128 envs, 5M steps): **771** in 8 min
- Best FastSAC (C51, 1024 envs, 100M steps): 582 in 108 min
- Best FastDSAC (Gaussian + DEM, 1024 envs, 30M steps): 567

**Why "Fast" was slower:** The FastSAC/FastTD3 recipe (C51, large batch, 1024 envs) is designed for massive-scale humanoid tasks where wall-clock speed matters more than sample efficiency. On CheetahRun (6 dims), vanilla SAC's scalar Q critic learns faster with less compute because: (1) no C51 quantization overhead, (2) gamma=0.99 gives a longer effective horizon than FastSAC's 0.97, (3) 128 envs with 8 gradient updates/step is a better compute ratio than 1024 envs with 12 updates.

**Lesson:** Run the simple baseline first. If vanilla SAC takes 8 minutes, run it before spending a day on distributional variants. A baseline that takes minutes can save hours of wasted tuning on a fancier algorithm that may not even be appropriate for the task.

---

### Verify Configs Against Source Code, Not Paper Text

**Problem:** Implemented FastTD3/FastSAC from the paper text. Results were poor (FastTD3 NaN'd at 31M on HumanoidRun, FastSAC plateaued at 582 on CheetahRun). Dispatched audit agents to compare against the paper's actual source code (holosoma repo).

**Found 7+ critical mismatches** the paper text doesn't mention:
- tau=0.125 (paper text just says "soft update") — 25x different from standard 0.005
- Tapered 3-layer networks (512→256→128 actor, 768→384→192 critic) — paper just says "MLP"
- SiLU activation — paper doesn't mention activation function
- 101 C51 atoms — paper text says "C51" without specifying atom count
- Policy delay=4 for FastSAC — not mentioned in paper, only in source code
- No gradient clipping — paper code sets max_grad_norm=0, we assumed 1.0
- No LR schedule — paper uses constant LR, we added cosine decay

**The tau=0.125 NaN proves configs matter:** FastTD3 HumanoidRun hit 395 eval then NaN'd at 31M steps with tau=0.005. Q went from 12.34 → NaN in one step. With 8 gradient steps per env step, the online network changes so fast that tau=0.005 can't track it — the target becomes stale, Q bootstraps diverge, NaN propagates. tau=0.125 updates the target 25x faster, preventing this.

**Lesson:** Papers omit implementation details that are critical for reproduction. Always check the source code repo. The `FastSACConfig` dataclass in holosoma had every parameter we were missing — it took 5 minutes to read vs days of debugging wrong configs.

---

### Read the Whole Recipe, Not Just the Key Ingredients

**Problem:** Implemented "FastSAC from the original paper" with α_init=0.001 and max_σ=1.0. Result: peaked at 447, degraded to 375. Worse than our FastDSAC attempts.

**Root cause:** We only copied two hyperparameters from the paper. Missed three others that matter just as much:
- **gamma=0.97** (we used 0.99) — shorter horizon makes the critic's job easier at scale
- **AdamW with β2=0.95** (we used Adam with β2=0.999) — better for large batch training
- **weight_decay=0.001** (we had none) — regularization for stability

**Also missed:** The paper is from a **different research group** (Berkeley/Amazon) than the FastDSAC paper (different group). They benchmark on **real robots**, not dm_control. Their claim that FastSAC works is for humanoid locomotion tasks, not necessarily CheetahRun.

**Lesson:** When reproducing a paper, extract ALL hyperparameters into a table before implementing. Missing "minor" params like gamma and optimizer settings can completely change behavior. A paper saying "we use Adam with lr=3e-4" might also mean β2=0.95 and weight_decay=0.001 — check the appendix/code.

---

### Target Entropy = 0 Is Not Optional for SAC at Scale

**Problem:** FastDSAC with target_entropy=-3.0 on CheetahRun (6-dim actions): alpha collapsed to 0.012, eval peaked at 401. Same run with target_entropy=0.0: alpha stayed at 0.65, eval peaked at 509.

**Root cause:** Target entropy controls what alpha converges to. With target_entropy=-3 and 6-dim actions, the policy easily achieves entropy of -3 (just be moderately deterministic). Once achieved, alpha decays toward zero since there's no pressure to maintain exploration. The policy loses its stochastic advantage and stagnates.

With target_entropy=0, entropy can never reach 0 (that would require a perfectly deterministic policy), so alpha stays positive and exploration pressure is maintained indefinitely.

**The intuition:** Negative target entropy means "be this deterministic." Zero means "stay as random as you naturally want to be." For exploration-driven algorithms like SAC, you want the latter.

**When to use negative target entropy:**
- Standard SAC at small scale (128 envs, 5M steps) — the classic `-dim(A)` heuristic works because training is short enough that alpha doesn't fully collapse
- Tasks where you want convergence to a near-deterministic policy

**When to use target_entropy=0:**
- Large-scale training (1024 envs, 100M+ steps) where alpha has time to collapse
- High-dim action spaces where entropy collapse is catastrophic
- Any SAC variant with DEM (DEM handles exploration allocation, alpha should just stay alive)

**Lesson:** The standard SAC target entropy heuristic (`-dim(A)`) was designed for small-scale single-env training. At scale with parallel envs, it causes alpha collapse. Use 0 as the default for large-scale SAC variants.

---

### GPU OOM at Scale Is Usually Not a Leak — Profile Before Assuming

**Problem:** Training OOM'd at 67-85M steps. Assumed memory leak, spent hours testing hypotheses (un-JIT'd buffer ops, eval recompilation, PRNG key accumulation). All disproved.

**Root cause:** No leak. The RTX 5080 (16GB) starts at **95% capacity** with 1024 envs. JAX pre-allocates ~75% of VRAM at startup. MuJoCo env state + 1M replay buffer + model + optimizer = 15.4GB used before training begins. Only ~900MB headroom. XLA command buffers (outside JAX's pool) slowly accumulate over 60+ minutes and push past the physical limit.

**How we found it:** Added both `jax.devices()[0].memory_stats()` (JAX-managed allocations) AND `nvidia-smi` (total CUDA usage) to the training loop. JAX memory was flat and bounded. CUDA memory was 15.4GB from startup — no growth. The "leak" was just the GPU being full from the start.

**Key insight:** `jax.devices()[0].memory_stats()` only reports JAX-managed memory. XLA compiled graphs, command buffers, and CUDA driver overhead live outside this pool. `nvidia-smi` shows the real total. Always check both.

**Fixes:**
- `XLA_CLIENT_MEM_FRACTION=0.7` — reduce JAX pre-allocation, leave more room for XLA/driver overhead
- `XLA_FLAGS=--xla_gpu_enable_command_buffer=` — disable CUDA graph caching to prevent command buffer accumulation
- Eval batch dim matching — eliminate unnecessary separate compilations
- JIT'd buffer add_batch — fewer XLA dispatches (good practice, not the cause)

**Lesson:** When debugging GPU OOM, start with `nvidia-smi` monitoring, not code review. If CUDA memory is flat, there's no leak — the GPU is just too small. The FastTD3 paper ran on A100s (40-80GB) for a reason.

---

### AdamW Requires `params` in optimizer.update() — Adam Does Not

**Problem:** FastDSAC crashed on first gradient step: `ValueError: You are using a transformation that requires the current value of parameters, but you are not passing params when calling update`.

**Root cause:** `optax.adamw` needs the current parameters to compute weight decay (`params * weight_decay`). `optax.adam` doesn't. All our previous algos used Adam, so `optimizer.update(grads, opt_state)` worked. AdamW needs `optimizer.update(grads, opt_state, params=params)`.

**Lesson:** When switching optimizers, check if the new one requires additional arguments. AdamW is the common case — weight decay is applied to the params themselves, not just the gradients.

---

### Integer Division Truncation in Training Loop Bounds

**Problem:** Final eval+checkpoint never fired. `total_env_steps=200000`, `num_envs=128`. Loop range: `range(0, 200000 // 128)` = `range(0, 1562)`. Last iteration: `total_steps = 1562 * 128 = 199936 < 200000`. The condition `total_steps >= total_env_steps` was never true.

**Fix:** Moved final eval outside the training loop. The in-loop eval triggers on episode count; the post-loop eval always runs.

**Lesson:** `total_steps // num_envs * num_envs != total_steps` when they don't divide evenly. Never rely on hitting an exact step count in a loop — use a post-loop finalizer instead.

---

## Future Topics to Explore

- [x] `jax.jit` compilation and when to use it
- [x] `jax.lax.scan` vs Python loops (efficiency) — 542x speedup on PPO update
- [x] Checkpointing with Orbax
- [x] Layer normalization in JAX/Linen — used in QHead for SAC/TD3
- [x] Deterministic eval rollouts (separate from training returns)
- [ ] Performance profiling with JAX
- [ ] Pytrees and how Linen models work as pytrees
- [ ] Device placement (CPU vs GPU)

---

### MJX Physics NaN at Scale — Not an Algo Bug

**Problem:** FastTD3 on HumanoidRun NaN'd at random step counts (315k, 600k, 1.2M) with identical configs. Looked like algo instability — tau too high, obs norm issues, C51 mismatch.

**Root cause:** MuJoCo's MJX physics solver produces NaN obs when the humanoid enters extreme states (contact solver failure, singular mass matrix). With 1024 parallel envs, at least one env crashes stochastically. NaN obs enter the replay buffer → Q network trains on NaN → cascade.

**How we found it:**
1. Debug script with `jnp.isnan()` checks at every step → survived 2M steps (sync points changed timing)
2. Real training without sync → NaN at 1.2M
3. Different step counts each run = stochastic input-side failure, not systematic algo divergence

**Fix (three layers):**
1. NaN-safe env step: `if NaN in obs → zero obs, zero reward, done=True`. Auto-resets crashed envs.
2. Action NaN guard: `if NaN in action → zero action` before env.step. Prevents NaN obs → actor → NaN action → MJX crash chain.
3. C51 log_prob clamp: `jnp.maximum(log_softmax(...), -30.0)` in cross-entropy loss. Prevents `-inf * 0 = NaN` when projected probability is 0 at an atom where log_softmax is -inf.

**Lesson:** When NaN happens at random step counts with identical configs, check inputs (env output) before gradients. Debug sync points masking the bug = timing/async issue. Physics engines crash at scale — guard the boundary.

### C51 Cross-Entropy NaN — `-inf * 0 = NaN`

**Problem:** Even with env NaN guards, FastTD3 still NaN'd at 1.97M steps. The env guard caught NaN obs, but NaN propagated through the C51 loss.

**Root cause:** `jax.nn.log_softmax` returns `-inf` for atoms with near-zero probability. The C51 cross-entropy loss computes `projected * log_probs`. When `projected[i] = 0.0` and `log_probs[i] = -inf`, IEEE 754 gives `0 * -inf = NaN`. This NaN propagates through the mean into the gradient → params → everything.

**Fix:** `jnp.maximum(log_softmax(...), -30.0)` — clamp log probabilities to -30 (effectively zero probability, `exp(-30) ≈ 1e-13`). This is standard in categorical RL implementations but easy to miss.

**Lesson:** Any cross-entropy loss using `log_softmax` needs a floor clamp. The `-inf * 0 = NaN` trap is silent — `log_softmax` looks correct, `projected` looks correct, but the product is NaN.

### FastDSAC Gaussian Critic Diverges on HumanoidRun

**Problem:** FastDSAC NaN'd at 6M steps on HumanoidRun. Q1, alpha, entropy — all NaN. Eval stuck at 4.6 (never learned).

**Root cause (hypothesis):** The Gaussian distributional critic outputs (mean, variance) via softplus. Near-zero variance → large `1/variance` in the NLL loss → gradient explosion. Unlike C51 (which has the log_softmax clamp fix), the Gaussian NLL has no floor on the variance denominator.

**Context:** FastDSAC worked on CheetahRun (567 eval) but failed on HumanoidRun (21-dim actions, 67-dim obs). The higher dimensionality likely pushes more variance estimates toward zero, especially early in training when the critic hasn't learned yet.

**Comparison:** FastSAC (C51 critic) scored 892 on the same task. The C51 categorical approach is more numerically stable than Gaussian parameterization for distributional RL at this scale.

**Lesson:** Gaussian distributional critics need aggressive variance flooring for high-dim tasks. The softplus + small eps isn't enough — consider `jnp.maximum(variance, min_variance)` with `min_variance` as a tunable hyperparameter, or switch to log-variance parameterization with clamping (like the actor's log_std).

---

*"The best way to learn is to break things, then fix them systematically."*
