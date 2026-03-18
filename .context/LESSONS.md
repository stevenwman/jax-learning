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

**Lesson:** On-policy normalization (update stats → normalize → use immediately) is safe. Off-policy normalization (update stats → normalize → store → sample much later) is fundamentally broken because the stored normalized values become stale as statistics drift. For off-policy methods: either normalize at sample time with current stats (complex), or skip normalization and let LayerNorm handle it (simple, recommended).

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

*"The best way to learn is to break things, then fix them systematically."*
