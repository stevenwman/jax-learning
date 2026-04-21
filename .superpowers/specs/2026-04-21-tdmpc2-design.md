# TD-MPC2 JAX Reimplementation — Design Spec

**Date:** 2026-04-21 (3 review iterations; paper+code audit incorporated)
**Status:** Design approved; three rounds of independent review findings integrated; pending user sign-off
**Target:** Port TD-MPC2 (Hansen et al. 2024, `nicklashansen/tdmpc2`) to this JAX/Flax framework

---

## 1. Goals & Non-Goals

### Goals
- Reproduce TD-MPC2 on DM Control Suite with paper-comparable scores (CheetahRun ≥ 850, HumanoidRun ≥ 800).
- Provide an A/B-comparable baseline against existing FastTD3 (CheetahRun 880) and FastSAC (HumanoidRun 892) in this repo.
- Land algorithm + training script + configs + tests in the existing repo structure with no framework coupling.
- Wire diagnostic instrumentation from day 1 so future bottleneck analysis (world model vs planner vs Q vs prior) is empirical, not guesswork.
- Design with forward-compatibility seams for a later multi-task (MT) extension.

### Non-Goals (out of scope for this spec)
- Multi-task training (MT30/MT80 configs, task embeddings, language grounding). Seams only.
- Parameter scaling experiments (the 19M/48M/317M variants).
- Vision-based TD-MPC2 (pixel encoder, data augmentation). Vector obs only.
- Real Go2 deployment of TD-MPC2 policies. Deferred; Go2 sim training in scope later but deployability not required in MVP.
- PushT manipulation. Deferred.

---

## 2. Scope Phasing

| Phase | Target | Benchmark bar | Status |
|-------|--------|---------------|--------|
| **P1** | DMC single-task (CheetahRun, HumanoidRun, AcrobotSwingup) | Eval scores within 5% of paper | In scope this spec |
| **P2** | Go2 Warp Joystick Flat (single-task) | Compare vs FastSAC 283.8 / FastTD3 273.1 | In scope this spec |
| **P3** | PushT manipulation | Compare vs existing PushT SAC | Deferred |
| **P4** | Multi-task extension (C-version) | — | Deferred, seams only |

---

## 3. Architectural Decisions

### D1: Single-task port with C-compatible seams (vs. full multi-task now)
- **Chosen:** single-task implementation (“B”) with four forward-compatibility seams.
- **Why:** lowest risk path to a working algorithm; paper single-task mode is independently valuable; repo has no multi-task precedent to amortize.
- **Seams (retained from day 1):**
  1. All network forward fns accept `task_id: Optional[jnp.ndarray] = None`.
  2. Replay buffer accessed behind a minimal protocol (`add`, `sample`, `sample_sequence`).
  3. Config carries `num_tasks: int = 1` and `task_names: list[str] = ["single"]`.
  4. Return normalizer keyed by task name.
- **Migration cost B→C (estimated):** 3–5 days with seams in place vs. 1–2 weeks without.

### D2: Benchmark order DMC → Go2 (PushT deferred)
- **Chosen:** DMC first (paper-faithful, cleanest A/B), Go2 after DMC passes, PushT not prioritized.
- **Why:** DMC gives us a known-good sanity check before the Go2 domain-transfer risk layer.

### D3: Go2 observation handling — deferred with default
- **Chosen:** defer full design; when Go2 phase begins, **default to symmetric state-only** encoder (world model sees `state`, not `privileged_state`).
- **Why:** DMC is flat-vector obs, decision does not apply there; state-only keeps Go2 deploy story open (privileged obs only available in sim).
- **Revisit trigger:** if state-only Go2 run underperforms FastSAC by > 20%, consider asymmetric (privileged to world model + Q, state to policy).

### D4: Replay buffer — extend existing with per-episode slice sampling
- **Chosen:** add `sample_sequence(batch, H)` method to `jax_rl/buffers/jax_replay_buffer.py`; store a per-transition `episode_id` alongside existing fields.
- **Mechanism:** store transitions flat (existing). On `add`, increment `episode_id` whenever the previous transition's `done OR truncated` is true. On sample, draw starting indices uniformly from valid slots (`0 ≤ i ≤ size - (H+1)`), read `H+1` contiguous slots, and **reject any window whose `episode_id` is not constant across all H+1 slots**. Also reject windows crossing the buffer ring wrap. Resample until `batch` valid windows collected (paper source uses `torchrl.SliceSampler` keyed on `traj_key='episode'` — our per-episode `episode_id` match matches its guarantee).
- **Why not new class:** single buffer implementation to maintain; additive change breaks nothing; ~80 LOC.
- **Why not episode buffer:** Go2 episodes are 1000 steps; episode-as-unit wastes memory.
- **Reviewer-flagged bug avoided:** spec earlier proposed only rejecting wrap-around. That would silently sample across episode boundaries whenever the buffer was full of multiple episodes — a real bug. Per-episode `episode_id` match closes it.

### D5: Training script — standalone (FlashSAC precedent)
- **Chosen:** `train_tdmpc2.py` owns its own loop; does not delegate to `offpolicy_loop.py`.
- **Why:** sample shape differs (sequence vs. transition), action-selection path differs (MPPI vs. actor), loss composition differs (joint world model + Q + policy). Branching shared loop for these would hurt SAC/TD3 readability more than it saves LOC.

### D6: Eval uses both MPPI and policy prior every eval
- **Chosen:** each eval reports two scores — `mppi_return` and `prior_return` — plus their gap.
- **Why:** prior-eval is nearly free (one extra forward pass vs. MPPI's 1536 latent rollouts); the gap is a load-bearing diagnostic signal (is the planner actually helping?).
- **Paper-comparable score:** `mppi_return` is the paper-comparable number. `prior_return` is a diagnostic auxiliary only. Journals and benchmark tables report `mppi_return`; `prior_return` used for bottleneck analysis and deploy cost estimation.
- **Checkpoint contract:** `actor_params.npy` stores policy prior params only. MPPI requires the full world model checkpoint (saved as `world_model_params.npy`). Deploy paths that only load `actor_params.npy` continue to work.

### D7: Collect uses MPPI with small env count (paper-adjacent)
- **Chosen:** default `--num-envs 8 --collect-mode mppi`. Alternative mode `--collect-mode prior` (fast, non-paper) exposed for ablation.
- **Why:** TD-MPC2 is designed for 1-env-high-UTD regime, not 1024-env prior collect. Running 1024 envs with MPPI defeats its own sample-efficiency pitch (and is GPU-infeasible).
- **Deviation acknowledged:** source uses `num_envs=1`. Our default `num_envs=8` is a deliberate throughput deviation — trades some sample efficiency for wall-clock. Benchmark phase will A/B test `num_envs ∈ {1, 4, 8}` on CheetahRun to quantify the cost before locking.

### D8: Paper HPs come from source audit, not paper text
- **Chosen:** every HP cited in this spec was verified against `/tmp/tdmpc2/tdmpc2/config.yaml` and `tdmpc2/*.py`. No values paraphrased from paper text.
- **Why:** repo rule (AGENT_HANDOFF.md paper-audit pattern). Prior paper-vs-code mismatches cost days on FastDSAC. The draft-from-memory version of this spec had `consistency_coef=1` (vs. actual 20), `SAC auto-tune α` (vs. actual fixed `entropy_coef=1e-4`), and target-encoder consistency (vs. actual online-encoder stop-grad). Those would have silently trained-but-failed.

---

## 4. File Layout

```
jax_rl/algos/tdmpc2.py              # Pure algo: nets, loss, MPPI planner, update fns
jax_rl/configs/tdmpc2_config.py     # TDMPC2Config dataclass + preset helpers
jax_rl/buffers/jax_replay_buffer.py # Extended: sample_sequence(H) method added
jax_rl/utils/simnorm.py             # SimNorm activation (small, new)
jax_rl/utils/twohot.py              # Two-hot / HL-Gauss + symlog/symexp (new)
jax_rl/utils/qscale.py              # Running percentile Q-scale EMA tracker (new)
train_tdmpc2.py                     # Standalone train loop
tests/test_tdmpc2.py                # Unit + integration + probe tests
tests/test_replay_sequence.py       # Buffer extension tests (may collapse into above)
jax_rl/configs/env_presets.py       # Add TDMPC2 presets for DMC tasks + Go2
```

---

## 5. Components (audited HPs in bold)

### 5.1 Encoder `h(o) → z`
- Input: obs vector (DMC: flat; Go2: `state` key from dict obs, 48d).
- Arch: **2 hidden layers × `enc_dim=256`** → linear to `latent_dim=512` → **SimNorm** activation on output.
- Every hidden `NormedLinear` = Linear → LayerNorm → **Mish**.
- Dropout: 0 on encoder.

### 5.2 Latent dynamics `d(z, a) → z'`
- Input: concat `(z, a)`.
- Arch: 2 hidden × `mlp_dim=512` NormedLinear blocks → linear to `latent_dim=512` → SimNorm.

### 5.X Weight init (applies to all nets above)
- **Linear layer weights:** `trunc_normal(std=0.02)` (source `common/init.py:4-16`).
- **Biases:** zero.
- **Output layers with zero-init weights:** reward head final linear, each Q head final linear (source `common/world_model.py:31-32`). Zero-init keeps initial reward/value predictions at the bin center post-symlog.
- All other output layers: standard trunc_normal.
- SimNorm activation has no trainable parameters beyond the preceding Linear.

### 5.3 Reward head `R(z, a) → r̂_logits`
- Arch: 2 hidden × 512 NormedLinear → linear to **`num_bins=101`**.
- Target: two-hot over `symlog(r)` clamped to `[vmin=-10, vmax=+10]`.
- Bin size: computed as `(vmax - vmin) / (num_bins - 1) = 0.2` — do not hardcode; derive from config so changes to `vmin/vmax/num_bins` propagate.
- **Output layer weight init: zero** (source `common/init.py`; matches reward head in `world_model.py`). Biases zero.

### 5.4 Terminal Q `Q(z, a) → q_logits`
- Ensemble of **`num_q=5`** heads, each 2 hidden × 512 + 101-way logit output.
- **Dropout = 0.01 on first hidden layer of Q heads only.** Source `world_model.py:30` passes `dropout=cfg.dropout` to the Q-ensemble `mlp()` call; dynamics/reward/pi call `mlp()` without the kwarg (default `0`). Implementer note: do NOT wire dropout into the shared MLP builder's default — keep it an explicit Q-only argument.
- Targets: **min of 2 random Qs** for TD target. Subsample draws **a fresh random pair per update step** (per-batch, not per-batch-element — matches source `world_model.py:212-215`).
- Policy-update Q: **avg of 2 random Qs** using stop-grad alias `_detach_Qs` (online params, detached — NOT target params).
- Value loss: CE over all 5 heads, summed over H rollout steps, normalized by `H · num_q`.
- **Output layer weight init: zero** (source `common/init.py`). Biases zero.

### 5.5 Policy prior `π(z) → (μ, log_σ)`
- Tanh-squashed action, reparameterized sampling.
- Arch: 2 hidden × 512 NormedLinear → linear to `2 · action_dim`.
- `μ` unbounded. `log_σ` **tanh-bounded** between `log_std_min=-10` and `log_std_max=2` via `low + 0.5·(high−low)·(tanh(raw)+1)` (source `common/math.py:13-14`) — NOT a hard clamp, NOT free parameter.
- Action `a = tanh(μ + σ·ε)` with log-prob Jacobian correction `-Σ log(relu(1 − a²) + 1e-6)` — matches source `common/math.py` `squash()`. The `relu` + `1e-6` floor is load-bearing numerical safety (prevents log of ≤0 when `|tanh| ≈ 1`). A naive `1 - tanh²` would NaN at saturation.
- **No target entropy, no Lagrangian.** Fixed **`entropy_coef=1e-4`**.
- Entropy term uses **scaled_entropy** ≈ `-log_prob · action_dim` per sample (action-dim normalization). Source implements this as the ratio `(-log_prob · action_dim) / (log_prob + 1e-8)` multiplied back into `log_prob` — a numerical-identity trick that lets multi-task code zero out masked action dimensions without changing the scalar. Single-task implementation may simplify to `-log_prob · action_dim` directly; test must match the source's post-mask value in both single- and multi-task configurations. Source: `common/world_model.py:176-183`.

### 5.6 Target nets
- **EMA on encoder, dynamics, reward, Q.** No target policy.
- **`tau=0.01`**, soft update every gradient step.

### 5.7 Q-scale running tracker
- Tracks **5th and 95th percentiles** of Q outputs (batch-dim percentiles) across recent updates, EMA'd with **the same `tau=0.01`** as target networks (source `common/scale.py:42` reuses cfg `tau`).
- **Input:** the first of the two randomly subsampled detached Qs used in policy loss — `self.scale.update(qs[0])` in source `tdmpc2.py:222`, where `qs[0]` is shape `(batch,)` or `(H+1, batch)` of scalar Q values. Percentiles computed over the batch dimension.
- **Range clamp:** `range = max(p95 - p5, 1.0)` before the EMA step (source `scale.py` clamps `min=1.`). Without this, early-training small Q ranges (e.g. 0.01) divide the policy gradient by a tiny number and blow up. **Load-bearing.**
- Scales Q in policy loss: `qs_scaled = qs / range_ema`.
- Without this, `entropy_coef=1e-4` is the wrong magnitude — Q values must be scale-normalized before adding the entropy bonus.
- Lives in `jax_rl/utils/qscale.py`; part of `TrainState`.
- **Bootstrap:** initialize `range_ema` to `1.0` and populate during the `seed_steps` warmup gradient burst (§7 "Warmup"). By the time regular updates begin, the tracker is warm; no special-case in policy loss.

### 5.8 MPPI planner (non-network)
- **`num_samples=512`** trajectories; **`horizon=3`**; **`num_elites=64`**; **`iterations=6`** (+2 if `action_dim ≥ 20`).
- **`temperature=0.5`**; `min_std=0.05`, `max_std=2.0`.
- **`num_pi_trajs=24`** of the 512 samples are seeded by rolling out the policy prior forward in latent space.
- **Mean warm-start across env steps (`_prev_mean`):** the MPPI mean trajectory is persisted across calls. At each new planner call with `t0=False`, initialize `mean[:-1] = prev_mean[1:]` and `mean[-1] = 0`. On episode reset or first step (`t0=True`), zero-initialize. This matches source (`tdmpc2.py:41, 167-168, 206`) and is a load-bearing mechanism — without it, every env step restarts planning from scratch and planner quality drops visibly. **Intentional vectorization deviation from source:** source stores `_prev_mean` as `(horizon, action_dim)` (single env). We store it as `(num_envs, horizon, action_dim)` to support `num_envs > 1` collect. Reset logic indexes per-env on `t0[i]=True`. Lives in `TrainState`.
- **Per-iteration std update:** at each MPPI iteration, update mean and std from the elite-weighted empirical mean and variance, then clamp `std ∈ [min_std, max_std]`. Source `tdmpc2.py:194-195`.
- Scoring: `return(τ) = Σ_h γ^h · r̂(z_h, a_h) + γ^H · Q_avg_of_2(z_H, π(z_H))`, decoded through two-hot → symexp.
- **Action returned:** sample a **single elite trajectory** via Gumbel-softmax on elite scores (source `tdmpc2.py:201-205`), take its first action, and add exploration noise `std_{t=0} · ε` (the MPPI-updated std at horizon index 0; not an "elite std"). In `eval_mode=True`, skip the noise. Previous draft of this spec wrongly said "elite-weighted mean + elite_std noise" — corrected.
- Pure JAX, `jit`-compiled, `vmap` over batch of envs.

---

## 6. Loss Composition

Single forward pass per batch of H-sequences; one backward pass.

```
z_targets_h = sg(encoder_online(o_h))           # stop-grad, NOT target-encoder
z_0         = encoder_online(o_0)
for h in 0..H-1:
    ẑ_{h+1} = dynamics(ẑ_h, a_h)
    r̂_h     = reward(ẑ_h, a_h)
    q̂_h     = Q(ẑ_h, a_h)                       # all 5 heads

L_consistency = (1/H) · Σ_h  rho^h · MSE(ẑ_h, sg(z_targets_h))                     # unmasked (source)
L_reward      = (1/H) · Σ_h  rho^h · CE(r̂_logits_h, twohot(r_h))                   # unmasked (source)
L_value       = (1/(H·num_q)) · Σ_h  rho^h · Σ_q CE(q̂_logits_{h,q}, twohot(target_q_h))  # unmasked (source)
L_policy      = E_τ [ Σ_h rho^h · (entropy_coef · scaled_entropy_h - qs_scaled_h) ] / (H+1)

L_world_total = consistency_coef · L_consistency + reward_coef · L_reward + value_coef · L_value
```

**Normalization is load-bearing.** Source divides `consistency_loss / H`, `reward_loss / H`, and `value_loss / (H · num_q)` before applying coefficients. Without these divisions our tuned coefficients (`consistency=20, reward=0.1, value=0.1`) are silently miscalibrated by ~3×.

**No per-step terminal mask on ANY of the three world-model losses.** Source applies neither a `done` nor `truncated` mask to `L_consistency`, `L_reward`, or `L_value`. See `tdmpc2.py:273-305`. The `(1 - terminated)` factor appears ONLY inside the TD target computation (see "Target Q" below). Earlier draft of this spec wrongly applied `mask_h` to reward/value — corrected. The per-episode slice sampler (§D4) is what keeps sequence windows from crossing episode boundaries; no per-step mask is needed once sampling is correct.

### Coefficients (audited)
- **`consistency_coef=20`**
- **`reward_coef=0.1`**
- **`value_coef=0.1`**
- **`entropy_coef=1e-4`**
- **`rho=0.5`** (per-step loss discount on horizon rollout)
- `termination_coef=1` **skipped** (`episodic=false` for DMC/Go2)
- Grad clip: **`max_norm=20`** on both optimizers.

### Target Q
```
target_q_h = r_h + γ · (1 - terminated_h) · twohot_decode(
    Q_target_min_of_2( dynamics_target(z_h, a_h), π(dynamics_target(z_h, a_h)) )
)
```
`terminated_h` is the raw `terminated` flag from the buffer at step `h` (not a derived `done_natural`). `truncated` does NOT zero the bootstrap. For DMC with `episodic=false`, `terminated_h ≡ 0` → full bootstrap always.

### Truncation handling
- Buffer stores `terminated` and `truncated` flags per transition.
- **Loss side (reward, value, consistency):** NONE are masked by terminal or truncation flags. Windows are kept within-episode by the per-episode slice sampler (§D4); any `H+1` window fully inside a single episode has well-defined latents and transitions end-to-end.
- **TD-target side (only place `terminated` matters):** see "Target Q" below — the bootstrap term carries `(1 - terminated)`. `truncated` does NOT zero the bootstrap (truncation = time cutoff, future value still exists).
- **For DMC tasks (episodic=false):** `terminated` is always `False` (matches source `online_trainer.py:91-93` which hard-raises on any `terminated=True` when non-episodic). All windows behave identically. The per-episode sampler still excludes windows crossing episode boundaries (episodes end via truncation at step 500).
- **For Go2 tasks:** truncation at time limits is the dominant episode terminator; `terminated` fires only on actual fall. Same code path handles both.

---

## 7. Data Flow

### Collect loop (per env step, default `num_envs=8`)
1. Encode current `obs` → `z_0` (online encoder).
2. MPPI plan on `z_0` → action (or policy prior if `--collect-mode prior`).
3. `env.step(action)` → `(next_obs, r, done, truncated)`.
4. `buffer.add(obs, action, r, next_obs, done, truncated)`.

### Warmup
- First **`seed_steps = max(1000, 5·episode_length)`** env steps: random actions only, no updates.
- DMC: `5·500 = 2500` random-action steps.
- At `_step == seed_steps` boundary: **burst of `seed_steps` gradient updates** (pretraining on filled buffer) before normal UTD=1 begins.

### Update loop (UTD=1 after warmup)
1. Sample: `buffer.sample_sequence(batch=256, H=3)` → `(o_{0..H}, a_{0..H-1}, r_{0..H-1}, done, truncated)` shape `(H+1, B, …)`. Per-episode slice sampling guarantees all H+1 steps come from the same episode (§D4).
2. Encode entire observed sequence with online encoder under stop-grad → `z_targets_{0..H}`.
3. Forward-roll dynamics from `z_0 = encoder(o_0)` to get `ẑ_{1..H}`.
4. Compute reward/Q logits at each step.
5. Compute `L_world_total` (consistency + reward + value, with per-step `rho^h` discount, then per-H normalization per §6).
6. Backprop through **world model optimizer**:
   - **Single Adam, two param groups** (matches source `tdmpc2.py:23-31`). Implemented via Optax `multi_transform` or equivalent — NOT two separate optimizers.
   - Group A (scaled LR): **encoder only** at `lr · enc_lr_scale = 3e-4 · 0.3 = 9e-5`.
   - Group B (default LR): dynamics + reward + Qs **+ `task_emb`** at `lr = 3e-4`. (Source puts `task_emb` in the default group, not the scaled encoder group — important for Phase 4 C-migration, do not conflate.)
   - Grad clip `max_norm=20` applied to the combined gradient tree.
7. Compute `L_policy` on detached latents from step 3 (stop-grad on world model); backprop through **policy optimizer** (separate Adam, LR `3e-4`, `eps=1e-5`, grad clip `max_norm=20`).
8. Update Q-scale EMA (5th/95th percentile) with online Q outputs.
9. Soft-update target nets: `θ_target ← θ_target + tau · (θ - θ_target)` on encoder, dynamics, reward, Q only.

### Eval loop (every `--eval-every` env steps)
- `num_eval_envs=8`, both modes per episode:
  - `prior_return`: roll out policy prior greedily (`eval_mode=True`).
  - `mppi_return`: roll out with MPPI (`eval_mode=True`, no planner noise).
- Log: `prior_return`, `mppi_return`, `gap = mppi_return - prior_return`.
- Also log diagnostics (Section 8).
- Checkpoint on new best `mppi_return`.

---

## 8. Instrumentation

All logged to wandb + `metrics.csv`:

- Loss components: `L_consistency_raw`, `L_reward_raw`, `L_value_raw`, `L_world_total`, `L_policy`, `L_total_world_weighted` (post-coefficient)
- Latent rollout error per horizon: `latent_err_h1`, `latent_err_h2`, `latent_err_h3` = `mean((ẑ_h - sg(z_targets_h))^2)`
- Q diagnostics (existing `get_q_value()` hook): `q_bias`, `q_rmse`, `q_corr`
- Q-scale tracker: `q_p5`, `q_p95`, `q_scale_ema`
- Reward prediction accuracy: `reward_mae` = `|twohot_decode(r̂) - r|`
- MPPI diagnostics:
  - `mppi_elite_entropy` = entropy of softmax(elite scores) — peaked vs. flat
  - `mppi_mean_return` = predicted return of final mean trajectory
  - `mppi_iterations_effective` = at which iter the elite set stabilized
- Policy: `entropy(π)`, `log_prob_mean`, `action_norm`
- Target-online distance: `||θ - θ_target||_2` for each of encoder, dynamics, reward, Q
- Eval: `prior_return`, `mppi_return`, `mppi_prior_gap`
- Max-magnitude tracking: `max_|reward_observed|`, `max_|target_q|` — warn if near `vmax`

---

## 9. Config Schema

```python
@dataclass
class TDMPC2Config:
    # Architecture
    latent_dim: int = 512
    mlp_dim: int = 512
    enc_dim: int = 256
    num_enc_layers: int = 2
    simnorm_dim: int = 8
    dropout: float = 0.01  # Q heads only, first layer
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    # Loss
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 1e-4
    rho: float = 0.5
    grad_clip_norm: float = 20.0

    # Optimization
    lr: float = 3e-4
    enc_lr_scale: float = 0.3          # applied to encoder param group in world model optimizer
    pi_optim_eps: float = 1e-5
    tau: float = 0.01                   # shared by target EMA and Q-scale EMA
    batch_size: int = 256
    horizon: int = 3
    discount: float = 0.99              # DMC default; Go2 TBD

    # Policy prior bounds
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    # MPPI
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    mppi_iterations: int = 6
    mppi_temperature: float = 0.5
    mppi_min_std: float = 0.05
    mppi_max_std: float = 2.0

    # Training loop
    seed_steps: int = 2500            # random action warmup; source: max(1000, 5*episode_length)
    # Warmup gradient burst at _step == seed_steps: always performs `seed_steps` updates (multiplier=1 in source)
    utd: int = 1
    collect_mode: str = "mppi"        # "mppi" | "prior"
    num_envs: int = 8
    num_eval_envs: int = 8

    # Per-task discount heuristic (source `tdmpc2.py:58-71`): discount = clamp((frac-1)/frac, min, max) with frac = ep_len / discount_denom
    discount_denom: int = 5
    discount_min: float = 0.95
    discount_max: float = 0.995

    # Multi-task seams (B-mode defaults)
    num_tasks: int = 1
    task_names: tuple[str, ...] = ("single",)
    episode_lengths: tuple[int, ...] = (500,)   # per-task; feeds the discount heuristic above. For DMC: 500 → discount=0.99. For Go2 (ep_len=1000): 1000 → discount=0.995.
```

**`discount` is derived, not a flat default.** In B-mode the single-task discount is computed at config load as `discount = clamp((frac-1)/frac, discount_min, discount_max)` with `frac = episode_lengths[0] / discount_denom`. For DMC this yields `0.99`; for Go2 `0.995`. In C-mode this trivially extends to per-task discounts without config surgery — the seam is live, not dead.

Presets in `env_presets.py`:
- `TDMPC2_DMC_CheetahRun` (std defaults, discount=0.99)
- `TDMPC2_DMC_HumanoidRun` (std defaults)
- `TDMPC2_DMC_AcrobotSwingup` (std defaults)
- `TDMPC2_Go2WarpJoystickFlat` (symmetric state-only encoder; `num_envs=8`, `seed_steps=5000`)

---

## 10. Error Handling & Failure Modes

1. **NaN/Inf in latent or loss**
   - `jnp.isfinite` guard before MPPI scoring; non-finite samples get `-inf` score, drop from elites.
   - Clip encoder pre-SimNorm logits to `[-20, 20]` to prevent softmax overflow.
   - Grad clip `max_norm=20` on both optimizers.

2. **MPPI exploiting world model errors (OOD actions)**
   - `min_std=0.05` prevents std collapse.
   - Policy-prior seeding (24/512 samples) anchors to on-distribution actions.
   - Action clip to `[-1, 1]` post-sample.

3. **Two-hot bin saturation**
   - Log `max(|reward_observed|)` and `max(|target_q|)` after symlog.
   - Warn if within 10% of `vmax`.
   - If saturating, bump `vmax` via config — do not silently clip.

4. **Truncation mishandling**
   - Buffer stores `terminated` and `truncated` as separate flags (source stores only `terminated`; we extend).
   - `terminated` zeros the TD-target bootstrap via `(1 - terminated)`. `truncated` does NOT zero it (future value still exists). Neither flag masks the per-step world-model losses (consistency/reward/value are unmasked — matches source).
   - Unit test (Test 23): batch with mixed `terminated`/`truncated`/normal transitions; assert TD target uses correct `(1 - terminated)` multiplier and no per-step mask is applied to any of the three world-model losses.

5. **Consistency collapse (all z → same vector)**
   - SimNorm structurally prevents zero latent (each chunk is a softmax).
   - Monitor `std(z)` across batch; alert if < 0.01.

6. **Buffer edge cases**
   - Reject sequence windows that cross (a) buffer ring-wrap or (b) episode boundary (per-episode `episode_id` must be constant across all H+1 slots).
   - If `buffer.size < H + 1`, skip update.

7. **Policy prior entropy collapse**
   - `log_σ` tanh-bounded between **`log_std_min=-10`** and **`log_std_max=2`** (not a hard clamp — see §5.5).
   - Monitor `entropy(π)`; alert if < 0.01 for > 100 updates.

8. **GPU OOM in MPPI**
   - MPPI inner rollout batch: `num_envs × num_samples × H × latent_dim = 8 × 512 × 3 × 512 ≈ 6M elements` — fits comfortably.
   - If scaling `num_envs > 64`, group-batch MPPI to cap peak memory.

9. **Stale targets**
   - Log `||θ - θ_target||` per component.

10. **Q-scale EMA instability early in training**
    - Initialize scale to `1.0`; populated by the `seed_steps` warmup gradient burst before regular updates begin (see §5.7 + §7 "Warmup"). By the time policy updates start, the tracker is warm.

11. **MPPI `_prev_mean` staleness at episode boundary**
    - On episode reset (new `t0=True`), zero-initialize `_prev_mean` for that env. Without this, the planner warm-starts with trajectories from the prior episode's end — correlated with terminal states and actively misleading.
    - Unit test: simulate two sequential episodes through planner, assert `_prev_mean` resets at boundary.

---

## 11. Testing Strategy

See `tests/test_tdmpc2.py`. Target ~40 tests.

### Unit tests
1. SimNorm — output constraints, gradient flow, zero-input safety.
2. Two-hot + symlog/symexp roundtrip; boundary clamping.
3. Encoder/dynamics/reward/Q forward shapes across batch × H.
4. Single-batch loss — each component finite, matches hand-computed.
5. MPPI — (a) determinism with fixed seed; (b) bounded actions; (c) converges on toy reward landscape; (d) 24 π-seeds wired correctly; (e) single-elite Gumbel sampling matches source's categorical sample distribution; (f) `_prev_mean` warm-start applied on `t0=False` and reset on `t0=True`.
6. Q-scale EMA — percentile math + EMA follows `tau`.
7. Buffer sequence sampling — windows at boundaries (size 5/10/100); done-mask correctness; truncation vs. natural-done separation; **per-episode `episode_id` match rejects cross-episode windows**; wrap-around rejection.
7b. Tanh-Gaussian log-prob Jacobian — cross-check against source `common/math.py` `gaussian_logprob` + `squash` formulas numerically. Tests must cover the saturation edge case (`|tanh| → 1`): our impl uses `relu(1 - a²) + 1e-6` floor (matches source); a naive `1 - tanh²` version would return `-inf` at saturation. Random μ, σ, ε sweep; our log_prob ≈ source log_prob to 1e-5 across the range `|μ+σε| ∈ [0, 10]`.
8. Target EMA — one step of soft update moves params by `tau · Δ`.
9. Scaled entropy matches source formula (action-dim normalization).
10. Rho discount applied correctly per horizon step.

### Integration tests
11. Single update smoke — 1 collect, 1 update, no NaN, all scalars finite.
12. DMC CartpoleSwingup toy — 50k steps, eval > random-policy threshold.
13. Planner vs. prior action diff post-training — mean `||a_mppi - a_prior|| > 0.01`.

### Benchmark validation (blocking before claiming success)
14. DMC CheetahRun — 1M env steps, `mppi_return ≥ 850`. **Bar is a repo-level expectation**, not cited from the paper (paper was not accessible at spec-drafting time). Revise once paper score is confirmed.
15. DMC HumanoidRun — 1M env steps, `mppi_return ≥ 800`. Same caveat — repo-level expectation.
16. MPPI gap — `mppi_return − prior_return ≥ 10%` of prior at end of training.

### Failure probes
17. Force NaN into encoder output, assert MPPI skip-on-nonfinite works without crash.
18. Reward=1e4 env — assert saturation warning fires, symlog compresses, training stable.
19. Synthetic H=3 window with `terminated=True` at h=1 — assert (a) reward/value/consistency losses at h=2 are computed normally (NOT masked; matches source), (b) the `(1 - terminated)` factor in the TD target correctly zeros the bootstrap term at h=1 only. Earlier draft of this test asserted h=2 losses mask to zero — that was based on the pre-fix `mask_h` design and is now wrong; this test version matches the corrected §6.
20. Cross-episode buffer window — populate buffer with 2 episodes of length 3 each (total 6 transitions), assert sampler never returns a window crossing the boundary. Also log sampler rejection rate as a training diagnostic to catch pathological cases (buffer with few long episodes near capacity).
21. Loss-normalization regression — compute loss scalars for a hand-crafted batch with `H=3`; assert `L_consistency`, `L_reward`, `L_value` each include the `/H` (and `/num_q`) factor. Catches silent rescale if someone drops the division.
22. Q-scale range clamp — feed Q tensor with `p95 − p5 = 0.01`; assert EMA `range_ema` clamps to `1.0` not `0.01`. Prevents early-training gradient blowup.
23. Truncation-vs-terminated semantics — construct batch with window `[..., truncated=True, ...]`; assert (a) reward/value/consistency losses are NOT masked at or past the truncated slot, (b) `(1 - terminated)` in TD target is still 1 (truncation doesn't zero bootstrap), and conversely for `terminated=True` case: bootstrap zeroed but reward/value loss still applied.

---

## 12. Migration to Multi-Task (C) — Design Notes for Later

When/if multi-task is prioritized:
1. Replace `task_id=None` default with learned task embedding table of size `num_tasks × embed_dim`.
2. Inject FiLM conditioning from `task_emb(task_id)` into each MLP block (gain + bias modulation).
3. Pad action spaces to max across tasks; buffer carries per-task action masks.
4. Replace flat replay buffer with per-task ring buffers + weighted stratified sampler.
5. Replace scalar return normalizer with per-task return normalizer.
6. Config: `num_tasks > 1`, `task_names = [...]`.
7. Eval: per-task score + aggregate mean.

Everything else (losses, MPPI, Q, policy) unchanged — this is why the seams are cheap.

---

## 13. Open Questions / Known Gaps

1. **DMC env integration path:** existing `env_setup.py` uses Playground's registry. Does it already register DMC tasks, or is there new registration work? To be resolved during plan phase.
2. **Does `jax_rl/training/eval_runner.py` support two-mode eval out of the box**, or does TD-MPC2 need its own eval function? To be resolved during plan phase.
3. **Go2 integration not yet designed in detail** — this spec covers scope and obs-handling default only. Separate design pass or incremental plan addition when P2 begins.
4. **Optax `multi_transform` structure:** implementation must confirm the Optax recipe for single-optimizer + per-group LR (encoder at `lr·0.3`, rest at `lr`). If `multi_transform` is awkward for this shape, fall back to two Adam instances with distinct LRs (breaks source fidelity only in moment-state sharing; a tolerable deviation).

---

## 14. Acceptance Criteria

Spec is implementable when:
- [x] All P1 HPs have a source file:line citation
- [x] Loss formulas match source (audited) — including per-H normalization on consistency/reward and per-(H·num_q) on value
- [x] MPPI algorithm steps match source (audited) — including single-elite Gumbel sampling, `std_{t=0}·ε` noise, and `_prev_mean` warm-start across env steps
- [x] File layout fits existing repo philosophy
- [x] C-migration seams identified and cheap (~10% B overhead, ~80% C savings)
- [x] Failure modes + test plan cover known risks
- [x] Independent-review findings (MPPI action selection, `_prev_mean`, loss normalization, per-episode sampling, tanh-bounded log_std, weight init, single-Adam-with-param-groups, scaled_entropy formula, tanh-squash numerical safety, no per-step mask on world-model losses, Q-scale `min=1.` clamp, per-task discount heuristic, dropout scope, vectorized `_prev_mean`) resolved in §§5-11
- [x] Paper access caveat acknowledged — acceptance bars (§14) marked as repo-level expectations, not paper citations (paper was inaccessible at spec time)
- [ ] Implementation plan drafted (next step: `writing-plans` skill)
- [ ] Benchmark validation passes (P1): CheetahRun ≥ 850, HumanoidRun ≥ 800

---

## References

- Paper: Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* 2024.
- Source: `https://github.com/nicklashansen/tdmpc2` (cloned to `/tmp/tdmpc2/` for audit)
- Repo rules: `CLAUDE.md`, `.context/AGENT_HANDOFF.md` (paper-audit pattern, truncation handling, testing standards)
- Related specs: `.superpowers/specs/2026-04-10-offpolicy-loop-extraction.md` (why TD-MPC2 stays standalone)
