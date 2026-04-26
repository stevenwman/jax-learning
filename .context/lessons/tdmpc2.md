# TD-MPC2 — Porting & Debugging Lessons

Lessons from the JAX/Flax port of [`nicklashansen/tdmpc2`](https://github.com/nicklashansen/tdmpc2)
on DM Control benchmarks (J3 CheetahRun, J4 HumanoidRun). Five real
bugs surfaced over 2 days of debug; the patterns generalize to any
PyTorch→JAX port of a sequence-model RL algo.

## Verified results

| Task | Best mppi (40 eps) | Paper (Fig.4/15) |
|---|---|---|
| Cheetah Run @ 1M | **837 ± 1.5** | ~850 |
| Humanoid Run @ 1M | **557 ± 4** | trajectory band, paper trains 14M |

Both validated via `scripts/eval_tdmpc2.py --num-evals 5` (40 episodes).

---

## The 5 bugs (in order of impact, biggest first)

### 1. Replay buffer cross-env contamination

`add_batch` writes a `(num_envs,)` batch into consecutive indices, so
env e's transitions are at `ptr+e, ptr+e+num_envs, ptr+e+2*num_envs, ...`.
Old `sample_sequence` used offsets `[0, 1, 2, 3]` — returned a
4-frame "trajectory" stitched from 4 different envs. World model
trained on impossible cross-env transitions. ep_id check passed
because all envs reset on truncation simultaneously (sync ep
counters), so windows looked "valid" by ID alone.

**Why specific to TD-MPC2 / sequence models**: SAC/TD3 sample single
transitions (s, a, r, s') — each row self-contained, doesn't care
about neighbors. Cross-env interleaving harmless. TD-MPC2 needs H+1
contiguous-in-time obs from ONE env for latent rollout +
consistency loss against `encoded(real_obs[t+1])`. Sequence-sampling
breaks under round-robin storage.

**Fix**: add `stride` parameter to `sample_sequence`; offsets become
`[0, stride, 2*stride, 3*stride]`. Default 1 (single-env). With
multi-env, pass `stride=num_envs`.

**Impact**: 50× improvement at same step count. The kingpin bug.

### 2. action_repeat env-side parity

Source `dmcontrol.py:54-60` hardcodes `range(2)` action_repeat for ALL
DMC tasks. We ran with action_repeat=1, episode_length=1000. Same
physics duration (25s) but **40Hz agent decisions vs source's 20Hz**.
MPPI horizon=3 covers 0.075s vs 0.15s. For 21-DoF Humanoid (gait
period ~0.5s) that's a major planning-horizon shortfall.

**General lesson**: action_repeat is part of algo-tuning context. HPs
in source paper assume their action_repeat. Match it AND adjust
discount via the heuristic (`compute_discount(ep_len, denom)`) so the
effective time horizon stays equivalent.

For us: `episode_length=500` × `action_repeat=2` = 1000 control steps,
discount auto-computes to 0.99 (from 0.995 with the wrong setup).

**Impact**: lifted Humanoid from stuck-at-93 to peak-559 over 1M.

### 3. Truncated treated as terminated in TD target

Our `EpisodeWrapper` sets `done=1` at episode timeout, so
`batch["dones"] = terminated | truncated`. `compute_td_target` was
reading `terminated = batch["dones"]` and using `(1 - terminated)` to
zero the bootstrap. Source stores `terminated` SEPARATELY from
`truncated` (`common/buffer.py:98-106`) and uses ONLY terminated in
`(1 - terminated) * Q`.

For non-episodic DMC tasks (Cheetah, Humanoid), bootstrap MUST be
preserved across timeouts. We were zeroing it on every episode end.
Cheetah's dense reward (~0.85/step) absorbed the bias; Humanoid's
sparse early reward (<0.05/step) got crippled.

**Fix**: `terminated = clip(dones - truncations, 0, 1)`.

**General lesson**: this is a recurring class of bug across our repo
(see `lessons/offpolicy.md §Truncation Handling`). Any algo using
`(1 - done) * Q` must use TERMINATED, not done. Our wrapper's
conflation is convenient for things that don't care, dangerous for
things that do.

### 4. Q dropout disabled in policy_loss + qscale recompute

Source `_detach_Qs` (used by `update_pi` via `Q(..., detach=True)`) is
a deepcopy of `_Qs`. `world_model.py:74-80`'s `train()` override
forces ONLY `_target_Qs` to eval mode; `_detach_Qs` rides parent train
mode → dropout=0.01 active. Three call sites need attention:
value-loss (we got first), policy-loss (missed initially), qscale
recompute (also missed).

**Fix**: `deterministic=False + rngs={"dropout": key}` at all three.

**General lesson**: when porting from torch, be paranoid about what
`.eval()` / `.train()` toggles in source. PyTorch's implicit module
state has no JAX equivalent — we must explicitly thread
`deterministic` and `rngs` per call site.

### 5. MPPI temperature inverted

Source: `exp(temperature * Δ)`. Ours: `exp(Δ / temperature)`. With
`cfg.mppi_temperature=0.5`, ours was 4× sharper softmax over elites.
Concentrated weight on top elite — amplifies any Q-overestimation
peak.

**General lesson**: small one-line ports of mathematical formulas need
extra paranoia. Multiply-vs-divide, add-vs-subtract, sign flips. Diff
the actual operator.

---

## Patterns (reusable beyond TD-MPC2)

### Eval RNG isolation

Naive: eval branch consumes from training key (`key, eval_key = split(key)`).
Side effect: eval cadence (`cfg.eval_every`) silently affects training
trajectory. Two runs with same seed but different eval cadence
produce different training data.

Fix: derive eval_key from a dedicated `eval_base_key = PRNGKey(seed +
N)` folded with `eval_index`. Training key untouched.

```python
eval_base_key = jax.random.PRNGKey(seed + 9000)
# inside eval branch:
eval_index = (step_counter - 1) // cfg.eval_every
eval_key = jax.random.fold_in(eval_base_key, eval_index)
```

Verified within-process: log `int(jnp.sum(key))` before/after eval; should match. Cross-process verification impossible due to mujoco_warp non-determinism (see `lessons/determinism.md`).

**Apply to**: any algo with periodic eval that consumes from training
PRNG. PPO, SAC, TD3, FlashSAC etc. could benefit — currently each
has the same pattern.

### Multi-agent audit methodology

For non-trivial port debugging:
1. **Targeted audits in parallel** (3 opus agents, different angles —
   math/algo-flow/data-flow). Reduces chance of all missing the same
   class of bug.
2. **Independent validator** (4th agent) confirms findings line-by-line.
   Rejects 1-out-of-3 hallucinations.
3. **Dispatch only on real symptoms**, not preventively. Each round
   found 1-3 real bugs; pattern of "find bug → improve → still off"
   suggests more bugs exist.

Used to find Bug A+B (truncated/dropout) and the audit converged on
both with high agreement, validator confirmed.

### Cross-check env via existing repo benchmarks

Before suspecting env-side issues, check if other algos in this repo
have results on the SAME env. If FastSAC hit 892 on
mujoco_playground HumanoidRun, the env can support paper-level
performance. Init-randomization might be a TODO in the env, but it's
not blocking learning. This narrows the search.

```bash
grep -rn "HumanoidRun.*[678][0-9][0-9]" .context/
```

Also: `AGENT_HANDOFF.md` benchmark line is curated for exactly this
cross-check.

### Per-step diagnostic instrumentation (Tier B)

Beyond `eval_return`, log per-eval:
- `q_p5`, `q_p95`, `qscale_range_ema`
- `wm_grad_norm`, `pi_grad_norm` (pre-clip)
- `pi_entropy`, `scaled_entropy_mean`
- `latent_err_h0`, `latent_err_h1`, `latent_err_h2` (per-h consistency MSE)
- `max_reward_observed` (saturation watch)

When a run dips, these tell you WHY in one CSV. For our J3 1M
collapse window (step 950→1M), `wm_grad_norm` spiked 3× and
`latent_err_h2` tripled — pinpointed world-model momentary breakdown,
not Q-overestimation as initially hypothesized. Then re-eval on saved
ckpt confirmed.

---

## Don't trust paper Table 2 — read figure captions

Paper appendix Table 2 lists Humanoid Run action_dim=24. Figure 15
caption (page 23) says "Humanoid (A ∈ R²¹)". Our env is 21-dim;
Figure 15 is the source of truth, Table 2 is a typo.

Also Table 6: DMControl trains for **4-14M env steps**, NOT 1M. Don't
set "paper match @ 1M" as a target without verifying — source
reports per-task curves at 4M-14M depending on task difficulty.

Paper PDF: https://arxiv.org/pdf/2310.16828

---

## Don't-do summary

- Don't share buffer storage layout between sequence-sampling algos
  and single-transition-sampling algos without a stride parameter
- Don't conflate `done` and `terminated` in TD bootstrapping
- Don't assume PyTorch `.train()` / `.eval()` semantics map for free —
  thread `deterministic` + `rngs` per call site in JAX/Flax
- Don't burn the training PRNG inside the eval branch
- Don't use Table values as the only source of truth — cross-check
  with figures
- Don't run eval at 1M against a paper that reports asymptotic results
  at 14M
