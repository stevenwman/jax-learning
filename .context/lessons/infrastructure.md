# Infrastructure Lessons

---

## Complete Your Migrations

**What happened:** We built `DomainRandWrapper` to replace the legacy DR stack (`go2_randomize.py` + `DomainRandomizationVmapWrapper` + `--domain-rand` flag). But instead of deleting the old path, we archived it "just in case" and kept `env_setup.py` dispatching to both. Three days later, 11 files still had legacy DR references, new engineers had two ways to enable DR with unclear differences, and the CLI had a dead flag.

**Root cause:** Archive ≠ delete. "Keeping the old path alive during migration" is correct — but finishing the migration means removing it once the new path is validated. We validated per_step DR weeks ago and never went back to delete.

**Symptom of incomplete migration:** the new feature has 2+ entry points, config fields, or CLI flags coexisting for the same purpose. If you see `if reset_mode in ("legacy", "per_step")` alongside `if getattr(cfg, 'domain_rand', False)`, that's two DR switches for one feature. Delete one.

**Rule:** When introducing a replacement, put a deadline on the old path. Once the new one works: same-day deletion, same PR as the last validation. Do NOT "archive" production code — archive is for things with uncertain future, not legacy.

**Counter-example (good):** After the train_offpolicy.py → per-algo script split, train_offpolicy.py was *kept* as reference because the user explicitly requested it. But that decision was explicit and documented, not a default "just in case."

---

## Orbax Checkpointing

- `ocp.StandardCheckpointer()` saves/restores arbitrary pytrees (Linen or NNX)
- **Must call `checkpointer.wait_until_finished()`** after save — process exit kills the async write thread
- Save `meta.json` alongside checkpoint for config reconstruction
- Checkpoint + metrics CSV + meta.json in timestamped directories: `checkpoints/{timestamp}_{env}_seed{seed}/`

---

## Orbax Restore Requires Exact Pytree Structure Match

**Problem:** `record_video.py` crashed restoring a checkpoint saved with `optax.chain(clip_by_global_norm, adam)` while restore target used bare `optax.adam`.

**Targetless restore (emergency escape hatch):**
```python
raw = ocp.StandardCheckpointer().restore(os.path.abspath(ckpt_dir))
actor_params = raw['training_state']['actor_params']
```

**Correct design:** Save inference artifact separately:
```python
np.save(os.path.join(ckpt_dir, "actor_params.npy"), {
    "actor_params": jax.device_get(training_state.actor_params),
    "norm_mean": ..., "norm_mean_of_squares": ..., "norm_count": ...,
}, allow_pickle=True)
```

**Lesson:** Separate inference artifacts from training artifacts. Orbax for training resume, plain numpy for inference.

---

## Checkpoint Should Be Fully Self-Describing

Store `dataclasses.asdict(cfg)` in meta.json. All hyperparameters in one place — sufficient to reproduce any run.

**Lesson:** A checkpoint that requires external knowledge to load is incomplete.

---

## Video Recording — Two-Phase Approach

MuJoCo Playground's `env.render()` is CPU-side and can't be JIT'd.

1. **Phase 1 (GPU):** `jax.lax.scan` the rollout — fast (~0.3s for 1000 steps)
2. **Phase 2 (CPU):** Render frames from saved states — slow (~50ms/frame)

---

## Save Trajectory Data Alongside Videos — Always (2026-03-25)

Two policies with identical eval=11.6 had completely different behaviors. Invisible from video, obvious from .npz trajectory data.

**Lesson:** Video for qualitative checks. Trajectory data (qpos, qvel, actions, rewards, commands) for quantitative diagnosis. `record_video.py` auto-saves `_traj.npz`.

---

## Eval/Recording Must Match Training Preprocessing Exactly (2026-03-26)

**Problem:** FastSAC eval showed 225.7 avg. `record_video.py` produced instant death (7-50 steps). Concluded "SAC isn't doing well on Go2."

**Root cause:** Recording didn't apply obs normalization. Policy trained on normalized obs saw raw obs → garbage actions.

**Fix:** One line — `obs = norm_normalize(frozen_norm, obs)`.

**Lesson:** Any eval/recording/deployment code must replicate the EXACT preprocessing pipeline from training. Test recordings BEFORE concluding a policy is bad.

---

## Always Use Scientific Notation for Metric Printouts

PPO's VLoss formatted as `{:8.2f}` printed `0.00` for values like 0.003. We thought the value function wasn't learning.

**Fix:** All metrics use `.3e`. `0.003` displays as `3.000e-03`.

---

## CycloneDDS Requires Python <3.13 — Use Separate Deploy Venv (2026-03-26)

`cyclonedds==0.10.x` Python bindings have a C extension that references `_Py_IsFinalizing` — a symbol that changed in Python 3.13. Building from source (pip or git) all fail with `undefined symbol`.

**Fix:** Separate deploy venv with Python 3.12. Training stays on 3.13 (JAX/MJX). Deploy code is pure numpy anyway — no JAX dependency at runtime.

```
.venv/        → Python 3.13, JAX/MJX/Flax (training)
deploy/.venv/ → Python 3.12, numpy/cyclonedds/unitree_sdk2 (deployment)
```

Setup: `bash deploy/setup_deploy_deps.sh`

---

## Integer Division Truncation in Training Loop Bounds

`total_env_steps=200000`, `num_envs=128`. `200000 // 128 * 128 = 199936 < 200000`. Final eval never fired.

**Fix:** Post-loop finalizer. Never rely on hitting an exact step count.

---

## Verify Training Budget Against Published Results Before Debugging

Spent hours debugging Go2 PPO at eval ~17. Playground paper shows Go1 reaching ~25 at 100M steps. Our 50M runs were simply undertrained.

**Lesson:** Check what the reference achieves at the same training budget before debugging.

---

## `--eval-every` Is Episodes, Not Steps (2026-03-29)

**What happened:** Ran `--eval-every 5000000` expecting eval every 5M steps. Got zero evals in a 50M step run. Eval output was "missing" — thought it was buried in Warp warning spam.

**Root cause:** `train_ppo_fast.py` line 367: `if n_eps_total >= last_eval_eps + cfg.eval_every_n_episodes`. The `--eval-every` CLI flag maps to `eval_every_n_episodes`, NOT steps. 5M episodes is never reached in a 50M step run (~500k episodes total).

**Fix:** Use episode-scale values: `--eval-every 50000` for ~10 evals in a typical Go2 run. Or `--eval-every 100000` for ~5 evals.

**Lesson:** Read the argparse help text AND trace the flag through to where it's used. `--eval-every` is ambiguous — it could mean steps, episodes, or wall-clock seconds. The flag name doesn't tell you.

---

## Env Wrappers Must Be Applied In All Consumers (2026-04-01)

**Context:** `FrameStackWrapper` is applied in `env_setup.py` (used by training scripts). But `record_video.py` loads the env directly via `pg_registry.load()` and does NOT go through `env_setup.py`. A frame-stacked checkpoint (obs_dim=144) will fail at inference because record_video feeds raw 48d obs to a 144d network.

**Pattern:** Any env transformation (wrappers, obs preprocessing) applied during training must also be applied during inference/eval/recording. Every consumer of the env must apply the same wrapping chain, or the checkpoint is incompatible.

**Fixed:** `record_video.py` now reads `n_frame_stack` from `meta.json` and applies `FrameStackWrapper` before rollout.

---

## Brax Auto-Reset Does NOT Reset state.info (2026-04-01)

**What happened:** `FrameStackWrapper` stored the frame stack in `state.info["frame_stack"]`. After episode termination, Brax's `AutoResetWrapper` replaced `pipeline_state` and `obs` with cached initial values, but left `state.info` untouched. The frame stack retained frames from the dead episode. The first N-1 policy inputs of every new episode were contaminated.

**Root cause:** `AutoResetWrapper.step()` (Brax source) only does `jp.where(done, first_obs, obs)` and `jp.where(done, first_pipeline_state, pipeline_state)`. No other fields are reset.

**Fix:** Any per-env state in `state.info` that should reset at episode boundaries must handle it explicitly. Pattern: `jp.where(state.done, reset_value, normal_value)` inside the wrapper's `step()`. This is JIT-safe and adds negligible overhead.

**Applies to:** Frame stacking, action delay buffers, any FIFO/history stored in `state.info`.

---

## Inference Artifacts Must Include ALL Model State — Not Just Params (2026-04-08)

**Problem:** FlashSAC checkpoint's `actor_params.npy` saved actor params but NOT BatchNorm `batch_stats`. The saving code was added in a commit AFTER the training run finished. `record_video.py` rollout produced instant falls (26 steps), while training eval showed 282.

**Root cause:** FlashSAC's actor uses BatchNorm. At inference (`train=False`), BN uses running mean/var from `batch_stats`. Without them, BN normalizes with init-time zeros/ones → completely different activations → garbage actions. The full orbax checkpoint had batch_stats (saved for training resume), but the lightweight inference artifact didn't.

**Fix:** Extracted batch_stats from orbax checkpoint and patched `actor_params.npy`. Going forward, `save_checkpoint` now includes `actor_batch_stats` when present on `training_state`.

**Lesson:** Any model state that affects inference output must be in the inference artifact — not just learned params. For BatchNorm: running mean/var. For LayerNorm: nothing extra (stateless). For weight norm: nothing extra (applied to params). Test the inference artifact independently from the training checkpoint.

---

## Wrapper Composition: New Features Are Untested Until Combined (2026-04-10)

**Two bugs caught in one session, same root cause:** features that worked in isolation broke silently when combined with another feature nobody had paired them with before.

### Bug 1: JaxReplayBuffer dropped extra obs in frame-stack JIT path

`_make_jit_sample_fs` (frame-stack sampler) returned only `batch`, while `_make_jit_sample_with_idx` (asymmetric critic sampler) returned `(batch, idx)` so `_gather_extra` could fetch `critic_obs` for the sampled indices. The frame-stack path had a comment: *"For now, extra obs with frame-stack is not supported."* Six months later, somebody (us) tried frame-stack + asymmetric critic. The comment was load-bearing, not a TODO. KeyError on first gradient step.

**Fix:** Make the frame-stack JIT fn return `(batch, idx)` and call `_gather_extra` in the outer `sample()`. ~10 lines.

### Bug 2: DomainRandWrapper bypassed intermediate wrappers

`_swap_model` was implemented as:
```python
env = self.env.unwrapped
old = env._mjx_model
env._mjx_model = mjx_model
yield env  # ← bypasses every wrapper between DomainRand and base env
```

When `DomainRandWrapper(FrameStackWrapper(WarpJoystick))` ran, `_reset_with_model` called `v_env.reset()` on the *unwrapped* base env, skipping FrameStackWrapper entirely. The actor was built with `obs_dim=51` instead of 153. Crash on first eval. Worse: the bypass also breaks ActionDelay, any future obs/action wrappers.

**Fix:** Mutate `_mjx_model` on `self.env.unwrapped` (where the field lives), but **yield `self.env`** (the wrapped chain) so reset/step still goes through every wrapper.

### Lesson

Wrapper compositions are a combinatorial test surface. If feature A and feature B both work alone but were never tested together, **assume they don't compose**. Both bugs would have been caught by a single 5-line integration test:

```python
def test_frame_stack_with_dr_and_critic():
    cfg = TrainConfig(env_name=..., n_frame_stack=3, reset_mode="per_step")
    env, _, env_state, *_, obs_dim, _, _ = make_envs(cfg, seed=0)
    assert obs_dim == raw_dim * 3  # bug 2 catches this
    # train one step → bug 1 catches the buffer KeyError
```

**Pattern to enforce:** when adding a new wrapper, write at least one test combining it with every other wrapper that's already in the codebase. Not crossable: O(N²) tests for N wrappers, but N is small (≤5) and the test is cheap.

---

## mkdocstrings requires Google-style docstrings with correct section headers

**Symptom:** `mkdocs build --strict` fails with warnings about unresolvable parameters or unknown params on Flax `nn.Module` classes.

**Root cause:** mkdocstrings (via griffe) parses docstrings strictly. Two gotchas:
1. Flax `nn.Module` class attributes look like constructor params but griffe doesn't recognize `Args:` for them — must use `Attributes:` section header instead.
2. Untyped function parameters generate warnings under `--strict`. Add type annotations to all public function params.

**Fix:** Use `Attributes:` (not `Args:`) for `nn.Module` dataclass-style fields. Add type annotations to public API functions. If warnings persist, `warn_unknown_params: false` in mkdocs.yml as a last resort.

**Applies to:** Any new `nn.Module` class or public function that should appear in API docs.

---

## Extract Shared Loops as Functions, Not Classes

**What happened:** 4 off-policy training scripts (SAC, TD3, FastSAC, FastTD3) shared 85-90% identical code (~200 of 250 lines each). A code review identified the duplication and initially recommended re-unifying them into a single `train_offpolicy.py` with `if family == "sac"` branches. That was rejected — it would trade duplication noise for dispatcher noise. A Trainer base class was also explicitly rejected (project philosophy: no ABCs, no inheritance).

**Fix:** Extract the shared loop body into a single function `run_offpolicy_loop()` in `jax_rl/training/offpolicy_loop.py`. The 4 per-algo scripts call it after building their algo-specific pieces (optimizer, explore closure, log fields). Each script went from ~300 lines to ~115. FlashSAC stays standalone because its loop has genuinely different state management (BN stats, Zeta noise, adaptive reward scaling).

**The pattern:**
1. Identify the variation points (what differs between the N scripts)
2. Make those variation points function parameters
3. Copy the shared body into a function — verbatim, not "improved"
4. Thin each script to: build variation-point values → call the function

**When NOT to use this:** When the "shared" code has hidden divergence that will require `if algo_name == "sac"` branches inside the helper. That's re-unification in disguise. The helper should be branch-free. If an algo doesn't fit, it stays standalone (FlashSAC).

**Applies to:** Any time N > 2 scripts share > 70% identical code with well-defined variation points. The variation points must be narrow (≤ 5 parameters) or the function signature becomes its own complexity.

---

## Ghost Refs: Docs Reflecting Uncommitted Code

**What happened:** `.context/AGENT_HANDOFF.md` was updated during a 2026-04-10 experiment to claim `Go2 dict obs: {"state": (51,), "privileged_state": (125,)}` — reflecting an uncommitted branch that added linvel + accelerometer to the state group. The experiment was reverted (kept 48d for sim2real obs alignment), but AGENT_HANDOFF was never reverted. Two days later, a docs-site sweep propagated 51/125 to 10+ files. Nothing was technically broken — the code still worked — but every obs-dim claim in the public docs was a lie. A user running `make_env_bundle(cfg)` and printing `bundle.obs_dim` would see 48, not 51.

**Root cause:** AGENT_HANDOFF was treated as authoritative for obs dim claims. It wasn't — it was a snapshot of an aspirational state. The subagents doing the doc sweep trusted it. Three reviewers trusted it. The fourth (undergrad CS persona) cross-referenced against `jax_rl/envs/locomotion/go2_warp_joystick.py` and found the gap.

**Fix:** Revert docs to 48/122. Add a note to the benchmark table that 285.1/280.1 results came from the reverted 51d experiment and are **not reproducible with current code**.

**Prevention:**
1. **Don't update AGENT_HANDOFF based on uncommitted changes.** Write "target dim: 51d, current: 48d (in progress)" if you must capture the plan. Or don't capture it in AGENT_HANDOFF at all — use a journal entry.
2. **When doing doc sweeps, cross-reference against code, not against internal handoff docs.** Grep the env file, don't trust the handoff.
3. **In reviews, one persona should be an "undergrad CS cross-referencer"** whose explicit job is to check every claim in the docs against the actual source code. The other personas (high schooler, frontend, even PhD) tend to trust the docs as ground truth.

**Applies to:** Any claim about code behavior in AGENT_HANDOFF, README, NEW_AGENT_PROMPT, or any doc that doesn't auto-generate from source. The higher the doc in the "authoritative reference" hierarchy, the more dangerous a ghost ref becomes — downstream docs will copy it.
