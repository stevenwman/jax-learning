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

**Recurrence (2026-04-12, same day):** Found another ghost ref in `docs/tutorials/train-locomotion.md` — claimed "~18,000 steps/second on RTX 4090" for FastSAC on Go2. Actual Go2WarpJoystickFlat FastSAC runs at 3.4-3.95k sps per journal entries. The 18k number was a real measurement, but from CheetahRun (a DM Control benchmark), not Go2. A writer cited it generically on the Go2 page. Same pattern as the obs dims bug: real number measured in context A, carried over to context B without the context. Fix: replaced with "~3-4k sps" + note that DM Control benchmarks run much faster. Lesson reinforced: **treat every specific number in docs as a citation requirement**. When copying a benchmark figure, copy the env name with it.

---

## Validation Blocks Catch Cross-Agent Drift

**What happened:** Two parallel subagents were dispatched — one adding docs content, one applying frontend polish. Frontend agent added a `validation:` block to `mkdocs.yml` with `unrecognized_links: warn`. During its own build, it caught a broken link (`api/algos.md` → `reference/training-loop.md`) — a file the OTHER agent was still writing. Without the validation block, the broken link would have shipped and only surfaced when a user clicked it.

**Prevention:** `validation.links.unrecognized_links: warn` (and `validation.nav.omitted_files: warn`) in `mkdocs.yml` should be enabled by default for any MkDocs site. They don't break the build (warn, not error) but surface drift immediately — especially useful when multiple agents or contributors are editing in parallel.

**Applies to:** Any static site generator with link-validation support. The cost is zero (warnings only appear when something is actually broken); the benefit is catching problems before deployment.

---

## Doc-Drift Test Suite

**What happened:** Four rounds of 4-persona docs review today. Each round caught the prior round's cleanup artifacts:
- Round 1 → obs dims 48/122 vs 51/125
- Round 2 → truncation bug + 18k sps ghost-ref
- Round 3 → Round 2's cleanup hallucinated `build_env_bundle` + forgot to update `handle_truncation` constructor docs
- Round 4 → FAQ PPO Go2 timing lie, FastSAC `policy_delay` missing from docs, silent-zero fallback on `info["truncation"]`

The pattern wasn't "these particular reviewers are better" — it was "cleanup of round N creates ghost refs for round N+1, independent of who reviews."

**Fix:** `tests/test_docs_code_blocks.py` + `tests/test_docs_drift.py` (commit `2d2c807`) mechanize the cross-checks that humans kept making:

1. **Python fence compile-check.** Every ` ```python ` fence in `docs/**/*.md` is compiled (not executed — too slow + too many variables). Skips fences with `...`, ellipsis, or `{: .no-test}` attr-list markers. Catches syntax errors + hallucinated import names.

2. **Import resolution.** Every `from X import Y` in a docs fence is tried. If `Y` doesn't exist in module `X`, test fails pointing at the exact file:line. Catches `build_env_bundle`-class hallucinations.

3. **Constructor kwarg consistency.** For each algo (SAC/TD3/FastSAC/FastTD3/FlashSAC), parse `docs/api/algos.md` for the constructor signature block, compare against `inspect.signature(AlgoClass.__init__)`. Any kwarg in docs that's not in the real constructor fails. Catches `handle_truncation`-class ghost refs.

4. **Reverted-symbol greps.** `test_no_reverted_obs_dims()` asserts `51d`, `(51,)`, `125d`, `(125,)` never appear in docs/.context (except archive/historical files). `test_no_archived_script_refs()` asserts `train_offpolicy.py` never appears. One-line protections against resurrecting fixed bugs.

5. **CLI flag drift.** For each `train_*.py`, parse its argparse via subprocess `--help`, compare against the flags listed for that script in `docs/reference/cli-flags.md`. Missing flags and phantom flags both fail. Catches flag drift AND misleads in the generator script.

6. **arXiv ID resolution** (`@pytest.mark.slow`-gated). HEAD-checks every `arXiv:NNNN.NNNNN` reference. Catches hallucinated citations. Added round 4 after undergrad reviewer falsely flagged real IDs as fake — worth the test as insurance for future writing.

**First run caught 16 real drift issues.** The test suite is the durable infrastructure output — future sessions get same-commit feedback instead of requiring a 4-persona review round.

**What the tests CANNOT catch** (still requires human review):
- FAQ claims that are factually wrong (PPO Go2 timing — the code doesn't lie about it, but the claim in docs is false)
- Algorithmic descriptions that omit features (FastSAC `policy_delay` — code has it, docs don't mention it, test has no way to know)
- Missing documentation of features (DomainRandWrapper absent from wrappers.md pipeline section)
- Wrong benchmark numbers cited generically (the 18k sps ghost — test can't know which env a number was measured on)

**Prevention of future drift:**
- New docs pages with code fences → tests validate them on next commit
- Code refactors that rename/delete symbols → tests flag any docs stragglers
- Quarterly: run the full 4-persona review to catch the semantic drift classes the tests can't

**Applies to:** Any docs site that makes claims about code. The cost is one-time (~600 lines of test code); the benefit compounds with every cleanup pass. Invest early.

---

## XLA Memory Fraction Has To Drop For Bigger-Network Algos

**What happened (2026-04-13):** FlashSAC Go2 OOM'd twice during MuJoCo Warp graph creation, around step 10-15k:
```
Warp CUDA error 2: out of memory (in function wp_cuda_graph_create_exec, ...)
```

GPU had 13.6 GiB free at launch. JAX preallocated 0.7 of total = ~11.4 GiB. FlashSAC's networks are 2-3x larger than FastSAC (Q params 1.1M vs 495k each, plus BatchNorm running stats for 4 critics = online + target × 2). JAX heap actually used more of that 0.7 fraction → less left for Warp's CUDA graph capture.

**Fix:** Lower `XLA_CLIENT_MEM_FRACTION` to 0.55 for FlashSAC. Worked first try.

**General rule:** the `XLA_CLIENT_MEM_FRACTION=0.7` default in `train_*.py` scripts is calibrated for FastSAC/FastTD3 sized networks. Bigger algos (FlashSAC) need 0.55-0.6. Smaller algos (vanilla SAC/TD3 at 128 envs) can run higher. The right per-script default is roughly `1 - (model_size / VRAM)` with some headroom for Warp's graph allocations.

**TODO:** consider bumping the `os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.7")` line in `train_flashsac.py` to `"0.55"` so users don't hit the OOM. Low-priority but a good sharp-edge to dull.

---

## CheckpointManager "Best" Tracking Excludes Final Eval

**What happened (2026-04-13):** Recording benchmark numbers for FlashSAC Go2 10M, found:
- Best in-loop eval (CheckpointManager.best_eval): 279.5
- Final eval (after training loop ends): 284.5

The 284.5 was higher but didn't trigger "New best!" in the logs because the final eval is a separate code path in `eval_runner.py::final_eval_and_checkpoint`, and that path doesn't compare against `ckpt_mgr.best_eval` to maybe update it.

**Implication:** Reporting "best eval" from grep-ing "New best!" lines undercounts the true peak performance. The right number is `max(in_loop_best, final_eval)`.

**Update (2026-04-14):** On closer inspection, the checkpoint artifact was already correct — `final_eval_and_checkpoint` already passed `eval_mean` to `ckpt_mgr.save()`, which internally updated `best_eval` and wrote to `best_dir` when the final eval beat the in-loop best. Only the stdout announcement was missing. Fixed in `eval_runner.py::final_eval_and_checkpoint` by capturing `is_best` and printing "New best!" — now matches the in-loop path's behavior.

**Applies retroactively to:** Any benchmark recorded before 2026-04-14 — grep "New best!" alone undercounted peaks; always cross-check with the final "Eval return: X" line. Post-fix, grep is sufficient.

---

## `record_video.py` Memory Fix: `PREALLOCATE=false` + Python Loop (Not `lax.scan`)

**What happened (2026-04-13):** Trying to render best-checkpoint videos for the 4 post-truncation-fix runs. GPU showed 13.6 GiB free (one other user's process at 1.1 GiB), should have been fine. Both `XLA_CLIENT_MEM_FRACTION=0.7` (default) and `=0.5` OOM'd:
```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.49GiB
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.12MiB  # ← even smaller!
```

The 1.12 MiB OOM is the giveaway: not about the requested size, it's about JAX's preallocator grabbing a large contiguous chunk that doesn't fit alongside the other process + Warp's graph-capture buffer.

**Quick fix (applied to `record_video.py`):** `os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")`. On-demand allocation lets JAX and Warp share memory instead of fighting for a pre-locked slab. Solves the symptom.

### Where the ~1.5 GiB actually went (corrected analysis)

Initial guess blamed "Warp graph capture" entirely with trajectory buffer dismissed as ~1 MB. That was wrong. Real breakdown:

1. **Scan output preallocation (500 MB–1 GB).** `jax.lax.scan(rollout_step, init, length=1000)` allocates output buffers for the *full mjx State × 1000* upfront. The State isn't just `qpos`/`qvel` — it's the full MJX `Data` struct with ~50 fields: `qM` (nv×nv), `cinert`/`crb` (nbody×10), `actuator_moment`, all `efc_*` and `contact.*` arrays padded to `nconmax`/`njmax`, plus the `info` dict (reward_components, command, last_act, ...). For Go2: ~0.5–1 MB per step × 1000 steps.

2. **Warp graph-capture scratch (100–500 MB).** `_default_nconmax` = 48, `_default_njmax` = 64 for Go2 (heuristic floors of 45/53 rounded to valid tile sizes). Plus `_get_padded_sizes` pads for JtDAJ tile alignment. Per-step these are tiny — but with `opt.iterations=100` + `ls_iterations=50` + elliptic cone (`opt.cone=1`) + hundreds of kernels across `collision_*`, `constraint`, `solver`, `smooth`, `sensor`, `passive`, `forward`, `derivative`, the captured graph carries substantial workspace.

3. **XLA compilation working buffers** (tens-hundreds of MB).

Sum = 1–2 GB, matching the 1.49 GiB ask.

### Why a Python loop is the right architectural fix (not `lax.scan`)

`lax.scan` is for *training* (amortize compile over many rollouts, full traj is your data). For *one-shot inference*:
- You pay full compile cost for one execution (no amortization)
- You preallocate the full 1000-step State buffer once, discard most of it
- You can't early-stop on `done` cleanly

Refactor applied to `record_video.py`: replace `lax.scan` with a Python loop calling `jit(rollout_step)`. Each step:
- JIT compiles `rollout_step` *once* on first call (Warp graph captured once, reused)
- Single State buffer reused each iter (no 1000-step preallocation)
- `np.asarray(state.data.qpos)`, `np.asarray(state.data.qvel)`, action, reward, and info fields copied to host-side lists — only what's needed
- `break` on `done` for natural early-stop

**Measured result:** Peak HBM drops ~500 MB–1 GB. NPZ output schema unchanged (22 keys identical). First-run compile cheaper than scan (~7s vs ~20s) because XLA doesn't analyze a 1000-step unrolled body. Dispatch overhead: ~100 ms total over 1000 steps — negligible vs ~30s CPU render phase.

Both fixes landed together — `PREALLOCATE=false` handles the immediate preallocator-vs-Warp contention (Warp graph creation inside `env.reset` needs memory *before* the scan/loop even starts), the Python loop drops steady-state peak so renders coexist with concurrent training on the same GPU.

### What I initially dismissed and was wrong about

Original note said the Python loop "adds complexity; the env var fix is simpler." Reality: the env var alone only fixes the contested-GPU case. If you run a render while training on the same GPU, training's preallocator plus the render's 1 GB scan peak still OOMs. The Python loop makes the render memory-coexistent with other workloads.

**Applies to:** Any one-shot Warp+JAX inference script. The `lax.scan` pattern copied from training is the wrong default for single-rollout use cases. Use Python loop + jitted step.

---

## `md_in_html` doesn't propagate `markdown` to child HTML elements

**What happened (2026-04-13):** Homepage video grid captions rendered broken on the deployed site:
```
Go2 locomotion — [FastSAC](api/algos.md#fastsac){.gl}, eval 276.5
```
— raw markdown link syntax showing as literal text.

**Setup:** The `glossary_links.py` hook (enabled in `mkdocs.yml:59`) auto-links the first occurrence of each glossary term with `[term](target){.gl}` markdown syntax. For typical prose, mkdocs' core markdown processor converts this to `<a>` tags as expected.

**The interaction bug:** The video grid used nested HTML wrappers:
```html
<div class="video-grid" markdown>  <!-- outer has markdown -->
<div>                              <!-- inner does NOT -->
<video>...</video>
<p class="caption">Go2 locomotion — FastSAC, eval 276.5</p>
</div>
</div>
```

The hook injected `[FastSAC](...){.gl}` into the caption text. But `md_in_html`'s `markdown` attribute only enables markdown processing for the **direct text children** of the annotated element — it does NOT propagate to nested `<div>`s or `<p>`s. The inner `<div>` and the `<p class="caption">` both needed their own `markdown` / `markdown="1"` attributes.

**Fix:** add `markdown` to the inner `<div>` AND `markdown="1"` to each `<p class="caption">`. Then hook-injected markdown renders correctly.

**Why this survived 6 rounds of review:**
- Each component works in isolation (hook, md_in_html, custom HTML all valid).
- Drift tests can't catch it — source markdown is valid, config is valid, only rendered output is broken.
- Frontend-persona reviewers read `site/*.html` but apparently skimmed the caption as-rendered without noticing `[FastSAC](...){.gl}` was literal rather than a link. Captions are small; easy to miss.

**Detection rule:** after building, grep `site/*.html` for raw unprocessed markdown:
```bash
grep -rE '\[[A-Za-z][^\]]*\]\([^)]+\)\{\.' site/ | head  # hook output that didn't render
```
Zero matches = clean. Any matches = an HTML wrapper is missing `markdown`.

**Applies to:** Any MkDocs Material site using `md_in_html` + a markdown-generating hook (glossary auto-linking, shortcode expansion, term replacement) + custom HTML wrappers (video grids, card layouts, hero banners, two-column sections). The combination is the failure mode. Single-component usage is fine.

---

## Schema-from-Checkpoint for Deploy Obs (2026-04-24)

**Problem:** deploy code hardcoded the obs layout. The only sim↔deploy contract was `obs_dim` (an int). When sim added/removed/reordered an obs term, deploy stayed silently mis-wired as long as the total dim still matched. Twice (linvel removed 2026-04-10, accelerometer added later) → ~14 days of Go2 ckpts where deploy passed real `gyro` data into the slot the policy interpreted as `accelerometer`. No test caught it because dim equality held.

**Fix:** treat the obs term list as part of the checkpoint. At save time, serialize `env._obs_groups` to `meta.json["obs_schema"]["state"]` = ordered list of term names (resolving `IncludeGroup` references). At load time, `ObsBuilder.from_checkpoint(ckpt_dir)` reads the schema and composes obs in the saved order via a sensor-fetcher registry keyed by term name. Adding a sim term = 1 line in env + 1 line in deploy registry; new ckpts deploy automatically; old ckpts ignore unknown terms.

**Pattern, when applicable:**
- The producer (training env) and consumer (deploy / inference) are decoupled in time and codebase.
- The interface is a positional vector with named components.
- A hash check on dim alone is insufficient — content can drift while dim is preserved.

→ Serialize the schema, not just the dim. Deploy reads it. Lookup table at the consumer maps name → fetcher fn. Unknown name = clear error pointing at the file to edit, not silent miswiring.

**Anti-pattern this replaces:** "we just need to remember to update both files together." Twice in our codebase, that broke. A second source of truth for the obs layout (in deploy code) cannot stay in sync with the env definition by convention alone — the env is allowed to change, deploy is allowed to be stale, and there's no compiler check.

**Cost:** ~150 LOC across `obs_spec.py:schema_from_obs_groups`, `checkpointing.py` (5-line addition piggybacking on existing env-load for dr_specs), `deploy/obs_builder.py` (sensor registry + factory). Plus 9 unit tests. Backward-compat: schema-less ckpts fall back to a printed default.

**Generalizes to:** any environment with composable obs (privileged_state, contraction obs groups, vision tokens). Same `_obs_groups` mechanism, same `schema_from_obs_groups` flatten. Bongo and pusht envs would benefit immediately if/when they get deployed.
