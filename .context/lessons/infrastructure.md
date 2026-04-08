# Infrastructure Lessons

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

## mkdocstrings requires Google-style docstrings with correct section headers

**Symptom:** `mkdocs build --strict` fails with warnings about unresolvable parameters or unknown params on Flax `nn.Module` classes.

**Root cause:** mkdocstrings (via griffe) parses docstrings strictly. Two gotchas:
1. Flax `nn.Module` class attributes look like constructor params but griffe doesn't recognize `Args:` for them — must use `Attributes:` section header instead.
2. Untyped function parameters generate warnings under `--strict`. Add type annotations to all public function params.

**Fix:** Use `Attributes:` (not `Args:`) for `nn.Module` dataclass-style fields. Add type annotations to public API functions. If warnings persist, `warn_unknown_params: false` in mkdocs.yml as a last resort.

**Applies to:** Any new `nn.Module` class or public function that should appear in API docs.
