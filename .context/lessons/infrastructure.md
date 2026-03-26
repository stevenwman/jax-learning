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

## Integer Division Truncation in Training Loop Bounds

`total_env_steps=200000`, `num_envs=128`. `200000 // 128 * 128 = 199936 < 200000`. Final eval never fired.

**Fix:** Post-loop finalizer. Never rely on hitting an exact step count.

---

## Verify Training Budget Against Published Results Before Debugging

Spent hours debugging Go2 PPO at eval ~17. Playground paper shows Go1 reaching ~25 at 100M steps. Our 50M runs were simply undertrained.

**Lesson:** Check what the reference achieves at the same training budget before debugging.
