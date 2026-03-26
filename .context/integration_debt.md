# Integration Debt

Technical debt from the dict obs / asymmetric actor-critic changes (2026-03-23).
These are working but fragile — fix before adding more features.

---

## 1. `select_action` dual role — training vs eval

**Problem:** `PPO.select_action()` computes both action AND value. Eval only needs action but must pass `critic_obs` (privileged_state) to avoid shape mismatch. Current workaround: eval bypasses `select_action` and calls `actor.apply` directly.

**Fix:** Split into `select_action_train(obs, critic_obs, key) → (action, log_prob, value)` and `select_action_eval(obs) → action`. Clean separation, no workaround needed.

**Risk if not fixed:** Any new eval caller (record_video, diagnostic scripts) must know to bypass `select_action`. Easy to forget → shape error.

---

## 2. ~~`record_video.py` dict obs support~~ — DONE (2026-03-25)

Fixed. Also saves `_traj.npz` + command arrow overlay.

---

## 3. ~~`critic_obs_buf` allocated as zeros every rollout~~ — NO LONGER RELEVANT

`train_ppo_fast.py` uses `lax.scan` — no Python-level buffer allocation. This item only applies to the legacy `train_ppo.py` path, which is not the active training script.

---

## 4. `_extract_obs` is a free function, not part of env interface

**Problem:** `_extract_obs(obs)` in `train_ppo.py` hardcodes `"state"` and `"privileged_state"` key names. If another env uses different keys, it breaks silently. Still relevant for `train_ppo.py`; `train_ppo_fast.py` bakes the key extraction at trace time so it fails loudly at compile rather than silently at runtime.

**Fix:** Either standardize the key names in all envs (document as convention), or have envs expose `policy_obs_key` / `critic_obs_key` properties.

---

## 5. Obs normalization: two separate normalizers

**Problem:** `train_ppo.py` maintains `norm_state` (policy) and `critic_norm_state` (critic) separately. Only `norm_state` is saved to checkpoint. Critic norm state is lost on resume.

**Fix:** Save both normalizer states in checkpoint, or compute critic norm from policy norm (since privileged_state contains state as a prefix).

**Risk if not fixed:** Resume from checkpoint will have stale critic normalization for the first few iterations.

---

## 6. PPO tests don't cover asymmetric mode

**Problem:** `test_ppo_setup.py` tests only symmetric PPO (same obs for actor and critic). The asymmetric path (different obs dims) is only tested indirectly via `test_go2_env.py` shape checks.

**Fix:** Add a test that runs PPO init + update with `critic_obs_dim != obs_dim`.

---

## 7. ~~`train_ppo_fast.py` online tracker shows 0.0 early~~ — DONE (2026-03-25)

Fixed: `.3g` format + `running_ep_return` persistence across scan calls. Validated on seed 2100 (shows real values from ~iter 50).
