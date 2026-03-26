# Integration Debt

Technical debt from the dict obs / asymmetric actor-critic changes (2026-03-23).
These are working but fragile — fix before adding more features.

---

## 1. ~~`select_action` dual role~~ — DONE (2026-03-26)

Added `select_action_eval(actor_params, obs) → action` to PPO. No critic_obs needed, no value computation. `select_action()` still exists for training (returns action, log_prob, value). Eval callers use `select_action_eval()` instead.

---

## 2. ~~`record_video.py` dict obs support~~ — DONE (2026-03-25)

Fixed. Also saves `_traj.npz` + command arrow overlay.

---

## 3. ~~`critic_obs_buf` allocated as zeros every rollout~~ — NO LONGER RELEVANT

`train_ppo_fast.py` uses `lax.scan` — no Python-level buffer allocation. This item only applies to the legacy `train_ppo.py` path, which is not the active training script.

---

## 4. ~~`_extract_obs` hardcoded keys~~ — RESOLVED BY CONVENTION (2026-03-25)

All envs use `"state"` / `"privileged_state"` keys, matching Playground Go1. Convention documented. Three scripts use it consistently.

---

## 5. ~~Critic norm state lost on resume~~ — DONE (2026-03-26)

`save_checkpoint` and `load_checkpoint` now accept optional `critic_norm_state`. Saved to orbax checkpoint alongside policy norm state.

---

## 6. ~~PPO tests don't cover asymmetric mode~~ — DONE (2026-03-26)

Added `test_asymmetric_ppo()` — tests PPO init, select_action, and select_action_eval with critic_obs_dim=116 (different from actor obs_dim=17).

---

## 7. ~~`train_ppo_fast.py` online tracker shows 0.0 early~~ — DONE (2026-03-25)

Fixed: `.3g` format + `running_ep_return` persistence across scan calls. Validated on seed 2100 (shows real values from ~iter 50).
