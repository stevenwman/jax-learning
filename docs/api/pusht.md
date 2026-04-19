# PushT (vendored gym-pusht)

Diffusion Policy push-T benchmark (Chi et al. 2023), vendored from HuggingFace [gym-pusht](https://github.com/huggingface/gym-pusht) into the repo so we can pin the physics, extend rewards for RL, and bundle the expert demos. Apache 2.0, license retained.

Old, static benchmark. The point of vendoring is that old benchmarks **should** be static — no floating pip version, no silent API breaks from upstream deps (pymunk 7 already broke upstream once), reproducible across years.

!!! note "Location"
    `jax_rl/envs/manipulation/pusht/` — env + contact helper + LICENSE + 206 expert demos (0.29 MB).

---

## Usage

```python
import gymnasium as gym
from jax_rl.envs.manipulation.pusht import PushTEnv

env = PushTEnv(obs_type="state", reward_mode="contact_gated")
env = gym.wrappers.TimeLimit(env, max_episode_steps=300)   # CRITICAL
obs, info = env.reset(seed=0)
for _ in range(300):
    action = env.action_space.sample()     # (2,), XY target in [0, 512] px
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break
```

!!! danger "TimeLimit is not applied by direct construction"
    `gym.make("gym_pusht/PushT-v0")` auto-wraps with `TimeLimit(300)` via the env registry. Direct `PushTEnv(...)` does NOT. Without TimeLimit, failed episodes run indefinitely during RL training — the critic bootstraps infinite future and diverges. **Always wrap with `gym.wrappers.TimeLimit(env, max_episode_steps=300)` when constructing directly.** We caught this the hard way: missed the wrapper for 8 training runs, hit a "14% coverage ceiling." One-line fix, 6× jump to 88%.

- Single-env, CPU, pymunk 2D physics. ~2k sps single process.
- `action_space = Box([0, 0], [512, 512])` — absolute agent target in pixel coords.
- `observation_space` varies by `obs_type`: `"state"` (5d), `"pixels"` (96×96×3), `"pixels_agent_pos"` (dict), `"environment_state_agent_pos"` (dict with 16d keypoints + 2d agent).

---

## Reward modes

Added to the vendored env via `reward_mode=` kwarg. Coverage is the DP default; the rest are for RL-from-scratch experiments where pure coverage is too sparse.

| Mode | Reward per step | Use case |
|------|-----------------|----------|
| `coverage` (default) | `clip(coverage / 0.95, 0, 1)` — DP's exact formula | Matches DP paper baseline; sparse for pure RL |
| `sparse` | `1.0 if coverage > 0.95 else 0.0` | Cleanest signal; hardest to learn from scratch |
| `shaped` | `coverage + 0.01 * n_contacts - 0.001 * pusher_to_block` | Mild shaping atop coverage |
| `approach` | `coverage + 0.05 * (1 - pusher_to_block / diag)` | Adds smooth proximity bonus |
| `dense` | `r_coverage + 0.01·(prev_pos_err - pos_err) + 0.5·(prev_angle_err - angle_err) + 0.01·n_contacts + 5·is_success` | Delta shaping; telescoping |
| `contact_gated` | `r_coverage + 0.2·(1-in_contact)·exp(-d/50) + 0.15·correct_side + 0.3·in_contact·v̂·ĝ + 2·angle_delta_gated + 50·is_success` | **Recommended for RL from scratch** — Cong 2023 + Ferrari 2025 recipe |

All modes terminate on `coverage > 0.95` per DP convention. `info` dict always includes:

- `coverage`, `is_success`, `n_contact_points` — common diagnostics
- `pusher_to_block`, `block_to_goal`, `angle_err`, `block_vel_toward` — geometry signals
- `r_coverage`, `r_pos`, `r_angle`, `r_approach`, `r_block_vel`, `r_contact`, `r_success` — per-component reward split (regardless of selected mode)

Per-component breakdown enables post-hoc analysis like "was the policy earning mostly r_pos or mostly r_angle" without a retrain. Zero-cost instrumentation — always on.

### Dense reward term definitions

- `r_pos = 1 - min(block_to_goal / 512, 1)` — block in the goal zone ≈ 1, far away ≈ 0
- `r_angle = 1 - min(angle_err / π, 1)` — aligned yaw ≈ 1, flipped ≈ 0
- `r_approach = 1 - min(pusher_to_block / (512·√2), 1)` — touching block ≈ 1
- `r_block_vel = clip(block_vel · unit_to_goal / 50, -1, 1)` — block moving toward goal gives positive reward
- `r_contact = 0.01 * n_contact_points` — small reward per pymunk contact point per step
- `r_success = 5.0 if coverage > 0.95 else 0` — large terminal bonus

Weights (0.3, 0.2, 0.1, 0.1, 0.01) tuned so no single shaping term exceeds the 1.0 coverage signal. `r_success = 5` gives the solving episode a clear one-shot advantage.

---

## Expert demos (LeRobot)

206 human-teleop demonstrations (25,650 transitions) bundled at `demos/pusht_demos.npz`. Source: [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) (Apache 2.0).

```python
from jax_rl.envs.manipulation.pusht.demos import load_demos, get_episode, n_episodes

n_episodes()        # → 206
d = load_demos()    # dict of numpy arrays: obs_state, actions, rewards, dones, successes, ...
ep0 = get_episode(0)   # one episode's slice: 161 frames for ep 0
```

Keys in packed `.npz`:

| Key | Shape | Dtype | Meaning |
|-----|-------|-------|---------|
| `obs_state` | (25650, 2) | float32 | Agent XY at each step (DP teleop cursor pos) |
| `actions` | (25650, 2) | float32 | Target XY commanded by human at each step |
| `rewards` | (25650,) | float32 | DP coverage reward per step (max 0.9489 across all demos) |
| `dones` | (25650,) | bool | Episode terminal flag |
| `successes` | (25650,) | bool | Success flag (all False — humans didn't cross 0.95 threshold) |
| `episode_idx` | (25650,) | int32 | Episode index each row belongs to |
| `frame_idx` | (25650,) | int32 | Frame index within episode |
| `ep_bounds` | (206, 2) | int32 | (start, end) row indices per episode |

!!! info "Demo quality caveat"
    None of the 206 human demos actually cross the 0.95 coverage threshold — max observed 0.9489. Human teleop is imperfect; demos approach the target but don't consistently solve it. Consistent with DP paper. BC-only policies trained on these will have < 100% success rate.

### Refresh demos

Re-download from HuggingFace:

```bash
uv run python jax_rl/envs/manipulation/pusht/demos/download.py
```

---

## Parity with upstream

[tests/test_pusht_parity.py](https://github.com/stevenwman/jax-learning/blob/main/tests/test_pusht_parity.py) confirms:

- **Byte-exact obs and reward match** with `pip install gym-pusht` in `coverage` mode over 100 random steps (seed 42).
- 4 reward modes produce distinct values.
- Unknown `reward_mode` raises `ValueError` at construction.

Run with `uv run pytest tests/test_pusht_parity.py -v`.

---

## Why vendor?

- **Upstream is static.** Paper from 2023, little ongoing development. No value in floating pip version.
- **Upstream deps break.** gym-pusht requires pymunk<7; pymunk released 7.2 with API removal, `pip install gym-pusht` now fails out-of-box. Pinning pymunk in our `pyproject.toml` is a ticking time bomb.
- **API extension.** `reward_mode` kwarg is 20 LOC but critical for RL experiments; forking upstream would be overkill.
- **Demos size.** 0.29 MB fits in repo without git-lfs.
- **Reproducibility.** A future user clones this repo in 2030 and everything still works.

---

## Training from Scratch — Working Recipe

Pure-RL SAC trained from scratch with no demos reaches **~84% mean coverage / 89% peak stochastic** on push-T at 2M env steps (~35 min wall-clock). Published BC+RL (DPPO) exceeds 95%; our number is a pure-RL reference point.

### Use `train_pusht.py`

```bash
uv run python train_pusht.py --reward-mode contact_gated \
    --obs-type environment_state_agent_pos --frame-stack 3 --action-repeat 2 \
    --total-timesteps 2000000 --num-envs 8 --buffer-size 500000 \
    --batch-size 1024 --grad-updates-per-step 2 \
    --reward-scale 0.1 --grad-clip-norm 1.0 --target-entropy-scale 2.0 \
    --lr 1e-4 --gamma 0.995 \
    --eval-every-n-steps 50000 --wandb
```

### Why each knob matters

1. **`contact_gated` reward** — gates shaping on pusher-block contact state. Not farmable like absolute distance shaping (which we tried first, policy hovered near block and collected shaping bonuses without pushing).
2. **`environment_state_agent_pos` obs** — 16d T-vertex keypoints + 2d agent. Captures geometry 5d state obs misses.
3. **`frame_stack=3`** — implicit velocity from obs history. gym-pusht state obs has no velocities.
4. **Obs normalization** (automatic in `train_pusht.py`) — pixel coords [0, 512] → [-1, 1]. Without, SAC critic explodes early.
5. **`action_repeat=2`** — commits each policy decision for 2 env steps (FiGAR). Lets the policy complete a short push before reconsidering. Halves policy frequency 10Hz → 5Hz.
6. **`reward_scale=0.1`** — brings Q values to a tractable range. Raw sum over 300 steps otherwise hits Q ~ 150.
7. **`grad_clip_norm=1.0`** — damps early Q-loss explosion during warmup.
8. **`target_entropy_scale=2.0`** — target_entropy = -4 (= 2 × action_dim). Default (-action_dim) pushes alpha to near-zero; slightly higher keeps exploration alive.
9. **`gamma=0.995`** — effective horizon ~200 steps (vs ~100 with 0.99). Matches 300-step episode length better.
10. **`lr=1e-4`** — `3e-4` (SAC default) was unstable with this reward scale; `1e-4` converges cleanly.
11. **`TimeLimit(300)`** — see danger note above. Without this, everything above fails.

### Individual contributions (ablation sketch, not rigorously re-run)

- `state` obs (5d) + no frame_stack: ~14% sto coverage ceiling
- +keypoint obs + obs normalization: minor gain, unstable without TimeLimit
- +frame_stack: helps policy temporally reason about motion
- +action_repeat: enables cleaner multi-contact sub-sequences
- +TimeLimit: unlocks everything, **6× overall jump to 84%**

The single highest-leverage fix was TimeLimit. Everything else delivered 1-2% each.

### Eval result interpretation

Our 5-episode deterministic eval diagnostic (`tools/pusht_eval_diag.py`):

| Episode | Final coverage | Position error (px) | Angle error |
|---------|---------------|--------------------|--------------| 
| 0       | 87.4%         | 6.3                | 0.75°        |
| 1       | 80.4%         | 8.5                | 0.24°        |
| 2       | 86.5%         | 6.6                | 1.25°        |
| 3       | 80.2%         | 8.9                | 0.02°        |
| 4       | 85.8%         | 7.0                | 4.66°        |

**Mean: 84% cov, 7.5 px pos error, 1.4° angle error.** Angle control is tight. The remaining gap is **position refinement** — policy stops ~7 px short of perfect alignment.

!!! note "95% threshold vs human performance"
    The bundled `lerobot/pusht` dataset contains 206 human teleop demos. Max coverage across all 25,650 frames in those demos: **0.9489**. Not a single human demo frame crosses the 0.95 threshold. Humans don't "solve" push-T under this success criterion either. Our RL policy hits ~94% of human peak. Published >95% results require BC + gradient-based refinement.

## Comparison to `push_env.py`

`jax_rl/envs/manipulation/push_env.py` is our custom shape-agnostic pushing benchmark for RL adaptability studies (train on T, zero-shot on L / circle / plus). Different purpose.

| | PushTEnv (vendored) | PushEnv (ours) |
|---|---|---|
| Purpose | DP paper parity, BC comparisons | Shape-transfer adaptability benchmark |
| Backend | pymunk 2D (CPU, single-env) | MuJoCo Warp (GPU, 1024+ envs vmap) |
| Shapes | T only | T, L, circle, plus (swap via config) |
| Obs | 5d state (DP) or pixels 96×96 | 16d state (pusher+block+vels+last_act) |
| Action modes | Position-PD (fixed) | Position-PD / velocity-delta / teleport |
| Reward modes | 4 (coverage/sparse/shaped/approach) | 3 (dense/sparse/shaped with 7 components) |
| Expert demos | 206 teleops bundled | None |
| Throughput | ~2k sps single process | 360k sps @ 1024 envs |

Rule of thumb: **vendored pusht** for publishing numbers comparable to DP; **push_env** for large-batch GPU RL research.
