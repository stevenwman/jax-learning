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

Pure-RL SAC trained from scratch with no demos reaches **~93% mean coverage stochastic** on push-T at 2M env steps (~35 min wall-clock). Published BC+RL (DPPO) exceeds 95%; our number is a pure-RL reference point, within 2pp of the 95% threshold (which is above human teleop peak of 0.9489).

!!! tip "Latest (2026-04-20): log-barrier coverage reward"
    Switching `coverage_shape="log_barrier"` lifts ceiling from 0.85 → 0.93 sto (+8pp). Default is still `"linear"` for literature parity; pass `--coverage-shape log_barrier` to train_pusht.py for the new best config.

### Use `train_pusht.py`

```bash
# Full-stack config (0.933 sto)
uv run python train_pusht.py --reward-mode contact_gated \
    --obs-type environment_state_agent_pos --frame-stack 3 --action-repeat 2 \
    --coverage-shape log_barrier --coverage-eps 0.01 \
    --total-timesteps 2000000 --num-envs 8 --buffer-size 500000 \
    --batch-size 1024 --grad-updates-per-step 2 \
    --reward-scale 0.1 --grad-clip-norm 1.0 --target-entropy-scale 2.0 \
    --lr 1e-4 --gamma 0.995 \
    --eval-every-n-steps 50000 --wandb

# Minimal shape-agnostic config (0.939 sto) — better than full stack and
# uses 5d state obs that generalizes across shapes.
uv run python train_pusht.py --reward-mode contact_gated \
    --obs-type state --frame-stack 1 --action-repeat 2 \
    --coverage-shape log_barrier --coverage-eps 0.01 \
    --total-timesteps 2000000 --num-envs 8 --buffer-size 500000 \
    --batch-size 1024 --grad-updates-per-step 2 \
    --reward-scale 0.1 --grad-clip-norm 1.0 --target-entropy-scale 2.0 \
    --lr 1e-4 --gamma 0.995
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

### Ablation study (rigorously re-run 2026-04-20)

9-run study isolating each component. All at 2M steps, seed 0.

| Run | Config diff from baseline_logbar | Det | Sto |
|---|---|---|---|
| v9 (linear) | full stack, reward `linear` | 0.841 | 0.852 |
| **baseline_logbar** | full stack, reward `log_barrier` | **0.867** | **0.933** |
| A1 linear | strip keypoints (5d state obs) | 0.585 | 0.658 |
| A2 linear | strip frame_stack (FS=1) | 0.795 | 0.821 |
| A1_logbar | strip keypoints | 0.865 | 0.882 |
| A2_logbar | strip frame_stack | 0.854 | 0.922 |
| B_logbar | strip action_repeat (AR=1) | 0.326 | 0.522 |
| bigbonus_logbar | success_bonus 50→200 | 0.914 | 0.933 |
| **minimal_logbar** | state + FS=1 (cross-shape) | **0.906** | **0.939** |

**Effect-size ranking:**
1. **action_repeat** (−41pp stripped) — dominant. Manipulation needs sustained directional force; AR=1 lets policy oscillate and kills push impulse.
2. **log_bar reward** (+8pp) — matches reward curvature to coverage curvature (see below).
3. **keypoints** (−5pp under log_bar; −19pp under linear) — log_bar compensates for obs sparsity.
4. **frame_stack** (−1pp) — marginal under log_bar.
5. **success_bonus** — 50→200 tightens det std 5× but doesn't raise sto ceiling (policy never crosses 0.95 threshold during training → bigger bonus is unsampled).

Ceiling is 0.933 sto (full stack + log_bar) or 0.939 sto (minimal config).

### Why log-barrier reward helps

Linear `r_coverage = cov` gives flat marginal reward: gaining 1 percentage point of coverage earns 0.01 regardless of whether you're at 30% or 90%. But the coverage metric is geometrically nonlinear — 2px displacement drops 5% coverage near goal (see calibration below). Linear reward underpays precision work, policy plateaus at 85%.

`coverage_shape="log_barrier"`: `r = -log(1 - clip(cov/thresh, 0, 1) + ε)`, with `coverage_eps=0.01` giving ceiling 4.6.

| cov | linear | log_bar |
|---|---|---|
| 0.5 | 0.53 | 0.73 |
| 0.7 | 0.74 | 1.30 |
| 0.9 | 0.95 | 2.77 |
| 0.95+ | 1.00 | 4.60 (ceiling) |

Marginal reward ∝ `1/(1-cov+ε)`: 9× steeper at cov=0.9 than cov=0.5. Policy actually chases the last few percent. Q magnitudes grow ~3-5× but remain stable at `reward_scale=0.1, grad_clip_norm=1.0`.

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

!!! tip "Tunable `success_threshold` kwarg"
    The DP default 0.95 termination threshold is too strict for sparse-reward RL — it would give zero learning signal because no policy ever crosses it from scratch. Use a lower threshold for tractable sparse training:
    ```python
    env = PushTEnv(reward_mode="sparse", success_threshold=0.85)
    ```
    Default `0.95` preserves DP/literature parity for evaluation runs. Reporting convention in published push-T work is **max coverage achieved per episode**, not binary success rate — compare against ~0.91 (DP) / ~0.55-0.74 (BC LSTM) numbers, not the 0.95 termination flag.

### Coverage sensitivity (calibrated 2026-04-19)

Coverage is geometrically very sensitive to small pose perturbations because the T has thin bars. Direct env probing (set block pose, measure coverage):

| Coverage | Pos error (x) | Pos error (diag) | Yaw error |
|----------|---------------|------------------|-----------|
| 0.999    | 0 px (identity) | 0 px           | 0°        |
| 0.95     | **1.9 px**    | 2.6 px           | 2.5°      |
| 0.90     | 3.8 px        | 5.3 px           | 5.1°      |
| 0.85     | 5.7 px        | 7.9 px           | 7.8°      |
| 0.80     | 7.8 px        | 10.5 px          | 10.5°     |

A policy at mean 7.5 px position error sits at ~80-85% coverage even with near-perfect yaw. The remaining "final-mile" 2 px is the gap between RL-from-scratch (~85%) and BC-refined methods (>95%).

Generate calibration figure: `PYTHONPATH=. uv run python tools/pusht_coverage_calibration.py`. Output: `.temp/pusht_coverage_calibration.png`.

Also note pymunk CoG offset gotcha: `block.position = (256, 256)` is **NOT** the goal pose. The T's center of gravity is offset (0, 45) from body origin in body frame, so setting body position+angle leaves the actual world COG elsewhere. True identity state (for `reset_to_state`): `[agent_x, agent_y, 224.2, 242.8, π/4]` → coverage 0.9995.

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
