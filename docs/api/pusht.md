# PushT (vendored gym-pusht)

Diffusion Policy push-T benchmark (Chi et al. 2023), vendored from HuggingFace [gym-pusht](https://github.com/huggingface/gym-pusht) into the repo so we can pin the physics, extend rewards for RL, and bundle the expert demos. Apache 2.0, license retained.

Old, static benchmark. The point of vendoring is that old benchmarks **should** be static — no floating pip version, no silent API breaks from upstream deps (pymunk 7 already broke upstream once), reproducible across years.

!!! note "Location"
    `jax_rl/envs/manipulation/pusht/` — env + contact helper + LICENSE + 206 expert demos (0.29 MB).

---

## Usage

```python
from jax_rl.envs.manipulation.pusht import PushTEnv

env = PushTEnv(obs_type="state", reward_mode="shaped")
obs, info = env.reset(seed=0)
for _ in range(300):
    action = env.action_space.sample()     # (2,), XY target in [0, 512] px
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break
```

- Single-env, CPU, pymunk 2D physics. ~2k sps single process.
- `action_space = Box([0, 0], [512, 512])` — absolute agent target in pixel coords.
- `observation_space` varies by `obs_type`: `"state"` (5d), `"pixels"` (96×96×3), `"pixels_agent_pos"` (dict), `"environment_state_agent_pos"` (dict).

---

## Reward modes

Added to the vendored env via `reward_mode=` kwarg. Coverage is the DP default; the rest are for RL-from-scratch experiments where pure coverage is too sparse.

| Mode | Reward per step | Use case |
|------|-----------------|----------|
| `coverage` (default) | `clip(coverage / 0.95, 0, 1)` — DP's exact formula | Matches DP paper baseline; sparse for pure RL |
| `sparse` | `1.0 if coverage > 0.95 else 0.0` | Cleanest signal; hardest to learn from scratch |
| `shaped` | `coverage + 0.01 * n_contacts - 0.001 * pusher_to_block` | Encourages contact + approach |
| `approach` | `coverage + 0.05 * (1 - pusher_to_block / diag)` | Adds smooth proximity bonus to coverage |

All modes terminate on `coverage > 0.95` per DP convention. `info` dict always includes `coverage`, `is_success`, `pusher_to_block`, `n_contact_points` so post-hoc analysis works regardless of mode.

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
