# Documentation Site Agent Prompt

---

You are building the public-facing documentation website for **jax-learning**, a JAX-based reinforcement learning framework for robot control (Unitree Go2 quadruped).

## Setup

1. Read the design spec: `.superpowers/specs/2026-04-06-docs-site-design.md`
2. Read the implementation plan: `.superpowers/plans/2026-04-06-docs-site.md`
3. Read the project overview: `.context/AGENT_HANDOFF.md`

These three docs give you everything you need. Do NOT pre-read all of `.context/` — use the lookup pattern from `NEW_AGENT_PROMPT.md` (read on demand).

## Quick facts

- `uv run python` (not python3)
- No Co-Authored-By in commits
- MkDocs + Material theme + mkdocstrings[python]
- Install docs deps: `uv sync --group docs`
- Build: `uv run mkdocs build --strict`
- Serve: `uv run mkdocs serve`
- Existing `docs/superpowers/` must be excluded from built site

## Current project state (as of 2026-04-08)

### Algorithms (6 total)
| Algo | Script | Key feature |
|------|--------|-------------|
| PPO | `train_ppo.py`, `train_ppo_fast.py` | On-policy, asymmetric critic |
| SAC | `train_offpolicy.py --algo sac` | Vanilla off-policy |
| TD3 | `train_offpolicy.py --algo td3` | Deterministic off-policy |
| FastTD3 | `train_offpolicy.py --algo fast_td3` | C51 distributional + TD3 |
| FastSAC | `train_offpolicy.py --algo fast_sac` | C51 distributional + SAC, eval 276.5 Go2 |
| **FlashSAC** | `train_flashsac.py` | Inverted residual + BatchNorm + weight norm, eval 282.4 Go2 |

### Environments
- **Go2WarpJoystickFlat** — primary locomotion env (Warp backend, unitree MJCF)
- **Go2BongoHandstand** — bongo board balance task
- MuJoCo Playground: CartpoleBalance, CheetahRun, WalkerWalk, HumanoidRun

### Key infrastructure
- 5 root scripts: `train_ppo_fast.py`, `train_ppo.py`, `train_offpolicy.py`, `train_flashsac.py`, `record_video.py`
- Replay buffer: `jax_rl/buffers/jax_replay_buffer.py`
- Env presets: `jax_rl/configs/env_presets.py`
- Network blocks: `jax_rl/networks/` (MLP encoders, distributional heads, FlashSAC blocks)
- Wrappers: `jax_rl/envs/wrappers/` (frame stack, action delay, DR, training wrappers)
- Tests: 243 tests across `tests/`

### API docs — what to autodoc
The `api/` pages should use mkdocstrings to render docstrings from:
- `jax_rl/algos/*.py` — all 6 algorithm classes
- `jax_rl/envs/locomotion/*.py` — Go2 envs
- `jax_rl/configs/*.py` — all config dataclasses
- `jax_rl/buffers/*.py` — replay buffer
- `jax_rl/envs/wrappers/*.py` — wrapper classes
- `jax_rl/networks/builders.py`, `jax_rl/networks/encoders/`, `jax_rl/networks/heads/`, `jax_rl/networks/flash_blocks.py`

### Content sources
Synthesize tutorial/reference prose from these `.context/` docs (read on demand, don't pre-load):
- `.context/AGENT_HANDOFF.md` — project overview, codebase map, benchmarks
- `.context/lessons/*.md` — framework lessons (PPO, off-policy, distributional, etc.)
- `.context/go2/` — Go2-specific docs
- `.context/references/` — vision RL design, RL framework plan
- `deploy/README.md` — deployment pipeline

### FlashSAC-specific docs (NEW — not in original spec)
The docs site spec was written before FlashSAC was ported. Add FlashSAC to:
- `api/algos.md` — autodoc `jax_rl.algos.flash_sac.FlashSAC`
- `api/configs.md` — autodoc `jax_rl.configs.flash_sac_config.FlashSACConfig`
- `api/networks.md` — autodoc `jax_rl.networks.flash_blocks` (FlashSACBlock, normalize_weights, etc.)
- `reference/env-presets.md` — FlashSAC presets table
- `reference/cli-flags.md` — `train_flashsac.py` CLI args
- `tutorials/train-locomotion.md` — mention FlashSAC as alternative to FastSAC for Go2

## Success criteria

1. `uv run mkdocs build --strict` passes
2. `uv run mkdocs serve` shows navigable site with all pages
3. API pages render docstrings from `jax_rl` modules (including FlashSAC)
4. No internal docs (`.context/`, `docs/superpowers/`) leak into public site
5. A new user can follow getting-started → quickstart logically

## Execution

Follow the implementation plan task by task. Each task has exact file paths and content descriptions. The plan was already reviewed and approved.

Say "Ready" and wait for instructions.
