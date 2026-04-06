# Documentation Site Design — jax-learning

**Date:** 2026-04-06
**Status:** Approved
**Author:** Steven + Claude

---

## Overview

Public-facing documentation website for jax-learning, a JAX-based RL framework for robot learning (Unitree Go2 quadruped). Target audience: undergrads, high schoolers, researchers who want to train RL policies and deploy them on real robots.

Built with MkDocs + Material theme. Local-only for now — GitHub Pages deployment deferred until a permanent repo home is chosen.

## Decisions

- **Tool:** MkDocs + Material theme + mkdocstrings[python]
- **Deployment:** Local only (`mkdocs serve`). GH Actions workflow created but dormant (TODO comment).
- **Scope:** Additive only — no existing files deleted or modified except `pyproject.toml` (new optional dep group).
- **Content depth:** Structural first pass — all pages exist with correct headings, navigation works, API autodoc renders. Prose synthesized from `.context/` docs. Not final-draft quality.
- **Internal docs:** `docs/superpowers/` excluded from built site via mkdocs config. Not touched.

## Dependencies (pyproject.toml)

```toml
[dependency-groups]
docs = [
    "mkdocs-material>=9.6",
    "mkdocstrings[python]>=0.29",
]
```

Matches existing `[dependency-groups]` pattern (uv-native). Install with `uv sync --group docs`.

## Site Structure

```
docs/
├── index.md                    # Landing page — what is jax-learning, why it exists, key results
├── getting-started/
│   ├── installation.md         # uv setup, GPU deps (JAX CUDA), verify installation
│   ├── quickstart.md           # Train CartpoleBalance in 5 min, see eval, record video
│   └── concepts.md             # Key abstractions: envs, algos, configs, wrappers, RewardSpec, ObsSpec
├── tutorials/
│   ├── train-locomotion.md     # Go2WarpJoystickFlat end-to-end
│   ├── custom-env.md           # Add a new env (go2_bongo_handstand as example)
│   ├── custom-rewards.md       # RewardSpec + ObsSpec composability
│   ├── sim2real.md             # Deploy pipeline: train → checkpoint → sim2sim → deploy_go2.py → real robot
│   └── asymmetric-critic.md   # When and how to use privileged observations
├── api/
│   ├── algos.md                # Auto-generated: PPO, SAC, TD3, FastSAC, FastTD3
│   ├── envs.md                 # Go2 envs, bongo board, env registration
│   ├── configs.md              # TrainConfig, PPOConfig, SACConfig, FastSACConfig, etc.
│   ├── buffers.md              # JaxReplayBuffer, RolloutBuffer
│   ├── wrappers.md             # jax_rl/envs/wrappers/ — FrameStack, ActionDelay, training, pipeline
│   └── networks.md             # Encoder, heads (Gaussian, Deterministic, Value, Q, Distributional)
├── reference/
│   ├── cli-flags.md            # All CLI args for train scripts + record_video
│   ├── env-presets.md          # Table of all presets with proven HPs and eval scores
│   ├── architecture.md         # System architecture, data flow, how pieces connect
│   └── lessons-learned.md      # Curated selection from .context/lessons/
└── contributing.md             # How to contribute, run tests, code style
```

## mkdocs.yml Configuration

- Material theme: dark/light toggle, navigation tabs, search, code copy button
- mkdocstrings: python handler pointed at `jax_rl` package
- Repo URL: placeholder (`# TODO: set repo_url when repo is finalized`)
- `docs/superpowers/` excluded via `exclude_docs: superpowers/` (prevents build, not just nav hiding)
- Navigation structure mirrors the file tree above

## GitHub Actions (dormant)

`.github/workflows/docs.yml` — triggers on push to main for `docs/**`, `mkdocs.yml`, `jax_rl/**`. Uses `mkdocs gh-deploy`. File created with a `# TODO` header noting it's inactive until repo is set up.

## Deferred Work

- [ ] Finalize GitHub repo home → uncomment GH Actions workflow, set `repo_url`
- [ ] Polish prose content (second pass after skeleton is validated)
- [ ] Add screenshots/videos of trained policies if available
- [ ] Custom 404 page, favicon, logo

## Success Criteria

- `uv run mkdocs build --strict` passes (warnings = errors, catches broken autodoc refs)
- `uv run mkdocs serve` shows navigable site with all pages
- API pages render docstrings from `jax_rl` modules
- No internal docs (`.context/`, `docs/superpowers/`) leak into public site
- A new user can follow getting-started → quickstart logically
