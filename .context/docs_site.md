# Documentation Site

The project has a public-facing docs site built with MkDocs + Material theme.

## Quick reference

- **Config:** `mkdocs.yml` (root)
- **Source files:** `docs/` (all website content; superpowers specs/plans live in `.superpowers/`, not here)
- **Assets:** `docs/assets/videos/` (embedded MP4s)
- **Generators:** `docs/scripts/gen_cli_reference.py`, `docs/scripts/gen_env_presets.py`
- **GH Actions:** `.github/workflows/docs.yml` (dormant — no repo yet)
- **Deps:** `uv sync --group docs` installs mkdocs-material + mkdocstrings

## Local dev

```bash
uv run mkdocs serve          # live reload at localhost:8000
uv run mkdocs build --strict  # build, warnings = errors
```

## What auto-updates vs what's manual

### Auto-updates from code (via mkdocstrings at build time)

All 6 API reference pages (`docs/api/*.md`) pull docstrings from `jax_rl/` source. Change a docstring → rebuild → docs update.

### Auto-generated via scripts (run manually when code changes)

| Page | Generator | When to re-run |
|---|---|---|
| `docs/reference/cli-flags.md` | `uv run python docs/scripts/gen_cli_reference.py` | Any argparse change in `train_offpolicy.py`, `train_ppo_fast.py`, or `record_video.py` |
| `docs/reference/env-presets.md` | `uv run python docs/scripts/gen_env_presets.py` | Any change in `jax_rl/configs/env_presets.py` |

**Important:** The generators mirror the argparse definitions (they don't import the parsers directly because they're inside `if __name__ == "__main__"` blocks). If you add/remove/rename a CLI flag, update both the training script AND the corresponding parser in `docs/scripts/gen_cli_reference.py`.

### Fully manual (must edit the markdown)

| Page | Goes stale when... |
|---|---|
| `docs/index.md` | Benchmark scores change, new features added |
| `docs/getting-started/quickstart.md` | Training commands or expected output changes |
| `docs/getting-started/concepts.md` | New abstractions added (new wrapper, algo, etc.) |
| `docs/tutorials/*.md` | API changes, new env patterns, deploy workflow changes |
| `docs/reference/architecture.md` | System architecture changes (rare) |
| `docs/reference/lessons-learned.md` | New lessons in `.context/lessons/` |
| `docs/contributing.md` | Process or conventions change |

## Videos

Two MP4s embedded on the site (landing page, tutorials):
- `docs/assets/videos/go2_joystick_walk.mp4` — FastSAC eval 276.5 locomotion
- `docs/assets/videos/go2_bongo_handstand.mp4` — PPO bongo board handstand

Embedded via `<video autoplay loop muted playsinline>` tags.

## Deployment (deferred)

No GitHub repo yet. When one exists:
1. Set `repo_url` in `mkdocs.yml`
2. Uncomment the `on: push` trigger in `.github/workflows/docs.yml`
3. Enable GitHub Pages (Settings → Pages → Source: GitHub Actions)
4. Push to main

## Doc sync rule

When modifying code, check if docs need updating:

| Code area | Also update |
|---|---|
| `jax_rl/algos/*.py` docstrings | API docs auto-update on rebuild |
| `train_*.py` argparse | Run `docs/scripts/gen_cli_reference.py` |
| `jax_rl/configs/env_presets.py` | Run `docs/scripts/gen_env_presets.py` |
| New env or wrapper | `docs/getting-started/concepts.md`, possibly new tutorial |
| New algo | `docs/api/algos.md` (add `::: jax_rl.algos.new_algo.NewAlgo`), `docs/index.md` |
| Deploy workflow | `docs/tutorials/sim2real.md` |
| New lesson in `.context/lessons/` | Consider adding to `docs/reference/lessons-learned.md` |
