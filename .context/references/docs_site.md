# Documentation Site — Agent Handoff

The project has a public-facing docs site built with MkDocs + Material theme. This doc tells you everything you need to maintain or extend it.

## Quick reference

- **Config:** `mkdocs.yml` (root)
- **Source files:** `docs/` — all website content
- **Assets:** `docs/assets/videos/` (embedded MP4s), `docs/stylesheets/code.css` (syntax highlighting)
- **Generators:** `docs/scripts/gen_cli_reference.py`, `docs/scripts/gen_env_presets.py`
- **GH Actions:** `.github/workflows/docs.yml` (dormant — no repo yet)
- **Deps:** `uv sync --group docs` installs mkdocs-material + mkdocstrings
- **Superpowers specs/plans:** `.superpowers/` (NOT `docs/superpowers/` — override in CLAUDE.md)

## Commands

```bash
uv sync --group docs              # install docs deps
uv run mkdocs serve               # live reload at localhost:8000
uv run mkdocs build --strict      # build, warnings = errors
uv run python docs/scripts/gen_cli_reference.py   # regenerate CLI flags table
uv run python docs/scripts/gen_env_presets.py      # regenerate presets table
```

## Site structure (20 pages)

```
docs/
├── index.md                    # Landing page — features, videos, quick links
├── getting-started/
│   ├── installation.md         # uv setup, GPU deps, verify
│   ├── quickstart.md           # CartpoleBalance in 5 min
│   └── concepts.md             # Three-layer arch, 6 algos, configs, wrappers, RewardSpec/ObsSpec
├── tutorials/
│   ├── train-locomotion.md     # Go2WarpJoystickFlat end-to-end (FastSAC + FlashSAC)
│   ├── custom-env.md           # Adding a new env (BongoHandstand example)
│   ├── custom-rewards.md       # RewardSpec + ObsSpec composability
│   ├── sim2real.md             # Deploy pipeline: train → sim2sim → real robot
│   └── asymmetric-critic.md   # Privileged observations, A/B results
├── api/                        # All autodoc via mkdocstrings ::: directives
│   ├── algos.md                # PPO, SAC, TD3, FastSAC, FastTD3, FlashSAC
│   ├── envs.md                 # WarpJoystick, BongoHandstand, reward_spec, obs_spec
│   ├── configs.md              # TrainConfig, PPOConfig, SACConfig, TD3Config, FastSACConfig, FastTD3Config, FlashSACConfig, EncoderConfig, PolicyHeadConfig
│   ├── buffers.md              # JaxReplayBuffer, FrameStackConfig, RolloutBuffer, compute_gae
│   ├── wrappers.md             # FrameStack, ActionDelay, pipeline, training wrappers
│   └── networks.md             # Actor, DeterministicActor, VCritic, MlpEncoder, heads, flash_blocks
├── reference/
│   ├── cli-flags.md            # AUTO-GENERATED — all CLI args for 4 train scripts + record_video
│   ├── env-presets.md          # AUTO-GENERATED — all presets with HPs and eval scores
│   ├── architecture.md         # Mermaid diagram, data flow, config system, checkpoint format
│   └── lessons-learned.md      # Curated lessons (PPO, off-policy, distributional, JAX, sim2real)
└── contributing.md             # Dev setup, tests, code style, PR process (placeholder)
```

## What auto-updates vs what's manual

### Auto-updates from code (mkdocstrings at build time)
All `docs/api/*.md` pages pull docstrings from `jax_rl/` source via `::: module.path` directives. Change a docstring, rebuild, docs update. `__init__` methods are filtered out.

### Auto-generated via scripts (run manually)
| Page | Generator | When to re-run |
|---|---|---|
| `docs/reference/cli-flags.md` | `gen_cli_reference.py` | Any argparse change in train scripts |
| `docs/reference/env-presets.md` | `gen_env_presets.py` | Any change in `env_presets.py` |

**Caveat:** The CLI generator mirrors argparse definitions (can't import them from `if __name__ == "__main__"` blocks). If you add a flag to a training script, update the matching `build_*_parser()` function in the generator too.

### Manual pages (edit the markdown directly)
| Page | Goes stale when... |
|---|---|
| `index.md` | New features, new algos, benchmark scores change |
| `concepts.md` | New algo, wrapper, or abstraction added |
| `tutorials/*.md` | API changes, new env patterns, deploy workflow |
| `architecture.md` | System architecture changes (rare) |
| `lessons-learned.md` | New lessons in `.context/lessons/` |

## How to add a new algorithm to the docs

When a new algo is added (like FlashSAC was):

1. **API autodoc:** Add `::: jax_rl.algos.new_algo.NewAlgo` to `docs/api/algos.md` with `filters: ["!__init__"]`
2. **Config autodoc:** Add `::: jax_rl.configs.new_config.NewConfig` to `docs/api/configs.md`
3. **Network blocks (if any):** Add `::: jax_rl.networks.new_module` to `docs/api/networks.md`
4. **CLI flags:** Add `build_new_parser()` to `docs/scripts/gen_cli_reference.py`, add to `sections` list, regenerate
5. **Presets:** Add `NEW_PRESETS` import + `render_offpolicy_presets(...)` call to `docs/scripts/gen_env_presets.py`, regenerate
6. **Concepts page:** Update algo count and table in `docs/getting-started/concepts.md`
7. **Index page:** Update algo count in features list
8. **Tutorials:** Mention in relevant tutorials (e.g., locomotion tutorial's "Next Steps")
9. **Verify:** `uv run mkdocs build --strict`

## How to add a new environment to the docs

1. **API autodoc:** Add `::: jax_rl.envs.locomotion.new_env.NewEnvClass` to `docs/api/envs.md`
2. **Presets:** If presets exist, add to `gen_env_presets.py` and regenerate
3. **Concepts page:** Mention in environments section if it's a major env
4. **Tutorial:** Consider a tutorial if the env is instructive (like bongo board)
5. **Video:** If a good rollout video exists, copy to `docs/assets/videos/` and embed

## Videos

Two MP4s embedded (landing page + tutorials):
- `docs/assets/videos/go2_joystick_walk.mp4` — FastSAC eval 276.5
- `docs/assets/videos/go2_bongo_handstand.mp4` — PPO bongo board

Embedded via `<video autoplay loop muted playsinline>` tags. Use absolute paths (`/assets/videos/...`) from pages in subdirectories to avoid 404s.

Note: `.gitignore` has `*.mp4` but with exception `!docs/assets/videos/*.mp4`.

## Style guidelines

- **Serious and concise.** No hype ("blazing fast"), no filler ("let's dive in"), no marketing ("from zero to X in N minutes").
- Use mkdocs admonitions (`!!! note`, `!!! tip`, `!!! warning`) sparingly and for genuinely useful info.
- All commands use `uv run python` (never `python` or `python3`).
- Syntax highlighting: custom CSS in `docs/stylesheets/code.css` (One Dark palette). All code blocks must have language tags (`python`, `bash`, `yaml`, etc.).

## mkdocs.yml key config

- Material theme: dark/light toggle, nav tabs, search, code copy
- `pymdownx.emoji` via `material.extensions.emoji` (not deprecated `materialx`)
- `pymdownx.superfences` with Mermaid fence support
- `pymdownx.highlight` with Pygments
- `mkdocstrings` python handler: `paths: [.]`, `show_source: true`, `docstring_style: google`, `warn_unknown_params: false`
- Custom CSS: `extra_css: [stylesheets/code.css]`

## Deployment (deferred)

No GitHub repo yet. When one exists:
1. Set `repo_url` in `mkdocs.yml`
2. Uncomment the `on: push` trigger in `.github/workflows/docs.yml`
3. Enable GitHub Pages (Settings → Pages → Source: GitHub Actions)
4. Push to main

## Common gotchas

- **Superpowers files leaking into `docs/`:** Other agents may ignore the CLAUDE.md override and write specs/plans to `docs/superpowers/`. Move them to `.superpowers/` and delete `docs/superpowers/`. The build warns about unnavigated files.
- **mkdocstrings `Attributes:` vs `Args:`:** For Flax `nn.Module` classes, use `Attributes:` section header in docstrings (not `Args:`). Griffe doesn't recognize dataclass-style fields as constructor params.
- **Video 404s in subdirectories:** Use absolute paths (`/assets/videos/file.mp4`) not relative (`../assets/videos/file.mp4`). MkDocs `use_directory_urls` makes relative paths resolve wrong.
- **`*.mp4` gitignore:** Videos in `docs/assets/videos/` are tracked via `!docs/assets/videos/*.mp4` exception. New videos elsewhere will be ignored by git.
