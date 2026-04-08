# Documentation Site — Agent Handoff

The project has a public docs site at **https://stevenwman.github.io/jax-learning/** built with MkDocs + Material theme. Deploys automatically on push to main via `.github/workflows/docs.yml`.

## Quick reference

- **Config:** `mkdocs.yml` (root)
- **Source files:** `docs/` — all website content (22 pages)
- **Assets:** `docs/assets/videos/` (embedded MP4s), `docs/stylesheets/extra.css` (syntax highlighting + layout)
- **Abbreviations:** `docs/includes/abbreviations.md` — global hover tooltips for jargon (UTD, GAE, MLP, etc.)
- **Generators:** `docs/scripts/gen_cli_reference.py`, `docs/scripts/gen_env_presets.py`
- **GH Actions:** `.github/workflows/docs.yml` — triggers on push to main for `docs/**`, `mkdocs.yml`, `jax_rl/**`
- **Deps:** `uv sync --group docs` installs mkdocs-material
- **Review pattern:** `.context/references/docs_review_pattern.md` — 4-persona parallel review

## Commands

```bash
uv sync --group docs              # install docs deps
uv run mkdocs serve               # live reload at localhost:8000
uv run mkdocs build               # build site to site/
uv run python docs/scripts/gen_cli_reference.py   # regenerate CLI flags table
uv run python docs/scripts/gen_env_presets.py      # regenerate presets table
```

## Architecture decisions

### Hand-crafted API pages (not mkdocstrings)

All API pages (`docs/api/*.md`) are hand-written Markdown — **not** auto-generated from docstrings. This was deliberate: mkdocstrings dumped docstring prose as flat unstyled paragraphs with no visual separation. Hand-crafted pages use summary tables, structured sections, import lines, field tables, and definition-list methods.

**Trade-off:** API pages can drift from source code. After changing a constructor signature, config field, or method, update the corresponding API page manually.

The `mkdocstrings` plugin has been removed from `mkdocs.yml`.

### Abbreviation tooltips

`docs/includes/abbreviations.md` defines ~25 terms (UTD, GAE, MLP, MJCF, PD gains, etc.). Via the `abbr` + `pymdownx.snippets` extensions, every occurrence of these terms across the entire site gets a dotted underline and hover tooltip — zero per-page effort.

**To add a new abbreviation:** Edit `docs/includes/abbreviations.md`, add a line like `*[TERM]: Definition here.`

### Glossary cross-links

Key pages (concepts, quickstart, train-locomotion) link first-use jargon to the glossary page (`docs/glossary.md`). Links use MkDocs anchor IDs from `###` headings (e.g., `../glossary.md#utd-ratio-update-to-data`). The glossary is grouped into 4 sections: JAX, RL Fundamentals, Algorithms & Architecture, Environments & Hardware.

### CSS theming

`docs/stylesheets/extra.css` provides:
- **Syntax highlighting:** Separate One Dark (slate) and One Light (default) palettes using `[data-md-color-scheme]` selectors
- **API doc separation:** Top borders between `.doc-object` siblings, left-border indentation for methods
- **Video grid:** `.video-grid` flexbox class for homepage video embedding
- **Table scroll:** `overflow-x: auto` for wide tables on mobile
- **Contrast fixes:** Light-mode comments `#717580` (WCAG AA), video captions use theme-aware `var(--md-default-fg-color--light)`

## Site structure (22 pages)

```
docs/
├── index.md                    # Landing — features, videos, quick links (incl. glossary)
├── glossary.md                 # 28 terms in 4 domain sections, TOC-navigable
├── faq.md                      # Installation, training, recording troubleshooting
├── contributing.md             # Dev setup, tests, code style
├── getting-started/
│   ├── installation.md         # uv setup, GPU deps, verify
│   ├── quickstart.md           # CartpoleBalance in 5 min
│   └── concepts.md             # Three-layer arch, 6 algos, configs, wrappers, RewardSpec/ObsSpec
├── tutorials/
│   ├── train-locomotion.md     # Go2WarpJoystickFlat end-to-end (FastSAC + FlashSAC)
│   ├── custom-env.md           # Adding a new env (BongoHandstand example)
│   ├── custom-rewards.md       # RewardSpec + ObsSpec composability
│   ├── sim2real.md             # Deploy pipeline: train → sim2sim → real robot
│   └── asymmetric-critic.md    # Privileged observations, A/B results, frame stacking
├── api/                        # Hand-crafted API reference (NOT mkdocstrings)
│   ├── algos.md                # Summary table + per-algo sections with constructor/methods
│   ├── envs.md                 # WarpJoystick, BongoHandstand, reward_spec, obs_spec
│   ├── configs.md              # 9 config classes as field tables with types/defaults
│   ├── buffers.md              # JaxReplayBuffer, RolloutBuffer, compute_gae
│   ├── wrappers.md             # Grouped: obs/action wrappers, pipeline, training wrappers
│   └── networks.md             # Builders, encoders, heads, flash blocks
├── reference/
│   ├── cli-flags.md            # AUTO-GENERATED — all CLI args
│   ├── env-presets.md          # AUTO-GENERATED — all presets with HPs
│   ├── architecture.md         # Mermaid diagram, data flow, config, checkpoint format
│   └── lessons-learned.md      # Curated lessons from training
└── includes/
    └── abbreviations.md        # Global tooltip definitions
```

## What auto-updates vs what's manual

### Auto-generated via scripts (run manually)
| Page | Generator | When to re-run |
|---|---|---|
| `docs/reference/cli-flags.md` | `gen_cli_reference.py` | Any argparse change in train scripts |
| `docs/reference/env-presets.md` | `gen_env_presets.py` | Any change in `env_presets.py` |

### Manual pages (edit the markdown directly)
| Page | Goes stale when... |
|---|---|
| `api/*.md` | Constructor signatures, config fields, or methods change |
| `index.md` | New features, new algos, benchmark scores change |
| `concepts.md` | New algo, wrapper, or abstraction added |
| `tutorials/*.md` | API changes, new env patterns, deploy workflow |
| `architecture.md` | System architecture changes (rare) |
| `glossary.md` | New framework-specific terms introduced |
| `abbreviations.md` | New jargon used in docs |

## How to add a new algorithm to the docs

1. **API page:** Add a new `## AlgoName` section in `docs/api/algos.md` with constructor, methods, and summary table row
2. **Config page:** Add a new `## AlgoConfig` section in `docs/api/configs.md` with field table
3. **Network blocks (if any):** Add to `docs/api/networks.md`
4. **CLI flags:** Add `build_new_parser()` to `gen_cli_reference.py`, regenerate
5. **Presets:** Add to `gen_env_presets.py`, regenerate
6. **Concepts page:** Update algo count and table
7. **Index page:** Update algo count in features list
8. **Architecture page:** Add to mermaid diagram and config list
9. **Abbreviations:** Add acronym to `docs/includes/abbreviations.md` if it uses new jargon
10. **Verify:** `uv run mkdocs build`

## How to add a new environment to the docs

1. **API page:** Add section to `docs/api/envs.md` with methods and summary table row
2. **Presets:** If presets exist, add to `gen_env_presets.py` and regenerate
3. **Concepts page:** Mention in environments section if major
4. **Tutorial:** Consider a tutorial if instructive
5. **Video:** Copy to `docs/assets/videos/` and embed

## Review process

Use the 4-persona parallel review pattern documented in `.context/references/docs_review_pattern.md`:
1. High schooler (accessibility)
2. Undergrad CS (factual accuracy — cross-references everything against code)
3. PhD researcher (code correctness, benchmark rigor, adoption readiness)
4. Frontend engineer (visual design, CSS, navigation, accessibility)

Dispatch all 4 as background Opus agents. Compile into prioritized action list grouped by effort (quick/medium/large).

## Videos

Two MP4s embedded:
- `docs/assets/videos/go2_joystick_walk.mp4` — FastSAC eval 276.5
- `docs/assets/videos/go2_bongo_handstand.mp4` — PPO bongo board

Use relative paths from page location (e.g., `../../assets/videos/` from tutorial pages). `.gitignore` has `*.mp4` with exception `!docs/assets/videos/*.mp4`.

## Style guidelines

- **Serious and concise.** No hype, no filler, no marketing.
- Use admonitions (`!!! note`, `!!! tip`, `!!! warning`) sparingly.
- All commands use `uv run python` (never `python` or `python3`).
- Code blocks must have language tags (`python`, `bash`, `yaml`, `text`).
- API pages: summary table at top, structured sections, field tables, definition-list methods.

## mkdocs.yml key config

- Material theme: dark/light toggle, deep purple/amber, robot-industrial logo
- `navigation.tabs`, `navigation.footer`, `navigation.instant`, `content.code.copy`
- `abbr` + `pymdownx.snippets` with `auto_append` for global abbreviation tooltips
- `pymdownx.superfences` with Mermaid fence support
- `pymdownx.highlight` with Pygments (One Dark / One Light in `extra.css`)
- `site_url: https://stevenwman.github.io/jax-learning/`
- **No mkdocstrings** — all API pages are hand-crafted

## Common gotchas

- **API page drift:** Hand-crafted API pages can fall behind code changes. After modifying signatures or defaults, update the corresponding `docs/api/*.md` page.
- **Superpowers files leaking into `docs/`:** Other agents may write to `docs/superpowers/`. Move to `.superpowers/`.
- **Video 404s:** Use relative paths that go up from the page's directory (e.g., `../../assets/videos/` from `tutorials/page/`). Absolute paths break on GH Pages at `/jax-learning/`.
- **`*.mp4` gitignore:** Videos in `docs/assets/videos/` are tracked via exception. New videos elsewhere will be ignored.
- **Light-mode comment contrast:** Must be at least `#717580` to pass WCAG AA on white backgrounds.
