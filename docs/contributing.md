# Contributing

## Development Setup

```bash
# Clone the repo
git clone <repo-url>
cd jax-learning

# Install all dependencies (requires Python 3.13+ and NVIDIA GPU)
uv sync --group dev

# Verify
uv run python -c "import jax; print(jax.devices())"
```

## Tests

The default lane is CPU-only and hermetic — runs ~600 tests in 1-2 min:

```bash
uv run python -m pytest
```

To run subsets:

```bash
uv run python -m pytest -m gpu              # MJX/CUDA tests (~200)
uv run python -m pytest -m "warp or go2"   # Go2 Warp surface (~70)
uv run python -m pytest -m deploy           # Deploy contract tests
uv run python -m pytest -m "not network"   # Everything except network-bound
uv run python -m pytest -m slow             # Long-running (CPU or GPU)
```

Markers are defined in `pyproject.toml`. See `pytest --markers` for
the live list with descriptions.

## Code Style

### Architecture principles

- **No base classes.** Algorithms are self-contained — no `BaseAlgorithm` ABC, no `Trainer` class. Shared utilities live in `jax_rl/networks/builders.py` and `jax_rl/utils/`.
- **Envs are black boxes.** Algorithms never import from env modules. Training scripts are the glue layer.
- **Configs over code.** Hyperparameters live in dataclass configs (`jax_rl/configs/`), not scattered in training loops.

### Conventions

- Always use `uv run python` — never `python` or `python3`
- Use [conventional commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
- Keep functions pure where possible — JAX JIT requires functional style
- Use `jnp.where` for conditional logic inside JIT'd functions, not Python `if`
- Use `@flax.struct.dataclass` for training state, not `@dataclass`

## Building Docs

```bash
# Install docs dependencies
uv sync --group docs

# Serve locally (auto-reloads on changes)
uv run mkdocs serve

# Build static site
uv run mkdocs build --strict
```

## Pull Request Process

!!! note
    PR process will be defined once the project has a permanent GitHub home. For now, work on feature branches and coordinate with the maintainer.
