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

## Running Tests

```bash
# Full test suite (~241 tests)
uv run python -m pytest tests/ -v

# Run a specific test file
uv run python -m pytest tests/test_replay_buffer.py -v

# Run a specific test
uv run python -m pytest tests/test_replay_buffer.py::test_add_and_sample -v
```

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
