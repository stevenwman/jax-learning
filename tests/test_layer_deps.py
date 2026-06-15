"""Layer-dependency lint — enforces the one-way import rule from the repo
redesign (docs/redesign-linen-refactor.html):

    jax_rl/algos  ←  jax_rl/envs  ←  projects/

The library core (``jax_rl/``) is reusable and experiment-agnostic; research
code (``projects/``) consumes it. The arrow must never point back: if core
imports an experiment, the experiment's vocabulary ("mud", a specific reward)
has bled into the library. This test is the CI guard that keeps that from
silently regressing during the refactor.

Scope note: this checks the *source-level* rule (no ``import projects`` /
``from projects`` statements under ``jax_rl/``). It is deliberately simple and
fast — a string scan, not an import-graph build — so it can't be fooled into a
false pass but also won't catch dynamic/`importlib` imports (none exist today).
"""
from __future__ import annotations

import re
from pathlib import Path

# repo root = two levels up from this file (tests/ -> repo).
_REPO = Path(__file__).resolve().parents[1]
_CORE = _REPO / "jax_rl"

# Matches `import projects...` or `from projects... import ...` at any indent.
_BLEED = re.compile(r"^\s*(?:import\s+projects\b|from\s+projects\b)", re.MULTILINE)


def test_core_does_not_import_projects():
    """No file under jax_rl/ may import projects/ (the one-way-dep rule)."""
    offenders = []
    for py in _CORE.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        if _BLEED.search(text):
            offenders.append(str(py.relative_to(_REPO)))
    assert not offenders, (
        "jax_rl/ (library core) must not import projects/ (experiments). "
        "Move the shared code into jax_rl/, or invert the dependency.\n"
        "Offending files:\n  " + "\n  ".join(offenders)
    )
